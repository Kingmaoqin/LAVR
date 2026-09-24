"""
Task-utility oracle labels on the Nemotron backbone.

THE LABEL THE BRIEF ASKED FOR AND THE FIRST TWO BACKBONES COULD NOT PROVIDE
    A_task(i | s_t) = P(correct answer | commit (i, x_hat_i) at s_t, then pi_ref)
                    - P(correct answer |                              pi_ref)
estimated by CRN-coupled paired rollouts run to COMPLETION, with the reward
being exact-match on the GSM8K final numeric answer.

WHY ROLLING TO COMPLETION REMOVES A WHOLE BUG CLASS
A task reward only exists once a full answer exists, so both branches must run
to the end. Both therefore fill EVERY remaining masked position, which makes
them automatically matched in the number of positions consumed. The denominator
mismatch and the depletion artifact that had to be repaired on the MDLM
backbone cannot arise here by construction.

Both labels come from the SAME rollouts:
  * A_task  -- the binary task-utility advantage (the new external-validity target)
  * A_full  -- the Path-LL advantage, comparable in spirit to the other backbones
"""
import re
from dataclasses import dataclass

import numpy as np
import torch

from features import PROJ_DIM, ProjCache, TOPK_FEATS
from nemotron_local import MASK_TOKEN_ID
from nemotron_policy import (BlockPiRefConfig, block_rollout, make_state,
                             topk_logprobs, sample_tokens, expected_logp,
                             active_block)

SUFFIX = "\n\nEnd your reply with the final numeric answer alone on the last line."


@dataclass
class TaskCollectConfig:
    gen_len: int = 160
    block_length: int = 32
    threshold: float = 0.9
    temperature: float = 0.2
    top_k: int = 50
    record_fracs: tuple = (0.15, 0.35, 0.55, 0.75)
    n_cand: int = 4
    n_cand_conf: int = 2
    K: int = 8
    rollout_batch: int = 2       # hard memory ceiling on the shared GPUs
    store_layers: tuple = tuple(range(0, 26, 2))   # 13 of 26, to bound storage
    seed_base: int = 7_777_777

    def pi(self):
        return BlockPiRefConfig(block_length=self.block_length,
                                threshold=self.threshold,
                                temperature=self.temperature,
                                top_k=self.top_k)


def gold_answer(a):
    return a.split("####")[-1].strip().replace(",", "")


def extract_number(s):
    n = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return n[-1].rstrip(".") if n else None


def task_reward(text, g):
    p = extract_number(text)
    if p is None:
        return 0.0
    try:
        return float(abs(float(p) - float(g)) < 1e-6)
    except ValueError:
        return 0.0


@torch.no_grad()
def project_logprobs(logits, device, chunk=8):
    """Fixed JL projection of the full log-probability vector (control C3)."""
    V = logits.shape[-1]
    P = ProjCache.get(V, device)
    colsum = P.sum(0)
    B, L, _ = logits.shape
    out = torch.empty(B, L, P.shape[1], dtype=torch.float32, device=device)
    for a in range(0, L, chunk):
        sl = logits[:, a:a + chunk].float()
        lse = sl.logsumexp(-1)
        out[:, a:a + chunk] = sl @ P - lse[..., None] * colsum
    return out


@torch.no_grad()
def snapshot(model, ids, mask, step, n_steps, hist, cfg, gen_start):
    """Everything measurable at state s_t, before any rollout."""
    logits, hs = model(ids, output_hidden_states=True)
    lp_top, idx = topk_logprobs(logits, cfg.top_k, 8)
    # Keep the actual raw top-k logits.  The previous implementation copied
    # log-probabilities here, making C1.logit_max a duplicate of logp1 and
    # weakening the cheap baseline on task data.
    lg_top = logits.float().gather(-1, idx)
    proj = project_logprobs(logits, ids.device)
    argmax = idx[..., 0]
    p1 = lp_top[..., 0].exp()

    if hist["prev_lp"] is None:
        tkl = torch.zeros_like(p1); dp1 = torch.zeros_like(p1)
    else:
        # Temporal KL over the current top-k support. The vocabulary is 131072,
        # so a dense previous-step distribution is not affordable; instead look
        # each current top-k token id up inside the PREVIOUS step's top-k ids
        # and fall back to a floor when it was not in that set. Looking it up
        # with `gather` would be a bounds error -- `idx` holds vocab ids, while
        # the stored tensor has only k columns.
        prev_idx, prev_lp = hist["prev_idx"], hist["prev_lp"]
        srt, order = torch.sort(prev_idx, dim=-1)
        pos = torch.searchsorted(srt, idx.contiguous())
        pos = pos.clamp(max=srt.shape[-1] - 1)
        hit = srt.gather(-1, pos) == idx
        prev = prev_lp.gather(-1, order.gather(-1, pos))
        prev = torch.where(hit, prev, torch.full_like(prev, -30.0))
        p = lp_top.exp()
        tkl = (p * (lp_top - prev)).sum(-1)
        dp1 = p1 - hist["prev_p1"]

    flip_count, persistence = history_counts_at_state(hist, argmax, mask)
    snap = {"ids": ids.clone(), "mask": mask.clone(), "step": step,
            "n_steps": n_steps, "lp_top": lp_top, "lg_top": lg_top,
            "idx": idx, "proj": proj, "argmax": argmax,
            "temporal_kl": tkl, "dp1": dp1,
            "flip_count": flip_count.clone(),
            "persistence": persistence.clone(),
            "hidden": [hs[l] for l in cfg.store_layers],
            "prefix_len": gen_start}
    return snap, (lp_top, idx, p1, argmax)


def new_hist(B, L, device):
    return {"prev_lp": None, "prev_idx": None, "prev_p1": None,
            "prev_argmax": None,
            "flip_count": torch.zeros(B, L, device=device),
            "persistence": torch.zeros(B, L, device=device)}


def history_counts_at_state(hist, argmax, mask):
    """Stability counters including the transition into the current state."""
    flip_count = hist["flip_count"]
    persistence = hist["persistence"]
    if hist["prev_argmax"] is not None:
        flip = (argmax != hist["prev_argmax"]).float()
        flip_count = flip_count + flip * mask.float()
        persistence = torch.where(
            flip.bool(), torch.zeros_like(persistence), persistence + 1.0)
    return flip_count, persistence


def update_hist(hist, lp_top, idx, p1, argmax, mask):
    hist["flip_count"], hist["persistence"] = history_counts_at_state(
        hist, argmax, mask)
    # keep only the top-k slice with its ids; a dense (B, L, 131072) history
    # would be ~70 GB and is not affordable
    hist["prev_lp"] = lp_top.clone()
    hist["prev_idx"] = idx.clone()
    hist["prev_p1"] = p1
    hist["prev_argmax"] = argmax


def pick_candidates(snap, cfg, gen):
    """Half uniformly-random masked positions, half high-confidence x unstable."""
    mask = snap["mask"]
    B, L = mask.shape
    dev = mask.device
    n_nat = cfg.n_cand - cfg.n_cand_conf
    conf = snap["lp_top"][..., 0]
    instab = snap["temporal_kl"] + 0.5 * snap["flip_count"]
    info = torch.where(mask, conf + instab, torch.full_like(conf, -1e30))
    rnd = torch.where(mask, torch.rand(mask.shape, generator=gen, device=dev),
                      torch.full_like(conf, -1e30))
    cb, ci, cs = [], [], []
    for b in range(B):
        nm = int(mask[b].sum())
        if nm == 0:
            continue
        k_nat = min(n_nat, nm)
        nat = rnd[b].topk(k_nat).indices
        k_inf = min(cfg.n_cand_conf, nm - k_nat)
        if k_inf > 0:
            sc = info[b].clone(); sc[nat] = -1e30
            inf = sc.topk(k_inf).indices
        else:
            inf = torch.empty(0, dtype=torch.long, device=dev)
        for p in nat.tolist():
            cb.append(b); ci.append(p); cs.append(0)
        for p in inf.tolist():
            cb.append(b); ci.append(p); cs.append(1)
    return (torch.tensor(cb, device=dev), torch.tensor(ci, device=dev),
            np.array(cs, dtype=np.int8))

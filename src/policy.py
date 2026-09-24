"""
Reference decoding policy pi_ref and POKE-style Path-LL accounting for MDLM.

================= FROZEN ESTIMAND =======================
Fixed before any probe was fitted. Do not change after label collection starts.

pi_ref -- MDLM's native ancestral (reverse-diffusion) sampler, one position per
step, which is the T = |M| limit of the absorbing-state reverse process:

  * unmasking ORDER: a uniformly random permutation of the masked positions.
    Realised as a per-rollout random key `order_noise[pos]`; at each step the
    masked position with the largest key is unmasked. This is exactly a uniform
    random order, and it makes the order identical across coupled branches.
  * TOKEN at the chosen position: sampled from the top_k=50 truncated,
    temperature-1.0 softmax, via Gumbel-max with noise keyed by TOKEN IDENTITY
    (not by rank), so coupled branches that agree about a token draw it alike.
  * Substrate note (Phase S): confidence-ordered decoding COLLAPSES into
    repetition loops on MDLM-owt (distinct-2 0.22 vs 0.86 for real text), so a
    confidence-ordered pi_ref would fail the coherence gate. The ancestral
    sampler is the model's own generative process and is coherent. A further
    benefit: the reference order is position-agnostic, so the resulting
    advantage label is not mechanically entangled with confidence.

G -- POKE-style Path-LL. Every commitment contributes the log-probability the
FROZEN model assigns to the committed token in the state immediately before the
commit, always under the FULL untruncated softmax even though pi_ref proposes
from a truncated one:

    PathLL = sum_over_commits  log p_theta(committed token | state before commit)

which is the any-order autoregressive log-likelihood of the produced sequence
under its own realised unmasking order.

RAO-BLACKWELLISATION (primary estimator). The realised-token Path-LL is an
extremely noisy Monte-Carlo estimate: at a single state its rollout noise
exceeds the between-candidate signal (measured in the Phase-0A pilot). We
therefore score each commit by the CONDITIONAL EXPECTATION of its contribution,

    rb_t = E_{x ~ pi_ref(.|s_t)} [ log p_theta(x | s_t) ]
         = sum_{v in top_k} p_trunc(v) * log p_full(v),

while still SAMPLING the token to continue the trajectory. By the tower
property E[sum_t log p_theta(x_t|s_t)] = E[sum_t rb_t], so this estimates the
SAME Path-LL expectation with strictly lower variance (Rao-Blackwell). The
realised-token Path-LL is retained and reported as a consistency check.

CRN -- common random numbers. Both the order key and the token Gumbel noise are
indexed by absolute POSITION (not by step), so the Q-branch and the V-branch of
a pair see identical noise at the same position even though the Q-branch is one
commit ahead. Each position is unmasked exactly once per rollout, so no noise is
consumed twice and the marginal law of pi_ref is exact.

Q / V / A -- for candidate position i at state s_t, with x_hat_i = argmax_v
p_theta(v | s_t), and horizon H:
    Q-branch: force the commit (i, x_hat_i), then H pi_ref commits
              -> H+1 positions consumed, Path-LL = logp_action + S_Q
    V-branch: H+1 pi_ref commits
              -> H+1 positions consumed, Path-LL = S_V
    A_full   = [logp_action + S_Q - S_V] / (H+1)
    A_future = [S_Q - (S_V - v_first)] / H        (both sides are H commits
               made AFTER the decision point; v_first is the V-branch's own
               commit at the decision point)

BOTH BRANCHES MUST CONSUME THE SAME NUMBER OF POSITIONS. An earlier version ran
the V-branch for H commits against the Q-branch's H+1, which was wrong twice
over: (a) the per-token denominators differed (H+1 vs H), and the resulting
(1/(H+1) - 1/H) mismatch injected a -V/(H+1) term into A_full, so the label was
algebraically contaminated with the state value it was supposed to be
independent of; (b) the branches depleted the mask pool by different amounts, so
committing an easy token appeared harmful purely because it removed an easy
commit from the future pool. Both artifacts are removed by matching the counts.
=============================================================================
"""
from dataclasses import dataclass

import torch

from crn import gumbel_by_token, uniforms
from mdlm_local import MASK_TOKEN_ID

NEG_INF = -1e30
ORDER_STREAM = 0       # CRN stream id for the unmasking permutation
TOKEN_STREAM = 1       # CRN stream id for the token Gumbel noise
ORDER_NOISE_STREAM = 2  # CRN stream id for confidence-order Gumbel noise


@dataclass
class PiRefConfig:
    seq_len: int = 256
    prefix_len: int = 64
    temperature: float = 1.0
    top_k: int = 50
    logit_chunk: int = 32     # sequence-chunk for the fp32 log-softmax
    order: str = "ancestral"  # "ancestral" | "confidence"
    order_temp: float = 1.0   # only used by "confidence"

    def __post_init__(self):
        if self.order not in ("ancestral", "confidence"):
            raise ValueError(self.order)


@torch.no_grad()
def forward_raw(model, ids, output_hidden_states=False):
    out = model(ids, output_hidden_states=output_hidden_states)
    return out if output_hidden_states else (out, None)


@torch.no_grad()
def topk_logprobs(logits, k, chunk=32):
    """Memory-lean top-k of the FULL log-softmax.

    Materialising log_softmax over (B, L, 50258) in fp32 costs ~1 GB at B=32,
    L=256; we chunk over the sequence axis instead and keep only the top-k.
    Returns (lp_top (B,L,k) fp32, idx (B,L,k), lse (B,L) fp32, lg_top (B,L,k)).
    """
    B, L, _ = logits.shape
    dev = logits.device
    lp_top = torch.empty(B, L, k, dtype=torch.float32, device=dev)
    lg_top = torch.empty(B, L, k, dtype=torch.float32, device=dev)
    idx = torch.empty(B, L, k, dtype=torch.int64, device=dev)
    lse = torch.empty(B, L, dtype=torch.float32, device=dev)
    for a in range(0, L, chunk):
        sl = logits[:, a:a + chunk].float()
        z = sl.logsumexp(-1)
        v, i = sl.topk(k, -1)
        lse[:, a:a + chunk] = z
        lg_top[:, a:a + chunk] = v
        lp_top[:, a:a + chunk] = v - z[..., None]
        idx[:, a:a + chunk] = i
    return lp_top, idx, lse, lg_top


def unmask_order(seeds, positions):
    """Per-rollout uniformly random unmasking permutation key."""
    return uniforms(seeds, ORDER_STREAM, positions, 1)[:, :, 0]     # (B, L)


def order_score(lp_top, mask, order_key, seeds, positions, cfg):
    """Score by which the next position to unmask is chosen.

    "ancestral"  -- a fixed per-rollout random permutation key (MDLM's own
                    reverse process, one position per step).
    "confidence" -- TOP-1 MAX log-probability, which is the standard MaskGIT /
                    LLaDA convention and the decoder POKE / LookUM operate on.
                    NOTE: an earlier Phase-S measurement here ranked by the
                    log-probability of the SAMPLED token instead, which is a
                    different and much more degenerate rule -- it exaggerated
                    the repetition collapse by roughly 2x (distinct-2 0.202 vs
                    0.447 under identical prompts and seeds). Gumbel noise
                    scaled by `order_temp` keeps the order stochastic and keeps
                    the two branches of a Q/V pair coupled.
    """
    if cfg.order == "ancestral":
        return order_key
    sc = lp_top[..., 0] / max(cfg.order_temp, 1e-6)
    g = gumbel_by_token(seeds, ORDER_NOISE_STREAM, positions,
                        positions[None, :, None].expand(sc.shape[0], -1, 1))
    return sc + g[..., 0]


def expected_logp(lp_top, cfg):
    """E_{x ~ truncated pi_ref}[ log p_full(x) ] -- the Rao-Blackwellised score."""
    w = torch.softmax(lp_top / cfg.temperature, dim=-1)
    return (w * lp_top).sum(-1)


def sample_tokens(lp_top, idx, seeds, positions, cfg):
    """Gumbel-max over the top_k truncated distribution, token-identity CRN."""
    g = gumbel_by_token(seeds, TOKEN_STREAM, positions, idx)
    pick = (lp_top / cfg.temperature + g).argmax(-1)
    proposed = idx.gather(-1, pick[..., None]).squeeze(-1)
    logp = lp_top.gather(-1, pick[..., None]).squeeze(-1)   # FULL-softmax logp
    return proposed, logp


@torch.no_grad()
def rollout(model, ids, mask, seeds, cfg, horizon=None, return_trace=False):
    """Run pi_ref to completion (or `horizon` commits) from (ids, mask).

    Returns path_ll (B,) float64 -- the future Path-LL -- and n_commit (B,).
    """
    ids = ids.clone()
    mask = mask.clone()
    B, L = ids.shape
    positions = torch.arange(L, device=ids.device, dtype=torch.int64)
    order = unmask_order(seeds, positions)

    path_ll = torch.zeros(B, device=ids.device, dtype=torch.float64)
    path_ll_rb = torch.zeros(B, device=ids.device, dtype=torch.float64)
    # the FIRST commit is scored separately: the V-branch's first commit is
    # pi_ref's own choice at the decision point, and excluding it is what makes
    # the "downstream effect only" label comparable to the Q-branch, whose
    # first commit was forced.
    first_ll = torch.zeros(B, device=ids.device, dtype=torch.float64)
    first_ll_rb = torch.zeros(B, device=ids.device, dtype=torch.float64)
    n_done = torch.zeros(B, device=ids.device, dtype=torch.long)
    trace = []
    steps = 0
    while mask.any() and (horizon is None or steps < horizon):
        logits, _ = forward_raw(model, ids)
        lp_top, idx, _, _ = topk_logprobs(logits, cfg.top_k, cfg.logit_chunk)
        proposed, logp = sample_tokens(lp_top, idx, seeds, positions, cfg)
        rb = expected_logp(lp_top, cfg)
        # commit the masked position with the largest order score
        raw = order_score(lp_top, mask, order, seeds, positions, cfg)
        score = torch.where(mask, raw, torch.full_like(raw, -1e30))
        sel = torch.zeros_like(mask)
        sel.scatter_(1, score.argmax(1, keepdim=True), True)
        sel &= mask
        if return_trace:
            pos_sel = score.argmax(1)
            tok_sel = proposed.gather(1, pos_sel[:, None]).squeeze(1)
            trace.append((pos_sel.detach().cpu(), tok_sel.detach().cpu()))
        step_ll = (logp * sel).sum(1).double()
        step_rb = (rb * sel).sum(1).double()
        if steps == 0:
            first_ll += step_ll
            first_ll_rb += step_rb
        path_ll += step_ll
        path_ll_rb += step_rb
        n_done += sel.sum(1)
        ids = torch.where(sel, proposed, ids)
        mask = mask & ~sel
        steps += 1
    out = {"path_ll": path_ll, "path_ll_rb": path_ll_rb,
            "first_ll": first_ll, "first_ll_rb": first_ll_rb,
            "n_commit": n_done, "ids": ids,
            "incomplete": int(mask.any(1).sum().item())}
    if return_trace:
        out["trace"] = trace
    return out


def make_initial_state(token_windows, cfg, device):
    """Prefix observed, suffix fully masked."""
    ids = token_windows.clone().to(device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, cfg.prefix_len:] = True
    ids[:, cfg.prefix_len:] = MASK_TOKEN_ID
    return ids, mask

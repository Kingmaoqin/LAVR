"""
Reference policy pi_ref for the Nemotron backbone: BLOCK DIFFUSION.

WHY IT DIFFERS FROM THE FIRST TWO BACKBONES
MDLM and SEDD used a one-commit-per-step ancestral sampler. Nemotron's native
decoding is block diffusion: work left to right in blocks of `block_length`,
and inside the active block commit every masked position whose confidence
clears `threshold`, in parallel. Reproducing that is both ~3-7x cheaper (28-66
forward evaluations instead of one per token) and closer to how the model is
actually run -- and to what POKE / LookUM operate on. This is a DELIBERATE
deviation from the first two backbones' estimand, recorded as such: the
Nemotron labels are not step-for-step comparable with the MDLM/SEDD labels.

ONE THING THE DEVIATION FIXES FOR FREE
Task utility can only be scored once a full answer exists, so both branches are
rolled to COMPLETION. Both therefore fill every remaining position, which makes
the Q and V branches automatically matched in the number of positions consumed.
The denominator mismatch and the depletion artifact that had to be repaired on
the first backbone cannot arise here by construction.

CRN is unchanged: Gumbel noise keyed by (seed, stream, absolute position,
token id), so a Q branch and its paired V branch see identical noise at the
same position even though one is a commit ahead.
"""
from dataclasses import dataclass

import torch

from crn import gumbel_by_token
from nemotron_local import MASK_TOKEN_ID

TOKEN_STREAM = 1


@dataclass
class BlockPiRefConfig:
    block_length: int = 32
    threshold: float = 0.9      # confidence needed to commit inside a block
    temperature: float = 1.0
    top_k: int = 50
    logit_chunk: int = 16       # sequence chunk for the fp32 log-softmax
    max_steps: int = 400        # hard stop; a block always commits >= 1


@torch.no_grad()
def topk_logprobs(logits, k, chunk=16):
    """Memory-lean top-k of the full log-softmax. vocab is 131072 here, so a
    dense fp32 log-softmax over (B, L, V) is not affordable -- chunk over L."""
    B, L, _ = logits.shape
    dev = logits.device
    lp = torch.empty(B, L, k, dtype=torch.float32, device=dev)
    idx = torch.empty(B, L, k, dtype=torch.int64, device=dev)
    for a in range(0, L, chunk):
        sl = logits[:, a:a + chunk].float()
        z = sl.logsumexp(-1, keepdim=True)
        v, i = sl.topk(k, -1)
        lp[:, a:a + chunk] = v - z
        idx[:, a:a + chunk] = i
    return lp, idx


def sample_tokens(lp_top, idx, seeds, positions, cfg):
    g = gumbel_by_token(seeds, TOKEN_STREAM, positions, idx)
    pick = (lp_top / cfg.temperature + g).argmax(-1)
    proposed = idx.gather(-1, pick[..., None]).squeeze(-1)
    logp = lp_top.gather(-1, pick[..., None]).squeeze(-1)
    return proposed, logp


def expected_logp(lp_top, cfg):
    """Rao-Blackwellised per-commit score: E_{x~pi_ref}[log p_full(x)]."""
    w = torch.softmax(lp_top / cfg.temperature, dim=-1)
    return (w * lp_top).sum(-1)


def active_block(mask, gen_start, cfg):
    """Leftmost block that still holds a masked position, as a boolean mask."""
    B, L = mask.shape
    dev = mask.device
    pos = torch.arange(L, device=dev)[None, :].expand(B, L)
    blk = torch.where(mask, (pos - gen_start).clamp(min=0) // cfg.block_length,
                      torch.full_like(pos, 10 ** 6))
    first = blk.min(dim=1, keepdim=True).values
    return mask & (blk == first)


@torch.no_grad()
def block_rollout(model, ids, mask, seeds, cfg, gen_start):
    """Run block-diffusion pi_ref to COMPLETION.

    Returns the finished ids plus the path log-likelihood accumulated over
    every commit, under both the Rao-Blackwellised and the realised-token
    scoring, and the number of commits made.
    """
    ids = ids.clone()
    mask = mask.clone()
    B, L = ids.shape
    dev = ids.device
    positions = torch.arange(L, device=dev, dtype=torch.int64)

    path_rb = torch.zeros(B, device=dev, dtype=torch.float64)
    path_mc = torch.zeros(B, device=dev, dtype=torch.float64)
    n_commit = torch.zeros(B, device=dev, dtype=torch.long)
    nfe = 0

    while mask.any() and nfe < cfg.max_steps:
        logits = model(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        lp_top, idx = topk_logprobs(logits, cfg.top_k, cfg.logit_chunk)
        nfe += 1
        proposed, logp = sample_tokens(lp_top, idx, seeds, positions, cfg)
        rb = expected_logp(lp_top, cfg)

        blk = active_block(mask, gen_start, cfg)
        conf = lp_top[..., 0].exp()
        sel = blk & (conf >= cfg.threshold)
        # a block must always advance: if nothing clears the threshold, commit
        # the single most confident masked position in it
        empty = ~sel.any(1)
        if empty.any():
            sc = torch.where(blk, conf, torch.full_like(conf, -1.0))
            force = torch.zeros_like(sel)
            force.scatter_(1, sc.argmax(1, keepdim=True), True)
            sel = sel | (force & blk & empty[:, None])

        path_rb += (rb * sel).sum(1).double()
        path_mc += (logp * sel).sum(1).double()
        n_commit += sel.sum(1)
        ids = torch.where(sel, proposed, ids)
        mask = mask & ~sel

    return {"ids": ids, "path_ll_rb": path_rb, "path_ll": path_mc,
            "n_commit": n_commit, "nfe": nfe,
            "incomplete": int(mask.any(1).sum().item())}


@torch.no_grad()
def safe_block_rollout(model, ids, mask, seeds, cfg, gen_start, micro=2,
                       max_wait=180):
    """Chunked block_rollout that survives a co-tenant memory spike.

    These GPUs are shared. The 3B checkpoint leaves only ~0.3 GB of headroom on
    a single card, and a first Phase-S attempt was killed mid-run when the other
    tenant grew. Rather than sharding the model -- which is robust but wastes
    two thirds of the GPUs, since pipeline stages run one at a time and measured
    3.5x lower total throughput -- keep the model on one GPU and make the
    ROLLOUT resilient: halve the micro-batch on OOM, and if even a single
    sequence will not fit, wait for the co-tenant to release memory rather than
    dying and losing the run.
    """
    import time
    n = ids.shape[0]
    outs = {k: [] for k in ("path_ll_rb", "path_ll", "n_commit")}
    ids_out = []
    incomplete = 0
    nfe_tot = 0
    a = 0
    while a < n:
        mb = max(int(micro), 1)
        try:
            r = block_rollout(model, ids[a:a + mb], mask[a:a + mb],
                              seeds[a:a + mb], cfg, gen_start)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if mb > 1:
                micro = mb // 2
                continue
            waited = 0
            while waited < max_wait:            # single sequence will not fit
                time.sleep(15); waited += 15
                torch.cuda.empty_cache()
                try:
                    r = block_rollout(model, ids[a:a + 1], mask[a:a + 1],
                                      seeds[a:a + 1], cfg, gen_start)
                    break
                except torch.cuda.OutOfMemoryError:
                    continue
            else:
                raise RuntimeError("co-tenant did not release memory in "
                                   f"{max_wait}s; aborting rather than "
                                   "silently dropping rollouts")
            mb = 1
        for k in outs:
            outs[k].append(r[k])
        ids_out.append(r["ids"])
        incomplete += r["incomplete"]
        nfe_tot += r["nfe"]
        a += mb
    res = {k: torch.cat(v) for k, v in outs.items()}
    res["ids"] = torch.cat(ids_out)
    assert res["n_commit"].shape[0] == n, \
        f"rollout returned {res['n_commit'].shape[0]} of {n} rows"
    res["incomplete"] = incomplete
    res["nfe"] = nfe_tot
    return res


def make_state(prompt_ids, gen_len, device, pad_id=MASK_TOKEN_ID):
    """Prompt observed, `gen_len` masked positions appended for the answer."""
    B = prompt_ids.shape[0]
    ids = torch.cat([prompt_ids.to(device),
                     torch.full((B, gen_len), MASK_TOKEN_ID, device=device,
                                dtype=prompt_ids.dtype)], 1)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, prompt_ids.shape[1]:] = True
    return ids, mask

"""
Control-feature hierarchy.

The hidden-state probe must beat STRONG output-side controls, not merely scalar
confidence. Blocks, in increasing strength:

  C1  scalar confidence     top-1 prob, log p1, entropy, top1-top2 margins, ...
  C2  trajectory            timestep, mask ratio, locality of observed context,
                            temporal KL, argmax flip count, persistence, ...
  C3  output distribution   sorted top-16 log-probs + a FIXED Johnson-
                            Lindenstrauss projection (d=64) of the FULL
                            log-probability vector
  H   hidden representation [h_{i,t} ; h_global,t]  -- the object under test

Primary comparison:  C1+C2+C3   versus   C1+C2+C3+H.
"""
import numpy as np
import torch

TOPK_FEATS = 16
PROJ_DIM = 64
PROJ_SEED = 20260819


class ProjCache:
    """A single fixed JL matrix, identical across runs, seeds and splits."""
    _P = None
    _dev = None
    _vocab_size = None

    @classmethod
    def get(cls, vocab_size, device):
        if (cls._P is None or cls._dev != str(device)
                or cls._vocab_size != int(vocab_size)):
            g = torch.Generator(device="cpu").manual_seed(PROJ_SEED)
            P = torch.randn((vocab_size, PROJ_DIM), generator=g) / np.sqrt(PROJ_DIM)
            cls._P = P.to(device)
            cls._dev = str(device)
            cls._vocab_size = int(vocab_size)
        return cls._P


C1_NAMES = ["p1", "logp1", "entropy_topk", "margin_p", "margin_logp", "p2",
            "logit_max", "logit_top_std"]
C2_NAMES = ["t_norm", "mask_ratio", "pos_norm", "local_obs_ratio",
            "dist_left", "dist_right", "dist_min", "temporal_kl",
            "flip_count", "persistence", "dp1", "n_remaining_norm"]
C3_NAMES = ([f"toplogp_{i}" for i in range(TOPK_FEATS)]
            + [f"proj_{i}" for i in range(PROJ_DIM)])


def locality_feats(mask, cand_b, cand_i, window=8):
    """Observed-context structure around each candidate. mask True = masked."""
    B, L = mask.shape
    dev = mask.device
    obs = (~mask).float()
    kernel = torch.ones(1, 1, 2 * window + 1, device=dev)
    local = torch.nn.functional.conv1d(
        obs[:, None], kernel, padding=window)[:, 0] / (2 * window + 1)

    ar = torch.arange(L, device=dev)
    big = L + 1
    left_src = torch.where(~mask, ar[None].expand(B, L),
                           torch.full((B, L), -big, device=dev, dtype=torch.long))
    left = torch.cummax(left_src, dim=1).values
    right_src = torch.where(~mask, ar[None].expand(B, L),
                            torch.full((B, L), big, device=dev, dtype=torch.long))
    right = torch.flip(torch.cummin(torch.flip(right_src, [1]), dim=1).values, [1])
    d_l = (ar[None] - left).float().clamp(max=float(L))
    d_r = (right - ar[None]).float().clamp(max=float(L))
    sel = (cand_b, cand_i)
    return local[sel], d_l[sel], d_r[sel], torch.minimum(d_l, d_r)[sel]


def assemble_cheap(snap, cand_b, cand_i, n_steps):
    """C1 / C2 / C3 blocks for one snapshot's candidates."""
    lp_top = snap["lp_top"][cand_b, cand_i]        # (N, k) full-softmax log-probs
    lg_top = snap["lg_top"][cand_b, cand_i]        # (N, k) raw logits
    N = lp_top.shape[0]
    dev = lp_top.device

    p = lp_top.exp()
    ent = -(p * lp_top).sum(1)                     # entropy over the top-k support
    c1 = torch.stack([
        lp_top[:, 0].exp(), lp_top[:, 0], ent,
        lp_top[:, 0].exp() - lp_top[:, 1].exp(),
        lp_top[:, 0] - lp_top[:, 1], lp_top[:, 1].exp(),
        lg_top[:, 0], lg_top.std(1)], 1)

    mask = snap["mask"]
    L = mask.shape[1]
    pre = snap["prefix_len"]
    local, d_l, d_r, d_m = locality_feats(mask, cand_b, cand_i)
    c2 = torch.stack([
        torch.full((N,), snap["step"] / max(n_steps, 1), device=dev),
        mask.float().mean(1)[cand_b],
        (cand_i.float() - pre) / max(L - pre, 1),
        local, d_l, d_r, d_m,
        snap["temporal_kl"][cand_b, cand_i],
        snap["flip_count"][cand_b, cand_i],
        snap["persistence"][cand_b, cand_i],
        snap["dp1"][cand_b, cand_i],
        mask.sum(1).float()[cand_b] / L,
    ], 1)

    c3 = torch.cat([lp_top[:, :TOPK_FEATS], snap["proj"][cand_b, cand_i]], 1)
    return (c1.float().cpu().numpy(), c2.float().cpu().numpy(),
            c3.float().cpu().numpy())


def assemble_hidden(snap, cand_b, cand_i):
    """Per-layer [h_{i,t}] and [h_global,t]; h_global = mean over positions."""
    h_i, h_g = [], []
    for h in snap["hidden"]:                        # each (B, L, D)
        # With a device-mapped backbone, hidden layers can live on a different
        # GPU from the candidate indices.  Index locally, then bring only the
        # selected features to CPU.  The single-GPU path is unchanged.
        hb = cand_b.to(h.device)
        hi = cand_i.to(h.device)
        h_i.append(h[hb, hi].float().cpu().numpy().astype(np.float16))
        h_g.append(h.mean(1)[hb].float().cpu().numpy().astype(np.float16))
    return np.stack(h_i, 1), np.stack(h_g, 1)       # (N, n_layers, D) each

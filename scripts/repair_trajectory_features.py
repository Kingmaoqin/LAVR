"""
Repair the C2 trajectory-stability controls (`flip_count`, `persistence`).

The first collection run accumulated these only inside `snapshot()`, i.e. at the
6 recorded checkpoints rather than at every denoising step, which coarsened them
and made the control baseline WEAKER than specified. A weaker control biases the
study in favour of its own hypothesis, so it must be repaired rather than
excused.

`pi_ref` trajectories are fully deterministic given (prompt window, rollout
seed), and the repair needs no rollouts at all -- only the forward passes along
the reference trajectory. So the correct per-step statistics can be recomputed
exactly and written back over the stored columns; every other feature, every
hidden state and every label is untouched.

Run AFTER collection, BEFORE any experiment.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from mdlm_local import load_mdlm                                   # noqa: E402
import data as datamod                                             # noqa: E402
import collect as C                                                # noqa: E402
import features as F                                               # noqa: E402
from policy import (forward_raw, topk_logprobs, unmask_order,      # noqa: E402
                    sample_tokens, make_initial_state)

FLIP_COL = F.C2_NAMES.index("flip_count")
PERSIST_COL = F.C2_NAMES.index("persistence")


@torch.no_grad()
def recompute(model, windows, idx, cfg, pcfg, dev, seed_offset, record_at):
    """Correct per-step flip_count / persistence at each recorded step."""
    w = torch.as_tensor(windows[idx], device=dev)
    ids, mask = make_initial_state(w, pcfg, dev)
    B, L = ids.shape
    positions = torch.arange(L, device=dev, dtype=torch.int64)
    traj_seeds = torch.arange(B, device=dev, dtype=torch.int64) + seed_offset
    order = unmask_order(traj_seeds, positions)

    flip_count = torch.zeros(B, L, device=dev)
    persistence = torch.zeros(B, L, device=dev)
    prev_argmax = None
    out = {}
    for step in range(L - cfg.prefix_len):
        logits, _ = forward_raw(model, ids)
        lp_top, tk_idx, _, _ = topk_logprobs(logits, cfg.top_k, 32)
        argmax = tk_idx[..., 0]
        if prev_argmax is not None:
            flip = (argmax != prev_argmax).float()
            flip_count = flip_count + flip * mask.float()
            persistence = torch.where(flip.bool(),
                                      torch.zeros_like(persistence),
                                      persistence + 1.0)
        if step in record_at:
            out[step] = (flip_count.cpu().numpy().copy(),
                         persistence.cpu().numpy().copy())
        proposed, _ = sample_tokens(lp_top, tk_idx, traj_seeds, positions, pcfg)
        sel = torch.zeros_like(mask)
        sel.scatter_(1, torch.where(mask, order,
                                    torch.full_like(order, -1.0)
                                    ).argmax(1, keepdim=True), True)
        sel &= mask
        ids = torch.where(sel, proposed, ids)
        mask = mask & ~sel
        prev_argmax = argmax
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["a", "b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    dev = args.device
    model, _ = load_mdlm(device=dev)

    for tag in args.tags:
        ddir = os.path.join(ROOT, "data", f"labels_{tag}")
        meta = json.load(open(os.path.join(ddir, "meta.json")))
        cfg = C.CollectConfig(seq_len=meta["seq_len"],
                              prefix_len=meta["prefix_len"],
                              traj_batch=meta["traj_batch"])
        pcfg = cfg.pi()
        n_steps = cfg.seq_len - cfg.prefix_len
        record_at = sorted({max(1, int(f * n_steps))
                            for f in meta["record_fracs"]})
        windows, doc_ids = datamod.get_windows(seq_len=cfg.seq_len, n_docs=3000)
        splits = datamod.doc_level_split(doc_ids)
        order = np.concatenate([splits["train"], splits["val"], splits["test"]])
        order = order[meta["offset"]:meta["offset"] + meta["n_prompts"]]

        # prompt_row -> {step: (flip, persist) over positions}
        table = {}
        for b0 in range(0, len(order), cfg.traj_batch):
            idx = order[b0:b0 + cfg.traj_batch]
            seed_off = cfg.seed_base + meta["offset"] * 131 + b0
            got = recompute(model, windows, idx, cfg, pcfg, dev, seed_off,
                            set(record_at))
            for j, pr in enumerate(idx):
                table[int(pr)] = {st: (v[0][j], v[1][j]) for st, v in got.items()}
            print(f"  [{tag}] recomputed {b0 + len(idx)}/{len(order)}", flush=True)

        for f in sorted(glob.glob(os.path.join(ddir, "shard_*.npz"))):
            z = dict(np.load(f))
            C2 = z["C2"].copy()
            n_fix = n_miss = 0
            old_flip = C2[:, FLIP_COL].copy()
            for r in range(len(C2)):
                pr, st, po = (int(z["prompt_row"][r]), int(z["step"][r]),
                              int(z["position"][r]))
                e = table.get(pr, {}).get(st)
                if e is None:
                    n_miss += 1
                    continue
                C2[r, FLIP_COL] = e[0][po]
                C2[r, PERSIST_COL] = e[1][po]
                n_fix += 1
            print(f"  [{tag}] {os.path.basename(f)}: repaired {n_fix}, "
                  f"unmatched {n_miss};  flip_count mean "
                  f"{old_flip.mean():.2f} -> {C2[:, FLIP_COL].mean():.2f}, "
                  f"max {old_flip.max():.0f} -> {C2[:, FLIP_COL].max():.0f}")
            if n_miss:
                raise RuntimeError(f"{n_miss} records could not be matched -- "
                                   f"trajectory reproduction is not exact")
            if not args.dry_run:
                z["C2"] = C2
                np.savez_compressed(f, **z)
        if not args.dry_run:
            meta["c2_repaired"] = ("flip_count and persistence recomputed "
                                   "per-step; see repair_trajectory_features.py")
            json.dump(meta, open(os.path.join(ddir, "meta.json"), "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()

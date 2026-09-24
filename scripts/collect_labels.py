"""
Phase 0A driver -- walk pi_ref trajectories, snapshot states, run CRN-coupled
paired rollouts, and write feature/label shards.

Usage
  python scripts/collect_labels.py --n_prompts 300 --K 4 --n_cand 6
  python scripts/collect_labels.py --pilot            # label-variance diagnostic
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from mdlm_local import load_mdlm                                  # noqa: E402
from sedd_local import load_sedd                                  # noqa: E402
import data as datamod                                            # noqa: E402
from policy import (forward_raw, topk_logprobs, unmask_order,      # noqa: E402
                    sample_tokens, make_initial_state, order_score)
import collect as C                                               # noqa: E402
import features as F                                              # noqa: E402


@torch.no_grad()
def walk_prompt_batch(model, windows, doc_ids, idx, cfg, pcfg, dev,
                      seed_offset, gen):
    """Decode one batch of prompts with pi_ref, labelling the recorded states."""
    w = torch.as_tensor(windows[idx], device=dev)
    ids, mask = make_initial_state(w, pcfg, dev)
    B, L = ids.shape
    n_steps = L - cfg.prefix_len
    record_at = sorted({max(1, int(f * n_steps)) for f in cfg.record_fracs})

    positions = torch.arange(L, device=dev, dtype=torch.int64)
    traj_seeds = torch.arange(B, device=dev, dtype=torch.int64) + seed_offset
    order = unmask_order(traj_seeds, positions)
    hist = C.new_hist(B, L, dev)

    rows = []
    for step in range(n_steps):
        want = step in record_at
        if want:
            snap, aux = C.snapshot(model, ids, mask, step, hist, cfg, pcfg)
            logits, lse, p1, argmax = aux
            lp_top, tk_idx = snap["lp_top"], snap["idx"]
        else:
            logits, _ = forward_raw(model, ids)
            lp_top, tk_idx, lse, _ = topk_logprobs(logits, cfg.top_k, 32)
            p1 = lp_top[..., 0].exp()
            argmax = tk_idx[..., 0]

        if want:
            cand_b, cand_i, strata = C.pick_candidates(snap, cfg, gen)
            if cand_b.numel():
                lab = C.branch_rollouts(model, snap, cand_b, cand_i, cfg, pcfg,
                                        seed_offset * 977 + step * 31)
                c1, c2, c3 = F.assemble_cheap(snap, cand_b, cand_i, n_steps)
                hi, hg = F.assemble_hidden(snap, cand_b, cand_i)
                cb = cand_b.cpu().numpy()
                rows.append(dict(
                    doc_id=doc_ids[idx][cb], prompt_row=idx[cb],
                    step=np.full(len(cb), step, dtype=np.int32),
                    mask_ratio=mask.float().mean(1).cpu().numpy()[cb],
                    position=cand_i.cpu().numpy(),
                    proposed_token=snap["argmax"][cand_b, cand_i].cpu().numpy(),
                    stratum=strata, C1=c1, C2=c2, C3=c3, H_i=hi, H_g=hg,
                    **{k: v for k, v in lab.items() if k != "incomplete"}))
            del snap

        # advance pi_ref one commit
        mask_before = mask.clone()
        proposed, _ = sample_tokens(lp_top, tk_idx, traj_seeds, positions, pcfg)
        raw = order_score(lp_top, mask, order, traj_seeds, positions, pcfg)
        score = torch.where(mask, raw, torch.full_like(raw, -1e30))
        sel = torch.zeros_like(mask)
        sel.scatter_(1, score.argmax(1, keepdim=True), True)
        sel &= mask
        ids = torch.where(sel, proposed, ids)
        mask = mask & ~sel
        C.update_hist(hist, logits, lse, p1, argmax, mask_before)
    return rows


def concat_rows(rows):
    out = {}
    for k in rows[0]:
        out[k] = np.concatenate([r[k] for r in rows], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=400)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--n_cand", type=int, default=6)
    ap.add_argument("--n_cand_conf", type=int, default=3)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--prefix_len", type=int, default=64)
    ap.add_argument("--traj_batch", type=int, default=6)
    ap.add_argument("--rollout_batch", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--order", choices=["ancestral", "confidence"],
                    default="ancestral")
    ap.add_argument("--order_temp", type=float, default=1.0)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", choices=["mdlm", "sedd"], default="mdlm")
    args = ap.parse_args()

    cfg = C.CollectConfig(
        seq_len=args.seq_len, prefix_len=args.prefix_len, K=args.K,
        n_cand=args.n_cand, n_cand_conf=args.n_cand_conf,
        traj_batch=args.traj_batch, rollout_batch=args.rollout_batch,
        horizon=args.horizon, order=args.order, order_temp=args.order_temp)
    if args.pilot:
        cfg.record_fracs = (0.3, 0.7)
        cfg.K = max(args.K, 8)
    pcfg = cfg.pi()
    dev = args.device

    outdir = os.path.join(ROOT, "data", f"labels_{args.tag}")
    os.makedirs(outdir, exist_ok=True)

    model, _ = (load_sedd(device=dev) if args.backbone == "sedd"
                else load_mdlm(device=dev))
    windows, doc_ids = datamod.get_windows(seq_len=cfg.seq_len, n_docs=3000)
    splits = datamod.doc_level_split(doc_ids)
    order = np.concatenate([splits["train"], splits["val"], splits["test"]])
    order = order[args.offset:args.offset + args.n_prompts]
    gen = torch.Generator(device=dev).manual_seed(4242)

    print(f"[0A] prompts={len(order)} K={cfg.K} n_cand={cfg.n_cand} "
          f"record_fracs={cfg.record_fracs} horizon={cfg.horizon}")
    t0 = time.time()
    shard, n_ex, shard_id = [], 0, 0
    for b0 in range(0, len(order), cfg.traj_batch):
        idx = order[b0:b0 + cfg.traj_batch]
        rows = walk_prompt_batch(model, windows, doc_ids, idx, cfg, pcfg, dev,
                                 cfg.seed_base + args.offset * 131 + b0, gen)
        shard.extend(rows)
        n_ex += sum(len(r["position"]) for r in rows)
        done = b0 + len(idx)
        el = time.time() - t0
        print(f"  prompts {done}/{len(order)}  examples {n_ex}  "
              f"{el:.0f}s  eta {el / done * (len(order) - done):.0f}s",
              flush=True)
        if sum(len(r["position"]) for r in shard) >= 2000 or done >= len(order):
            d = concat_rows(shard)
            np.savez_compressed(
                os.path.join(outdir, f"shard_{shard_id:03d}.npz"), **d)
            shard, shard_id = [], shard_id + 1

    meta = dict(vars(args))
    meta.update(record_fracs=list(cfg.record_fracs), n_examples=n_ex,
                seconds=time.time() - t0,
                C1_names=F.C1_NAMES, C2_names=F.C2_NAMES, C3_names=F.C3_NAMES,
                estimand="A^{pi_ref} per-token Path-LL advantage, "
                         "CRN-paired, policy-relative (NOT A*)")
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[0A] done: {n_ex} examples in {time.time()-t0:.0f}s -> {outdir}")


if __name__ == "__main__":
    main()

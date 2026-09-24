"""
Backfill per-seed replicates of the state value V for collections made before
`V_pertok_seeds` was stored.

Why it is needed: V was kept only as a mean over the K coupled rollouts, so it
has no noise ceiling and its Delta_R2 cannot be ceiling-corrected. Without that
it can only serve as a foil for the advantage result, not be reported as a
finding in its own right.

Why it is cheap: only the V branch has to be re-run. The V branch is B*K
rollout chains against the Q branch's N*K (N = B * n_cand), so this costs about
1/7 of a full collection. `pi_ref` trajectories and every rollout seed are
deterministic functions of (offset, batch index, step), so the V values
reproduce exactly -- the script asserts that the recomputed mean matches the
stored `V_pertok` before writing anything.
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
from sedd_local import load_sedd                                   # noqa: E402
import data as datamod                                             # noqa: E402
import collect as C                                                # noqa: E402
from policy import (forward_raw, topk_logprobs, unmask_order,      # noqa: E402
                    sample_tokens, make_initial_state, order_score,
                    rollout)


@torch.no_grad()
def v_for_batch(model, windows, idx, cfg, pcfg, dev, seed_off, record_at):
    """Per-seed V for every recorded step of one prompt batch."""
    w = torch.as_tensor(windows[idx], device=dev)
    ids, mask = make_initial_state(w, pcfg, dev)
    B, L = ids.shape
    K = cfg.K
    pos = torch.arange(L, device=dev, dtype=torch.int64)
    tseeds = torch.arange(B, device=dev, dtype=torch.int64) + seed_off
    order = unmask_order(tseeds, pos)

    out = {}
    for step in range(L - cfg.prefix_len):
        logits, _ = forward_raw(model, ids)
        lp_top, tk, lse, _ = topk_logprobs(logits, cfg.top_k, 32)

        if step in record_at:
            # exactly the seed stream branch_rollouts uses for the V branch
            off = seed_off * 977 + step * 31
            v_seeds = (torch.arange(B, device=dev).repeat_interleave(K) * K
                       + torch.arange(K, device=dev).repeat(B) + off)
            V = C._chunked_rollout(
                model, ids.repeat_interleave(K, 0), mask.repeat_interleave(K, 0),
                v_seeds, pcfg, cfg,
                horizon=(None if cfg.horizon is None else cfg.horizon + 1))
            per_tok = (V["path_ll_rb"] / V["n_commit"].clamp(min=1).double()
                       ).view(B, K).float().cpu().numpy()
            out[step] = per_tok

        proposed, _ = sample_tokens(lp_top, tk, tseeds, pos, pcfg)
        raw = order_score(lp_top, mask, order, tseeds, pos, pcfg)
        score = torch.where(mask, raw, torch.full_like(raw, -1e30))
        sel = torch.zeros_like(mask)
        sel.scatter_(1, score.argmax(1, keepdim=True), True)
        sel &= mask
        ids = torch.where(sel, proposed, ids)
        mask = mask & ~sel
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rollout_batch", type=int, default=48)
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="max allowed |recomputed mean - stored V_pertok|")
    args = ap.parse_args()
    dev = args.device

    for tag in args.tags:
        ddir = os.path.join(ROOT, "data", f"labels_{tag}")
        meta = json.load(open(os.path.join(ddir, "meta.json")))
        if meta.get("v_seeds_backfilled"):
            print(f"[{tag}] already backfilled, skipping")
            continue
        cfg = C.CollectConfig(
            seq_len=meta["seq_len"], prefix_len=meta["prefix_len"],
            K=meta["K"], horizon=meta["horizon"], traj_batch=meta["traj_batch"],
            rollout_batch=args.rollout_batch,
            order=meta.get("order") or "ancestral",
            order_temp=meta.get("order_temp", 1.0))
        pcfg = cfg.pi()
        model, _ = (load_sedd(device=dev) if meta.get("backbone") == "sedd"
                    else load_mdlm(device=dev))
        n_steps = cfg.seq_len - cfg.prefix_len
        record_at = {max(1, int(f * n_steps)) for f in meta["record_fracs"]}

        windows, doc_ids = datamod.get_windows(seq_len=cfg.seq_len, n_docs=3000)
        splits = datamod.doc_level_split(doc_ids)
        order = np.concatenate([splits["train"], splits["val"], splits["test"]])
        order = order[meta["offset"]:meta["offset"] + meta["n_prompts"]]

        table = {}
        for b0 in range(0, len(order), cfg.traj_batch):
            idx = order[b0:b0 + cfg.traj_batch]
            got = v_for_batch(model, windows, idx, cfg, pcfg, dev,
                              cfg.seed_base + meta["offset"] * 131 + b0,
                              record_at)
            for j, pr in enumerate(idx):
                table[int(pr)] = {st: v[j] for st, v in got.items()}
            print(f"  [{tag}] {b0 + len(idx)}/{len(order)}", flush=True)

        for f in sorted(glob.glob(os.path.join(ddir, "shard_*.npz"))):
            z = dict(np.load(f))
            Vs = np.zeros((len(z["V_pertok"]), cfg.K), dtype=np.float32)
            for r in range(len(Vs)):
                Vs[r] = table[int(z["prompt_row"][r])][int(z["step"][r])]
            err = np.abs(Vs.mean(1) - z["V_pertok"]).max()
            print(f"  [{tag}] {os.path.basename(f)}: max|recomputed - stored| "
                  f"= {err:.2e}")
            if err > args.tol:
                raise RuntimeError(
                    f"V did not reproduce for {f} (max err {err:.3e}); the "
                    f"trajectory or seed stream is not being replayed exactly")
            z["V_pertok_seeds"] = Vs
            np.savez_compressed(f, **z)
        meta["v_seeds_backfilled"] = True
        json.dump(meta, open(os.path.join(ddir, "meta.json"), "w"), indent=2)
        del model
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()

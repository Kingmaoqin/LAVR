"""
P11 temporal-difference probe + synthetic scenario C.

Tests the hypothesis that candidate-level future value is encoded in the
representation's **temporal difference** `dh_i = h_{i,t} - h_{i,t-1}` rather than in the static `h_{i,t}`.
Previous-step hidden states are recovered by `collect_prev_hidden.py` through
bit-exact trajectory replay (self-check max|dH_i| = 0).

Two parts:

1. **Synthetic C** (validate the probe first): plant `A = w.(h_t - h_{t-1}) + eps`.
      Requirement: static probe (h_t only) is weak, temporal-difference probe succeeds.
      If this pattern does not appear, a null result on real data is meaningless.

2. **Real data**: add dh_i / dh_g / cosine change / L2 change to the hidden block and
      compare with an equal-capacity static-only control. Three-block ridge (cheap | static | temporal),
      independent penalties per block; the temporal gamma grid includes 0 -> static is a nested sub-model,
      so temporal >= static on validation by construction.

No t+1 information is used: this script only reads `H_i_prev` / `H_g_prev`, which come from
the step **before** the record point.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import probes2 as P2           # noqa: E402
from rlib import probes3 as P3           # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

SLICES = {"cheap_only": [True, False, False],
          "static": [True, True, False],
          "temporal": [True, True, True]}


def align(d, dp):
    """Align prev shards to the row order of the main shards by (prompt_row, step, position)."""
    key = {(int(a), int(b), int(c)): i for i, (a, b, c) in
           enumerate(zip(dp["prompt_row"], dp["step"], dp["position"]))}
    idx = np.array([key.get((int(a), int(b), int(c)), -1) for a, b, c in
                    zip(d["prompt_row"], d["step"], d["position"])])
    return idx


def temporal_feats(hi_t, hi_p, hg_t, hg_p):
    """dh_i, dh_g, plus cosine / L2 change scalars."""
    dhi = hi_t - hi_p
    dhg = hg_t - hg_p
    def _cos(a, b):
        na = np.linalg.norm(a, axis=1) + 1e-8
        nb = np.linalg.norm(b, axis=1) + 1e-8
        return ((a * b).sum(1) / (na * nb))[:, None]
    sc = np.concatenate([
        _cos(hi_t, hi_p), _cos(hg_t, hg_p),
        np.linalg.norm(dhi, axis=1)[:, None],
        np.linalg.norm(dhg, axis=1)[:, None]], 1)
    return dhi, dhg, sc.astype(np.float32)


def build_blocks(d, dp, aidx, layer, sp, pca_dim, kron_unused=None):
    tr, va, te = sp["train"], sp["val"], sp["test"]
    cheap = RD.cheap_block(d)
    hi_t = RD.h_i(d, layer); hg_t = RD.h_g(d, layer)
    hi_p = dp["H_i_prev"][aidx][:, layer].astype(np.float32)
    hg_p = dp["H_g_prev"][aidx][:, layer].astype(np.float32)
    dhi, dhg, sc = temporal_feats(hi_t, hi_p, hg_t, hg_p)

    def split(X):
        return {"train": X[tr], "val": X[va], "test": X[te]}

    def pca_of(X, dim):
        p = RD.TrainPCA(min(dim, X.shape[1]), whiten=True).fit(X[tr])
        return {k: p.transform(v) for k, v in split(X).items()}

    mu = cheap[tr].mean(0, keepdims=True)
    sd = cheap[tr].std(0, keepdims=True); sd[sd < 1e-8] = 1.0
    B_cheap = {k: ((v - mu) / sd).astype(np.float32)
               for k, v in split(cheap).items()}
    Zi, Zg = pca_of(hi_t, pca_dim), pca_of(hg_t, pca_dim)
    B_static = {k: np.concatenate([Zi[k], Zg[k]], 1)
                for k in ("train", "val", "test")}
    Di, Dg = pca_of(dhi, pca_dim), pca_of(dhg, pca_dim)
    S = split(sc)
    B_temp = {k: np.concatenate([Di[k], Dg[k], S[k]], 1)
              for k in ("train", "val", "test")}
    return B_cheap, B_static, B_temp


def run(B_cheap, B_static, B_temp, y, sid, criterion, gm, gt):
    sel = P2.make_selector(criterion, sid["val"])
    res = P3.fit_ridge_blocks(
        [B_cheap["train"], B_static["train"], B_temp["train"]], y["train"],
        [B_cheap["val"], B_static["val"], B_temp["val"]], y["val"],
        [B_cheap["test"], B_static["test"], B_temp["test"]], sel,
        gamma_grids=[[1.0], gm, gt], record_slices=SLICES)
    _, g_te = M.group_slices(sid["test"])
    out = {}
    for k, m in res.items():
        if m is None:
            continue
        out[k] = M.full_report(y["test"], m["pred_test"], sid["test"], g_te)
        out[k]["hp"] = m["hp"]; out[k]["val_score"] = m["val_score"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["a3", "b3"])
    ap.add_argument("--prev_tags", nargs="+", default=["prev_a3", "prev_b3"])
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--criterion", default="within_r2")
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "temporal_probe.json"))
    args = ap.parse_args()
    t0 = time.time()

    d = RD.load_labels(args.tags)
    dp = RD.load_labels(args.prev_tags,
                        keys=["prompt_row", "step", "position",
                              "H_i_prev", "H_g_prev", "H_i_now", "H_g_now"])
    aidx = align(d, dp)
    ok = aidx >= 0
    print(f"[temporal] main shards {len(aidx)} rows, aligned {int(ok.sum())} rows", flush=True)
    if ok.sum() < len(aidx):
        d = {k: (v[ok] if isinstance(v, np.ndarray)
                 and v.shape[:1] == aidx.shape else v) for k, v in d.items()}
        aidx = aidx[ok]
    # verify alignment again: H_i_now in prev shards must equal H_i in the main shards
    L0 = 6
    dif = np.abs(dp["H_i_now"][aidx][:, L0].astype(np.float32)
                 - RD.h_i(d, L0))
    print(f"[temporal] alignment check max|dH_i(now)| = {dif.max():.3g}", flush=True)
    assert dif.max() < 1e-2, "alignment error: prev shards and main shards are not the same rows"

    gm = [0.0] + list(P3.GAMMAS_MAIN)
    gt = list(P3.GAMMAS_INT)
    sp = RD.doc_splits(d, seed=0)
    RD.check_split_disjoint(d, sp)
    _, groups, _ = RD.state_groups(d["state_id"])
    sid = SC.split_sid(d, sp)
    rep = {"config": vars(args), "n_rows": int(len(aidx)), "results": {}}

    for L in args.layers:
        Bc, Bs, Bt = build_blocks(d, dp, aidx, L, sp, args.pca_dim)

        # ---- synthetic C: plant A = w.(h_t - h_{t-1}) + eps ----
        if args.synthetic:
            rng = np.random.default_rng(11)
            hi_t = RD.h_i(d, L)
            hi_p = dp["H_i_prev"][aidx][:, L].astype(np.float32)
            dh = hi_t - hi_p
            U = np.linalg.eigh((dh - dh.mean(0)).T @ (dh - dh.mean(0)))[1][:, -64:]
            Z = (dh - dh.mean(0)) @ U
            Z = Z / (Z.std(0) + 1e-8)
            w = rng.normal(size=Z.shape[1])
            sig = Z @ w; sig = (sig - sig.mean()) / sig.std()
            ysyn = sig + rng.normal(0, 1 / np.sqrt(2.0), size=len(sig))
            ys = {k: ysyn[v] for k, v in sp.items()}
            o = run(Bc, Bs, Bt, ys, sid, args.criterion, gm, gt)
            rep["results"][f"syntheticC_L{L}"] = o
            print(f"[synC L{L}] cheap {o['cheap_only']['concordance']:.4f} | "
                  f"static {o['static']['concordance']:.4f} "
                  f"(within {o['static']['within_r2']:+.4f}) | "
                  f"**temporal {o['temporal']['concordance']:.4f} "
                  f"(within {o['temporal']['within_r2']:+.4f})**  "
                  f"Δ={o['temporal']['concordance']-o['static']['concordance']:+.4f}",
                  flush=True)

        # ---- real labels ----
        y = SC.split_targets(d, sp, args.target)
        o = run(Bc, Bs, Bt, y, sid, args.criterion, gm, gt)
        rep["results"][f"real_L{L}"] = o
        dt = o["temporal"]["concordance"] - o["static"]["concordance"]
        dh_ = o["static"]["concordance"] - o["cheap_only"]["concordance"]
        print(f"[real L{L}] cheap {o['cheap_only']['concordance']:.4f} | "
              f"static {o['static']['concordance']:.4f} (Δ{dh_:+.4f}) | "
              f"temporal {o['temporal']['concordance']:.4f}  "
              f"**TEMPORAL Δconc {dt:+.4f}**  γ={o['temporal']['hp']['gammas']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(rep, open(args.out, "w"), indent=2, default=float)

    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

"""
Synthetic self-tests A-F.

All built on the **real hidden-state geometry** (layer-L h_i / h_g of a3/b3, real
state/doc structure), replacing only the labels with known constructions. This tests whether "the probe family can read
this structure under the real covariance of this data", rather than on idealised Gaussian data.

Expected patterns (any mismatch means the probe family is broken and null results on real data are untrustworthy):

    A linear     : linear succeeds, nonlinear also succeeds
    B bilinear   : h_i-linear fails, additive [h_i;h_g] linear fails, **bilinear succeeds**,
                                shuffled h_g fails                      <-- the most important self-test
    D rank-only  : pooled regression is swamped by state-level variance, **ranking probes recover candidate order**,
                                and the within_r2 selection criterion clearly beats pooled_r2
    E nonlinear  : linear weak, MLP/kernel succeed
    F null       : all deltas ~ 0, permutation calibrated, no systematic false positives

    C temporal   : needs h_{i,t-1}, not stored on disk -> marked PENDING (see temporal_probe.py)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

OUT = os.path.join(HERE, "synthetic")
os.makedirs(OUT, exist_ok=True)


def global_pca(X, dim, seed=0):
    """Direction basis used to construct labels (not part of the probe; may use all data)."""
    Xc = X - X.mean(0, keepdims=True)
    G = Xc.T @ Xc
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:dim]
    return V[:, idx], np.sqrt(np.clip(w[idx], 1e-12, None) / len(X))


def scale_to_snr(signal, snr, rng):
    """Given a signal, add noise so that var(signal)/var(noise) = snr."""
    s = signal - signal.mean()
    sd_n = np.sqrt(max(s.var(), 1e-12) / max(snr, 1e-9))
    return s + rng.normal(0, sd_n, size=len(s))


def build(scenario, hi, hg, sid, rng, snr=2.0, dim=64):
    """Returns y. hi/hg are the raw 768-dim vectors."""
    Ui, si = global_pca(hi, dim)
    Ug, sg = global_pca(hg, dim)
    Zi = (hi - hi.mean(0, keepdims=True)) @ Ui / si[None, :]
    Zg = (hg - hg.mean(0, keepdims=True)) @ Ug / sg[None, :]

    if scenario == "A_linear":
        w = rng.normal(size=dim)
        sig = Zi @ w
    elif scenario == "B_bilinear":
        r = 4
        U = rng.normal(size=(dim, r)); V = rng.normal(size=(dim, r))
        sig = ((Zi @ U) * (Zg @ V)).sum(1)
    elif scenario == "D_rank_only":
        # large state-level signal (from h_g only) + small within-state signal (from h_i only).
        # Noise must be scaled to the **candidate-level** component: otherwise the 5x larger state-level component would raise noise
        # to a level that swamps the candidate signal, the scenario would degenerate into "no signal at all", and nothing
        # could be learned about pooled-vs-ranking selection criteria.
        v = rng.normal(size=dim); w = rng.normal(size=dim)
        state_part = Zg @ v
        cand_part = Zi @ w
        state_part = state_part / (state_part.std() + 1e-12)
        cand_part = cand_part - _state_mean(cand_part, sid)
        cand_part = cand_part / (cand_part.std() + 1e-12)
        noise = rng.normal(0, 1.0 / np.sqrt(max(snr, 1e-9)), size=len(Zi))
        return 5.0 * state_part + 1.0 * cand_part + noise
    elif scenario == "E_nonlinear":
        # frequency must be mild: sin(Zi@u/sqrt(dim)*3) has argument variance ~9,
        # giving a high-frequency function that is **not learnable** at this sample size -- testing sample size,
        # not the probe. The projection is standardised to unit variance before a mild nonlinearity.
        u = rng.normal(size=dim); v = rng.normal(size=dim); w = rng.normal(size=dim)
        zi = Zi @ u; zi /= (zi.std() + 1e-12)
        zg = Zg @ v; zg /= (zg.std() + 1e-12)
        zw = Zi @ w; zw /= (zw.std() + 1e-12)
        sig = np.tanh(1.5 * zi) + 0.7 * (zw ** 2 - 1.0) + 0.5 * np.cos(zg)
    elif scenario == "F_null":
        return rng.normal(size=len(hi))
    else:
        raise KeyError(scenario)
    return scale_to_snr(sig, snr, rng)


def _state_mean(x, sid):
    _, groups = M.group_slices(sid)
    out = np.zeros_like(x)
    for g in groups:
        out[g] = x[g].mean()
    return out


PROBE_SETS = {
    "A_linear":    ["P0", "P2", "P4"],
    "B_bilinear":  ["P0", "P2", "P3", "P4", "P7"],
    "D_rank_only": ["P0", "P2", "P7", "P8", "P9"],
    "E_nonlinear": ["P0", "P4", "P5", "P6"],
    "F_null":      ["P0", "P2", "P4", "P7"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["a3", "b3"])
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--snr", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--scenarios", nargs="+",
                    default=["A_linear", "B_bilinear", "D_rank_only",
                             "E_nonlinear", "F_null"])
    ap.add_argument("--out", default=os.path.join(OUT, "synthetic_results.json"))
    args = ap.parse_args()

    print(f"[syn] loading {args.tags} layer {args.layer}", flush=True)
    d = RD.load_labels(args.tags, keys=["H_i", "H_g", "C1", "C2", "C3",
                                        "prompt_row", "step", "doc_id",
                                        "stratum", "A_pertok"])
    sp = RD.doc_splits(d, seed=0)
    RD.check_split_disjoint(d, sp)
    _, groups, _ = RD.state_groups(d["state_id"])
    hi = RD.h_i(d, args.layer)
    hg = RD.h_g(d, args.layer)
    sid_all = d["state_id"]

    prep = SC.prepare(d, sp, args.layer, pca_dim=128, groups=groups)
    sid = SC.split_sid(d, sp)
    doc = {k: d["doc_id"][v] for k, v in sp.items()}
    rng_ctl = np.random.default_rng(999)
    ctl = SC.make_controls(prep, sid, rng_ctl)

    results = {"config": vars(args), "scenarios": {}}
    for sc in args.scenarios:
        t0 = time.time()
        rng = np.random.default_rng(hash(sc) % (2 ** 31))
        y_all = build(sc, hi, hg, sid_all, rng, snr=args.snr)
        y = {k: y_all[v] for k, v in sp.items()}
        entry = {}
        for selkind in ("pooled_r2", "within_r2"):
            R = SC.run_probes(prep, y, sid, doc, selkind,
                              which=PROBE_SETS[sc],
                              seeds=tuple(range(args.seeds)),
                              epochs=args.epochs)
            entry[selkind] = SC.score_all(R, y["test"], sid["test"])
            print(f"  [{sc}/{selkind}] {time.time()-t0:.0f}s "
                  f"{len(entry[selkind])} probes", flush=True)
        # key control for scenario B: shuffled h_global must go to zero
        if sc == "B_bilinear":
            R = SC.run_probes(prep, y, sid, doc, "within_r2",
                              which=["P0", "P2"],
                              seeds=tuple(range(args.seeds)),
                              epochs=args.epochs,
                              hg_override=ctl["shuffle_hg"])
            entry["ctl_shuffle_hg"] = SC.score_all(R, y["test"], sid["test"])
            R = SC.run_probes(prep, y, sid, doc, "within_r2",
                              which=["P0", "P2"],
                              seeds=tuple(range(args.seeds)),
                              epochs=args.epochs,
                              hg_override=ctl["gauss_hg"])
            entry["ctl_gauss_hg"] = SC.score_all(R, y["test"], sid["test"])
        results["scenarios"][sc] = entry
        json.dump(results, open(args.out, "w"), indent=2, default=float)
        print(f"[syn] {sc} done in {time.time()-t0:.0f}s", flush=True)

    results["scenarios"]["C_temporal"] = {
        "status": "PENDING",
        "reason": "needs h_{i,t-1}; label shards only store hidden states of the current step. "
                  "Trajectories are deterministic given the seed and can be replayed (collect_prev_hidden.py)."}
    json.dump(results, open(args.out, "w"), indent=2, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

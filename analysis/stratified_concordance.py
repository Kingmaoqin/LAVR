"""
Stratified concordance: does the stratified candidate set let the cheap baseline win "for free"?

The 6 candidates per state are 3 natural (uniformly sampled masked positions) + 3 informative
(high confidence x high instability). Of the 15 pairs, **the 9 cross-stratum natural x informative
pairs** may be separable by confidence alone -- and confidence is the first feature of C1.
If so, a substantial part of the cheap concordance of 0.7786 is this "stratum separability",
diluting the genuine candidate-level decision (within-stratum comparison) and compressing the room for hidden states.

This script splits concordance into three parts and evaluates each:
        within-natural    : 3 pairs among the 3 natural candidates
        within-informative: 3 pairs among the 3 informative candidates
        cross             : 9 natural x informative pairs
and reports cheap / cheap+H / perfect-predictor bounds for each.

Note: the `stratum` field is 0=natural, 1=informative (see src/collect.py:pick_candidates).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import probes2 as P            # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

ARMS = {"MDLM_anc": ["a3", "b3"], "SEDD_anc": ["s1", "s2"],
        "FRESH_MDLM_anc": ["freshA"]}


def conc_by_stratum(y, pred, groups, strat):
    """Returns concordance for {within_nat, within_inf, cross, all}."""
    acc = {k: [0.0, 0] for k in ("within_nat", "within_inf", "cross", "all")}
    for g in groups:
        yy, pp, ss = y[g], pred[g], strat[g]
        n = len(g)
        for a in range(n):
            for b in range(a + 1, n):
                dy = yy[a] - yy[b]
                if abs(dy) <= 1e-9:
                    continue
                dp = pp[a] - pp[b]
                v = 0.5 if abs(dp) <= 1e-12 else float(np.sign(dy) == np.sign(dp))
                if ss[a] == ss[b]:
                    key = "within_nat" if ss[a] == 0 else "within_inf"
                else:
                    key = "cross"
                acc[key][0] += v; acc[key][1] += 1
                acc["all"][0] += v; acc["all"][1] += 1
    return {k: (c / n if n else float("nan")) for k, (c, n) in acc.items()}, \
           {k: n for k, (c, n) in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["MDLM_anc", "SEDD_anc",
                                                  "FRESH_MDLM_anc"])
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--n_rep", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "stratified_concordance.json"))
    args = ap.parse_args()
    rep = {"config": vars(args), "arms": {}}

    for arm in args.arms:
        d = RD.load_labels(ARMS[arm])
        sp = RD.doc_splits(d, seed=0)
        RD.check_split_disjoint(d, sp)
        _, groups_all, _ = RD.state_groups(d["state_id"])
        sid = SC.split_sid(d, sp)
        y = SC.split_targets(d, sp, args.target)
        prep = SC.prepare(d, sp, args.layer, pca_dim=128, groups=groups_all)
        sel = P.make_selector("within_r2", sid["val"])
        m_ch = P.fit_ridge(prep["raw"]["cheap"]["train"], y["train"],
                           prep["raw"]["cheap"]["val"], y["val"],
                           prep["raw"]["cheap"]["test"], sel)
        H = {k: np.concatenate([prep["raw"]["hi"][k], prep["raw"]["hg"][k]], 1)
             for k in ("train", "val", "test")}
        m_h = P.fit_ridge_2block(prep["raw"]["cheap"]["train"], H["train"],
                                 y["train"], prep["raw"]["cheap"]["val"],
                                 H["val"], y["val"],
                                 prep["raw"]["cheap"]["test"], H["test"], sel)
        te = sp["test"]
        _, g_te = M.group_slices(sid["test"])
        strat = d["stratum"][te]
        yt = y["test"]

        c_cheap, npairs = conc_by_stratum(yt, m_ch["pred_test"], g_te, strat)
        c_hid, _ = conc_by_stratum(yt, m_h["pred_test"], g_te, strat)
        # use confidence only (C1 column 0, p1) as the ranker to quantify "stratum separability"
        p1 = d["C1"][te][:, 0]
        c_p1, _ = conc_by_stratum(yt, p1, g_te, strat)
        # split-half achievable bound (same stratification)
        sk = "A_full_seeds"
        seeds = d[sk][te].astype(np.float64)
        K = seeds.shape[1]; half = K // 2
        rng = np.random.default_rng(0)
        acc = {k: [] for k in ("within_nat", "within_inf", "cross", "all")}
        for _ in range(args.n_rep):
            perm = rng.permutation(K)
            ya = seeds[:, perm[:half]].mean(1)
            yb = seeds[:, perm[half:2 * half]].mean(1)
            c, _ = conc_by_stratum(yb, ya, g_te, strat)
            for k in acc:
                acc[k].append(c[k])
        c_split = {k: float(np.mean(v)) for k, v in acc.items()}

        ent = {"n_pairs": npairs, "cheap": c_cheap, "cheap+H": c_hid,
               "confidence_only": c_p1, "split_half_reliability": c_split,
               "delta_hidden": {k: c_hid[k] - c_cheap[k] for k in c_cheap}}
        rep["arms"][arm] = ent
        print(f"\n===== {arm} (layer {args.layer})")
        print(f"  {'subset':<18}{'n_pairs':>8}{'conf_only':>10}{'cheap':>9}"
              f"{'cheap+H':>10}{'dhidden':>10}{'split_half':>12}")
        for k in ("within_nat", "within_inf", "cross", "all"):
            print(f"  {k:<18}{npairs[k]:>8}{c_p1[k]:>10.4f}{c_cheap[k]:>9.4f}"
                  f"{c_hid[k]:>10.4f}{c_hid[k]-c_cheap[k]:>+10.4f}"
                  f"{c_split[k]:>12.4f}")
        json.dump(rep, open(args.out, "w"), indent=2, default=float)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

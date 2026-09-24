"""
**Ranking reliability ceiling** of the labels -- a key premise check for any null conclusion.

Motivation: every dconcordance is a "probe vs cheap control" difference. A more basic question is
rarely asked:

        Given that the label is a Monte Carlo mean over K=24 rollouts, what concordance can **any** predictor
        reach at most in within-state ranking?

If this ceiling is only 0.79 while cheap already reaches 0.7786, then "no probe beats cheap"
is a statement about **measurement precision**, not about the representation.
If the ceiling is 0.89, there is 0.11 of room above cheap that no probe captures, and the null result
is genuinely about the representation.

Method (no Gaussian approximation):
    1. Randomly split the K seeds into two disjoint halves A / B and average each to get A_bar_A, A_bar_B.
          Both share the signal with non-overlapping seed subsets; this is a reliability diagnostic, not an oracle label.
    2. `concordance(y = A_bar_B, pred = A_bar_A)` is the **split-half reliability** at noise level K/2,
          not an upper bound for predictors.
    3. Repeat for K/2 = 2, 3, 4, 6, 12 to trace concordance vs seeds per half,
          explaining the curve with an explicit reliability model; no linear "noise-free ceiling" extrapolation in 1/m.
    4. Also evaluate cheap / cheap+H probes against A_bar_B so the noise level is comparable.

Also reports a Gaussian reference: if (pred, label) are jointly Gaussian,
    concordance = 0.5 + arcsin(rho) / pi
convert the within-state noise ceiling (explainable variance fraction) into a concordance ceiling
and compare with the empirical value.
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

ARMS = {"MDLM_anc": ["a3", "b3"], "MDLM_conf": ["c3", "d3"],
        "SEDD_anc": ["s1", "s2"], "FRESH_MDLM_anc": ["freshA"]}
SEEDKEY = {"A_pertok": "A_full_seeds", "A_future": "A_future_seeds"}


def split_half_concordance(seeds, sid, groups, m, n_rep, rng):
    """Randomly split K seeds into two disjoint subsets of m; return the distribution of concordance."""
    K = seeds.shape[1]
    assert 2 * m <= K
    out = []
    for _ in range(n_rep):
        perm = rng.permutation(K)
        a, b = perm[:m], perm[m:2 * m]
        ya = seeds[:, a].mean(1)
        yb = seeds[:, b].mean(1)
        out.append(M.concordance(yb, ya, sid, groups))
    return np.array(out)


def gaussian_conc_from_rho(rho):
    return 0.5 + np.arcsin(np.clip(rho, -1, 1)) / np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["MDLM_anc", "SEDD_anc",
                                                  "FRESH_MDLM_anc"])
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--n_rep", type=int, default=40)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "label_reliability.json"))
    args = ap.parse_args()
    rep = {"config": vars(args), "arms": {}}

    for arm in args.arms:
        d = RD.load_labels(ARMS[arm])
        sid = d["state_id"]
        uniq, groups, _ = RD.state_groups(sid)
        sk = SEEDKEY[args.target]
        seeds = d[sk].astype(np.float64)
        K = seeds.shape[1]
        rng = np.random.default_rng(0)
        ent = {"tags": ARMS[arm], "K": int(K), "n_states": int(len(uniq)),
               "n_rows": int(len(sid)), "curve": {}}

        # ---- 1. split-half curve ----
        ms = [m for m in (2, 3, 4, 6, 8, 12) if 2 * m <= K]
        for m in ms:
            v = split_half_concordance(seeds, sid, groups, m, args.n_rep, rng)
            ent["curve"][str(m)] = {"mean": float(v.mean()),
                                    "sd": float(v.std()),
                                    "n_rep": int(len(v))}
            print(f"[{arm}] m={m:2d} seeds per half: concordance "
                  f"{v.mean():.4f} ± {v.std():.4f}", flush=True)

        # ---- 2. infer the noise model from the split-half curve, then compute the perfect-predictor ceiling ----
        #
        # Note: linearly extrapolating conc(1/m) (which gives 0.79) is wrong for two reasons:
        #   (a) conc is convex in 1/m, so linear extrapolation to 1/m->0 is biased low;
        #   (b) more fundamentally, it answers the wrong question -- split-half concordance measures
        #       ranking with an **equally noisy** measurement of A, not a perfect predictor.
        #       As m->inf both halves become noise-free and the true limit is 1.0.
        #
        #
        # Correct approach: split-half correlation is the reliability r(m) = s2_s / (s2_s + s2_n/m).
        # If (prediction, label) are approximately jointly Gaussian, concordance = 0.5 + arcsin(rho)/pi.
        # Hence:
        #   * noisy predictor (m seeds) vs noisy label (m seeds): rho = r(m)
        #   * **perfect** predictor vs the K-seed label actually used: rho = sqrt(r(K))
        # The latter is the achievable ceiling for any probe on this data.
        x = np.array([1.0 / m for m in ms])
        y = np.array([ent["curve"][str(m)]["mean"] for m in ms])
        ent["curve_fit_note"] = ("conc(1/m) is convex; linear extrapolation is meaningless; "
                                 "use the reliability model + arcsin transform instead")
        # within-state noise ceiling gives s2_n/s2_s
        wc = RD.within_state_noise_ceiling(seeds, sid, groups)
        rK = float(np.clip(wc["ceiling"], 1e-9, 1 - 1e-9))   # r(K)
        ratio = (1.0 / rK - 1.0) * K                          # σ²_n / σ²_s
        ent["noise_to_signal_ratio"] = float(ratio)
        pred_curve = {str(m): float(gaussian_conc_from_rho(1.0 / (1.0 + ratio / m)))
                      for m in ms}
        ent["curve_predicted_by_model"] = pred_curve
        err = max(abs(pred_curve[str(m)] - ent["curve"][str(m)]["mean"])
                  for m in ms)
        ent["curve_model_max_abs_error"] = float(err)
        print(f"[{arm}] noise/signal ratio = {ratio:.2f}; max deviation between model-predicted and observed"
              f" split-half curve {err:.4f}", flush=True)
        for m in ms:
            print(f"          m={m:2d} observed {ent['curve'][str(m)]['mean']:.4f}"
                  f"  model {pred_curve[str(m)]:.4f}", flush=True)

        # ---- 3. concordance ceiling of a perfect predictor ----
        ent["within_ceiling_r2"] = wc
        rho_max = float(np.sqrt(rK))
        ent["conc_ceiling_perfect_predictor"] = float(
            gaussian_conc_from_rho(rho_max))
        print(f"[{arm}] within-state R2 ceiling r(K)={rK:.4f} "
              f"-> correlation of perfect predictor with K={K} label rho={rho_max:.4f} "
              f"-> **concordance ceiling {ent['conc_ceiling_perfect_predictor']:.4f}**",
              flush=True)

        # ---- 4. probes under the same convention ----
        sp = RD.doc_splits(d, seed=0)
        RD.check_split_disjoint(d, sp)
        s_ = SC.split_sid(d, sp)
        y_full = {k: d[args.target][v] for k, v in sp.items()}
        prep = SC.prepare(d, sp, args.layer, pca_dim=128, groups=groups)
        sel = P.make_selector("within_r2", s_["val"])
        m_ch = P.fit_ridge(prep["raw"]["cheap"]["train"], y_full["train"],
                           prep["raw"]["cheap"]["val"], y_full["val"],
                           prep["raw"]["cheap"]["test"], sel)
        H = {k: np.concatenate([prep["raw"]["hi"][k], prep["raw"]["hg"][k]], 1)
             for k in ("train", "val", "test")}
        m_h = P.fit_ridge_2block(prep["raw"]["cheap"]["train"], H["train"],
                                 y_full["train"], prep["raw"]["cheap"]["val"],
                                 H["val"], y_full["val"],
                                 prep["raw"]["cheap"]["test"], H["test"], sel)
        te = sp["test"]
        _, g_te = M.group_slices(s_["test"])
        # evaluate with the full-K label (standard convention)
        c_cheap = M.concordance(y_full["test"], m_ch["pred_test"],
                                s_["test"], g_te)
        c_hid = M.concordance(y_full["test"], m_h["pred_test"],
                              s_["test"], g_te)
        ceil = ent["conc_ceiling_perfect_predictor"]
        ent["probe_vs_full_label"] = {
            "cheap": c_cheap, "cheap+H": c_hid, "ceiling": ceil,
            "headroom_above_cheap": float(ceil - c_cheap),
            "hidden_gain": float(c_hid - c_cheap),
            "frac_of_headroom_captured_by_hidden":
                float((c_hid - c_cheap) / max(ceil - c_cheap, 1e-9))}
        print(f"[{arm}] full-K label: cheap {c_cheap:.4f}  cheap+H {c_hid:.4f}"
              f"  ceiling {ceil:.4f}  ->  room above cheap {ceil-c_cheap:+.4f}, "
              f"captured by hidden states: "
              f"{100*(c_hid-c_cheap)/max(ceil-c_cheap,1e-9):.1f}%", flush=True)
        # evaluate with the half-B label (same noise level as the split-half ceiling)
        half = K // 2
        rng2 = np.random.default_rng(7)
        cb, chh, ceil_te = [], [], []
        for _ in range(args.n_rep):
            perm = rng2.permutation(K)
            a, b = perm[:half], perm[half:2 * half]
            yb = seeds[te][:, b].mean(1)
            ya = seeds[te][:, a].mean(1)
            cb.append(M.concordance(yb, m_ch["pred_test"], s_["test"], g_te))
            chh.append(M.concordance(yb, m_h["pred_test"], s_["test"], g_te))
            ceil_te.append(M.concordance(yb, ya, s_["test"], g_te))
        ent["on_test_half_label"] = {
            "cheap": float(np.mean(cb)), "cheap+H": float(np.mean(chh)),
            "split_half_reliability": float(np.mean(ceil_te)),
            "gap_over_cheap_NOT_HEADROOM": float(np.mean(ceil_te) - np.mean(cb))}
        print(f"[{arm}] on test (half-B label as truth): cheap "
              f"{np.mean(cb):.4f}  cheap+H {np.mean(chh):.4f}  "
              f"split-half reliability {np.mean(ceil_te):.4f}  "
              f"difference from cheap {np.mean(ceil_te)-np.mean(cb):+.4f}", flush=True)

        rep["arms"][arm] = ent
        json.dump(rep, open(args.out, "w"), indent=2, default=float)

    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

"""
Experiment 2 -- matched candidates.

Mine candidate PAIRS (i, j) from the same state s_t whose exposed prediction
signals are tightly matched -- confidence, entropy, top1-top2 margin, output
distribution projection distance, temporal KL / persistence / flip history,
timestep and mask ratio -- but whose ORACLE advantages differ materially.

Then ask whether the frozen hidden state's linear probe still says which of the
two has the larger true advantage, after the exposed signals have been matched
away. A result only counts if hidden adds separation AFTER matching.

Baselines: confidence; combined scalar+history controls; output-distribution
controls. (A TraceLock-style future-stability signal is reported where the
trajectory history supports it -- flip count / persistence are exactly the
realised-trajectory stability statistics that method keys on.)
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import dataset as D                                                # noqa: E402
import probes as P                                                 # noqa: E402


def mine_pairs(d, idx, y, cheap_pred, tol, min_gap_sd, mode="caliper",
               max_per_state=6, seed=0):
    """Pairs within one state s_t: matched on exposed signals, split on oracle A.

    Two matching modes, both reported:

    `caliper` (primary) -- match on the CHEAP-CONTROL PREDICTION, the single
      sufficient statistic that all exposed signals (confidence, entropy,
      margins, trajectory history, top-k log-probs, full log-prob projection)
      collectively provide, plus a hard caliper on raw confidence and entropy.
      This asks the decision-relevant question: among candidates the exposed
      signals rate EQUALLY, does the frozen hidden state know which is better?

    `strict` (secondary) -- every one of the 18 individual exposed coordinates
      must agree within `tol` standardized units. Far fewer pairs survive, so
      it is reported as a stricter but lower-powered check.
    """
    rng = np.random.default_rng(seed)
    gap_scale = y[idx].std()

    coords = np.concatenate([
        d["C1"][:, [0, 1, 2, 3, 4]],        # p1, logp1, entropy, both margins
        d["C2"][:, [0, 1, 7, 8, 9]],        # t, mask ratio, tKL, flips, persist
        d["C3"][:, :8],                     # top-8 log-probs
    ], 1)[idx]
    cz = (coords - coords.mean(0)) / np.maximum(coords.std(0), 1e-8)
    proj = d["C3"][idx, 16:]
    pz = (proj - proj.mean(0)) / np.maximum(proj.std(0), 1e-8)
    proj_tol = np.sqrt(pz.shape[1]) * tol

    cp = cheap_pred[idx]
    cp_scale = max(cp.std(), 1e-12)
    conf_z = cz[:, :3]                       # p1, logp1, entropy

    sid = d["state_id"][idx]
    pairs = []
    for st in np.unique(sid):
        loc = np.where(sid == st)[0]
        if len(loc) < 2:
            continue
        got = 0
        unordered = list(itertools.combinations(loc.tolist(), 2))
        for z in rng.permutation(len(unordered)):
            if got >= max_per_state:
                break
            a, b = unordered[z]
            if abs(y[idx][a] - y[idx][b]) < min_gap_sd * gap_scale:
                continue
            if mode == "caliper":
                if abs(cp[a] - cp[b]) > tol * cp_scale:
                    continue
                if np.max(np.abs(conf_z[a] - conf_z[b])) > tol:
                    continue
            else:
                if np.max(np.abs(cz[a] - cz[b])) > tol:
                    continue
                if np.linalg.norm(pz[a] - pz[b]) > proj_tol:
                    continue
            pairs.append((idx[a], idx[b]))
            got += 1
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def pair_accuracy(y, score, pairs):
    """Fraction of matched pairs whose ordering the score gets right."""
    dy = y[pairs[:, 0]] - y[pairs[:, 1]]
    ds = score[pairs[:, 0]] - score[pairs[:, 1]]
    ok = np.abs(dy) > 1e-12
    dy, ds = dy[ok], ds[ok]
    ties = np.abs(ds) <= 1e-12          # ties score as chance, not as wrong
    acc = (float((np.sign(dy[~ties]) == np.sign(ds[~ties])).sum())
           + 0.5 * ties.sum()) / max(len(dy), 1)
    return float(acc), int(len(dy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["a", "b"])
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--match_tol", type=float, default=0.25)
    ap.add_argument("--min_gap_sd", type=float, default=1.0)
    ap.add_argument("--mode", choices=["caliper", "strict"], default="caliper")
    ap.add_argument("--ladder", action="store_true",
                    help="report the matching-tolerance ladder and stop")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--placebo", action="store_true",
                    help="PLACEBO CONTROL: replace the hidden block with "
                         "Gaussian noise of the same shape. Pairs are mined by "
                         "requiring the cheap prediction to be EQUAL while the "
                         "true label differs, i.e. deliberately selecting the "
                         "pairs cheap gets WRONG. Any block that was not "
                         "matched on could therefore look good regardless of "
                         "content. If the placebo also beats the controls, the "
                         "procedure is biased and no matched-pair result from "
                         "it can be trusted.")
    args = ap.parse_args()

    outdir = args.out or os.path.join(ROOT, "results", "exp2")
    if args.tag:
        outdir = outdir + "_" + args.tag
    os.makedirs(outdir, exist_ok=True)
    d = D.load_labels(args.tags)
    y = d[args.target]
    nL = d["n_layers"]
    print(f"[exp2] {len(y)} examples, target={args.target}, "
          f"match_tol={args.match_tol} sd, min_gap={args.min_gap_sd} sd")

    if args.ladder:
        sp = D.doc_splits(d, seed=0)
        X = D.block(d, "cheap")
        m = P.fit_linear(X[sp["train"]], y[sp["train"]],
                         X[sp["val"]], y[sp["val"]])
        cp = P.predict(m, X)
        print(f"\n{'mode':>9}{'tol':>7}{'min_gap_sd':>12}{'n_pairs':>9}")
        for mode in ("caliper", "strict"):
            for tol in (0.05, 0.1, 0.25, 0.5, 1.0):
                for g in (0.5, 1.0):
                    n = len(mine_pairs(d, sp["test"], y, cp, tol, g, mode=mode))
                    print(f"{mode:>9}{tol:>7.2f}{g:>12.1f}{n:>9}")
        return

    rows, all_pairs, primary = {}, [], None
    for seed in range(args.seeds):
        sp = D.doc_splits(d, seed=seed)
        tr, va, te = sp["train"], sp["val"], sp["test"]

        # choose the probe layer on validation only
        Xc = D.block(d, "cheap")
        val = []
        for l in range(nL):
            Xh = D.block(d, "H", l)
            val.append(P.fit_linear_2block(Xc[tr], Xh[tr], y[tr],
                                           Xc[va], Xh[va], y[va])["val_r2"])
        L = int(np.argmax(val))

        scores = {}
        scores["confidence (p1)"] = d["C1"][:, 0]
        scores["logp1"] = d["C1"][:, 1]
        for nm, blk in [("scalar+history controls", "C1C2"),
                        ("output-distribution controls", "C3"),
                        ("ALL cheap controls", "cheap")]:
            X = D.block(d, blk)
            m = P.fit_linear(X[tr], y[tr], X[va], y[va])
            scores[nm] = P.predict(m, X)
        # trajectory-stability signal (TraceLock-style): persistence / flips
        scores["future-stability (persist - flips)"] = (
            d["C2"][:, 9] - d["C2"][:, 8])
        Xh = D.block(d, "H", L)
        Xl = D.block(d, "H_local", L)
        if args.placebo:
            rng_p = np.random.default_rng(1000 + seed)
            Xh = rng_p.standard_normal(Xh.shape).astype(np.float32)
            Xl = rng_p.standard_normal(Xl.shape).astype(np.float32)
        m2 = P.fit_linear_2block(Xc[tr], Xh[tr], y[tr], Xc[va], Xh[va], y[va])
        scores["HIDDEN [h_i; h_global]"] = P.predict_2block(m2, Xc, Xh)
        # The decisive candidate-level score. h_global is CONSTANT across the
        # candidates of a state, so it cannot discriminate a matched pair
        # directly -- but including it changes the fitted coefficients on the
        # cheap block and can therefore move the ranking anyway. Only h_i is
        # genuinely per-candidate, so this is the score that tests the actual
        # hypothesis, and it is the one kept as "HIDDEN linear probe".
        m3 = P.fit_linear_2block(Xc[tr], Xl[tr], y[tr], Xc[va], Xl[va], y[va])
        scores["HIDDEN linear probe"] = P.predict_2block(m3, Xc, Xl)

        pairs = mine_pairs(d, te, y, scores["ALL cheap controls"],
                           args.match_tol, args.min_gap_sd,
                           mode=args.mode, seed=seed)
        all_pairs.append(len(pairs))
        if len(pairs) < 20:
            print(f"  seed {seed}: only {len(pairs)} matched pairs -- "
                  f"loosen --match_tol or collect more states")
            continue
        print(f"  seed {seed}: layer {L}, {len(pairs)} matched pairs "
              f"on held-out test ({args.mode} matching, tol={args.match_tol})")
        for nm, sc in scores.items():
            acc, n = pair_accuracy(y, sc, pairs)
            rows.setdefault(nm, []).append(acc)

        # Select the comparison control on VALIDATION matched pairs, never on
        # the test pairs whose gap will be reported.  Seed 0 is the single
        # primary estimand; other overlapping re-splits are sensitivity only.
        val_pairs = mine_pairs(d, va, y, scores["ALL cheap controls"],
                               args.match_tol, args.min_gap_sd,
                               mode=args.mode, seed=10_000 + seed)
        controls = [k for k in scores if "HIDDEN" not in k]
        best_ctrl = (max(controls, key=lambda k: pair_accuracy(y, scores[k], val_pairs)[0])
                     if len(val_pairs) else "ALL cheap controls")
        if seed == 0:
            primary = {"pairs": pairs, "scores": scores, "best_ctrl": best_ctrl,
                       "layer": L, "n_val_pairs": len(val_pairs)}

    print(f"\n{'score used to order the matched pair':<40}{'accuracy':>10}"
          f"{'per-seed':>28}")
    summary = {}
    for nm, accs in rows.items():
        summary[nm] = {"mean": float(np.mean(accs)), "per_seed": accs}
        print(f"{nm:<40}{np.mean(accs):>10.4f}   {np.round(accs,4).tolist()}")

    # ---- inference on one pre-specified primary split (seed 0) --------------
    if primary is not None:
        pairs, scores = primary["pairs"], primary["scores"]
    if primary is not None and len(pairs) >= 20:
        docs_p = d["doc_id"][pairs[:, 0]]

        # NOTE on circularity: `caliper` matching equalises the CHEAP-CONTROL
        # PREDICTION across the pair by construction, so the cheap predictor is
        # forced to ~chance on exactly these pairs. The "gap over cheap" is
        # therefore partly mechanical under that mode. The primary quantity is
        # the direct one: is the HIDDEN probe above chance among candidates the
        # exposed signals rate equally? `strict` mode does not match on the
        # cheap prediction, so its gap is not circular -- but far fewer pairs
        # survive it, so it is lower powered.
        def f_abs(sel):
            return pair_accuracy(y, scores["HIDDEN linear probe"],
                                 pairs[sel % len(pairs)])[0] - 0.5

        ma, la, ha = P.cluster_bootstrap(f_abs, docs_p, n_boot=args.n_boot,
                                         seed=0)
        print(f"\n  hidden probe accuracy above chance: {0.5+ma:.4f} "
              f"(excess {ma:+.4f})  95% CI on excess [{la:+.4f}, {ha:+.4f}]  "
              f"{'EXCLUDES 0' if (la>0 or ha<0) else 'includes 0'}")
        summary["_hidden_above_chance"] = {
            "accuracy": 0.5 + ma, "excess": ma, "ci_lo": la, "ci_hi": ha,
            "ci_excludes_zero": bool(la > 0 or ha < 0)}

        def f(sel):
            sub = pairs[sel % len(pairs)]
            a = pair_accuracy(y, scores["HIDDEN linear probe"], sub)[0]
            b = pair_accuracy(y, scores["ALL cheap controls"], sub)[0]
            return a - b

        m, lo, hi = P.cluster_bootstrap(f, docs_p, n_boot=args.n_boot, seed=0)
        print(f"  matched-pair accuracy gap (hidden - cheap): "
              f"{m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'EXCLUDES 0' if (lo>0 or hi<0) else 'includes 0'}")
        summary["_gap_hidden_minus_cheap"] = {
            "mean": m, "ci_lo": lo, "ci_hi": hi,
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}

        # strongest single exposed-signal baseline, selected on VALIDATION
        best_ctrl = primary["best_ctrl"]
        def f_bc(sel):
            sub = pairs[sel % len(pairs)]
            return (pair_accuracy(y, scores["HIDDEN linear probe"], sub)[0]
                    - pair_accuracy(y, scores[best_ctrl], sub)[0])
        mb, lb, hb = P.cluster_bootstrap(f_bc, docs_p, n_boot=args.n_boot,
                                         seed=0)
        print(f"  gap over the BEST single control ({best_ctrl}): "
              f"{mb:+.4f}  95% CI [{lb:+.4f}, {hb:+.4f}]  "
              f"{'EXCLUDES 0' if (lb>0 or hb<0) else 'includes 0'}")
        summary["_gap_vs_best_control"] = {
            "control": best_ctrl, "mean": mb, "ci_lo": lb, "ci_hi": hb,
            "ci_excludes_zero": bool(lb > 0 or hb < 0)}
        primary_acc = pair_accuracy(y, scores["HIDDEN linear probe"], pairs)[0]
        summary["_primary_seed0"] = {
            "hidden_accuracy": primary_acc, "n_unique_pairs": int(len(pairs)),
            "selected_layer": int(primary["layer"]),
            "best_control_selected_on_validation": best_ctrl,
            "n_validation_pairs": int(primary["n_val_pairs"])}

    summary["_n_pairs_per_seed"] = all_pairs
    summary["_config"] = vars(args)
    with open(os.path.join(outdir, "exp2_report.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {outdir}/exp2_report.json")


if __name__ == "__main__":
    main()

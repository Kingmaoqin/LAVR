"""
Experiment 1 -- is rollout-defined future path value linearly decodable from the
frozen state, BEYOND strong output-side and trajectory controls?

The key comparison is NOT `hidden > confidence`. It is

    cheap controls + output distribution
        versus
    cheap controls + output distribution + hidden representation

reported as Delta_R2 and Delta_Spearman with cluster-bootstrap CIs, plus the
layer x diffusion-timestep decodability heatmap.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import dataset as D                                                # noqa: E402
import probes as P                                                 # noqa: E402


def _score(d, te, y, pred, extra=None):
    conc, npairs = P.within_state_concordance(y[te], pred, d["state_id"][te])
    out = {"r2": P.r2_score(y[te], pred), "spearman": P.spearman(y[te], pred),
           "within_r2": P.within_state_r2(y[te], pred, d["state_id"][te]),
           "concordance": conc, "n_pairs": npairs, "pred": pred, "test": te}
    out.update(extra or {})
    return out


def eval_block(d, sp, name, target, layer=None, mlp=False):
    """Single-block probe (controls, or hidden alone)."""
    X = D.block(d, name, layer)
    y = d[target]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    if mlp:
        m = P.fit_mlp(X[tr], y[tr], X[va], y[va])
        pred = P.predict_mlp(m, X[te])
    else:
        m = P.fit_linear(X[tr], y[tr], X[va], y[va])
        pred = P.predict(m, X[te])
    return _score(d, te, y, pred, {"alpha": float(m.get("alpha", np.nan))})


def eval_2block(d, sp, hidden_name, target, layer):
    """Controls + hidden with SEPARATE regularisation per block.

    The gamma grid includes 0, so this model NESTS the controls-only model and
    Delta_R2 is not biased negative by the hidden block's dimensionality. This
    was verified on synthetic data: with a pure-noise hidden block the shared-
    alpha version loses 0.06 R^2, the nested version loses 0.0004.
    """
    y = d[target]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    Xc, Xh = D.block(d, "cheap"), D.block(d, hidden_name, layer)
    m = P.fit_linear_2block(Xc[tr], Xh[tr], y[tr], Xc[va], Xh[va], y[va])
    pred = P.predict_2block(m, Xc[te], Xh[te])
    return _score(d, te, y, pred,
                  {"alpha": float(m["alpha"]), "gamma": float(m["gamma"])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["a", "b"])
    # V is a pre-registered SECONDARY target: separating "the state value is
    # decodable" from "the action advantage is decodable" is the sharpest
    # distinction this study can draw.
    ap.add_argument("--targets", nargs="+",
                    default=["A_pertok", "A_future", "V_pertok"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--refit_boot", type=int, default=0,
                    help="bootstrap replicates that RE-SPLIT and RE-FIT both "
                         "models. The default bootstrap holds both fitted "
                         "models fixed and resamples only the test set, so it "
                         "captures test-sampling variability but NOT fitting "
                         "variability, and its CI is too narrow. Applied to "
                         "the first target only. Each replicate repeats layer "
                         "selection, so this is intentionally expensive.")
    ap.add_argument("--null", choices=["none", "global", "within_state"],
                    default="none",
                    help="negative control: permute the labels. The pipeline "
                         "MUST report Delta_R2 ~ 0 and concordance ~ 0.5, "
                         "otherwise there is a leak or a bug.")
    args = ap.parse_args()

    outdir = args.out or os.path.join(
        ROOT, "results", "exp1" if args.null == "none" else f"exp1_null_{args.null}")
    os.makedirs(outdir, exist_ok=True)
    d = D.load_labels(args.tags)
    nL = d["n_layers"]
    if args.null != "none":
        rng = np.random.default_rng(20260819)
        for tgt in args.targets:
            if args.null == "global":
                d[tgt] = rng.permutation(d[tgt])
            else:                       # destroy within-state ranking only
                y = d[tgt].copy()
                for s_ in np.unique(d["state_id"]):
                    m = np.where(d["state_id"] == s_)[0]
                    y[m] = rng.permutation(y[m])
                d[tgt] = y
        print(f"[exp1] NEGATIVE CONTROL active: labels permuted ({args.null})")
    print(f"[exp1] {len(d['A_pertok'])} examples, {len(np.unique(d['doc_id']))} "
          f"docs, {len(np.unique(d['state_id']))} states, {nL} layers")

    report = {"n_examples": int(len(d["A_pertok"])),
              "n_docs": int(len(np.unique(d["doc_id"]))),
              "n_states": int(len(np.unique(d["state_id"])))}

    # ---- label quality: what R^2 is even attainable? -----------------------
    SEEDKEY = {"A_pertok": "A_full_seeds", "A_future": "A_future_seeds",
               "A_task": "A_task_seeds"}
    for tgt in args.targets:
        sk = SEEDKEY.get(tgt)
        if sk is None or sk not in d:
            report[f"noise_ceiling_{tgt}"] = float("nan")
            print(f"[exp1] noise ceiling for {tgt}: n/a "
                  f"(no per-seed replicates stored)")
            continue
        ceil, noise, obs = P.noise_ceiling(d[sk])
        print(f"[exp1] noise ceiling for {tgt}: max attainable R2 = {ceil:.3f} "
              f"(label noise var {noise:.2e} of observed {obs:.2e})")
        report[f"noise_ceiling_{tgt}"] = ceil

    blocks = [("C1", "confidence scalars"),
              ("C1C2", "confidence + trajectory"),
              ("C3", "output distribution"),
              ("cheap", "ALL cheap controls (C1+C2+C3)")]

    for target in args.targets:
        print(f"\n{'='*74}\nTARGET {target}\n{'='*74}")
        res = {b: [] for b, _ in blocks}
        for k in ("cheap+H", "H", "H_local", "H_global",
                  "cheap+H_local", "cheap+H_mlp"):
            res[k] = []
        best_layers = []
        for seed in range(args.seeds):
            sp = D.doc_splits(d, seed=seed)
            # pick the probe layer on VALIDATION only, never on test
            tr_, va_ = sp["train"], sp["val"]
            Xc = D.block(d, "cheap")
            val_r2 = []
            for l in range(nL):
                Xh = D.block(d, "H", l)
                val_r2.append(P.fit_linear_2block(
                    Xc[tr_], Xh[tr_], d[target][tr_],
                    Xc[va_], Xh[va_], d[target][va_])["val_r2"])
            L = int(np.argmax(val_r2))
            best_layers.append(L)
            for b, _ in blocks:
                res[b].append(eval_block(d, sp, b, target))
            res["H"].append(eval_block(d, sp, "H", target, L))
            res["H_local"].append(eval_block(d, sp, "H_local", target, L))
            res["H_global"].append(eval_block(d, sp, "H_global", target, L))
            res["cheap+H_local"].append(
                eval_2block(d, sp, "H_local", target, L))
            res["cheap+H"].append(eval_2block(d, sp, "H", target, L))
            res["cheap+H_mlp"].append(
                eval_block(d, sp, "cheap+H", target, L, mlp=True))
            print(f"    gamma (hidden-block scale) = "
                  f"{res['cheap+H'][-1]['gamma']:.3g}")
            note = "  (NB layer 0 h_i is the MASK embedding: constant across "\
                   "masked positions, so it can carry no per-candidate signal)" \
                if L == 0 else ""
            print(f"  seed {seed}: probe layer chosen on val = {L}{note}")

        print(f"\n{'feature block':<34}{'R2':>9}{'Spearman':>10}"
              f"{'within-R2':>11}{'concord':>9}")
        rows = {}
        for key, lbl in blocks + [
                ("H_global", "h_global only (cannot rank within a state)"),
                ("H_local", "h_i only (candidate position)"),
                ("H", "hidden only [h_i; h_global]"),
                ("cheap+H_local", "cheap + h_i only"),
                ("cheap+H", "cheap + HIDDEN  <-- primary"),
                ("cheap+H_mlp", "cheap + hidden, MLP (secondary)")]:
            r2 = np.mean([r["r2"] for r in res[key]])
            sp_ = np.mean([r["spearman"] for r in res[key]])
            wr = np.mean([r["within_r2"] for r in res[key]])
            cc = np.mean([r["concordance"] for r in res[key]])
            rows[key] = {"r2": r2, "spearman": sp_, "concordance": cc,
                         "within_r2": wr,
                         "r2_by_seed": [r["r2"] for r in res[key]],
                         "within_r2_by_seed": [r["within_r2"] for r in res[key]],
                         "spearman_by_seed": [r["spearman"] for r in res[key]],
                         "concordance_by_seed": [r["concordance"] for r in res[key]]}
            fmt = lambda v, w: (f"{v:>{w}.4f}" if np.isfinite(v)
                                else f"{'n/a':>{w}}")
            print(f"{lbl:<34}{r2:>9.4f}{sp_:>10.4f}"
                  f"{fmt(wr,11)}{fmt(cc,9)}")

        # ---- primary incremental statistic, paired cluster bootstrap -------
        print(f"\n  PRIMARY: Delta = (cheap + hidden) - (cheap)")
        deltas = {}
        for stat in ["r2", "spearman", "within_r2", "concordance"]:
            per_seed = [res["cheap+H"][s][stat] - res["cheap"][s][stat]
                        for s in range(args.seeds)]
            # bootstrap on seed 0's test set, resampling documents
            r_h, r_c = res["cheap+H"][0], res["cheap"][0]
            te = r_h["test"]
            y = d[target][te]
            docs_te = d["doc_id"][te]
            sid = d["state_id"][te]

            def make(stat=stat, y=y, r_h=r_h, r_c=r_c, sid=sid):
                def f(idx):
                    if stat == "r2":
                        return (P.r2_score(y[idx], r_h["pred"][idx])
                                - P.r2_score(y[idx], r_c["pred"][idx]))
                    if stat == "spearman":
                        return (P.spearman(y[idx], r_h["pred"][idx])
                                - P.spearman(y[idx], r_c["pred"][idx]))
                    if stat == "within_r2":
                        return (P.within_state_r2(y[idx], r_h["pred"][idx], sid[idx])
                                - P.within_state_r2(y[idx], r_c["pred"][idx], sid[idx]))
                    a = P.within_state_concordance(y[idx], r_h["pred"][idx], sid[idx])[0]
                    b = P.within_state_concordance(y[idx], r_c["pred"][idx], sid[idx])[0]
                    return a - b
                return f

            if not np.isfinite(np.mean(per_seed)):
                deltas[stat] = {"per_seed": per_seed, "mean": float("nan"),
                                "ci_excludes_zero": False}
                print(f"    Delta_{stat:<12} n/a (undefined for this target)")
                continue
            mB, lo, hi = P.cluster_bootstrap(make(), docs_te,
                                             n_boot=args.n_boot, seed=0)
            excl = (lo > 0) or (hi < 0)
            deltas[stat] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)),
                            "boot_mean": mB, "ci_lo": lo, "ci_hi": hi,
                            "ci_excludes_zero": bool(excl)}
            print(f"    Delta_{stat:<12} per-seed {np.round(per_seed,4).tolist()}  "
                  f"mean {np.mean(per_seed):+.4f}  boot 95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"  {'EXCLUDES 0' if excl else 'includes 0'}")

        # ---- calibration of the predicted advantage ------------------------
        rh = res["cheap+H"][0]
        te0, pr0 = rh["test"], rh["pred"]
        y0 = d[target][te0]
        qs = np.quantile(pr0, np.linspace(0, 1, 11))
        qs[-1] += 1e-9
        binid = np.clip(np.digitize(pr0, qs[1:-1]), 0, 9)
        cal = []
        for b in range(10):
            m = binid == b
            if m.sum():
                cal.append((float(pr0[m].mean()), float(y0[m].mean()),
                            int(m.sum())))
        slope = float(np.polyfit(pr0, y0, 1)[0]) if np.std(pr0) > 1e-12 else 0.0
        print(f"  calibration of predicted advantage (decile bins), "
              f"slope of actual on predicted = {slope:.3f} (1.0 = calibrated)")
        print("     pred  " + " ".join(f"{c[0]:+7.4f}" for c in cal))
        print("     true  " + " ".join(f"{c[1]:+7.4f}" for c in cal))
        report[f"calibration_{target}"] = {"slope": slope, "deciles": cal}

        # ---- SECONDARY diagnostic: Gaussian conditional MI -----------------
        # Explicitly NOT the gate (we do not make high-dimensional
        # CMI the decision rule -- it is estimator-sensitive). Reported as the
        # linear-Gaussian transform of the same incremental R2.
        r2c = rows["cheap"]["r2"]
        r2h = rows["cheap+H"]["r2"]
        frac = (r2h - r2c) / max(1.0 - r2c, 1e-9)
        cmi = (-0.5 * np.log(max(1.0 - frac, 1e-12))
               if 0 < frac < 1 else max(frac, 0.0))
        print(f"  [secondary, non-decisive] Gaussian I(A ; hidden | cheap) "
              f"~ {cmi:.4f} nats")
        report[f"cmi_gaussian_{target}"] = float(cmi)

        # ---- naturally-sampled stratum only --------------------------------
        te = res["cheap+H"][0]["test"]
        nat = d["stratum"][te] == 0
        y = d[target][te]
        nat_h = P.r2_score(y[nat], res["cheap+H"][0]["pred"][nat])
        nat_c = P.r2_score(y[nat], res["cheap"][0]["pred"][nat])
        print(f"  held-out NATURAL stratum only (n={int(nat.sum())}): "
              f"R2 cheap {nat_c:.4f} -> cheap+hidden {nat_h:.4f} "
              f"(Delta {nat_h-nat_c:+.4f})")

        report[target] = {"blocks": rows, "deltas": deltas,
                          "best_layers": best_layers,
                          "natural_stratum": {"r2_cheap": nat_c,
                                              "r2_cheap_hidden": nat_h,
                                              "n": int(nat.sum())}}

    # ---- honest inference: bootstrap that RE-SPLITS and RE-FITS ------------
    if args.refit_boot:
        tgt = args.targets[0]
        Xc = D.block(d, "cheap")
        y = d[tgt]
        # Each replicate creates NEW disjoint document pools, resamples only
        # within those pools, and repeats validation layer selection.  This is
        # a full pipeline refit bootstrap rather than the former fixed-pool,
        # fixed-layer procedure.
        docs = np.unique(d["doc_id"])
        n_tr, n_va = int(.60*len(docs)), int(.15*len(docs))
        rows_by_doc = {x: np.where(d["doc_id"] == x)[0]
                       for x in docs}
        rng = np.random.default_rng(7)
        stats = {"r2": [], "within_r2": [], "concordance": []}
        chosen_layers = []
        for it in range(args.refit_boot):
            perm = rng.permutation(docs)
            pools = {"train": perm[:n_tr], "val": perm[n_tr:n_tr+n_va],
                     "test": perm[n_tr+n_va:]}
            def draw(k):
                pk = rng.choice(pools[k], len(pools[k]), replace=True)
                return np.concatenate([rows_by_doc[x] for x in pk])
            tr_, va_, te_ = draw("train"), draw("val"), draw("test")
            assert not (set(d["doc_id"][tr_]) & set(d["doc_id"][te_])), \
                "train/test document overlap in bootstrap replicate"
            if min(len(tr_), len(va_), len(te_)) < 50:
                continue
            val_scores = []
            layer_models = []
            for l in range(nL):
                Xhl = D.block(d, "H", l)
                ml = P.fit_linear_2block(Xc[tr_], Xhl[tr_], y[tr_],
                                         Xc[va_], Xhl[va_], y[va_])
                val_scores.append(ml["val_r2"]); layer_models.append(ml)
            L = int(np.argmax(val_scores)); chosen_layers.append(L)
            Xh = D.block(d, "H", L)
            mc = P.fit_linear(Xc[tr_], y[tr_], Xc[va_], y[va_])
            pc = P.predict(mc, Xc[te_])
            mh = P.fit_linear_2block(Xc[tr_], Xh[tr_], y[tr_],
                                     Xc[va_], Xh[va_], y[va_])
            ph = P.predict_2block(mh, Xc[te_], Xh[te_])
            sid_ = d["state_id"][te_]
            stats["r2"].append(P.r2_score(y[te_], ph) - P.r2_score(y[te_], pc))
            stats["within_r2"].append(P.within_state_r2(y[te_], ph, sid_)
                                      - P.within_state_r2(y[te_], pc, sid_))
            stats["concordance"].append(
                P.within_state_concordance(y[te_], ph, sid_)[0]
                - P.within_state_concordance(y[te_], pc, sid_)[0])
        print(f"\n  REFIT bootstrap ({len(stats['r2'])} replicates, "
              f"re-split + re-fit + layer re-selection) -- includes full "
              f"pipeline variability")
        report[f"refit_boot_{tgt}"] = {"chosen_layers": chosen_layers,
                                        "full_layer_reselection": True}
        for k, v in stats.items():
            v = np.array([x for x in v if np.isfinite(x)])
            if not len(v):
                continue
            lo, hi = np.percentile(v, [2.5, 97.5])
            excl = (lo > 0) or (hi < 0)
            print(f"    Delta_{k:<12} {v.mean():+.4f}  95% CI [{lo:+.4f}, "
                  f"{hi:+.4f}]  {'EXCLUDES 0' if excl else 'includes 0'}")
            report[f"refit_boot_{tgt}"][k] = {
                "mean": float(v.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                "ci_excludes_zero": bool(excl), "n": int(len(v))}

    # ---- mechanistic centerpiece: layer x timestep heatmap -----------------
    print(f"\n{'='*74}\nlayer x diffusion-timestep decodability "
          f"(Delta R2 over cheap controls)\n{'='*74}")
    tgt = args.targets[0]
    bins = D.timestep_bins(d)
    sp = D.doc_splits(d, seed=0)
    heat = np.full((nL, len(bins)), np.nan)
    heat_abs = np.full((nL, len(bins)), np.nan)
    steps = sorted(bins)
    for bi, st in enumerate(steps):
        m = d["step"] == st
        sub = {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] ==
                   d["step"].shape else v) for k, v in d.items()}
        s2 = D.doc_splits(sub, seed=0)
        if min(len(s2["train"]), len(s2["val"]), len(s2["test"])) < 30:
            continue
        base = eval_block(sub, s2, "cheap", tgt)
        for l in range(nL):
            r = eval_2block(sub, s2, "H", tgt, l)
            heat[l, bi] = r["r2"] - base["r2"]
            heat_abs[l, bi] = r["r2"]
    hdr = "layer " + "".join(f"{f't={s}':>9}" for s in steps)
    print(hdr)
    for l in range(nL):
        print(f"{l:>5} " + "".join(
            f"{heat[l,b]:>9.3f}" if np.isfinite(heat[l, b]) else f"{'--':>9}"
            for b in range(len(steps))))
    # SECONDARY: the preregistered protocol -- the same linear readout fitted
    # timestep-bin-by-timestep-bin, with the held-out predictions POOLED into a
    # single Delta_R2. This differs from the primary statistic, which fits ONE
    # global probe across all timesteps; if the readout is timestep-conditional
    # the global probe is mis-specified and will understate it. Clearly labelled
    # as secondary: the pre-registered gate uses the pooled global probe.
    print(f"\n[secondary] per-timestep-bin probes, held-out predictions pooled")
    for tgt2 in args.targets:
        yh, yc, yy, sids = [], [], [], []
        for st in steps:
            m = d["step"] == st
            sub = {k: (v[m] if isinstance(v, np.ndarray) and
                       v.shape[:1] == d["step"].shape else v)
                   for k, v in d.items()}
            s2 = D.doc_splits(sub, seed=0)
            if min(len(s2["train"]), len(s2["val"]), len(s2["test"])) < 30:
                continue
            Lb = int(np.argmax([
                P.fit_linear_2block(
                    D.block(sub, "cheap")[s2["train"]],
                    D.block(sub, "H", l)[s2["train"]], sub[tgt2][s2["train"]],
                    D.block(sub, "cheap")[s2["val"]],
                    D.block(sub, "H", l)[s2["val"]], sub[tgt2][s2["val"]]
                )["val_r2"] for l in range(nL)]))
            rb = eval_2block(sub, s2, "H", tgt2, Lb)
            rc = eval_block(sub, s2, "cheap", tgt2)
            yh.append(rb["pred"]); yc.append(rc["pred"])
            yy.append(sub[tgt2][s2["test"]]); sids.append(sub["state_id"][s2["test"]])
        if not yy:
            continue
        yh, yc = np.concatenate(yh), np.concatenate(yc)
        yy, sids = np.concatenate(yy), np.concatenate(sids)
        dd = P.r2_score(yy, yh) - P.r2_score(yy, yc)
        dw = (P.within_state_r2(yy, yh, sids) - P.within_state_r2(yy, yc, sids))
        dc = (P.within_state_concordance(yy, yh, sids)[0]
              - P.within_state_concordance(yy, yc, sids)[0])
        print(f"   {tgt2:<10} Delta_R2 {dd:+.4f}   Delta_within_R2 {dw:+.4f}"
              f"   Delta_concord {dc:+.4f}")
        report[f"per_bin_pooled_{tgt2}"] = {"delta_r2": float(dd),
                                            "delta_within_r2": float(dw),
                                            "delta_concordance": float(dc)}

    report["heatmap"] = {"steps": [int(s) for s in steps],
                         "delta_r2": heat.tolist(),
                         "abs_r2": heat_abs.tolist(), "target": tgt}

    with open(os.path.join(outdir, "exp1_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    np.savez(os.path.join(outdir, "heatmap.npz"), delta_r2=heat,
             abs_r2=heat_abs, steps=np.array(steps))
    print(f"\nwrote {outdir}/exp1_report.json")


if __name__ == "__main__":
    main()

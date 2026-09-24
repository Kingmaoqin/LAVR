"""
Does the frozen state predict whether this generation will end up CORRECT?

Three probes, all linear, all on a FROZEN backbone, all selected on a
validation split that is disjoint by DOCUMENT (never by row):

  cheap        control features only  -- what you can already read off the
                                         output distribution for free
  hidden       mean-pooled hidden state of one layer only
  cheap+hidden two-block ridge, gamma grid includes 0 so it NESTS cheap

Plus two falsification controls that must come out at ~0:
  placebo      hidden block replaced by Gaussian noise of matched scale
  permuted     labels shuffled within the training split

And the part a reviewer actually cares about -- the SELECTIVE-GENERATION
curve: rank in-flight generations by predicted value, abandon the worst
ones, and report what that buys in accuracy and costs in coverage.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from probes import (fit_linear, fit_linear_2block, predict, predict_2block,   # noqa
                    r2_score, spearman, noise_ceiling, cluster_bootstrap)

TAGS = sys.argv[1:] or ["vreadA", "vreadB"]


def load():
    P = []
    for t in TAGS:
        P += sorted(glob.glob(os.path.join(ROOT, "data", f"labels_{t}", "shard_*.npz")))
    if not P:
        sys.exit("no shards")
    D = [np.load(p) for p in P]
    out = {k: np.concatenate([d[k] for d in D], 0)
           for k in ["doc_id", "step", "V_reward", "V_seeds", "n_masked", "cheap", "H_g", "H_m"]}
    return out


def auc(y, s):
    """Mann-Whitney AUC against the binarised outcome (correct vs not)."""
    b = y > 0.5
    if b.all() or not b.any():
        return np.nan
    r = np.empty(len(s)); r[np.argsort(s, kind="stable")] = np.arange(len(s)) + 1
    npos, nneg = b.sum(), (~b).sum()          # NOT `np` -- that shadows numpy
    return (r[b].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def split(doc, seed=0):
    u = np.unique(doc); rng = np.random.default_rng(seed); rng.shuffle(u)
    a, b = int(0.60 * len(u)), int(0.75 * len(u))
    s = {"tr": set(u[:a]), "va": set(u[a:b]), "te": set(u[b:])}
    return {k: np.array([d in v for d in doc]) for k, v in s.items()}


def evaluate(y, p):
    return {"r2": float(r2_score(y, p)), "spearman": float(spearman(y, p)),
            "auc": float(auc(y, p))}


def main():
    d = load()
    y, doc, step = d["V_reward"], d["doc_id"], d["step"]
    n_docs = len(np.unique(doc))
    nc, nc_noise, nc_obs = noise_ceiling(d["V_seeds"])
    print(f"states={len(y)}  prompts={n_docs}  states/prompt={len(y)/n_docs:.2f}")
    print(f"mean V_reward={y.mean():.4f}  sd={y.std():.4f}  "
          f"frac(all-8-correct)={(y==1).mean():.3f}  frac(all-8-wrong)={(y==0).mean():.3f}")
    print(f"noise ceiling (K=8) = {nc:.4f}  (label noise var {nc_noise:.4f} / observed var {nc_obs:.4f})\n")

    m = split(doc)
    tr, va, te = m["tr"], m["va"], m["te"]
    print(f"split docs {len(np.unique(doc[tr]))}/{len(np.unique(doc[va]))}/"
          f"{len(np.unique(doc[te]))}  rows {tr.sum()}/{va.sum()}/{te.sum()}\n")

    C = d["cheap"]
    res, best_layer, best_val = {}, None, -9e9

    # ---- controls only
    mc = fit_linear(C[tr], y[tr], C[va], y[va])
    pc_te = predict(mc, C[te])
    res["cheap"] = evaluate(y[te], pc_te)
    print(f"cheap        alpha={mc['alpha']:.3g}  " +
          "  ".join(f"{k}={v:.4f}" for k, v in res["cheap"].items()))

    # ---- hidden alone: pick the layer on VALIDATION only
    print("\nlayer sweep (validation R^2, hidden alone, mean-pooled over all "
          "positions | over masked positions):")
    for L in range(d["H_g"].shape[1]):
        for name, key in (("glob", "H_g"), ("mask", "H_m")):
            H = d[key][:, L].astype(np.float32)
            mm = fit_linear(H[tr], y[tr], H[va], y[va])
            v = r2_score(y[va], predict(mm, H[va]))
            if v > best_val:
                best_val, best_layer = v, (L, key, mm)
            if L % 5 == 0 and name == "glob":
                print(f"  L{L:2d} glob={v:+.4f}", end="")
        if L % 5 == 0:
            print()
    L, key, mh = best_layer
    print(f"\nselected layer {L} pooling={key}  val R^2={best_val:.4f}")
    H = d[key][:, L].astype(np.float32)
    ph_te = predict(mh, H[te])
    res["hidden"] = evaluate(y[te], ph_te)
    print(f"hidden       " + "  ".join(f"{k}={v:.4f}" for k, v in res["hidden"].items()))

    # ---- two-block
    m2 = fit_linear_2block(C[tr], H[tr], y[tr], C[va], H[va], y[va])
    p2_te = predict_2block(m2, C[te], H[te])
    res["cheap+hidden"] = evaluate(y[te], p2_te)
    print(f"cheap+hidden gamma={m2['gamma']}  alpha={m2['alpha']:.3g}  " +
          "  ".join(f"{k}={v:.4f}" for k, v in res["cheap+hidden"].items()))
    dR2 = res["cheap+hidden"]["r2"] - res["cheap"]["r2"]
    dAUC = res["cheap+hidden"]["auc"] - res["cheap"]["auc"]
    print(f"  Delta over cheap:  R^2 {dR2:+.4f}   AUC {dAUC:+.4f}")

    # ---- falsification controls
    rng = np.random.default_rng(7)
    Hp = rng.normal(0, 1, H.shape).astype(np.float32) * H.std()
    mp = fit_linear_2block(C[tr], Hp[tr], y[tr], C[va], Hp[va], y[va])
    r_pl = r2_score(y[te], predict_2block(mp, C[te], Hp[te]))
    yperm = y.copy(); yperm[tr] = rng.permutation(y[tr])
    mn = fit_linear_2block(C[tr], H[tr], yperm[tr], C[va], H[va], yperm[va])
    r_ng = r2_score(y[te], predict_2block(mn, C[te], H[te]))
    print(f"\nplacebo (gaussian hidden) Delta R^2 = {r_pl - res['cheap']['r2']:+.4f}"
          f"   permuted-label R^2 = {r_ng:+.4f}   [both must be ~0]")

    # ---- cluster bootstrap on the test split
    dte, yte = doc[te], y[te]
    # probes.cluster_bootstrap takes a SCALAR statistic; these three deltas
    # must be resampled jointly on the same replicate, so do it here.
    ud = np.unique(dte); rb = np.random.default_rng(1); bs = []
    for _ in range(2000):
        pick = rb.choice(ud, len(ud), replace=True)
        i = np.concatenate([np.where(dte == u)[0] for u in pick])
        bs.append([r2_score(yte[i], p2_te[i]) - r2_score(yte[i], pc_te[i]),
                   auc(yte[i], p2_te[i]) - auc(yte[i], pc_te[i]),
                   auc(yte[i], ph_te[i]) - auc(yte[i], pc_te[i])])
    bs = np.array(bs, dtype=float)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], 0)
    print(f"cluster bootstrap 95% CI  dR2 [{lo[0]:+.4f},{hi[0]:+.4f}]  "
          f"dAUC(2blk) [{lo[1]:+.4f},{hi[1]:+.4f}]  dAUC(hidden-only) "
          f"[{lo[2]:+.4f},{hi[2]:+.4f}]")

    # ---- stratified by denoising progress
    print("\nby denoising progress (test split):")
    q = np.quantile(step[te], [0.25, 0.5, 0.75])
    bins = np.digitize(step[te], q)
    strat = []
    for b in range(4):
        i = bins == b
        if i.sum() < 20 or (yte[i] > 0.5).all() or not (yte[i] > 0.5).any():
            continue
        row = {"bin": b, "n": int(i.sum()), "step_med": float(np.median(step[te][i])),
               "auc_cheap": auc(yte[i], pc_te[i]), "auc_hidden": auc(yte[i], ph_te[i]),
               "auc_both": auc(yte[i], p2_te[i]),
               "r2_cheap": r2_score(yte[i], pc_te[i]), "r2_both": r2_score(yte[i], p2_te[i])}
        strat.append(row)
        print(f"  step~{row['step_med']:5.0f} n={row['n']:4d}  AUC cheap={row['auc_cheap']:.4f}"
              f" hidden={row['auc_hidden']:.4f} both={row['auc_both']:.4f}"
              f"  dR2={row['r2_both']-row['r2_cheap']:+.4f}")

    # ---- IS IT JUST PROMPT DIFFICULTY?
    # A probe can score a high AUC by reading the QUESTION and judging it hard,
    # never looking at the partially-denoised answer at all. That would still be
    # useful, but it is a different claim. Removing the per-prompt mean from both
    # label and prediction isolates the part that tracks THIS TRAJECTORY: how
    # V_reward moves across the 6 record points of one prompt.
    def within(y_, p_, g_):
        y2, p2 = y_.astype(float).copy(), p_.astype(float).copy()
        for u in np.unique(g_):
            i = g_ == u
            if i.sum() < 2:
                y2[i] = p2[i] = 0.0
                continue
            y2[i] -= y2[i].mean(); p2[i] -= p2[i].mean()
        ss = ((y2 - p2) ** 2).sum(); tt = (y2 ** 2).sum()
        return float(1 - ss / tt) if tt > 1e-12 else float("nan")

    vy = np.array([y[doc == u].std() for u in np.unique(doc)])
    print(f"\nwithin-prompt variation of V_reward: mean sd across prompts "
          f"= {vy.mean():.4f}  (between-prompt sd = {np.array([y[doc==u].mean() for u in np.unique(doc)]).std():.4f})")
    wr = {n: within(yte, pp, dte) for n, pp in
          (("cheap", pc_te), ("hidden", ph_te), ("both", p2_te))}
    print("within-prompt R^2 (prompt difficulty removed): " +
          "  ".join(f"{n}={v:+.4f}" for n, v in wr.items()))

    # ---- SELECTIVE GENERATION: abandon the worst-scoring generations
    print("\nselective generation (test split) -- keep the top-c fraction by "
          "predicted value, report accuracy of what you kept:")
    curves = {}
    base = float(yte.mean())
    for nm, s in (("cheap", pc_te), ("hidden", ph_te), ("both", p2_te),
                  ("oracle", yte), ("random", rng.normal(size=len(yte)))):
        o = np.argsort(-s, kind="stable")
        cum = np.cumsum(yte[o]) / np.arange(1, len(o) + 1)
        curves[nm] = cum
    print(f"  base accuracy (keep everything) = {base:.4f}")
    print("  coverage   " + "".join(f"{n:>10s}" for n in curves))
    for c in (0.9, 0.8, 0.7, 0.6, 0.5, 0.3):
        k = max(1, int(c * len(yte)))
        print(f"  {c:6.0%}     " + "".join(f"{curves[n][k-1]:10.4f}" for n in curves))
    # What aborting actually buys. Abort at median progress f: the dropped
    # generations skip their remaining (1 - f) of denoising, so the saved
    # compute is (1 - coverage) * (1 - f) of the total decode budget.
    f_ab = float(np.median(step[te]) / 160)
    print(f"\n  compute accounting, aborting at median progress f={f_ab:.1%}:")
    print("  coverage   saved_compute   acc(cheap)  acc(hidden)   gain_vs_cheap")
    for c in (0.9, 0.8, 0.7, 0.6, 0.5, 0.3):
        k = max(1, int(c * len(yte)))
        sv = (1 - c) * (1 - f_ab)
        print(f"  {c:6.0%}   {sv:11.1%}      {curves['cheap'][k-1]:.4f}      "
              f"{curves['hidden'][k-1]:.4f}       {curves['hidden'][k-1]-curves['cheap'][k-1]:+.4f}")
    aurc = {n: float(np.mean(v)) for n, v in curves.items()}
    print("  area under the risk-coverage curve: " +
          "  ".join(f"{n}={v:.4f}" for n, v in aurc.items()))

    # ---- PER-RECORD-POINT, PROMPT-LEVEL
    # Since V_reward barely moves within a prompt, the 6 states of one prompt are
    # ~6 copies of one observation: the EFFECTIVE sample size is the number of
    # PROMPTS, not states. So refit independently at each record point, where
    # every row is a distinct prompt. This also answers "how early can you tell?":
    # if the 10%-denoised probe already matches the 85% one, the signal was in the
    # question all along and no lookahead is involved.
    print("\nper-record-point, one row per prompt (independent fits):")
    order = np.argsort(step); ranks = {}
    for u in np.unique(doc):
        i = np.where(doc == u)[0]
        for r, j in enumerate(i[np.argsort(step[i])]):
            ranks[j] = r
    rk = np.array([ranks[j] for j in range(len(y))])
    per_point = []
    for r in range(int(rk.max()) + 1):
        sel = rk == r
        if sel.sum() < 60:
            continue
        yr, Cr, Hr, dr = y[sel], C[sel], d[key][sel, L].astype(np.float32), doc[sel]
        mr = split(dr, seed=0)
        t_, v_, e_ = mr["tr"], mr["va"], mr["te"]
        if (yr[e_] > 0.5).all() or not (yr[e_] > 0.5).any():
            continue
        mcx = fit_linear(Cr[t_], yr[t_], Cr[v_], yr[v_])
        mhx = fit_linear(Hr[t_], yr[t_], Hr[v_], yr[v_])
        m2x = fit_linear_2block(Cr[t_], Hr[t_], yr[t_], Cr[v_], Hr[v_], yr[v_])
        row = {"point": r, "n_prompts": int(sel.sum()),
               "median_progress": float(np.median(step[sel]) / 160),
               "auc_cheap": auc(yr[e_], predict(mcx, Cr[e_])),
               "auc_hidden": auc(yr[e_], predict(mhx, Hr[e_])),
               "auc_both": auc(yr[e_], predict_2block(m2x, Cr[e_], Hr[e_])),
               "r2_cheap": r2_score(yr[e_], predict(mcx, Cr[e_])),
               "r2_both": r2_score(yr[e_], predict_2block(m2x, Cr[e_], Hr[e_]))}
        per_point.append(row)
        row["auc_gap"] = row["auc_hidden"] - row["auc_cheap"]
        print(f"  point {r}  progress={row['median_progress']:5.1%}  n={row['n_prompts']:4d}"
              f"  AUC cheap={row['auc_cheap']:.4f} hidden={row['auc_hidden']:.4f}"
              f" both={row['auc_both']:.4f}  gap={row['auc_gap']:+.4f}"
              f"   R^2 cheap={row['r2_cheap']:+.4f} both={row['r2_both']:+.4f}")
    if len(per_point) >= 2:
        e, l = per_point[0]["auc_hidden"], per_point[-1]["auc_hidden"]
        ec, lc = per_point[0]["auc_cheap"], per_point[-1]["auc_cheap"]
        g = [q["auc_gap"] for q in per_point]
        print(f"  earliest->latest AUC: hidden {e:.4f}->{l:.4f} ({l-e:+.4f}), "
              f"cheap {ec:.4f}->{lc:.4f} ({lc-ec:+.4f})")
        print(f"  hidden-over-cheap gap across denoising: min={min(g):+.4f} "
              f"max={max(g):+.4f} mean={float(np.mean(g)):+.4f}  "
              "[roughly constant => the hidden state's ADVANTAGE does not grow "
              "as the answer becomes visible]")

    out = {"n_states": int(len(y)), "n_prompts": int(n_docs),
           "mean_V": base, "noise_ceiling": float(nc),
           "layer": int(L), "pooling": key, "gamma": float(m2["gamma"]),
           "metrics": res, "delta_r2": float(dR2), "delta_auc": float(dAUC),
           "placebo_delta_r2": float(r_pl - res["cheap"]["r2"]),
           "permuted_r2": float(r_ng),
           "ci": {"dR2": [float(lo[0]), float(hi[0])],
                  "dAUC_2blk": [float(lo[1]), float(hi[1])],
                  "dAUC_hidden": [float(lo[2]), float(hi[2])]},
           "strata": strat, "aurc": aurc, "abort_progress": f_ab, "per_point": per_point, "within_prompt_r2": wr,
           "within_prompt_sd": float(vy.mean()),
           "selective": {n: [float(curves[n][max(1, int(c*len(yte)))-1])
                             for c in (0.9, 0.8, 0.7, 0.6, 0.5, 0.3)] for n in curves}}
    p = os.path.join(ROOT, "data", "v_readout_results.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

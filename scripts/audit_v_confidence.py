"""Was the 08-30 read-out under-claimed?

Five checks I did not run before reporting:
  1  hidden-ALONE delta, not just the two-block one I quoted (the two-block fit
     scored LOWER than hidden alone, so quoting it understates the effect)
  2  per-record-point deltas WITH cluster-bootstrap CIs -- agreement across all
     six points is temporal consistency (the points share prompts, not six
     independent replications)
  3  the within-prompt noise FLOOR -- a negative within-prompt R^2 means nothing
     unless the within-prompt signal was above the label-noise floor to begin
     with. This decides whether "no trajectory signal" is a finding or a
     non-measurement.
  4  multi-layer probe -- selecting ONE layer may leave signal on the table;
     concatenating layers is still a LINEAR probe, so it is inside the protocol
  5  bootstrap CI on the risk-coverage area, and on the operating point
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from probes import (fit_linear, fit_linear_2block, predict, predict_2block,  # noqa
                    r2_score, spearman, noise_ceiling)
sys.path.insert(0, HERE)
from analyze_v_readout import auc, split, load                              # noqa

TAGS = ["vreadA", "vreadB"]
sys.argv = [sys.argv[0]] + TAGS
d = load()
y, doc, step = d["V_reward"], d["doc_id"], d["step"]
C = d["cheap"]
m = split(doc); tr, va, te = m["tr"], m["va"], m["te"]
nc, nvar, ovar = noise_ceiling(d["V_seeds"])


def boot(fn, g, n=2000, seed=1):
    u = np.unique(g); rb = np.random.default_rng(seed); out = []
    for _ in range(n):
        p = rb.choice(u, len(u), replace=True)
        i = np.concatenate([np.where(g == q)[0] for q in p])
        out.append(fn(i))
    return np.array(out, dtype=float)


print("=" * 78)
print("CHECK 3  -- is the within-prompt result a FINDING or a NON-MEASUREMENT?")
print("=" * 78)
# per-prompt: observed within variance vs the variance label noise alone injects
wv, nf, ns = [], [], []
for u in np.unique(doc):
    i = doc == u
    if i.sum() < 2:
        continue
    wv.append(y[i].var(ddof=1))
    nf.append((d["V_seeds"][i].var(1, ddof=1) / d["V_seeds"].shape[1]).mean())
    ns.append(i.sum())
wv, nf = np.array(wv), np.array(nf)
print(f"  prompts with >=2 states: {len(wv)}")
print(f"  mean observed within-prompt variance : {wv.mean():.6f}")
print(f"  mean variance injected by label noise: {nf.mean():.6f}")
sig = wv.mean() - nf.mean()
print(f"  => within-prompt SIGNAL variance      : {sig:+.6f}"
      f"   ({sig/max(ovar,1e-12)*100:+.2f}% of total observed variance)")
print(f"  between-prompt variance              : {np.array([y[doc==u].mean() for u in np.unique(doc)]).var():.6f}")
print("  reading: if signal variance is ~0 or negative, V_reward is genuinely")
print("  CONSTANT within a prompt -- there is nothing there for ANY probe to")
print("  read, so the negative within-prompt R^2 is not a probe failure.")

print()
print("=" * 78)
print("CHECK 3b -- FIT A PROBE TO THE WITHIN-PROMPT TARGET DIRECTLY")
print("=" * 78)
print("  The earlier within-prompt R^2 took a probe fitted to the GLOBAL target")
print("  and scored it on prompt-centred residuals. That is not what it was")
print("  optimised for, so a negative score there is not evidence of absence.")
print("  The honest test is to centre BOTH features and target per prompt and")
print("  fit the probe on that.")


def centre(X, g):
    Z = X.astype(np.float32).copy()
    for u in np.unique(g):
        i = g == u
        Z[i] -= Z[i].mean(0, keepdims=True)
    return Z


yw = y.astype(np.float32).copy()
for u in np.unique(doc):
    i = doc == u
    yw[i] -= yw[i].mean()
wceil = sig / max(wv.mean(), 1e-12)
print(f"  within-prompt noise ceiling: {wceil:.4f}  (a perfect probe caps here)")
Cw = centre(C, doc)
for L2 in range(0, d["H_g"].shape[1], 3):
    for k2 in ("H_g", "H_m"):
        Hw = centre(d[k2][:, L2], doc)
        mw = fit_linear_2block(Cw[tr], Hw[tr], yw[tr], Cw[va], Hw[va], yw[va])
        mw0 = fit_linear(Cw[tr], yw[tr], Cw[va], yw[va])
        rw = r2_score(yw[te], predict_2block(mw, Cw[te], Hw[te]))
        rw0 = r2_score(yw[te], predict(mw0, Cw[te]))
        if k2 == "H_m":
            print(f"    L{L2:2d}  cheap-within R2={rw0:+.4f}   +hidden={rw:+.4f}"
                  f"   delta={rw-rw0:+.4f}   (ceiling {wceil:.3f})")

print()
print("=" * 78)
print("CHECK 1+4 -- hidden alone, and whether ONE layer left signal on the table")
print("=" * 78)
cand = {}
mc = fit_linear(C[tr], y[tr], C[va], y[va]); pc = predict(mc, C[te])
cand["cheap"] = pc
best = (-9e9, None)
for L in range(d["H_g"].shape[1]):
    for key in ("H_g", "H_m"):
        H = d[key][:, L].astype(np.float32)
        mm = fit_linear(H[tr], y[tr], H[va], y[va])
        v = r2_score(y[va], predict(mm, H[va]))
        if v > best[0]:
            best = (v, (L, key, mm))
L, key, mh = best[1]
H1 = d[key][:, L].astype(np.float32)
cand["hidden_1layer"] = predict(mh, H1[te])

# multi-layer: concatenate an evenly spaced set of layers (still linear)
for nL in (3, 5, 9):
    ls = np.linspace(0, d["H_g"].shape[1] - 1, nL).round().astype(int)
    Hm = np.concatenate([d[key][:, l].astype(np.float32) for l in ls], 1)
    mmm = fit_linear(Hm[tr], y[tr], Hm[va], y[va])
    cand[f"hidden_{nL}layers"] = predict(mmm, Hm[te])
m2 = fit_linear_2block(C[tr], H1[tr], y[tr], C[va], H1[va], y[va])
cand["cheap+hidden"] = predict_2block(m2, C[te], H1[te])
ls5 = np.linspace(0, d["H_g"].shape[1] - 1, 5).round().astype(int)
H5 = np.concatenate([d[key][:, l].astype(np.float32) for l in ls5], 1)
m25 = fit_linear_2block(C[tr], H5[tr], y[tr], C[va], H5[va], y[va])
cand["cheap+hidden_5layers"] = predict_2block(m25, C[te], H5[te])

yte, dte = y[te], doc[te]
print(f"  selected layer {L} ({key}); ceiling R^2 = {nc:.4f}")
print(f"  {'probe':24s} {'R2':>8s} {'AUC':>8s} {'rho':>8s}   {'dR2 vs cheap':>14s} "
      f"{'dAUC':>10s}  {'% of ceiling':>12s}")
for n, p in cand.items():
    r, a = r2_score(yte, p), auc(yte, p)
    print(f"  {n:24s} {r:8.4f} {a:8.4f} {spearman(yte,p):8.4f}   "
          f"{r-r2_score(yte,cand['cheap']):14.4f} {a-auc(yte,cand['cheap']):10.4f}"
          f"  {r/nc*100:11.1f}%")
# The layer/pooling choice for hidden_1layer was made on validation data above.
# Do not select among the displayed multi-layer variants on the test set.
bestname = "hidden_1layer"
pb = cand[bestname]
b = boot(lambda i: [r2_score(yte[i], pb[i]) - r2_score(yte[i], pc[i]),
                    auc(yte[i], pb[i]) - auc(yte[i], pc[i])], dte)
lo, hi = np.nanpercentile(b, [2.5, 97.5], 0)
print(f"  best hidden probe = {bestname}:  dR2 95% CI [{lo[0]:+.4f},{hi[0]:+.4f}]"
      f"   dAUC 95% CI [{lo[1]:+.4f},{hi[1]:+.4f}]")

print()
print("=" * 78)
print("CHECK 2  -- is the effect temporally consistent at every record point?")
print("=" * 78)
ranks = {}
for u in np.unique(doc):
    i = np.where(doc == u)[0]
    for r, j in enumerate(i[np.argsort(step[i])]):
        ranks[j] = r
rk = np.array([ranks[j] for j in range(len(y))])
rows = []
print(f"  {'pt':>3s} {'prog':>6s} {'n':>5s} {'dR2':>8s} {'95% CI':>20s} "
      f"{'dAUC':>8s} {'95% CI':>20s} {'sig':>4s}")
for r in range(int(rk.max()) + 1):
    s_ = rk == r
    if s_.sum() < 60:
        continue
    yr, Cr, Hr, dr = y[s_], C[s_], d[key][s_, L].astype(np.float32), doc[s_]
    mr = split(dr, seed=0); t_, v_, e_ = mr["tr"], mr["va"], mr["te"]
    mcx = fit_linear(Cr[t_], yr[t_], Cr[v_], yr[v_])
    mhx = fit_linear(Hr[t_], yr[t_], Hr[v_], yr[v_])
    pcx, phx = predict(mcx, Cr[e_]), predict(mhx, Hr[e_])
    ye, de = yr[e_], dr[e_]
    bb = boot(lambda i: [r2_score(ye[i], phx[i]) - r2_score(ye[i], pcx[i]),
                         auc(ye[i], phx[i]) - auc(ye[i], pcx[i])], de, n=1500, seed=2 + r)
    l2, h2 = np.nanpercentile(bb, [2.5, 97.5], 0)
    sg = "**" if l2[0] > 0 and l2[1] > 0 else ("*" if l2[0] > 0 or l2[1] > 0 else "")
    rows.append({"pt": r, "prog": float(np.median(step[s_]) / 160), "n": int(s_.sum()),
                 "dR2": r2_score(ye, phx) - r2_score(ye, pcx),
                 "dR2_ci": [l2[0], h2[0]],
                 "dAUC": auc(ye, phx) - auc(ye, pcx), "dAUC_ci": [l2[1], h2[1]]})
    print(f"  {r:3d} {rows[-1]['prog']:6.1%} {s_.sum():5d} {rows[-1]['dR2']:+8.4f} "
          f"[{l2[0]:+.4f},{h2[0]:+.4f}] {rows[-1]['dAUC']:+8.4f} "
          f"[{l2[1]:+.4f},{h2[1]:+.4f}] {sg:>4s}")
k = sum(1 for r in rows if r["dR2_ci"][0] > 0)
print(f"  => dR2 CI excludes 0 at {k}/{len(rows)} record points")

print()
print("=" * 78)
print("CHECK 5  -- CI on the practical operating point")
print("=" * 78)
def curve(p, i):
    o = i[np.argsort(-p[i], kind="stable")]
    return np.cumsum(yte[o]) / np.arange(1, len(o) + 1)
for c in (0.7, 0.5, 0.3):
    bb = boot(lambda i: [curve(pb, i)[max(1, int(c*len(i)))-1],
                         curve(pc, i)[max(1, int(c*len(i)))-1],
                         curve(pb, i)[max(1, int(c*len(i)))-1]
                         - curve(pc, i)[max(1, int(c*len(i)))-1]], dte, n=1500, seed=9)
    l3, h3 = np.nanpercentile(bb, [2.5, 97.5], 0)
    print(f"  coverage {c:.0%}: hidden acc {curve(pb,np.arange(len(yte)))[max(1,int(c*len(yte)))-1]:.4f} "
          f"[{l3[0]:.4f},{h3[0]:.4f}]   gain over cheap "
          f"{curve(pb,np.arange(len(yte)))[max(1,int(c*len(yte)))-1]-curve(pc,np.arange(len(yte)))[max(1,int(c*len(yte)))-1]:+.4f} "
          f"[{l3[2]:+.4f},{h3[2]:+.4f}]")
ba = boot(lambda i: [curve(pb, i).mean() - curve(pc, i).mean()], dte, n=1500, seed=11)
l4, h4 = np.nanpercentile(ba, [2.5, 97.5], 0)
print(f"  AURC gain over cheap: {curve(pb,np.arange(len(yte))).mean()-curve(pc,np.arange(len(yte))).mean():+.4f}"
      f"  95% CI [{l4[0]:+.4f},{h4[0]:+.4f}]")

json.dump({"within_signal_var": float(sig), "within_obs_var": float(wv.mean()),
           "within_noise_var": float(nf.mean()), "ceiling": float(nc),
           "probes": {n: {"r2": float(r2_score(yte, p)), "auc": float(auc(yte, p))}
                      for n, p in cand.items()},
           "best": bestname, "best_ci": {"dR2": [float(lo[0]), float(hi[0])],
                                         "dAUC": [float(lo[1]), float(hi[1])]},
           "per_point": rows},
          open(os.path.join(ROOT, "data", "v_audit.json"), "w"), indent=2, default=float)
print("\nwrote data/v_audit.json")

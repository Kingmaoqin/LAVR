"""Was the KILL verdict decided by a MIS-SPECIFIED test?

exp1b_within_state.py fits the probe on the RAW target A -- whose variance is
dominated by the state-level channel -- and then scores it on within-state
residuals. Ridge picks alpha to minimise GLOBAL validation error, so it is free
to spend all its capacity on the between-state direction and leave the
within-state direction unmodelled. A near-zero within-state R^2 from that
procedure is not evidence that the candidate-level signal is absent.

The fair test centres BOTH the features and the target within each state and
fits the probe on THAT. Then the probe's whole objective is the ranking
decision a scheduler actually faces. If the answer is still ~0, the negative
result is real and now properly founded. If it is not, the KILL verdict was
decided by a methodological artefact and has to be revisited.

Also reports the WITHIN-STATE noise ceiling, so a low number can be attributed
to either a weak signal or a noisy label rather than left ambiguous.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import dataset as D                                                    # noqa
import probes as P                                                     # noqa


def centre(X, sid):
    Z = np.asarray(X, dtype=np.float32).copy()
    for u in np.unique(sid):
        i = sid == u
        Z[i] -= Z[i].mean(0, keepdims=True)
    return Z


def within_ceiling(seeds, sid):
    """Ceiling for the CENTRED target: centre every seed replicate, then the
    usual signal/(signal+noise) decomposition."""
    Yc = np.stack([centre(seeds[:, k][:, None], sid)[:, 0]
                   for k in range(seeds.shape[1])], 1)
    K = Yc.shape[1]
    noise = float((Yc.var(1, ddof=1) / K).mean())
    obs = float(Yc.mean(1).var())
    return max(0.0, (obs - noise) / max(obs, 1e-12)), noise, obs


def pair_counts(y, pred, sid, idx):
    """Return correct/total rankable pairs without joining state copies."""
    correct, total = 0.0, 0
    for state in np.unique(sid[idx]):
        take = idx[sid[idx] == state]
        yy, pp = y[take], pred[take]
        dy = yy[:, None] - yy[None, :]
        dp = pp[:, None] - pp[None, :]
        iu = np.triu_indices(len(take), 1)
        dy, dp = dy[iu], dp[iu]
        rankable = np.abs(dy) > 1e-9
        dy, dp = dy[rankable], dp[rankable]
        ties = np.abs(dp) <= 1e-12
        correct += float((np.sign(dy[~ties]) == np.sign(dp[~ties])).sum()
                         + 0.5 * ties.sum())
        total += int(len(dy))
    return correct, total


def run(tags, target, seedkey, split_seed=0):
    d = D.load_labels(tags)
    sid, doc = d["state_id"], d["doc_id"]
    y = d[target].astype(np.float32)
    sp = D.doc_splits(d, seed=split_seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    nstate = len(np.unique(sid))
    cands = np.array([np.sum(sid == u) for u in np.unique(sid)])

    yc = centre(y[:, None], sid)[:, 0]
    Cc = centre(D.block(d, "cheap"), sid)
    ceil, nvar, ovar = within_ceiling(d[seedkey].astype(np.float32), sid)

    print(f"\n{'='*78}\n{tags} target={target}\n{'='*78}")
    print(f"  rows={len(y)}  states={nstate}  cands/state={cands.mean():.2f}"
          f"  docs={len(np.unique(doc))}")
    print(f"  raw target sd={y.std():.4f}   within-state centred sd={yc.std():.4f}"
          f"  ({yc.var()/max(y.var(),1e-12)*100:.1f}% of raw variance)")
    print(f"  WITHIN-STATE noise ceiling = {ceil:.4f}"
          f"   (signal {ovar-nvar:.5f} / noise {nvar:.5f}; SNR {(ovar-nvar)/max(nvar,1e-12):.2f})")
    if ceil < 0.2:
        print("  !! ceiling is low: a null result here is UNINFORMATIVE, not negative")

    # baseline: cheap only, fit on the centred target
    mc = P.fit_linear(Cc[tr], yc[tr], Cc[va], yc[va])
    pc = P.predict(mc, Cc[te])
    r_c = P.r2_score(yc[te], pc)
    k_c = P.within_state_concordance(y[te], pc, sid[te])[0]

    best = (-9e9, None, None)
    for l in range(d["n_layers"]):
        Hc = centre(D.block(d, "H_local", l), sid)
        m = P.fit_linear_2block(Cc[tr], Hc[tr], yc[tr], Cc[va], Hc[va], yc[va])
        v = P.r2_score(yc[va], P.predict_2block(m, Cc[va], Hc[va]))
        if v > best[0]:
            best = (v, l, (m, Hc))
    l, (m, Hc) = best[1], best[2]
    ph = P.predict_2block(m, Cc[te], Hc[te])
    r_h = P.r2_score(yc[te], ph)
    k_h = P.within_state_concordance(y[te], ph, sid[te])[0]

    print(f"  selected layer {l} on validation; gamma={m['gamma']}, alpha={m['alpha']:.3g}")
    print(f"  {'probe':22s} {'within-R2':>10s} {'concordance':>12s}  {'% of ceiling':>12s}")
    print(f"  {'cheap (centred fit)':22s} {r_c:10.4f} {k_c:12.4f}  {r_c/max(ceil,1e-9)*100:11.1f}%")
    print(f"  {'cheap + h_i':22s} {r_h:10.4f} {k_h:12.4f}  {r_h/max(ceil,1e-9)*100:11.1f}%")
    print(f"  {'delta':22s} {r_h-r_c:+10.4f} {k_h-k_c:+12.4f}")

    # Cluster bootstrap over documents.  Pre-aggregate each document so that a
    # document sampled m times contributes weight m, rather than merging its
    # repeated state ids and accidentally creating O(m^2) cross-copy pairs.
    dte, yte_c, yte_r, ste = doc[te], yc[te], y[te], sid[te]
    u = np.unique(dte)
    rss_c, rss_h, tss, conc_c, conc_h, pairs = [], [], [], [], [], []
    for value in u:
        i = np.where(dte == value)[0]
        rss_c.append(float(((yte_c[i] - pc[i]) ** 2).sum()))
        rss_h.append(float(((yte_c[i] - ph[i]) ** 2).sum()))
        tss.append(float((yte_c[i] ** 2).sum()))
        c0, n0 = pair_counts(yte_r, pc, ste, i)
        c1, n1 = pair_counts(yte_r, ph, ste, i)
        if n0 != n1:
            raise AssertionError("pair denominators differ")
        conc_c.append(c0); conc_h.append(c1); pairs.append(n0)
    rss_c, rss_h, tss, conc_c, conc_h, pairs = map(
        np.asarray, (rss_c, rss_h, tss, conc_c, conc_h, pairs))
    rb = np.random.default_rng(3)
    pick = rb.integers(0, len(u), size=(1500, len(u)))
    bs = np.column_stack([
        (rss_c[pick].sum(1) - rss_h[pick].sum(1)) / tss[pick].sum(1),
        (conc_h[pick].sum(1) - conc_c[pick].sum(1)) / pairs[pick].sum(1),
    ])
    lo, hi = np.nanpercentile(np.array(bs, float), [2.5, 97.5], 0)
    print(f"  95% CI  d within-R2 [{lo[0]:+.4f},{hi[0]:+.4f}]"
          f"   d concordance [{lo[1]:+.4f},{hi[1]:+.4f}]")

    # negative control on the centred fit
    rng = np.random.default_rng(5)
    Hn = rng.normal(0, 1, Hc.shape).astype(np.float32) * Hc.std()
    mn = P.fit_linear_2block(Cc[tr], Hn[tr], yc[tr], Cc[va], Hn[va], yc[va])
    r_n = P.r2_score(yc[te], P.predict_2block(mn, Cc[te], Hn[te]))
    print(f"  placebo (gaussian h_i): delta = {r_n-r_c:+.4f}  [must be ~0]")
    return {"tags": tags, "target": target, "ceiling": ceil, "layer": int(l),
            "r2_cheap": r_c, "r2_hidden": r_h, "d_r2": r_h - r_c,
            "conc_cheap": k_c, "conc_hidden": k_h, "d_conc": k_h - k_c,
            "ci_r2": [lo[0], hi[0]], "ci_conc": [lo[1], hi[1]],
            "placebo": r_n - r_c, "n_states": int(nstate)}


out = []
for tags, name in ((["a3", "b3"], "arm1 ancestral"), (["c3", "d3"], "arm2 confidence")):
    # SEEDKEY per exp1_decodability.py:120 -- A_pertok is the PRIMARY target and
    # its per-seed replicates live under A_full_seeds.
    for target, sk in (("A_pertok", "A_full_seeds"), ("A_future", "A_future_seeds")):
        for ss in (0, 1, 2):          # G2 requires the SAME SIGN in all 3 splits
            try:
                r = run(tags, target, sk, ss)
                r["arm"] = name; r["split_seed"] = ss; out.append(r)
            except Exception as e:
                print(f"  {name}/{target}/s{ss} failed: {type(e).__name__}: {e}")

print("\n" + "=" * 78)
print("G2 VERDICT under the FAIR (centred-fit) test   [threshold: dConc >= 0.020,")
print("same sign in all 3 split seeds]")
print("=" * 78)
print(f"  {'arm':16s} {'target':10s} {'dConc s0':>9s} {'s1':>9s} {'s2':>9s} "
      f"{'mean':>8s} {'>=0.020?':>9s}")
for name in ("arm1 ancestral", "arm2 confidence"):
    for target in ("A_pertok", "A_future"):
        rs = [r for r in out if r["arm"] == name and r["target"] == target]
        if len(rs) != 3:
            continue
        cs = [r["d_conc"] for r in sorted(rs, key=lambda z: z["split_seed"])]
        mu = float(np.mean(cs))
        same = all(c > 0 for c in cs) or all(c < 0 for c in cs)
        print(f"  {name:16s} {target:10s} {cs[0]:+9.4f} {cs[1]:+9.4f} {cs[2]:+9.4f} "
              f"{mu:+8.4f} {('PASS' if mu >= 0.020 and same else 'FAIL'):>9s}")
        print(f"  {'':16s} {'':10s} same sign: {same};  "
              f"mean dWithinR2 {np.mean([r['d_r2'] for r in rs]):+.4f};  "
              f"ceiling {rs[0]['ceiling']:.3f}")
json.dump(out, open(os.path.join(ROOT, "data", "A_fairtest.json"), "w"),
          indent=2, default=float)
print("\nwrote data/A_fairtest.json")

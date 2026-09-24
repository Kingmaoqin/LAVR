"""Effect sizes with intervals from several estimators. No thresholds.

50 split seeds x 3 backbones x 2 targets x 8 probes, every fit on the
WITHIN-STATE CENTRED target so each probe is optimised for the ranking
decision it is scored on.

Statistics reported per (backbone, target, probe):
  1 naive t over seeds            (shown to expose how wrong it is)
  2 Nadeau-Bengio corrected t     (the right one for repeated splits)
  3 cluster bootstrap percentile  (resample documents)
  4 BCa bootstrap                 (bias + skew corrected)
  5 jackknife over documents
  6 sign test over seeds
  7 Wilcoxon signed-rank over seeds
  8 BH-FDR across the probe family
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import dataset as D                                                     # noqa
import probes as P                                                      # noqa
import stats_suite as S                                                 # noqa

ARMS = {"MDLM_ancestral": ["a3", "b3"], "MDLM_confidence": ["c3", "d3"],
        "SEDD_ancestral": ["s1", "s2"]}
TARGETS = {"A_pertok": "A_full_seeds", "A_future": "A_future_seeds"}
# task-grounded advantage: same schema, different reward
TASK_ARMS = {"Nemotron_GSM8K": ["taskC"], "Nemotron_SVAMP": ["taskE_svamp"]}
TASK_TARGETS = {"A_task": "A_task_seeds"}


def centre(X, sid):
    Z = np.asarray(X, dtype=np.float32).copy()
    for u in np.unique(sid):
        i = sid == u
        Z[i] -= Z[i].mean(0, keepdims=True)
    return Z


def within_state_shuffle(X, sid, rng):
    """Destroy WHICH candidate got WHICH vector, keep everything else."""
    Z = X.copy()
    for u in np.unique(sid):
        i = np.where(sid == u)[0]
        Z[i] = Z[rng.permutation(i)]
    return Z


def ceiling(seeds, sid):
    Y = np.stack([centre(seeds[:, k][:, None], sid)[:, 0]
                  for k in range(seeds.shape[1])], 1)
    K = Y.shape[1]
    n = float((Y.var(1, ddof=1) / K).mean()); o = float(Y.mean(1).var())
    return max(0.0, (o - n) / max(o, 1e-12))


def run(arm, tags, target, seedkey, n_seeds):
    d = D.load_labels(tags)
    sid, doc = d["state_id"], d["doc_id"]
    y = d[target].astype(np.float32)
    yc = centre(y[:, None], sid)[:, 0]
    ceil = ceiling(d[seedkey].astype(np.float32), sid)
    C = centre(D.block(d, "cheap"), sid)

    sel = D.doc_splits(d, seed=99)                    # layer picked ONCE, held fixed
    best = (-9e9, 0)
    for l in range(d["n_layers"]):
        H = centre(D.block(d, "H_local", l), sid)
        m = P.fit_linear_2block(C[sel["train"]], H[sel["train"]], yc[sel["train"]],
                                C[sel["val"]], H[sel["val"]], yc[sel["val"]])
        v = P.r2_score(yc[sel["val"]], P.predict_2block(m, C[sel["val"]], H[sel["val"]]))
        if v > best[0]:
            best = (v, l)
    L = best[1]
    rng = np.random.default_rng(0)
    HL = centre(D.block(d, "H_local", L), sid)
    HG = centre(D.block(d, "H_global", L), sid)
    nL = d["n_layers"]
    ls = sorted({max(0, L - 3), L, min(nL - 1, L + 3)})
    HM = centre(np.concatenate([D.block(d, "H_local", l) for l in ls], 1), sid)
    a = np.linalg.svd(HL - HL.mean(0), full_matrices=False)[2][:12]
    b = np.linalg.svd(HG - HG.mean(0), full_matrices=False)[2][:12]
    pa, pb = (HL - HL.mean(0)) @ a.T, (HG - HG.mean(0)) @ b.T
    BLOCKS = {
        "cheap+h_local": HL,
        "cheap+h_local(3layers)": HM,
        "cheap+h_local+h_global": np.concatenate([HL, HG], 1),
        "cheap+interaction(kron12x12)": (pa[:, :, None] * pb[:, None, :]
                                         ).reshape(len(pa), -1).astype(np.float32),
        "CONTROL:gaussian_h_local": (rng.normal(0, 1, HL.shape).astype(np.float32)
                                     * HL.std()),
        "CONTROL:within_state_shuffle": within_state_shuffle(HL, sid, rng),
        "CONTROL:across_row_shuffle": HL[rng.permutation(len(HL))],
    }

    per_seed = {k: {"d_r2": [], "d_conc": []} for k in BLOCKS}
    per_seed["CONTROL:label_permutation"] = {"d_r2": [], "d_conc": []}
    keep = {}
    n_tr = n_te = 0
    for s in range(n_seeds):
        sp = D.doc_splits(d, seed=s)
        tr, va, te = sp["train"], sp["val"], sp["test"]
        n_tr = len(np.unique(doc[tr])); n_te = len(np.unique(doc[te]))
        mc = P.fit_linear(C[tr], yc[tr], C[va], yc[va]); pc = P.predict(mc, C[te])
        r_c = P.r2_score(yc[te], pc)
        k_c = P.within_state_concordance(y[te], pc, sid[te])[0]
        for name, X in BLOCKS.items():
            m = P.fit_linear_2block(C[tr], X[tr], yc[tr], C[va], X[va], yc[va])
            p = P.predict_2block(m, C[te], X[te])
            per_seed[name]["d_r2"].append(P.r2_score(yc[te], p) - r_c)
            per_seed[name]["d_conc"].append(
                P.within_state_concordance(y[te], p, sid[te])[0] - k_c)
            if s == 0 and name == "cheap+h_local":
                keep = {"te": te, "pc": pc, "ph": p}
        yp = yc.copy(); yp[tr] = rng.permutation(yc[tr])
        m = P.fit_linear_2block(C[tr], HL[tr], yp[tr], C[va], HL[va], yp[va])
        p = P.predict_2block(m, C[te], HL[te])
        per_seed["CONTROL:label_permutation"]["d_r2"].append(P.r2_score(yc[te], p) - r_c)
        per_seed["CONTROL:label_permutation"]["d_conc"].append(
            P.within_state_concordance(y[te], p, sid[te])[0] - k_c)
        if (s + 1) % 10 == 0:
            print(f"    seed {s+1}/{n_seeds}", flush=True)

    # ---- bootstrap-family statistics on the seed-0 fit
    te, pc, ph = keep["te"], keep["pc"], keep["ph"]
    yte, ste, dte = y[te], sid[te], doc[te]
    ycte = yc[te]

    def stat_r2(i):
        return P.r2_score(ycte[i], ph[i]) - P.r2_score(ycte[i], pc[i])

    def stat_cc(i):
        return (P.within_state_concordance(yte[i], ph[i], ste[i])[0]
                - P.within_state_concordance(yte[i], pc[i], ste[i])[0])

    boots = {}
    for nm, fn in (("d_r2", stat_r2), ("d_conc", stat_cc)):
        cb = S.cluster_boot_percentile(fn, dte, n_boot=4000, seed=1)
        th = fn(np.arange(len(yte)))
        bc = S.bca(fn, dte, th, cb["draws"])
        boots[nm] = {"observed": float(th),
                     "cluster_boot_ci": [float(x) for x in cb["ci"]],
                     "bca_ci": [float(x) for x in bc["ci"]],
                     "bca_z0": bc["z0"], "bca_a": bc["a"],
                     "jackknife_se": bc["jackknife_se"]}

    out = {"arm": arm, "target": target, "layer": int(L), "ceiling": float(ceil),
           "n_seeds": n_seeds, "n_train_docs": int(n_tr), "n_test_docs": int(n_te),
           "seed0_bootstraps": boots, "probes": {}}
    for name, v in per_seed.items():
        e = {}
        for m_ in ("d_r2", "d_conc"):
            dd = np.array(v[m_], float)
            e[m_] = {"per_seed": [float(x) for x in dd],
                     "naive_t": S.naive_t(dd),
                     "nadeau_bengio": S.nadeau_bengio_t(dd, n_tr, n_te),
                     "sign_rank": S.sign_and_rank(dd)}
            for k in ("naive_t", "nadeau_bengio"):
                e[m_][k] = {kk: ([float(x) for x in vv] if isinstance(vv, tuple)
                                 else float(vv)) for kk, vv in e[m_][k].items()}
        out["probes"][name] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", default="data/estimate_all.json")
    ap.add_argument("--task", action="store_true",
                    help="run the task-grounded arms (A_task) instead")
    a = ap.parse_args()
    arms, targets = (TASK_ARMS, TASK_TARGETS) if a.task else (ARMS, TARGETS)
    res = []
    for arm, tags in arms.items():
        for target, sk in targets.items():
            print(f"=== {arm} / {target} ===", flush=True)
            res.append(run(arm, tags, target, sk, a.seeds))
            r = res[-1]
            print(f"    layer {r['layer']}  ceiling {r['ceiling']:.3f}", flush=True)
    # BH-FDR across probes, within each arm x target, on the corrected p-values
    for r in res:
        names = [n for n in r["probes"] if not n.startswith("CONTROL")]
        for m_ in ("d_r2", "d_conc"):
            ps = [r["probes"][n][m_]["nadeau_bengio"]["p"] for n in names]
            keep, adj = S.bh_fdr(ps)
            for n, k, q in zip(names, keep, adj):
                r["probes"][n][m_]["bh_fdr_q"] = float(q)
                r["probes"][n][m_]["bh_fdr_pass"] = bool(k)
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

"""
Confirmatory test -- strictly follows the frozen preregistered protocol (paper Appendix A).

Run once. Two designs:
    D1 frozen transfer: probes fitted and frozen on the original data a3/b3, applied to all freshA rows.
    D2 same-protocol refit: refit and evaluate on freshA's own document-level split.

Fresh data are passed directly as `X_test` to the fitting function, so D1 cannot
"fit on fresh" -- by design, not by accident.
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
from rlib import probes2 as P            # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402
from rlib import stats2 as S             # noqa: E402

LAYER = 6                    # locked by preregistration
PCA_D = 32                   # locked by preregistration
ALPHAS = np.logspace(-2, 9, 34)
GAMMAS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
CRITERION = "within_r2"      # locked by preregistration
TARGET = "A_pertok"          # locked by preregistration


def build_blocks(d_tr, idx_tr, idx_va, d_te, idx_te, layer, pca_d):
    """train/val come from one dataset, test may come from another (used by D1)."""
    def get(d, idx):
        return {"cheap": RD.cheap_block(d)[idx],
                "hi": RD.h_i(d, layer)[idx],
                "hg": RD.h_g(d, layer)[idx]}
    tr, va, te = get(d_tr, idx_tr), get(d_tr, idx_va), get(d_te, idx_te)
    out = {"raw": {}, "pca": {}}
    for nm in ("cheap", "hi", "hg"):
        out["raw"][nm] = {"train": tr[nm], "val": va[nm], "test": te[nm]}
    # PCA is fitted on train only
    for nm in ("hi", "hg"):
        pca = RD.TrainPCA(pca_d, whiten=True).fit(tr[nm])
        out["pca"][nm] = {k: pca.transform(out["raw"][nm][k])
                          for k in ("train", "val", "test")}
    mu = tr["cheap"].mean(0, keepdims=True)
    sd = tr["cheap"].std(0, keepdims=True); sd[sd < 1e-8] = 1.0
    out["pca"]["cheap"] = {k: ((out["raw"]["cheap"][k] - mu) / sd
                               ).astype(np.float32)
                           for k in ("train", "val", "test")}
    return out


def fit_family(B, y, sid, probe_seeds=3, epochs=400, hg_override=None):
    """Fit all probes per preregistration; returns {name: (pred_test, meta)}."""
    sel = P.make_selector(CRITERION, sid["val"])
    raw, pca = dict(B["raw"]), dict(B["pca"])
    if hg_override is not None:
        raw["hg"] = hg_override["raw"]; pca["hg"] = hg_override["pca"]
    out = {}

    def keep(nm, m):
        out[nm] = {"pred": m["pred_test"], "pred_val": m["pred_val"],
                   "val_score": m["val_score"], "hp": m.get("hp"),
                   "n_params": m.get("n_params")}

    Zi = {k: pca["hi"][k][:, :PCA_D] for k in ("train", "val", "test")}
    Zg = {k: pca["hg"][k][:, :PCA_D] for k in ("train", "val", "test")}
    Xa = {k: np.concatenate([Zi[k], Zg[k]], 1) for k in Zi}
    KR = {k: (Zi[k][:, :, None] * Zg[k][:, None, :]
              ).reshape(len(Zi[k]), -1).astype(np.float32) for k in Zi}
    Xk = {k: np.concatenate([Zi[k], Zg[k], KR[k]], 1) for k in Zi}
    Hf = {k: np.concatenate([raw["hi"][k], raw["hg"][k]], 1) for k in Zi}

    keep("cheap", P.fit_ridge(raw["cheap"]["train"], y["train"],
                              raw["cheap"]["val"], y["val"],
                              raw["cheap"]["test"], sel, ALPHAS))
    keep("additive_pca", P.fit_ridge_2block(
        pca["cheap"]["train"], Xa["train"], y["train"],
        pca["cheap"]["val"], Xa["val"], y["val"],
        pca["cheap"]["test"], Xa["test"], sel, ALPHAS, GAMMAS))
    keep("kron_pca", P.fit_ridge_2block(
        pca["cheap"]["train"], Xk["train"], y["train"],
        pca["cheap"]["val"], Xk["val"], y["val"],
        pca["cheap"]["test"], Xk["test"], sel, ALPHAS, GAMMAS))
    keep("cheap+H", P.fit_ridge_2block(
        raw["cheap"]["train"], Hf["train"], y["train"],
        raw["cheap"]["val"], Hf["val"], y["val"],
        raw["cheap"]["test"], Hf["test"], sel, ALPHAS, GAMMAS))

    d_c = pca["cheap"]["train"].shape[1]
    d_i = pca["hi"]["train"].shape[1]
    d_g = pca["hg"]["train"].shape[1]
    F = {"c": pca["cheap"], "hi": pca["hi"], "hg": pca["hg"]}
    rn = P._Runner({k: F[k]["train"] for k in F}, y["train"], sid["train"],
                   {k: F[k]["val"] for k in F}, y["val"], sid["val"],
                   {k: F[k]["test"] for k in F}, loss_kind="pairwise")
    m = rn.run(lambda: P.MLPProbe([d_c], (256,), keys=("c",)), sel,
               seeds=tuple(range(probe_seeds)), epochs=epochs)
    if m is not None:
        keep("rank_cheap", m)

    rn2 = P._Runner({k: F[k]["train"] for k in F}, y["train"], sid["train"],
                    {k: F[k]["val"] for k in F}, y["val"], sid["val"],
                    {k: F[k]["test"] for k in F}, loss_kind="mse")
    best = None
    for rk in (2, 4, 8, 16):
        mm = rn2.run(lambda rk=rk: P.BilinearC(d_i, d_g, rk, d_c), sel,
                     seeds=tuple(range(probe_seeds)), epochs=epochs)
        if mm is not None and (best is None
                               or mm["val_score"] > best["val_score"]):
            best = mm; best["hp"]["rank"] = rk
    if best is not None:
        keep("bilinear", best)
    return out


def evaluate(fam, y_te, sid_te, doc_te, n_perm, n_boot, tag):
    _, groups = M.group_slices(sid_te)
    state_doc = np.array([doc_te[g[0]] for g in groups])
    # best_cheap: chosen on the **validation set** from {cheap, rank_cheap}
    cands = [k for k in ("cheap", "rank_cheap") if k in fam]
    best_cheap = max(cands, key=lambda k: fam[k]["val_score"])
    rep = {"tag": tag, "best_cheap": best_cheap,
           "val_scores": {k: fam[k]["val_score"] for k in fam},
           "probes": {}, "comparisons": {}}
    for k, v in fam.items():
        rep["probes"][k] = M.full_report(y_te, v["pred"], sid_te, groups,
                                         p_base=fam["cheap"]["pred"])
        rep["probes"][k]["n_params"] = v["n_params"]
        rep["probes"][k]["hp"] = v["hp"]

    comps = [
        ("PRIMARY_additive_vs_bestcheap", "additive_pca", best_cheap),
        ("additive_vs_cheapridge", "additive_pca", "cheap"),
        ("cheapH_vs_cheapridge", "cheap+H", "cheap"),
        ("RELATIONAL_kron_vs_additive", "kron_pca", "additive_pca"),
        ("bilinear_vs_bestcheap", "bilinear", best_cheap),
        ("rankcheap_vs_cheapridge", "rank_cheap", "cheap"),
    ]
    for name, a, b in comps:
        if a not in fam or b not in fam or a == b:
            rep["comparisons"][name] = {"skipped": True}
            continue
        c = S.compare_probes(y_te, fam[a]["pred"], fam[b]["pred"], sid_te,
                             state_doc, groups, n_perm=n_perm, seed=0)

        def stat_fn(idx, a=a, b=b):
            return {
                "delta_concordance":
                    M.concordance(y_te[idx], fam[a]["pred"][idx], sid_te[idx])
                    - M.concordance(y_te[idx], fam[b]["pred"][idx], sid_te[idx]),
                "delta_top1":
                    M.topk_metrics(y_te[idx], fam[a]["pred"][idx],
                                   sid_te[idx])["top1"]
                    - M.topk_metrics(y_te[idx], fam[b]["pred"][idx],
                                     sid_te[idx])["top1"],
                "delta_within_r2":
                    M.within_state_r2(y_te[idx], fam[a]["pred"][idx],
                                      sid_te[idx])
                    - M.within_state_r2(y_te[idx], fam[b]["pred"][idx],
                                        sid_te[idx]),
                "delta_regret_norm":
                    M.regret(y_te[idx], fam[b]["pred"][idx],
                             sid_te[idx])["regret_norm_mean"]
                    - M.regret(y_te[idx], fam[a]["pred"][idx],
                               sid_te[idx])["regret_norm_mean"],
            }
        c["cluster_bootstrap"] = S.cluster_bootstrap_fixed(
            stat_fn, doc_te, n_boot=n_boot, seed=0)
        rep["comparisons"][name] = c
        cb = c["cluster_bootstrap"]
        print(f"  [{tag}/{name}] Dconc={c['delta_concordance']['mean']:+.4f} "
              f"CI[{cb['delta_concordance']['ci_lo']:+.4f},"
              f"{cb['delta_concordance']['ci_hi']:+.4f}] "
              f"p={c['delta_concordance']['permutation']['p']:.4f} | "
              f"Dtop1={c['delta_top1']['mean']:+.4f} "
              f"CI[{cb['delta_top1']['ci_lo']:+.4f},"
              f"{cb['delta_top1']['ci_hi']:+.4f}]", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_tags", nargs="+", default=["a3", "b3"])
    ap.add_argument("--fresh_tag", default="freshA")
    ap.add_argument("--probe_seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "CONFIRMATORY_freshA.json"))
    args = ap.parse_args()
    t0 = time.time()

    print("[confirm] loading old + fresh", flush=True)
    d_old = RD.load_labels(args.old_tags)
    d_new = RD.load_labels([args.fresh_tag])
    print(f"[confirm] old n={len(d_old[TARGET])} "
          f"fresh n={len(d_new[TARGET])} "
          f"fresh states={len(np.unique(d_new['state_id']))} "
          f"fresh docs={len(np.unique(d_new['doc_id']))}", flush=True)
    # no-overlap assertion: fresh prompt_rows must not intersect the original data
    ov = set(np.unique(d_old["prompt_row"]).tolist()) & \
        set(np.unique(d_new["prompt_row"]).tolist())
    assert not ov, f"fresh prompts overlap the original data: {sorted(ov)[:10]}"
    print("[confirm] no prompt overlap OK", flush=True)

    report = {"config": vars(args), "layer": LAYER, "pca_d": PCA_D,
              "criterion": CRITERION, "target": TARGET,
              "n_old": int(len(d_old[TARGET])),
              "n_fresh": int(len(d_new[TARGET])),
              "n_fresh_states": int(len(np.unique(d_new["state_id"]))),
              "n_fresh_docs": int(len(np.unique(d_new["doc_id"])))}
    sk = "A_full_seeds"
    if sk in d_new:
        _, gnew, _ = RD.state_groups(d_new["state_id"])
        report["fresh_ceiling"] = RD.noise_ceiling(d_new[sk])
        report["fresh_within_ceiling"] = RD.within_state_noise_ceiling(
            d_new[sk], d_new["state_id"], gnew)
        print(f"[confirm] fresh ceiling pooled "
              f"{report['fresh_ceiling']['ceiling']:.3f} "
              f"(SNR {report['fresh_ceiling']['snr']:.2f}) within "
              f"{report['fresh_within_ceiling']['ceiling']:.3f}", flush=True)

    # ---------------- D1 frozen transfer ----------------
    sp_old = RD.doc_splits(d_old, seed=0)
    RD.check_split_disjoint(d_old, sp_old)
    idx_fresh_all = np.arange(len(d_new[TARGET]))
    B1 = build_blocks(d_old, sp_old["train"], sp_old["val"],
                      d_new, idx_fresh_all, LAYER, PCA_D)
    y1 = {"train": d_old[TARGET][sp_old["train"]],
          "val": d_old[TARGET][sp_old["val"]],
          "test": d_new[TARGET][idx_fresh_all]}
    sid1 = {"train": d_old["state_id"][sp_old["train"]],
            "val": d_old["state_id"][sp_old["val"]],
            "test": d_new["state_id"]}
    print(f"\n[confirm] === D1 frozen transfer (fit on old, eval on ALL fresh) ===",
          flush=True)
    fam1 = fit_family(B1, y1, sid1, args.probe_seeds, args.epochs)
    report["D1_frozen_transfer"] = evaluate(
        fam1, y1["test"], sid1["test"], d_new["doc_id"], args.n_perm,
        args.n_boot, "D1")
    json.dump(report, open(args.out, "w"), indent=2, default=float)

    # ---------------- D2 same-protocol refit ----------------
    sp_new = RD.doc_splits(d_new, seed=0)
    RD.check_split_disjoint(d_new, sp_new)
    B2 = build_blocks(d_new, sp_new["train"], sp_new["val"],
                      d_new, sp_new["test"], LAYER, PCA_D)
    y2 = {k: d_new[TARGET][v] for k, v in sp_new.items()}
    sid2 = SC.split_sid(d_new, sp_new)
    print(f"\n[confirm] === D2 same-protocol refit (fresh split) ===", flush=True)
    fam2 = fit_family(B2, y2, sid2, args.probe_seeds, args.epochs)
    report["D2_refit"] = evaluate(fam2, y2["test"], sid2["test"],
                                  d_new["doc_id"][sp_new["test"]],
                                  args.n_perm, args.n_boot, "D2")
    json.dump(report, open(args.out, "w"), indent=2, default=float)

    # ---------------- falsification controls (D2 design) ----------------
    print(f"\n[confirm] === falsification controls ===", flush=True)
    rng = np.random.default_rng(20260831)
    ctl = SC.make_controls(B2, sid2, rng)
    report["controls"] = {}
    for cname in ("shuffle_hg", "gauss_hg", "gauss_hi"):
        ov_ = ctl[cname]
        B = {"raw": dict(B2["raw"]), "pca": dict(B2["pca"])}
        blk = "hi" if cname == "gauss_hi" else "hg"
        B["raw"][blk] = ov_["raw"]; B["pca"][blk] = ov_["pca"]
        f = fit_family(B, y2, sid2, args.probe_seeds, args.epochs)
        _, g2 = M.group_slices(sid2["test"])
        a = M.concordance(y2["test"], f["additive_pca"]["pred"],
                          sid2["test"], g2)
        b = M.concordance(y2["test"], f["cheap"]["pred"], sid2["test"], g2)
        report["controls"][cname] = {
            "delta_concordance_additive_vs_cheap": float(a - b),
            "concordance_additive": float(a), "concordance_cheap": float(b)}
        print(f"  [ctl {cname}] Dconc={a-b:+.4f}", flush=True)
    yp = SC.permute_labels_within_state(y2, sid2, rng)
    f = fit_family(B2, yp, sid2, args.probe_seeds, args.epochs)
    _, g2 = M.group_slices(sid2["test"])
    a = M.concordance(yp["test"], f["additive_pca"]["pred"], sid2["test"], g2)
    b = M.concordance(yp["test"], f["cheap"]["pred"], sid2["test"], g2)
    report["controls"]["label_perm_within"] = {
        "delta_concordance_additive_vs_cheap": float(a - b),
        "concordance_additive": float(a), "concordance_cheap": float(b)}
    print(f"  [ctl label_perm_within] Dconc={a-b:+.4f}", flush=True)

    json.dump(report, open(args.out, "w"), indent=2, default=float)
    print(f"\n[confirm] done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()

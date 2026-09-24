"""
Inference layer: run full S1-S6 + multiple comparisons on a small set of **pre-specified** probe comparisons.

Division of labour with the broad screen:
    * broad screen   -- breadth exploration, dozens of probes, point estimates only (exploratory).
    * `r2_stats.py`  -- proper inference on **a few** comparisons, saving per-state details.

Four pre-defined comparisons (each a fair "same objective, same capacity" comparison):
    C1  cheap ridge            vs  cheap+[h_i;h_g] ridge     (original G1/G2 convention)
    C2  cheap ridge            vs  cheap+h_i ridge           (pure candidate-level channel)
    C3  ranknet(cheap)         vs  ranknet(cheap+hidden)     (**objective-matched**)
    C4  additive(cheap+Zi,Zg)  vs  kron(cheap+Zi,Zg,Zi x Zg)   (**capacity-matched relational test**)

C3/C4 are the two key controls:
    - C3 answers "is the hidden-state gain just a more suitable loss function";
    - C4 answers "does the state-modulation term itself contribute anything".
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
from rlib import probes2 as P            # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402
from rlib import stats2 as S             # noqa: E402

ARMS = {"MDLM_anc": ["a3", "b3"], "MDLM_conf": ["c3", "d3"],
        "SEDD_anc": ["s1", "s2"], "MDLM_H8": ["h8a", "h8b"],
        "MDLM_H32": ["h32a", "h32b"], "FRESH_MDLM_anc": ["freshA"]}
SEEDKEY = {"A_pertok": "A_full_seeds", "A_future": "A_future_seeds"}


def fit_all(prep, y, sid, criterion, kron_d, probe_seeds, epochs):
    """Returns {name: pred_test}, only the probes needed for the four comparisons."""
    sel = P.make_selector(criterion, sid["val"])
    pca, raw = prep["pca"], prep["raw"]
    out, meta = {}, {}

    def keep(name, m):
        out[name] = m["pred_test"]
        meta[name] = {"val_score": m.get("val_score"), "hp": m.get("hp"),
                      "n_params": m.get("n_params")}

    # --- C1 / C2 ---
    keep("cheap", P.fit_ridge(raw["cheap"]["train"], y["train"],
                              raw["cheap"]["val"], y["val"],
                              raw["cheap"]["test"], sel))
    H = {k: np.concatenate([raw["hi"][k], raw["hg"][k]], 1)
         for k in ("train", "val", "test")}
    keep("cheap+H", P.fit_ridge_2block(
        raw["cheap"]["train"], H["train"], y["train"],
        raw["cheap"]["val"], H["val"], y["val"],
        raw["cheap"]["test"], H["test"], sel))
    keep("cheap+hi", P.fit_ridge_2block(
        raw["cheap"]["train"], raw["hi"]["train"], y["train"],
        raw["cheap"]["val"], raw["hi"]["val"], y["val"],
        raw["cheap"]["test"], raw["hi"]["test"], sel))

    # --- C3 objective-matched ranking MLP ---
    d_c = pca["cheap"]["train"].shape[1]
    d_i = pca["hi"]["train"].shape[1]
    d_g = pca["hg"]["train"].shape[1]
    F = {"c": pca["cheap"], "hi": pca["hi"], "hg": pca["hg"]}
    for nm, keys, dims in (("rank_cheap", ("c",), [d_c]),
                           ("rank_cheap_hidden", ("c", "hi", "hg"),
                            [d_c, d_i, d_g])):
        rn = P._Runner({k: F[k]["train"] for k in F}, y["train"], sid["train"],
                       {k: F[k]["val"] for k in F}, y["val"], sid["val"],
                       {k: F[k]["test"] for k in F}, loss_kind="pairwise")
        m = rn.run(lambda keys=keys, dims=dims: P.MLPProbe(dims, (256,),
                                                           keys=keys),
                   sel, seeds=tuple(range(probe_seeds)), epochs=epochs)
        if m is not None:
            keep(nm, m)

    # --- C4 capacity-matched relational test ---
    Zi = {k: pca["hi"][k][:, :kron_d] for k in ("train", "val", "test")}
    Zg = {k: pca["hg"][k][:, :kron_d] for k in ("train", "val", "test")}
    KR = {k: (Zi[k][:, :, None] * Zg[k][:, None, :]
              ).reshape(len(Zi[k]), -1).astype(np.float32) for k in Zi}
    Xa = {k: np.concatenate([Zi[k], Zg[k]], 1) for k in Zi}
    Xk = {k: np.concatenate([Zi[k], Zg[k], KR[k]], 1) for k in Zi}
    keep("additive_pca", P.fit_ridge_2block(
        pca["cheap"]["train"], Xa["train"], y["train"],
        pca["cheap"]["val"], Xa["val"], y["val"],
        pca["cheap"]["test"], Xa["test"], sel))
    keep("kron_pca", P.fit_ridge_2block(
        pca["cheap"]["train"], Xk["train"], y["train"],
        pca["cheap"]["val"], Xk["val"], y["val"],
        pca["cheap"]["test"], Xk["test"], sel))

    # --- low-rank bilinear (the only probe that reads relational structure on synthetic B) ---
    rn = P._Runner({k: F[k]["train"] for k in F}, y["train"], sid["train"],
                   {k: F[k]["val"] for k in F}, y["val"], sid["val"],
                   {k: F[k]["test"] for k in F}, loss_kind="mse")
    best = None
    for rk in (2, 4, 8, 16):
        m = rn.run(lambda rk=rk: P.BilinearC(d_i, d_g, rk, d_c), sel,
                   seeds=tuple(range(probe_seeds)), epochs=epochs)
        if m is not None and (best is None or m["val_score"] > best["val_score"]):
            best = m; best["hp"]["rank"] = rk
    if best is not None:
        keep("bilinear", best)
    return out, meta


COMPARISONS = [
    ("C1_cheapH_vs_cheap", "cheap+H", "cheap",
     "original G1/G2 convention: hidden block vs strong control set"),
    ("C2_cheaphi_vs_cheap", "cheap+hi", "cheap",
     "pure candidate-level channel: add h_i only"),
    ("C3_rank_hidden_vs_rank_cheap", "rank_cheap_hidden", "rank_cheap",
     "objective-matched: MLPs with the same pairwise ranking loss, differing only in the hidden block"),
    ("C4_kron_vs_additive", "kron_pca", "additive_pca",
     "capacity-matched relational test: differing only in the h_i x h_g interaction"),
    ("C5_bilinear_vs_cheap", "bilinear", "cheap",
     "low-rank bilinear (the only effective relational probe on synthetic B)"),
    ("C6_rank_cheap_vs_cheap", "rank_cheap", "cheap",
     "objective upgrade **without hidden states**: ranking loss vs pooled MSE"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="MDLM_anc")
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--criterion", default="within_r2")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--probe_seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--kron_d", type=int, default=32)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--hg_kind", default="hg")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tags = ARMS[args.arm]
    out = args.out or os.path.join(HERE, "results",
                                   f"R2stats_{args.arm}_{args.target}_"
                                   f"L{args.layer}_{args.criterion}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()

    d = RD.load_labels(tags)
    _, groups_all, _ = RD.state_groups(d["state_id"])
    sp = RD.doc_splits(d, seed=args.split_seed)
    RD.check_split_disjoint(d, sp)
    sid = SC.split_sid(d, sp)
    y = SC.split_targets(d, sp, args.target)
    prep = SC.prepare(d, sp, args.layer, pca_dim=args.pca_dim,
                      hg_kind=args.hg_kind, groups=groups_all)
    print(f"[R2] arm={args.arm} layer={args.layer} crit={args.criterion} "
          f"n_test={len(y['test'])}", flush=True)

    preds, meta = fit_all(prep, y, sid, args.criterion, args.kron_d,
                          args.probe_seeds, args.epochs)
    print(f"[R2] fitted {len(preds)} probes ({time.time()-t0:.0f}s)", flush=True)

    yt, st = y["test"], sid["test"]
    uniq_states, groups_te = M.group_slices(st)
    # document of each state (for S5 hierarchical analysis and cluster bootstrap)
    doc_te = d["doc_id"][sp["test"]]
    state_doc = np.array([doc_te[g[0]] for g in groups_te])

    ceil = RD.noise_ceiling(d[SEEDKEY[args.target]]) \
        if SEEDKEY.get(args.target) in d else None
    wceil = RD.within_state_noise_ceiling(d[SEEDKEY[args.target]],
                                          d["state_id"], groups_all) \
        if SEEDKEY.get(args.target) in d else None

    report = {"config": vars(args), "n_test": int(len(yt)),
              "n_test_states": int(len(uniq_states)),
              "n_test_docs": int(len(np.unique(doc_te))),
              "ceiling": ceil, "within_ceiling": wceil,
              "probe_meta": meta, "probes": {}, "comparisons": {}}

    for k, p in preds.items():
        report["probes"][k] = M.full_report(
            yt, p, st, groups_te, p_base=preds["cheap"],
            ceiling=(ceil or {}).get("ceiling"),
            within_ceiling=(wceil or {}).get("ceiling"))

    # ---- S2 / S3 / S5 per comparison ----
    perm_null_store = {}
    for name, a, b, desc in COMPARISONS:
        if a not in preds or b not in preds:
            report["comparisons"][name] = {"skipped": f"missing {a} or {b}"}
            continue
        cmp_ = S.compare_probes(yt, preds[a], preds[b], st, state_doc,
                                groups_te, n_perm=args.n_perm, seed=0)
        # S1 fixed-model cluster bootstrap (resample whole documents; all statistics share the same replicate)
        def stat_fn(idx, a=a, b=b):
            return {
                "delta_concordance":
                    M.concordance(yt[idx], preds[a][idx], st[idx])
                    - M.concordance(yt[idx], preds[b][idx], st[idx]),
                "delta_within_r2":
                    M.within_state_r2(yt[idx], preds[a][idx], st[idx])
                    - M.within_state_r2(yt[idx], preds[b][idx], st[idx]),
                "delta_top1":
                    M.topk_metrics(yt[idx], preds[a][idx], st[idx])["top1"]
                    - M.topk_metrics(yt[idx], preds[b][idx], st[idx])["top1"],
            }
        cmp_["cluster_bootstrap"] = S.cluster_bootstrap_fixed(
            stat_fn, doc_te, n_boot=args.n_boot, seed=0)
        cmp_["description"] = desc
        report["comparisons"][name] = cmp_
        pa = S.per_state_metrics(yt, preds[a], st, groups_te)
        pb = S.per_state_metrics(yt, preds[b], st, groups_te)
        perm_null_store[name] = (pa["concordance"] - pb["concordance"])
        print(f"  [{name}] Dconc={cmp_['delta_concordance']['mean']:+.4f} "
              f"p={cmp_['delta_concordance']['permutation']['p']:.4f}  "
              f"Dtop1={cmp_['delta_top1']['mean']:+.4f} "
              f"p={cmp_['delta_top1']['permutation']['p']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- multiple comparisons: BH-FDR / BY / Westfall-Young max-T ----
    names = [n for n in perm_null_store]
    if names:
        pvals = np.array([report["comparisons"][n]["delta_concordance"]
                          ["permutation"]["p"] for n in names])
        rej_bh, adj_bh = S.bh_fdr(pvals)
        rej_by, adj_by = S.by_fdr(pvals)
        # max-T: all comparisons share the same sign flips, preserving the correlation structure
        rng = np.random.default_rng(0)
        D = np.stack([perm_null_store[n] for n in names], 1)   # (S, m)
        ok = np.isfinite(D).all(1)
        D = D[ok]
        B = 5000
        signs = rng.choice([-1.0, 1.0], size=(B, D.shape[0]))
        null = np.stack([(signs * D[:, j][None, :]).mean(1)
                         for j in range(D.shape[1])], 1)        # (B, m)
        obs = D.mean(0)
        p_wy = S.westfall_young(obs, null, alternative="greater")
        report["multiple_testing"] = {
            "endpoint": "delta_concordance",
            "names": names,
            "p_uncorrected": pvals.tolist(),
            "p_BH": adj_bh.tolist(), "reject_BH": rej_bh.tolist(),
            "p_BY": adj_by.tolist(), "reject_BY": rej_by.tolist(),
            "p_westfall_young": p_wy.tolist()}

    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"\n[R2] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

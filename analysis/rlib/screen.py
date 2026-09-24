"""
Unified probe screen: run P0-P13 on the same (train/val/test) split and produce the same metric set.

Reused in two places:
    * `synthetic_tests.py` -- verify on labels with a **known answer** that the probe family can / cannot read them out;
    * broad screens      -- exploratory screens on real labels.

Key rule: `select_by` sets the validation criterion for choosing layer and hyperparameters. By default both
`pooled_r2` and `within_r2` (and optionally `concordance`) are run,
because the decision is based on within-state ranking.
"""
import numpy as np
import torch

from . import metrics as M
from . import probes2 as P
from . import rdata as RD


# ------------------------------------------------------------ data preparation --
def prepare(d, sp, layer, pca_dim=128, hg_kind="hg", groups=None, seed=0):
    """Split one layer's hidden states + control block into three parts and apply train-only PCA.

    hg_kind: 'hg'  = stored all-position pooled h_global
                      'hgm' = **leave-one-out** mean of other candidates' h_i in the same state (masked-position pooling proxy)
                      'both'= concatenation of the two
    """
    tr, va, te = sp["train"], sp["val"], sp["test"]
    cheap = RD.cheap_block(d)
    hi = RD.h_i(d, layer)
    if hg_kind == "hg":
        hg = RD.h_g(d, layer)
    elif hg_kind == "hgm":
        hg = RD.h_gm(d, layer, groups)
    elif hg_kind == "both":
        hg = np.concatenate([RD.h_g(d, layer), RD.h_gm(d, layer, groups)], 1)
    else:
        raise KeyError(hg_kind)

    out = {"raw": {}, "pca": {}}
    for nm, X in (("cheap", cheap), ("hi", hi), ("hg", hg)):
        out["raw"][nm] = {"train": X[tr], "val": X[va], "test": X[te]}

    for nm in ("hi", "hg"):
        pca = RD.TrainPCA(min(pca_dim, out["raw"][nm]["train"].shape[1]),
                          whiten=True).fit(out["raw"][nm]["train"])
        out["pca"][nm] = {k: pca.transform(out["raw"][nm][k])
                          for k in ("train", "val", "test")}
    # the cheap block is low-dimensional; plain standardisation suffices
    mu = cheap[tr].mean(0, keepdims=True)
    sd = cheap[tr].std(0, keepdims=True); sd[sd < 1e-8] = 1.0
    out["pca"]["cheap"] = {k: ((cheap[v] - mu) / sd).astype(np.float32)
                           for k, v in (("train", tr), ("val", va), ("test", te))}
    out["idx"] = {"train": tr, "val": va, "test": te}
    return out


def split_targets(d, sp, target):
    y = d[target].astype(np.float64)
    return {k: y[v] for k, v in sp.items()}


def split_sid(d, sp):
    return {k: d["state_id"][v] for k, v in sp.items()}


# -------------------------------------------------------------------- controls ---
def make_controls(prep, sid, rng):
    """Feature-replacement versions for three falsification controls (only the hidden block is replaced)."""
    ctl = {}
    # (a) shuffled h_global: permute h_g as whole blocks **across states**.
    #     If a probe relies on real state context from h_g, this control must go to zero.
    for space in ("raw", "pca"):
        hg = prep[space]["hg"]
        new = {}
        for k in ("train", "val", "test"):
            s = sid[k]
            uniq, groups = M.group_slices(s)
            perm = rng.permutation(len(groups))
            X = np.array(hg[k], copy=True)
            src = [hg[k][groups[p][0]] for p in perm]
            for gi, g in enumerate(groups):
                X[g] = src[gi]
            new[k] = X
        ctl.setdefault("shuffle_hg", {})[space] = new
    # (b) Gaussian replacement (same shape and scale)
    for name, blk in (("gauss_hi", "hi"), ("gauss_hg", "hg")):
        for space in ("raw", "pca"):
            ref = prep[space][blk]
            sd = ref["train"].std(0, keepdims=True)
            mu = ref["train"].mean(0, keepdims=True)
            ctl.setdefault(name, {})[space] = {
                k: (mu + sd * rng.standard_normal(ref[k].shape)).astype(np.float32)
                for k in ("train", "val", "test")}
    return ctl


def permute_labels_within_state(y, sid, rng):
    out = {}
    for k in y:
        yy = np.array(y[k], copy=True)
        _, groups = M.group_slices(sid[k])
        for g in groups:
            yy[g] = rng.permutation(yy[g])
        out[k] = yy
    return out


# ------------------------------------------------------------------- probes ----
def _feats(prep, keys, space="pca", override=None):
    src = dict(prep[space])
    if override:
        src.update(override)
    return {k: {sp: src[k][sp] for sp in ("train", "val", "test")} for k in keys}


def _center(y, sid):
    """Within-state centering (same function for training and evaluation)."""
    _, groups = M.group_slices(sid)
    out = np.asarray(y, np.float64).copy()
    for g in groups:
        out[g] -= out[g].mean()
    return out


def _pack(f, split):
    return {k: v[split] for k, v in f.items()}


def run_probes(prep, y, sid, doc, selector_kind, which=None, seeds=(0, 1, 2),
               epochs=400, ranks=(2, 4, 8, 16, 32, 64), widths=(128, 512),
               kron_dims=(16, 32), hg_override=None, verbose=False):
    """Run a set of probes on a given split; returns {probe_name: {'pred_test','val_score',...}}.

    hg_override: replace h_g with a control block (shuffle/gauss), dict(space->split->array).
    """
    sel = P.make_selector(selector_kind, sid["val"])
    R = {}
    raw = dict(prep["raw"]); pca = dict(prep["pca"])
    if hg_override is not None:
        raw = dict(raw); raw["hg"] = hg_override["raw"]
        pca = dict(pca); pca["hg"] = hg_override["pca"]
    if which is None:
        which = ["P0", "P1", "P2k", "P2", "P3", "P4", "P5", "P6", "P7",
                 "P8", "P9", "P13"]

    def ridge1(X, name):
        R[name] = P.fit_ridge(X["train"], y["train"], X["val"], y["val"],
                              X["test"], sel)

    def ridge2(Xc, Xh, name):
        R[name] = P.fit_ridge_2block(Xc["train"], Xh["train"], y["train"],
                                     Xc["val"], Xh["val"], y["val"],
                                     Xc["test"], Xh["test"], sel)

    # ---------------- P0 plain linear family ----------------
    if "P0" in which:
        ridge1(raw["cheap"], "P0_cheap")
        ridge1(raw["hi"], "P0_hi")
        ridge1(raw["hg"], "P0_hg")
        cat = {k: np.concatenate([raw["hi"][k], raw["hg"][k]], 1)
               for k in ("train", "val", "test")}
        ridge1(cat, "P0_hi_hg")
        ridge2(raw["cheap"], raw["hi"], "P0_cheap+hi")
        ridge2(raw["cheap"], cat, "P0_cheap+H")

    # ---------------- P1 residualised ----------------
    if "P1" in which:
        allidx = np.concatenate([prep["idx"]["train"], prep["idx"]["val"],
                                 prep["idx"]["test"]])
        Xc_all = np.concatenate([raw["cheap"][k] for k in
                                 ("train", "val", "test")], 0)
        y_all = np.concatenate([y[k] for k in ("train", "val", "test")])
        doc_all = np.concatenate([doc[k] for k in ("train", "val", "test")])
        resid, _ = P.cross_fit_residual(Xc_all, y_all, doc_all)
        n_tr, n_va = len(y["train"]), len(y["val"])
        r = {"train": resid[:n_tr], "val": resid[n_tr:n_tr + n_va],
             "test": resid[n_tr + n_va:]}
        sel_r = P.make_selector(selector_kind, sid["val"])
        for nm, X in (("hi", raw["hi"]),
                      ("hi_hg", {k: np.concatenate([raw["hi"][k], raw["hg"][k]], 1)
                                 for k in ("train", "val", "test")})):
            R[f"P1_resid_{nm}"] = P.fit_ridge(
                X["train"], r["train"], X["val"], r["val"], X["test"], sel_r)
            R[f"P1_resid_{nm}"]["residual_target"] = True
        R["_residual"] = r

    # ---------------- shared inputs for the torch family ----------------
    d_i = pca["hi"]["train"].shape[1]
    d_g = pca["hg"]["train"].shape[1]
    d_c = pca["cheap"]["train"].shape[1]
    F = {"hi": pca["hi"], "hg": pca["hg"], "c": pca["cheap"]}

    def runner(loss_kind):
        return P._Runner({k: F[k]["train"] for k in F}, y["train"], sid["train"],
                         {k: F[k]["val"] for k in F}, y["val"], sid["val"],
                         {k: F[k]["test"] for k in F}, loss_kind=loss_kind)

    # ---------------- P2 low-rank bilinear ----------------
    if "P2" in which:
        rn = runner("mse")
        best = None
        for rk in ranks:
            res = rn.run(lambda rk=rk: P.BilinearC(d_i, d_g, rk, d_c),
                         sel, seeds=seeds, epochs=epochs)
            if res is None:
                continue
            res["hp"]["rank"] = rk
            R[f"P2_bilinear_r{rk}"] = res
            if best is None or res["val_score"] > best["val_score"]:
                best = res
        if best is not None:
            R["P2_bilinear_best"] = best
        # equal-capacity control without the bilinear term (linear + cheap only)
        res = rn.run(lambda: P.BilinearC(d_i, d_g, 1, d_c, use_linear=True),
                     sel, seeds=seeds, epochs=epochs)
        if res is not None:
            R["P2_bilinear_r1"] = res

    # ---------------- P2k Kronecker ridge (convex form of the bilinear) ----------------
    # h_i^T U V^T h_g = <W, h_i h_g^T> is **linear** in the outer-product features,
    # so it can be solved in closed form with regularisation, far more reliably than training a low-rank parameterisation.
    # The low-rank version (P2_bilinear_r* above) is kept as a capacity-controlled comparison.
    if "P2k" in which:
        for dk in kron_dims:
            Zi = {k: pca["hi"][k][:, :dk] for k in ("train", "val", "test")}
            Zg = {k: pca["hg"][k][:, :dk] for k in ("train", "val", "test")}
            KR = {k: (Zi[k][:, :, None] * Zg[k][:, None, :]
                      ).reshape(len(Zi[k]), -1).astype(np.float32) for k in Zi}
            Xh = {k: np.concatenate([Zi[k], Zg[k], KR[k]], 1) for k in Zi}
            ridge2(pca["cheap"], Xh, f"P2k_kron_d{dk}")
            # interaction only, no main effects: does the interaction carry information **by itself**
            ridge2(pca["cheap"], KR, f"P2k_kroneOnly_d{dk}")
            # additive control (same PCA dims, no interaction) -- the capacity gap is the interaction contribution
            Xa = {k: np.concatenate([Zi[k], Zg[k]], 1) for k in Zi}
            ridge2(pca["cheap"], Xa, f"P2k_additive_d{dk}")
            # within-state centred target: directly optimise the candidate-level component
            yc = {k: _center(y[k], sid[k]) for k in ("train", "val", "test")}
            sel_w = P.make_selector("within_r2", sid["val"])
            m = P.fit_ridge_2block(pca["cheap"]["train"], Xh["train"], yc["train"],
                                   pca["cheap"]["val"], Xh["val"], yc["val"],
                                   pca["cheap"]["test"], Xh["test"], sel_w)
            m["centered_target"] = True
            R[f"P2k_kron_wc_d{dk}"] = m
            m = P.fit_ridge_2block(pca["cheap"]["train"], Xa["train"], yc["train"],
                                   pca["cheap"]["val"], Xa["val"], yc["val"],
                                   pca["cheap"]["test"], Xa["test"], sel_w)
            m["centered_target"] = True
            R[f"P2k_additive_wc_d{dk}"] = m

    # ---------------- P3 FiLM ----------------
    if "P3" in which:
        rn = runner("mse")
        res = rn.run(lambda: P.FiLM(d_i, d_g, hid=64, d_c=d_c), sel,
                     seeds=seeds, epochs=epochs)
        if res is not None:
            R["P3_film"] = res

    # ---------------- P4 relational MLP ----------------
    if "P4" in which:
        rel = {k: RD.relational_block(pca["hi"][k], pca["hg"][k])
               for k in ("train", "val", "test")}
        Frel = {"x": rel, "c": pca["cheap"]}
        rn = P._Runner({"x": rel["train"], "c": pca["cheap"]["train"]},
                       y["train"], sid["train"],
                       {"x": rel["val"], "c": pca["cheap"]["val"]}, y["val"],
                       sid["val"],
                       {"x": rel["test"], "c": pca["cheap"]["test"]},
                       loss_kind="mse")
        d_rel = rel["train"].shape[1]
        for w in widths:
            res = rn.run(lambda w=w: P.MLPProbe([d_rel, d_c], (w,),
                                                keys=("x", "c")),
                         sel, seeds=seeds, epochs=epochs)
            if res is not None:
                res["hp"]["width"] = w
                R[f"P4_relmlp_w{w}"] = res
        # equal-capacity cheap-only control
        rn_c = P._Runner({"c": pca["cheap"]["train"]}, y["train"], sid["train"],
                         {"c": pca["cheap"]["val"]}, y["val"], sid["val"],
                         {"c": pca["cheap"]["test"]}, loss_kind="mse")
        res = rn_c.run(lambda: P.MLPProbe([d_c], (widths[-1],), keys=("c",)),
                       sel, seeds=seeds, epochs=epochs)
        if res is not None:
            R["P4_cheaponly_mlp"] = res

    # ---------------- P5 kernel methods ----------------
    if "P5" in which:
        base = {k: np.concatenate([pca["cheap"][k], pca["hi"][k][:, :64],
                                   pca["hg"][k][:, :64]], 1)
                for k in ("train", "val", "test")}
        cheap_only = pca["cheap"]
        d0 = base["train"].shape[1]
        for gname, gamma in (("g0.1", 0.1 / d0), ("g1", 1.0 / d0),
                             ("g10", 10.0 / d0)):
            Z = {k: P.rff_features(base[k], 1024, gamma) for k in base}
            R[f"P5_rff_{gname}"] = P.fit_ridge(
                Z["train"], y["train"], Z["val"], y["val"], Z["test"], sel)
            Zc = {k: P.rff_features(cheap_only[k], 1024,
                                    gamma * d0 / cheap_only["train"].shape[1])
                  for k in base}
            R[f"P5_rff_cheaponly_{gname}"] = P.fit_ridge(
                Zc["train"], y["train"], Zc["val"], y["val"], Zc["test"], sel)
        small = {k: np.concatenate([pca["cheap"][k], pca["hi"][k][:, :24],
                                    pca["hg"][k][:, :24]], 1) for k in base}
        Zp = {k: P.poly2_features(small[k]) for k in small}
        R["P5_poly2"] = P.fit_ridge(Zp["train"], y["train"], Zp["val"],
                                    y["val"], Zp["test"], sel)
        Zpc = {k: P.poly2_features(pca["cheap"][k]) for k in small}
        R["P5_poly2_cheaponly"] = P.fit_ridge(Zpc["train"], y["train"],
                                              Zpc["val"], y["val"],
                                              Zpc["test"], sel)

    # ---------------- P6 boosted trees ----------------
    if "P6" in which:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            X = {k: np.concatenate([pca["cheap"][k], pca["hi"][k][:, :64],
                                    pca["hg"][k][:, :64]], 1)
                 for k in ("train", "val", "test")}
            Xc = pca["cheap"]
            for nm, XX in (("P6_boost", X), ("P6_boost_cheaponly", Xc)):
                bb = None
                for lr_ in (0.05, 0.1):
                    for leaves in (15, 31):
                        m = HistGradientBoostingRegressor(
                            learning_rate=lr_, max_leaf_nodes=leaves,
                            max_iter=400, early_stopping=False,
                            random_state=0).fit(XX["train"], y["train"])
                        pv = m.predict(XX["val"])
                        s = P._safe(sel(y["val"], pv))
                        if bb is None or s > bb["val_score"]:
                            bb = {"val_score": s, "pred_val": pv,
                                  "pred_test": m.predict(XX["test"]),
                                  "hp": {"lr": lr_, "leaves": leaves},
                                  "n_params": -1}
                R[nm] = bb
        except Exception as e:                                # noqa: BLE001
            R["P6_error"] = {"error": str(e)}

    # ---------------- P7 pairwise linear ----------------
    if "P7k" in which:
        dk = kron_dims[-1]
        Zi = {k: pca["hi"][k][:, :dk] for k in ("train", "val", "test")}
        Zg = {k: pca["hg"][k][:, :dk] for k in ("train", "val", "test")}
        KR = {k: (Zi[k][:, :, None] * Zg[k][:, None, :]
                  ).reshape(len(Zi[k]), -1).astype(np.float32) for k in Zi}
        # the main effect of h_g cancels in pairwise differences, so only h_i and the interaction are included
        rn = P._Runner({"hi": Zi["train"], "kr": KR["train"]}, y["train"],
                       sid["train"], {"hi": Zi["val"], "kr": KR["val"]},
                       y["val"], sid["val"],
                       {"hi": Zi["test"], "kr": KR["test"]},
                       loss_kind="pairwise")
        res = rn.run(lambda: P.MLPProbe([dk, dk * dk], (), keys=("hi", "kr")),
                     sel, seeds=seeds, epochs=epochs)
        if res is not None:
            R[f"P7k_pair_kron_d{dk}"] = res
        rn2 = P._Runner({"hi": Zi["train"]}, y["train"], sid["train"],
                        {"hi": Zi["val"]}, y["val"], sid["val"],
                        {"hi": Zi["test"]}, loss_kind="pairwise")
        res = rn2.run(lambda: P.MLPProbe([dk], (), keys=("hi",)), sel,
                      seeds=seeds, epochs=epochs)
        if res is not None:
            R[f"P7k_pair_hi_d{dk}"] = res


    if "P7" in which:
        # score_i = w1.h_i + w3.(h_i * h_g): the h_g-only term cancels in pairwise differences, so it is omitted
        inter = {k: pca["hi"][k] * pca["hg"][k] for k in ("train", "val", "test")}
        for nm, keys, dims in (("P7_pair_hi", ("hi",), [d_i]),
                               ("P7_pair_hi_x_hg", ("hi", "xg"), [d_i, d_g])):
            feats_tr = {"hi": pca["hi"]["train"], "xg": inter["train"]}
            feats_va = {"hi": pca["hi"]["val"], "xg": inter["val"]}
            feats_te = {"hi": pca["hi"]["test"], "xg": inter["test"]}
            rn = P._Runner(feats_tr, y["train"], sid["train"], feats_va,
                           y["val"], sid["val"], feats_te,
                           loss_kind="pairwise")
            res = rn.run(lambda keys=keys, dims=dims:
                         P.MLPProbe(dims, (), keys=keys), sel,
                         seeds=seeds, epochs=epochs)
            if res is not None:
                R[nm] = res

    # ---------------- P8 pairwise MLP (Siamese) ----------------
    if "P8" in which:
        for nm, keys, dims in (
                ("P8_ranknet_cand", ("hi",), [d_i]),
                ("P8_ranknet_cand_state", ("hi", "hg"), [d_i, d_g]),
                ("P8_ranknet_cheap", ("c",), [d_c]),
                ("P8_ranknet_cheap_hidden", ("c", "hi", "hg"),
                 [d_c, d_i, d_g])):
            rn = P._Runner({k: F[k]["train"] for k in F}, y["train"],
                           sid["train"], {k: F[k]["val"] for k in F}, y["val"],
                           sid["val"], {k: F[k]["test"] for k in F},
                           loss_kind="pairwise")
            res = rn.run(lambda keys=keys, dims=dims:
                         P.MLPProbe(dims, (256,), keys=keys), sel,
                         seeds=seeds, epochs=epochs)
            if res is not None:
                R[nm] = res

    # ---------------- P9 listwise ----------------
    if "P9" in which:
        for nm, keys, dims in (("P9_list_hidden", ("c", "hi", "hg"),
                                [d_c, d_i, d_g]),
                               ("P9_list_cheap", ("c",), [d_c])):
            rn = P._Runner({k: F[k]["train"] for k in F}, y["train"],
                           sid["train"], {k: F[k]["val"] for k in F}, y["val"],
                           sid["val"], {k: F[k]["test"] for k in F},
                           loss_kind="listwise")
            res = rn.run(lambda keys=keys, dims=dims:
                         P.MLPProbe(dims, (128,), keys=keys), sel,
                         seeds=seeds, epochs=epochs)
            if res is not None:
                R[nm] = res

    # ---------------- P13 PLS / CCA ----------------
    if "P13" in which:
        try:
            from sklearn.cross_decomposition import PLSRegression
            X = {k: np.concatenate([pca["hi"][k], pca["hg"][k]], 1)
                 for k in ("train", "val", "test")}
            bb = None
            for nc in (2, 4, 8, 16, 32):
                m = PLSRegression(n_components=nc, scale=False).fit(
                    X["train"], y["train"])
                pv = m.predict(X["val"]).ravel()
                s = P._safe(sel(y["val"], pv))
                if bb is None or s > bb["val_score"]:
                    bb = {"val_score": s, "pred_val": pv,
                          "pred_test": m.predict(X["test"]).ravel(),
                          "hp": {"n_components": nc}, "n_params": -1}
            R["P13_pls"] = bb
        except Exception as e:                                # noqa: BLE001
            R["P13_error"] = {"error": str(e)}

    return R


# ------------------------------------------------------------------ scoring -----
def score_all(R, y_test, sid_test, base_key="P0_cheap", ceiling=None,
              within_ceiling=None):
    _, groups = M.group_slices(sid_test)
    base = R.get(base_key, {}).get("pred_test")
    out = {}
    for k, v in R.items():
        if k.startswith("_") or not isinstance(v, dict) or "pred_test" not in v:
            continue
        yy = y_test
        if v.get("residual_target"):
            continue          # residual probes are scored separately
        out[k] = M.full_report(yy, v["pred_test"], sid_test, groups,
                               p_base=base, ceiling=ceiling,
                               within_ceiling=within_ceiling)
        out[k]["val_score"] = v.get("val_score")
        out[k]["hp"] = v.get("hp")
        out[k]["n_params"] = v.get("n_params")
        if v.get("centered_target"):
            # the target is within-state-centred y: pooled metrics (r2/spearman/
            # mae/rmse) are meaningless, ranking metrics (within_r2/concordance/top1/
            # regret) remain fully valid -- they only look at relative order within a state.
            out[k]["centered_target"] = True
            for bad in ("r2", "spearman", "mae", "rmse", "partial_r2",
                        "r2_ceiling_norm"):
                out[k].pop(bad, None)
    return out

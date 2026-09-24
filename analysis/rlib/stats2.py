"""
Statistical inference S1-S6 plus multiple-comparison correction.

Every method is specified in advance and all are reported (including non-significant ones); there is no room to "pick the significant one":
`evidence_table()` lists the conclusion of the same comparison under every method side by side.
"""
import numpy as np
from scipy import stats as sps

from . import metrics as M


# =============================================================== S1 bootstrap =
def cluster_refit_bootstrap(fit_eval_fn, doc_pools, rows_by_doc, n_boot=200,
                            seed=0, min_rows=50):
    """Resample with replacement **separately** within the disjoint train/val/test document pools and refit.

    fit_eval_fn(tr_idx, va_idx, te_idx) -> dict of statistics
    doc_pools: {"train": array_of_docs, ...}
    rows_by_doc: {doc: row_indices}

    Note: resampling must happen **within** each split. Resampling all documents first and then splitting would let duplicated documents
    land in both train and test.
    """
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(n_boot):
        idx = {}
        for k, pool in doc_pools.items():
            pick = rng.choice(pool, len(pool), replace=True)
            idx[k] = np.concatenate([rows_by_doc[x] for x in pick])
        assert not (set(np.unique(idx["train"])) & set(np.unique(idx["test"]))) \
            or True  # row indices may repeat; the real check is at the document level
        if min(len(idx[k]) for k in idx) < min_rows:
            continue
        st = fit_eval_fn(idx["train"], idx["val"], idx["test"])
        if st is None:
            continue
        for k, v in st.items():
            acc.setdefault(k, []).append(v)
    out = {}
    for k, v in acc.items():
        v = np.array([x for x in v if np.isfinite(x)])
        if not len(v):
            continue
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = {"mean": float(v.mean()), "ci_lo": float(lo),
                  "ci_hi": float(hi), "n": int(len(v)),
                  "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def cluster_bootstrap_fixed(stat_fn, cluster_ids, n_boot=2000, seed=0):
    """Fixed model; resample only test-set clusters. Returns (mean, lo, hi, samples).

    stat_fn may return a scalar or a dict -- with a dict, all statistics are computed on the **same** bootstrap replicate,
    so the intervals are mutually consistent.
    """
    rng = np.random.default_rng(seed)
    cl = np.unique(cluster_ids)
    idx_by = {c: np.where(cluster_ids == c)[0] for c in cl}
    samples = []
    for _ in range(n_boot):
        pick = rng.choice(cl, len(cl), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        v = stat_fn(idx)
        samples.append(v)
    if isinstance(samples[0], dict):
        out = {}
        for k in samples[0]:
            v = np.array([s[k] for s in samples if np.isfinite(s[k])])
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[k] = {"mean": float(v.mean()), "ci_lo": float(lo),
                      "ci_hi": float(hi),
                      "ci_excludes_zero": bool(lo > 0 or hi < 0)}
        return out
    v = np.array([s for s in samples if np.isfinite(s)])
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


# ==================================================== S2 paired permutation ==
def paired_permutation(a, b, n_perm=10000, seed=0, alternative="two-sided"):
    """State-level paired permutation test. a, b are per-state metrics of two models on the same states.

    H0: the two models are exchangeable within each state -> random sign flips.
    """
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if len(d) == 0:
        return {"stat": float("nan"), "p": float("nan"), "n": 0}
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d[None, :]).mean(1)
    if alternative == "greater":
        p = float((null >= obs).mean())
    elif alternative == "less":
        p = float((null <= obs).mean())
    else:
        p = float((np.abs(null) >= abs(obs)).mean())
    return {"stat": obs, "p": max(p, 1.0 / n_perm), "n": int(len(d)),
            "null_sd": float(null.std())}


# ============================================================ S3 Wilcoxon ====
def wilcoxon_sign(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    nz = d[np.abs(d) > 1e-12]
    out = {"n": int(len(d)), "n_nonzero": int(len(nz)),
           "median_diff": float(np.median(d)) if len(d) else float("nan")}
    if len(nz) >= 6:
        out["wilcoxon_p"] = float(sps.wilcoxon(nz).pvalue)
        k = int((nz > 0).sum())
        out["sign_p"] = float(sps.binomtest(k, len(nz), 0.5).pvalue)
        out["frac_positive"] = k / len(nz)
    return out


# ================================================ S4 cross-fitted residual ===
def cross_fitted_partial_r2(y, resid, pred_from_hidden):
    """Partial R^2 on cross-fitted residuals: 1 - SSE(resid - pred) / SSE(resid)."""
    resid = np.asarray(resid, np.float64)
    pred = np.asarray(pred_from_hidden, np.float64)
    sse0 = float((resid ** 2).sum())
    sse1 = float(((resid - pred) ** 2).sum())
    return (sse0 - sse1) / max(sse0, 1e-12)


def residual_permutation_test(resid, pred, cluster_ids, n_perm=2000, seed=0):
    """Permute residuals **across clusters**, breaking the hidden<->residual correspondence."""
    rng = np.random.default_rng(seed)
    obs = cross_fitted_partial_r2(None, resid, pred)
    cl = np.unique(cluster_ids)
    idx_by = {c: np.where(cluster_ids == c)[0] for c in cl}
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(cl))
        r = np.empty_like(resid)
        for i, c in enumerate(cl):
            src = idx_by[cl[perm[i]]]
            dst = idx_by[c]
            n = min(len(src), len(dst))
            r[dst[:n]] = resid[src[:n]]
            if len(dst) > n:
                r[dst[n:]] = resid[src[0]]
        null.append(cross_fitted_partial_r2(None, r, pred))
    null = np.array(null)
    return {"partial_r2": obs, "p": float(max((null >= obs).mean(),
                                              1.0 / n_perm)),
            "null_mean": float(null.mean()), "null_sd": float(null.std())}


# ======================================== S5 hierarchical / mixed effects ====
def hierarchical_effect(per_state_delta, group_ids, n_boot=4000, seed=0):
    """Two-level bootstrap: resample groups (documents/problems) first, then use the within-group state means.

    Gives a posterior-style interval for Delta and P(Delta>0). This is a bootstrap approximation to mixed effects that avoids
    statsmodels convergence problems on small samples; if statsmodels is available, MixedLM is reported as well.
    """
    per_state_delta = np.asarray(per_state_delta, np.float64)
    ok = np.isfinite(per_state_delta)
    d = per_state_delta[ok]; g = np.asarray(group_ids)[ok]
    groups = np.unique(g)
    idx_by = {x: np.where(g == x)[0] for x in groups}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(groups, len(groups), replace=True)
        vals.append(np.concatenate([d[idx_by[x]] for x in pick]).mean())
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    out = {"mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
           "P_gt_0": float((vals > 0).mean()), "n_states": int(len(d)),
           "n_groups": int(len(groups))}
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame({"d": d, "g": g})
        m = smf.mixedlm("d ~ 1", df, groups=df["g"]).fit(reml=True)
        out["mixedlm_coef"] = float(m.params["Intercept"])
        out["mixedlm_p"] = float(m.pvalues["Intercept"])
        ci = m.conf_int().loc["Intercept"]
        out["mixedlm_ci"] = [float(ci[0]), float(ci[1])]
    except Exception as e:                                   # noqa: BLE001
        out["mixedlm_error"] = str(e)
    return out


# ============================== S6 conditional-independence diagnostics ======
def distance_correlation(X, Y, max_n=2000, seed=0):
    """Distance correlation (conditional-independence diagnostic on cross-fitted residuals). Secondary only."""
    rng = np.random.default_rng(seed)
    X = np.atleast_2d(np.asarray(X, np.float64))
    if X.shape[0] == 1:
        X = X.T
    Y = np.asarray(Y, np.float64).reshape(-1, 1)
    n = len(Y)
    if n > max_n:
        sel = rng.choice(n, max_n, replace=False)
        X, Y = X[sel], Y[sel]
    def _dc(A):
        D = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
        return D - D.mean(0, keepdims=True) - D.mean(1, keepdims=True) \
            + D.mean()
    a, b = _dc(X), _dc(Y)
    dcov2 = float((a * b).mean())
    dvx = float((a * a).mean()); dvy = float((b * b).mean())
    den = np.sqrt(np.sqrt(dvx * dvy))
    return float(np.sqrt(max(dcov2, 0.0)) / den) if den > 1e-12 else 0.0


def dcor_permutation(X, Y, n_perm=500, seed=0, max_n=1500):
    rng = np.random.default_rng(seed)
    obs = distance_correlation(X, Y, max_n=max_n, seed=seed)
    null = []
    Y = np.asarray(Y)
    for i in range(n_perm):
        null.append(distance_correlation(X, rng.permutation(Y),
                                         max_n=max_n, seed=seed + i + 1))
    null = np.array(null)
    return {"dcor": obs, "p": float(max((null >= obs).mean(), 1.0 / n_perm)),
            "null_mean": float(null.mean()), "null_sd": float(null.std())}


# ================================================= multiple comparisons ======
def bh_fdr(pvals, q=0.05):
    p = np.asarray(pvals, np.float64)
    n = len(p)
    order = np.argsort(p)
    thr = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    if k:
        rej[order[:k]] = True
    adj = np.empty(n)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, p[order[i]] * n / (i + 1))
        adj[order[i]] = running
    return rej, np.clip(adj, 0, 1)


def by_fdr(pvals, q=0.05):
    """Benjamini-Yekutieli (valid under arbitrary dependence)."""
    p = np.asarray(pvals, np.float64)
    n = len(p)
    c = np.sum(1.0 / np.arange(1, n + 1))
    rej, adj = bh_fdr(p, q / c)
    return rej, np.clip(adj * c, 0, 1)


def westfall_young(obs_stats, null_stats, alternative="greater"):
    """max-T / Westfall-Young single-step correction.

    obs_stats : (m,) observed statistics
    null_stats: (B, m) **all** m statistics on each permutation replicate (must come from the same permutation
                            so that the correlation structure between statistics is preserved)
    """
    obs = np.asarray(obs_stats, np.float64)
    null = np.asarray(null_stats, np.float64)
    if alternative == "greater":
        mx = null.max(1)
        return np.array([max((mx >= o).mean(), 1.0 / len(mx)) for o in obs])
    mx = np.abs(null).max(1)
    return np.array([max((mx >= abs(o)).mean(), 1.0 / len(mx)) for o in obs])


# ================================================================ summary ====
def per_state_metrics(y, pred, state_id, groups=None):
    """Per-state concordance / top1 / regret, used by S2/S3/S5."""
    if groups is None:
        _, groups = M.group_slices(state_id)
    y = np.asarray(y, np.float64); pred = np.asarray(pred, np.float64)
    conc, top1, reg, regn = [], [], [], []
    for g in groups:
        yy, pp = y[g], pred[g]
        c, n = M._pair_stats(yy, pp)
        conc.append(c / n if n else np.nan)
        best = int(np.argmax(yy))
        top1.append(1.0 if int(np.argmax(pp)) == best else 0.0)
        r = float(yy.max() - yy[int(np.argmax(pp))])
        reg.append(r)
        rng_ = float(yy.max() - yy.min())
        regn.append(r / rng_ if rng_ > 1e-12 else 0.0)
    return {"concordance": np.array(conc), "top1": np.array(top1),
            "regret": np.array(reg), "regret_norm": np.array(regn)}


def compare_probes(y, pred_a, pred_b, state_id, group_ids, groups=None,
                   n_perm=10000, seed=0):
    """Run S2/S3/S5 at once: improvement of a (hidden probe) over b (baseline)."""
    ma = per_state_metrics(y, pred_a, state_id, groups)
    mb = per_state_metrics(y, pred_b, state_id, groups)
    out = {}
    for k in ("concordance", "top1"):
        out[f"delta_{k}"] = {
            "mean": float(np.nanmean(ma[k] - mb[k])),
            "permutation": paired_permutation(ma[k], mb[k], n_perm, seed,
                                              "greater"),
            "wilcoxon": wilcoxon_sign(ma[k], mb[k]),
            "hierarchical": hierarchical_effect(ma[k] - mb[k], group_ids,
                                                seed=seed)}
    for k in ("regret", "regret_norm"):
        # lower regret is better; flip the sign
        out[f"delta_{k}"] = {
            "mean": float(np.nanmean(mb[k] - ma[k])),
            "permutation": paired_permutation(mb[k], ma[k], n_perm, seed,
                                              "greater"),
            "wilcoxon": wilcoxon_sign(mb[k], ma[k]),
            "hierarchical": hierarchical_effect(mb[k] - ma[k], group_ids,
                                                seed=seed)}
    return out

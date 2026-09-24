"""Several estimators of the same quantity, so no single one decides anything.

WHY MORE THAN ONE
Each method fails differently. Cluster bootstrap trusts the resampling
distribution; BCa corrects its skew and bias; the jackknife is a different
resampling scheme; the permutation test builds an exact null by construction;
the sign and signed-rank tests assume almost nothing. When they agree the
conclusion is not an artefact of any one of them.

THE TRAP IN "JUST RUN MORE SEEDS"
Split seeds re-split the SAME data. Their train sets overlap heavily, so the
per-seed results are positively correlated. Treating them as independent draws
and running an ordinary t-test makes the interval shrink like 1/sqrt(J) when
the real information does not grow at all -- more seeds then manufacture
significance out of nothing. Nadeau & Bengio (2003) give the correction: the
variance of the mean is inflated by (1/J + n_test/n_train) instead of 1/J.
Both versions are reported so the gap is visible.
"""
import numpy as np
from scipy import stats as sps


def naive_t(deltas, alpha=0.05):
    d = np.asarray(deltas, float); J = len(d)
    se = d.std(ddof=1) / np.sqrt(J)
    t = sps.t.ppf(1 - alpha / 2, J - 1)
    return {"mean": d.mean(), "ci": (d.mean() - t * se, d.mean() + t * se),
            "se": se, "p": float(sps.ttest_1samp(d, 0).pvalue)}


def nadeau_bengio_t(deltas, n_train, n_test, alpha=0.05):
    """Corrected resampled t-test for REPEATED SPLITS OF THE SAME DATA."""
    d = np.asarray(deltas, float); J = len(d)
    var = d.var(ddof=1) * (1.0 / J + float(n_test) / float(n_train))
    se = np.sqrt(max(var, 1e-300))
    t = sps.t.ppf(1 - alpha / 2, J - 1)
    stat = d.mean() / se
    return {"mean": d.mean(), "ci": (d.mean() - t * se, d.mean() + t * se),
            "se": se, "p": float(2 * (1 - sps.t.cdf(abs(stat), J - 1))),
            "inflation_vs_naive": float(np.sqrt((1 / J + n_test / n_train) / (1 / J)))}


def cluster_boot_percentile(stat_fn, groups, n_boot=4000, seed=0):
    u = np.unique(groups); rb = np.random.default_rng(seed); out = []
    for _ in range(n_boot):
        pick = rb.choice(u, len(u), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        out.append(stat_fn(idx))
    out = np.asarray(out, float)
    return {"ci": tuple(np.nanpercentile(out, [2.5, 97.5])),
            "mean": float(np.nanmean(out)), "draws": out}


def bca(stat_fn, groups, theta_hat, draws, alpha=0.05):
    """Bias-corrected and accelerated interval (corrects skew of the bootstrap)."""
    draws = np.asarray(draws, float); draws = draws[np.isfinite(draws)]
    z0 = sps.norm.ppf(np.clip((draws < theta_hat).mean(), 1e-6, 1 - 1e-6))
    u = np.unique(groups)
    jack = np.array([stat_fn(np.where(groups != g)[0]) for g in u], float)
    jm = np.nanmean(jack)
    num = np.nansum((jm - jack) ** 3); den = 6 * (np.nansum((jm - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = sps.norm.ppf(q)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(np.nanpercentile(draws, 100 * sps.norm.cdf(adj)))
    return {"ci": tuple(out), "z0": float(z0), "a": float(a),
            "jackknife_mean": float(jm),
            "jackknife_se": float(np.sqrt((len(u) - 1) / len(u)
                                          * np.nansum((jm - jack) ** 2)))}


def sign_and_rank(deltas):
    d = np.asarray(deltas, float); d = d[np.isfinite(d)]
    npos = int((d > 0).sum()); n = int((d != 0).sum())
    sign_p = float(sps.binomtest(npos, n, 0.5).pvalue) if n else float("nan")
    try:
        w = float(sps.wilcoxon(d).pvalue)
    except Exception:
        w = float("nan")
    return {"n_positive": npos, "n": n, "sign_test_p": sign_p,
            "wilcoxon_p": w, "median": float(np.median(d))}


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg: control the false discovery rate across probes."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    keep = np.zeros(m, bool)
    if k:
        keep[order[:k]] = True
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(adj, 0, 1)
    return keep, out

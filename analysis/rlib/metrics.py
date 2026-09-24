"""
Full ranking / selection / decision / regression metric suite.

Design constraints:
    * all within-state metrics aggregate per state; the number of candidates per state may vary.
    * ties are handled as before (tied predictions count 0.5) so numbers are directly comparable.
    * every metric also returns per-state details for paired permutation / Wilcoxon tests.
"""
import numpy as np
from scipy import stats as sps


# ------------------------------------------------------------------ helpers --
def group_slices(state_id):
    uniq, inv = np.unique(state_id, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sinv = inv[order]
    bounds = np.flatnonzero(np.diff(sinv)) + 1
    return uniq, np.split(order, bounds)


# --------------------------------------------------------------- regression --
def r2_score(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def within_state_r2(y, p, state_id, groups=None):
    if groups is None:
        _, groups = group_slices(state_id)
    yc = np.asarray(y, np.float64).copy()
    pc = np.asarray(p, np.float64).copy()
    for g in groups:
        yc[g] -= yc[g].mean()
        pc[g] -= pc[g].mean()
    ss_tot = float((yc ** 2).sum())
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - float(((yc - pc) ** 2).sum()) / ss_tot


def mae(y, p):
    return float(np.abs(np.asarray(y) - np.asarray(p)).mean())


def rmse(y, p):
    return float(np.sqrt(((np.asarray(y) - np.asarray(p)) ** 2).mean()))


def partial_r2(y, p_full, p_base):
    """Partial R^2 relative to a baseline model: (SSE_base - SSE_full) / SSE_base."""
    y = np.asarray(y, np.float64)
    sse_b = float(((y - p_base) ** 2).sum())
    sse_f = float(((y - p_full) ** 2).sum())
    return (sse_b - sse_f) / max(sse_b, 1e-12)


# ------------------------------------------------------------------ ranking --
def _pair_stats(yy, pp):
    """Pairwise statistics within one state. Returns (n_concordant_weighted, n_pairs)."""
    n = len(yy)
    if n < 2:
        return 0.0, 0
    dy = yy[:, None] - yy[None, :]
    dp = pp[:, None] - pp[None, :]
    iu = np.triu_indices(n, 1)
    dy, dp = dy[iu], dp[iu]
    ok = np.abs(dy) > 1e-9
    dy, dp = dy[ok], dp[ok]
    if len(dy) == 0:
        return 0.0, 0
    ties = np.abs(dp) <= 1e-12
    conc = float((np.sign(dy[~ties]) == np.sign(dp[~ties])).sum()
                 + 0.5 * ties.sum())
    return conc, int(len(dy))


def concordance(y, p, state_id, groups=None, per_state=False):
    if groups is None:
        _, groups = group_slices(state_id)
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    tot_c = tot_n = 0
    per = []
    for g in groups:
        c, n = _pair_stats(y[g], p[g])
        tot_c += c; tot_n += n
        per.append((c / n) if n else np.nan)
    val = tot_c / tot_n if tot_n else float("nan")
    return (val, np.array(per, dtype=np.float64)) if per_state else val


def pairwise_auc(y, p, state_id, groups=None):
    """Same as concordance (pairwise AUC), but averaged with equal weight per state rather than per pair."""
    v, per = concordance(y, p, state_id, groups, per_state=True)
    per = per[np.isfinite(per)]
    return float(per.mean()) if len(per) else float("nan")


def kendall_tau(y, p, state_id, groups=None):
    if groups is None:
        _, groups = group_slices(state_id)
    vals = []
    for g in groups:
        if len(g) < 3:
            continue
        t = sps.kendalltau(y[g], p[g]).statistic
        if np.isfinite(t):
            vals.append(t)
    return float(np.mean(vals)) if vals else float("nan")


def within_spearman(y, p, state_id, groups=None):
    if groups is None:
        _, groups = group_slices(state_id)
    vals = []
    for g in groups:
        if len(g) < 3 or np.std(p[g]) < 1e-12:
            continue
        r = sps.spearmanr(y[g], p[g]).statistic
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


def pooled_spearman(y, p):
    if np.std(p) < 1e-12:
        return 0.0
    return float(sps.spearmanr(y, p).statistic)


# ---------------------------------------------------------------- selection --
def topk_metrics(y, p, state_id, groups=None, ks=(1, 2, 3)):
    """Oracle-best top-1 accuracy and top-k recall (whether the truly best candidate is hit)."""
    if groups is None:
        _, groups = group_slices(state_id)
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    hit = {k: [] for k in ks}
    for g in groups:
        if len(g) < 2:
            continue
        yy, pp = y[g], p[g]
        best = int(np.argmax(yy))
        order = np.argsort(-pp, kind="stable")
        for k in ks:
            hit[k].append(1.0 if best in order[:k] else 0.0)
    return {f"top{k}": float(np.mean(hit[k])) if hit[k] else float("nan")
            for k in ks}


def regret(y, p, state_id, groups=None, per_state=False):
    """Regret = A_oracle_best - A_selected (the candidate with the argmax prediction is selected).

    normalized: divide by (A_max - A_min), lies in [0,1].
    """
    if groups is None:
        _, groups = group_slices(state_id)
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    raw, norm = [], []
    for g in groups:
        if len(g) < 2:
            continue
        yy, pp = y[g], p[g]
        r = float(yy.max() - yy[int(np.argmax(pp))])
        raw.append(r)
        rng = float(yy.max() - yy.min())
        norm.append(r / rng if rng > 1e-12 else 0.0)
    raw = np.array(raw); norm = np.array(norm)
    out = {"regret_mean": float(raw.mean()) if len(raw) else float("nan"),
           "regret_median": float(np.median(raw)) if len(raw) else float("nan"),
           "regret_norm_mean": float(norm.mean()) if len(norm) else float("nan"),
           "regret_worst_quartile":
               float(raw[raw >= np.quantile(raw, 0.75)].mean())
               if len(raw) else float("nan")}
    return (out, raw, norm) if per_state else out


# ------------------------------------------------------------------ bundles --
def full_report(y, p, state_id, groups=None, p_base=None, ceiling=None,
                within_ceiling=None):
    """Compute all metrics at once."""
    if groups is None:
        _, groups = group_slices(state_id)
    out = {
        "r2": r2_score(y, p),
        "within_r2": within_state_r2(y, p, state_id, groups),
        "mae": mae(y, p), "rmse": rmse(y, p),
        "spearman": pooled_spearman(y, p),
        "concordance": concordance(y, p, state_id, groups),
        "pairwise_auc": pairwise_auc(y, p, state_id, groups),
        "kendall_tau": kendall_tau(y, p, state_id, groups),
        "within_spearman": within_spearman(y, p, state_id, groups),
    }
    out.update(topk_metrics(y, p, state_id, groups))
    out.update(regret(y, p, state_id, groups))
    if p_base is not None:
        out["partial_r2"] = partial_r2(y, p_full=p, p_base=p_base)
    if ceiling is not None and ceiling > 1e-9:
        out["r2_ceiling_norm"] = out["r2"] / ceiling
    if within_ceiling is not None and within_ceiling > 1e-9:
        out["within_r2_ceiling_norm"] = out["within_r2"] / within_ceiling
    return out


PRIMARY_KEYS = ("concordance", "within_r2", "top1", "regret_norm_mean", "r2")

"""
Linear probes and evaluation for Experiment 1 / 2.

PRIMARY probe is LINEAR:
    A_hat(i | s_t) = w^T [h_{i,t} ; h_global,t] + b
A <=2-layer MLP is fitted only as a SECONDARY nonlinear-headroom control. If
only the MLP works, we do not claim the value is linearly represented.

Everything is evaluated out of sample with DOCUMENT-level splits, so no two
states from the same generation can straddle a split boundary. Confidence
intervals come from a CLUSTER bootstrap over documents, because candidates
within a state -- and states within a trajectory -- are strongly dependent.
"""
import os

import numpy as np
from scipy import stats


_SOLVER_DEV = "unset"


def _solver_device():
    """Use a GPU for the ridge normal equations when one is usable.

    The label-collection jobs saturate the CPU, so the CPU BLAS path is an
    order of magnitude slower than it should be. Falls back to numpy silently.
    """
    global _SOLVER_DEV
    if _SOLVER_DEV == "unset":
        _SOLVER_DEV = None
        if os.environ.get("PROBE_SOLVER", "gpu") == "gpu":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.zeros(8, device="cuda")
                    _SOLVER_DEV = "cuda"
            except Exception:
                _SOLVER_DEV = None
    return _SOLVER_DEV


# ------------------------------------------------------------------ ridge ---
def _standardize(Xtr, *others):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return [(X - mu) / sd for X in (Xtr,) + others]


def ridge_path(Xtr, ytr, alphas):
    """Closed-form ridge for many alphas at once.

    Uses the eigendecomposition of the d x d Gram matrix X^T X rather than the
    SVD of X. Both are O(n d^2), but the Gram route is roughly an order of
    magnitude faster in practice at our shapes (n ~ 9k, d ~ 1.6k), which matters
    because the layer sweep fits 13 x 3 x 2 probes.

        w(a) = (X^T X + a I)^-1 X^T y = V diag(1/(lam + a)) V^T (X^T y)
    """
    ym = ytr.mean()
    yc = (ytr - ym).astype(np.float32)
    X32 = Xtr.astype(np.float32, copy=False)
    dev = _solver_device()
    if dev is not None:
        import torch
        Xg = torch.as_tensor(X32, device=dev)
        yg = torch.as_tensor(yc, device=dev)
        G = (Xg.T @ Xg).double()
        b = (Xg.T @ yg).double()
        lam, V = torch.linalg.eigh(G)
        lam = lam.clamp(min=0.0)
        Vtb = V.T @ b
        return [((V @ (Vtb / (lam + float(a)))).cpu().numpy(), ym)
                for a in alphas]
    G = (X32.T @ X32).astype(np.float64)
    b = (X32.T @ yc).astype(np.float64)
    lam, V = np.linalg.eigh(G)
    lam = np.clip(lam, 0.0, None)
    Vtb = V.T @ b
    return [(V @ (Vtb / (lam + a)), ym) for a in alphas]


def fit_linear(Xtr, ytr, Xva, yva, alphas=None):
    """Ridge with alpha chosen on the validation split by R^2."""
    if alphas is None:
        # must reach into the heavily-regularised regime: the hidden
        # blocks are 1536-dimensional
        alphas = np.logspace(-2, 9, 34)
    Xtr_s, Xva_s = _standardize(Xtr, Xva)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    best, best_r2, best_a = None, -np.inf, None
    for (w, b), a in zip(ridge_path(Xtr_s, ytr, alphas), alphas):
        pred = Xva_s @ w + b
        r2 = r2_score(yva, pred)
        if r2 > best_r2:
            best, best_r2, best_a = (w, b), r2, a
    return {"w": best[0], "b": best[1], "mu": mu, "sd": sd, "alpha": best_a,
            "val_r2": best_r2}


def predict(model, X):
    return ((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"]


def fit_linear_2block(Xc_tr, Xh_tr, ytr, Xc_va, Xh_va, yva,
                      alphas=None, gammas=None):
    """Ridge over TWO blocks with SEPARATE regularisation strengths.

    A single shared alpha cannot regularise a ~100-d control block and a
    1536-d hidden block at the same time. Measured on synthetic data where the
    hidden block is pure noise, the shared-alpha probe LOSES 0.06 R^2 purely
    from the added dimensionality -- so `Delta_R2` would be biased negative and
    the test would be under-powered rather than conservative-but-fair.

    Fitting  min ||y - Xc wc - Xh wh||^2 + a||wc||^2 + (a/g^2)||wh||^2
    is the same as a single-alpha ridge on [Xc, g*Xh], so we sweep g on the
    validation split alongside alpha. The grid INCLUDES g = 0, which exactly
    reproduces the controls-only fit; the augmented model therefore nests the
    baseline and `Delta_R2` cannot be negative except through validation noise.
    """
    if gammas is None:
        gammas = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    muc, sdc = Xc_tr.mean(0, keepdims=True), Xc_tr.std(0, keepdims=True)
    muh, sdh = Xh_tr.mean(0, keepdims=True), Xh_tr.std(0, keepdims=True)
    sdc[sdc < 1e-8] = 1.0
    sdh[sdh < 1e-8] = 1.0
    Ct, Ht = (Xc_tr - muc) / sdc, (Xh_tr - muh) / sdh
    Cv, Hv = (Xc_va - muc) / sdc, (Xh_va - muh) / sdh
    if alphas is None:
        alphas = np.logspace(-2, 9, 34)

    best = None
    for g in gammas:
        Xt = np.concatenate([Ct, g * Ht], 1) if g > 0 else Ct
        Xv = np.concatenate([Cv, g * Hv], 1) if g > 0 else Cv
        for (w, b), a in zip(ridge_path(Xt, ytr, alphas), alphas):
            r2 = r2_score(yva, Xv @ w + b)
            if best is None or r2 > best["val_r2"]:
                best = {"w": w, "b": b, "alpha": a, "gamma": g, "val_r2": r2}
    d = Xc_tr.shape[1]
    return {"muc": muc, "sdc": sdc, "muh": muh, "sdh": sdh, "d_c": d, **best}


def predict_2block(m, Xc, Xh):
    C = (Xc - m["muc"]) / m["sdc"]
    if m["gamma"] <= 0:
        return C @ m["w"] + m["b"]
    H = m["gamma"] * (Xh - m["muh"]) / m["sdh"]
    return np.concatenate([C, H], 1) @ m["w"] + m["b"]


# ------------------------------------------------------------------- MLP ----
def fit_mlp(Xtr, ytr, Xva, yva, seed=0, grid=None):
    """SECONDARY nonlinear-headroom control: projection + 2-layer MLP.

    This must be a CREDIBLE upper bound, otherwise a linear-probe null is open
    to the obvious objection that the probe was simply too weak. The first
    version was not credible: the targets have sd ~0.06, so the MSE is ~4e-3 and
    a weight decay of 1e-2 dominated the gradient and shrank the network toward
    the mean -- it scored BELOW the linear controls, which is diagnostic of an
    undertrained control rather than of absent information.

    Fixed by standardising the target, sweeping a small hyper-parameter grid,
    and selecting on validation.
    """
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    ym, ys = float(ytr.mean()), float(ytr.std()) or 1.0
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    Yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=dev)
    Xv = torch.tensor((Xva - mu) / sd, dtype=torch.float32, device=dev)
    d = Xt.shape[1]
    n = Xt.shape[0]
    if grid is None:
        grid = [(lr, wd, h) for lr in (3e-3, 1e-3, 3e-4)
                for wd in (1e-4, 1e-2) for h in (256, 512)]
    best = None
    for (lr, wd, hid) in grid:
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(d, min(512, d)), nn.GELU(),
                            nn.Linear(min(512, d), hid), nn.GELU(),
                            nn.Linear(hid, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        g = torch.Generator(device="cpu").manual_seed(seed)
        bad = 0
        for ep in range(80):
            net.train()
            perm = torch.randperm(n, generator=g).to(dev)
            for a_ in range(0, n, 256):
                idx = perm[a_:a_ + 256]
                opt.zero_grad()
                ((net(Xt[idx]).squeeze(-1) - Yt[idx]) ** 2).mean().backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(Xv).squeeze(-1).cpu().numpy() * ys + ym
            r2 = r2_score(yva, pv)
            if best is None or r2 > best["val_r2"]:
                best = {"state": {k: v.detach().clone()
                                  for k, v in net.state_dict().items()},
                        "val_r2": r2, "hid": hid, "lr": lr, "wd": wd}
                bad = 0
            else:
                bad += 1
                if bad > 25:
                    break
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(d, min(512, d)), nn.GELU(),
                        nn.Linear(min(512, d), best["hid"]), nn.GELU(),
                        nn.Linear(best["hid"], 1)).to(dev)
    net.load_state_dict(best["state"])
    net.eval()
    return {"net": net, "mu": mu, "sd": sd, "ym": ym, "ys": ys, "dev": dev,
            "val_r2": best["val_r2"], "hp": (best["lr"], best["wd"], best["hid"])}


def predict_mlp(model, X):
    import torch
    out = []
    with torch.no_grad():
        for a in range(0, len(X), 8192):
            Xt = torch.tensor((X[a:a + 8192] - model["mu"]) / model["sd"],
                              dtype=torch.float32, device=model["dev"])
            out.append(model["net"](Xt).squeeze(-1).cpu().numpy())
    return np.concatenate(out) * model["ys"] + model["ym"]


# ----------------------------------------------------------------- metrics --
def r2_score(y, pred):
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def spearman(y, pred):
    if np.std(pred) < 1e-12:
        return 0.0
    return float(stats.spearmanr(y, pred).statistic)


def within_state_r2(y, pred, state_id):
    """R^2 after removing each state's mean from BOTH label and prediction.

    This isolates the candidate-level signal -- "which of these positions is
    the better commit right now" -- from between-state variation. h_global is
    constant within a state, so it contributes exactly nothing here; only
    per-candidate structure can score. This is the decision a scheduler faces.
    """
    yc = y.astype(np.float64).copy()
    pc = pred.astype(np.float64).copy()
    for s in np.unique(state_id):
        m = state_id == s
        yc[m] -= yc[m].mean()
        pc[m] -= pc[m].mean()
    ss_tot = float((yc ** 2).sum())
    if ss_tot < 1e-18:
        # no within-state label variance at all -- e.g. the state value V, which
        # is shared by every candidate in a state. The quantity is undefined.
        return float("nan")
    return 1.0 - float(((yc - pc) ** 2).sum()) / ss_tot


def within_state_concordance(y, pred, state_id):
    """Pairwise ranking accuracy among candidates that share a state s_t.

    This is the decision a scheduler actually faces: given several maskable
    positions at the SAME state, which has the higher future value?
    """
    conc = tot = 0
    for s in np.unique(state_id):
        m = state_id == s
        yy, pp = y[m], pred[m]
        n = len(yy)
        if n < 2:
            continue
        dy = yy[:, None] - yy[None, :]
        dp = pp[:, None] - pp[None, :]
        iu = np.triu_indices(n, 1)
        dy, dp = dy[iu], dp[iu]
        ok = np.abs(dy) > 1e-9
        dy, dp = dy[ok], dp[ok]
        # a constant predictor (e.g. h_i at layer 0, which is the MASK
        # embedding and identical at every masked position) ties every pair;
        # ties score as chance, not as wrong
        ties = np.abs(dp) <= 1e-12
        conc += float((np.sign(dy[~ties]) == np.sign(dp[~ties])).sum()
                      + 0.5 * ties.sum())
        tot += int(len(dy))
    if tot == 0:                    # no rankable pairs (constant-within-state)
        return float("nan"), 0
    return conc / tot, tot


def noise_ceiling(y_seeds):
    """Max R^2 any predictor could reach given label noise.

    y_seeds: (N, K) per-seed label replicates. The observed label variance is
    signal + noise/K; only the signal part is predictable.
    """
    ybar = y_seeds.mean(1)
    K = y_seeds.shape[1]
    noise = float((y_seeds.var(1, ddof=1) / K).mean())
    obs = float(ybar.var())
    return max(0.0, (obs - noise) / max(obs, 1e-12)), noise, obs


# --------------------------------------------------------------- bootstrap --
def cluster_bootstrap(stat_fn, doc_ids, n_boot=1000, seed=0):
    """Resample DOCUMENTS with replacement; returns (mean, lo, hi) at 95%."""
    rng = np.random.default_rng(seed)
    docs = np.unique(doc_ids)
    idx_by_doc = {d: np.where(doc_ids == d)[0] for d in docs}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(docs, size=len(docs), replace=True)
        idx = np.concatenate([idx_by_doc[d] for d in pick])
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:          # statistic undefined on every resample (e.g. a
        nan = float("nan")   # within-state metric for a constant-within-state
        return nan, nan, nan  # target such as V)
    vals = np.array(vals)
    return float(vals.mean()), float(np.percentile(vals, 2.5)), \
        float(np.percentile(vals, 97.5))

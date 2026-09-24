"""
Probe family P0-P13.

Three rules applied to every probe:
    1. **Configurable selection criterion**. Selecting layer, alpha and gamma by *pooled R^2* while
          judging by *within-state ranking* is inconsistent.
          Every probe here accepts `select_by in {pooled_r2, within_r2, concordance}`
          and both conventions are reported side by side.
    2. **Test never participates in any selection**. Layer, hyperparameters and early stopping are all chosen on val.
    3. **Capacity controls**. Every nonlinear / relational probe has an equal-capacity cheap-only version,
          plus three falsification controls: shuffled-h_g / Gaussian / label permutation.

Convention: every fit_* function returns a dict containing at least
    {"pred_test", "pred_val", "val_score", "hp", "n_params"}
"""
import numpy as np
import torch

from . import metrics as M

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================ selection criteria ========
def make_selector(kind, state_id_val, groups_val=None):
    """Returns f(y_val, pred_val) -> scalar, larger is better."""
    if groups_val is None and state_id_val is not None:
        _, groups_val = M.group_slices(state_id_val)
    if kind == "pooled_r2":
        return lambda y, p: M.r2_score(y, p)
    if kind == "within_r2":
        return lambda y, p: M.within_state_r2(y, p, state_id_val, groups_val)
    if kind == "concordance":
        return lambda y, p: M.concordance(y, p, state_id_val, groups_val)
    raise KeyError(kind)


def _safe(v):
    return -1e18 if (v is None or not np.isfinite(v)) else float(v)


# ============================================================ ridge ==========
def _std(Xtr, *rest):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return [( (X - mu) / sd ).astype(np.float32) for X in (Xtr,) + rest], mu, sd


def ridge_path(Xtr, ytr, alphas):
    """Solutions for all alphas from a single eigendecomposition (GPU)."""
    ym = float(ytr.mean())
    yc = (ytr - ym).astype(np.float32)
    Xg = torch.as_tensor(np.ascontiguousarray(Xtr, dtype=np.float32), device=DEV)
    yg = torch.as_tensor(yc, device=DEV)
    G = (Xg.T @ Xg).double()
    b = (Xg.T @ yg).double()
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0.0)
    Vtb = V.T @ b
    out = []
    for a in alphas:
        w = (V @ (Vtb / (lam + float(a)))).float()
        out.append((w, ym))
    return out


def fit_ridge(Xtr, ytr, Xva, yva, Xte, selector, alphas=None):
    """P0 single-block ridge. selector determines alpha."""
    if alphas is None:
        alphas = np.logspace(-2, 9, 34)
    (Xt, Xv, Xs), mu, sd = _std(Xtr, Xva, Xte)
    Xt_g = torch.as_tensor(Xt, device=DEV)
    Xv_g = torch.as_tensor(Xv, device=DEV)
    Xs_g = torch.as_tensor(Xs, device=DEV)
    best = None
    for (w, ym), a in zip(ridge_path(Xt, ytr, alphas), alphas):
        pv = (Xv_g @ w + ym).cpu().numpy()
        s = _safe(selector(yva, pv))
        if best is None or s > best["val_score"]:
            best = {"val_score": s, "alpha": float(a), "w": w, "ym": ym,
                    "pred_val": pv}
    best["pred_test"] = (Xs_g @ best["w"] + best["ym"]).cpu().numpy()
    best["n_params"] = int(Xtr.shape[1] + 1)
    best["hp"] = {"alpha": best["alpha"]}
    del best["w"]
    return best


def fit_ridge_2block(Xc_tr, Xh_tr, ytr, Xc_va, Xh_va, yva, Xc_te, Xh_te,
                     selector, alphas=None, gammas=None):
    """Two-block ridge: control and hidden blocks regularised separately. The gamma grid includes 0 (nested baseline)."""
    if gammas is None:
        gammas = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    if alphas is None:
        alphas = np.logspace(-2, 9, 34)
    (Ct, Cv, Cs), _, _ = _std(Xc_tr, Xc_va, Xc_te)
    (Ht, Hv, Hs), _, _ = _std(Xh_tr, Xh_va, Xh_te)
    best = None
    for g in gammas:
        if g > 0:
            Xt = np.concatenate([Ct, g * Ht], 1)
            Xv = np.concatenate([Cv, g * Hv], 1)
            Xs = np.concatenate([Cs, g * Hs], 1)
        else:
            Xt, Xv, Xs = Ct, Cv, Cs
        Xv_g = torch.as_tensor(Xv, device=DEV)
        Xs_g = torch.as_tensor(Xs, device=DEV)
        for (w, ym), a in zip(ridge_path(Xt, ytr, alphas), alphas):
            pv = (Xv_g @ w + ym).cpu().numpy()
            s = _safe(selector(yva, pv))
            if best is None or s > best["val_score"]:
                best = {"val_score": s, "alpha": float(a), "gamma": float(g),
                        "pred_val": pv,
                        "pred_test": (Xs_g @ w + ym).cpu().numpy(),
                        "n_params": int(Xt.shape[1] + 1)}
    best["hp"] = {"alpha": best["alpha"], "gamma": best["gamma"]}
    return best


# ================================================= cross-fitted residualisation (P1) =======
def cross_fit_residual(X, y, doc_id, n_folds=5, alphas=None, seed=0):
    """Document-fold out-of-fold residuals r = y - f_cheap(X).

    Folds are split by **document**, so rows of one document are never in both the fitting and prediction folds.
    """
    if alphas is None:
        alphas = np.logspace(-2, 9, 34)
    docs = np.unique(doc_id)
    rng = np.random.default_rng(seed)
    docs = docs.copy(); rng.shuffle(docs)
    folds = np.array_split(docs, n_folds)
    pred = np.zeros(len(y), dtype=np.float64)
    for f in folds:
        te = np.isin(doc_id, f)
        tr = ~te
        # carve out a small inner split for alpha selection, still by document
        inner = np.unique(doc_id[tr])
        rng2 = np.random.default_rng(seed + 1)
        inner = inner.copy(); rng2.shuffle(inner)
        n_va = max(1, int(0.2 * len(inner)))
        va_docs = inner[:n_va]
        va = np.isin(doc_id, va_docs)
        tr2 = tr & ~va
        sel = lambda yy, pp: M.r2_score(yy, pp)
        m = fit_ridge(X[tr2], y[tr2], X[va], y[va], X[te], sel, alphas)
        pred[te] = m["pred_test"]
    return y - pred, pred


# ============================================ generic training loop for torch probes ========
class _Runner:
    """Pack (features dict, y, group structure) into tensors and train an arbitrary scoring model.

    loss_kind:
        'mse'      -- MSE on standardised y
        'mse_wc'   -- MSE on within-state-centred y
        'pairwise' -- within-state pairwise logistic: -log sigmoid((s_i-s_j)*sign(y_i-y_j))
        'listwise' -- ListNet/PL: cross-entropy between softmax(s) and softmax(y/tau)
    """

    def __init__(self, feats_tr, y_tr, sid_tr, feats_va, y_va, sid_va,
                 feats_te, loss_kind="mse", tau=1.0):
        self.loss_kind = loss_kind
        self.tau = tau
        self.ftr = {k: torch.as_tensor(v, dtype=torch.float32, device=DEV)
                    for k, v in feats_tr.items()}
        self.fva = {k: torch.as_tensor(v, dtype=torch.float32, device=DEV)
                    for k, v in feats_va.items()}
        self.fte = {k: torch.as_tensor(v, dtype=torch.float32, device=DEV)
                    for k, v in feats_te.items()}
        self.ym = float(np.mean(y_tr)); self.ys = float(np.std(y_tr)) or 1.0
        self.ytr_raw = np.asarray(y_tr, np.float64)
        self.ytr = torch.as_tensor((y_tr - self.ym) / self.ys,
                                   dtype=torch.float32, device=DEV)
        self.y_va = np.asarray(y_va, np.float64)
        # group structure (train set) -- each state has exactly 6 candidates, so it reshapes to (S, C)
        self.gidx_tr = self._group_tensor(sid_tr)
        if loss_kind == "mse_wc":
            yc = self.ytr.clone()
            G = self.gidx_tr
            gv = yc[G]                       # (S, C)
            yc[G] = gv - gv.mean(1, keepdim=True)
            self.ytr = yc

    @staticmethod
    def _group_tensor(sid):
        uniq, groups = M.group_slices(sid)
        sizes = {len(g) for g in groups}
        if len(sizes) != 1:
            raise ValueError(f"ragged groups {sorted(sizes)[:5]}; "
                             "this implementation assumes the same number of candidates per state")
        return torch.as_tensor(np.stack(groups), dtype=torch.long, device=DEV)

    def _loss(self, s):
        if self.loss_kind in ("mse", "mse_wc"):
            return ((s - self.ytr) ** 2).mean()
        G = self.gidx_tr
        sg = s[G]                                        # (S, C)
        yg = self.ytr[G]
        if self.loss_kind == "pairwise":
            ds = sg[:, :, None] - sg[:, None, :]
            dy = yg[:, :, None] - yg[:, None, :]
            iu = torch.triu_indices(sg.shape[1], sg.shape[1], offset=1)
            ds = ds[:, iu[0], iu[1]]; dy = dy[:, iu[0], iu[1]]
            m = dy.abs() > 1e-9
            if m.sum() == 0:
                return (s * 0).sum()
            z = ds[m] * torch.sign(dy[m])
            return torch.nn.functional.softplus(-z).mean()
        if self.loss_kind == "listwise":
            tgt = torch.softmax(yg / self.tau, dim=1)
            logp = torch.log_softmax(sg, dim=1)
            return -(tgt * logp).sum(1).mean()
        raise KeyError(self.loss_kind)

    def run(self, make_model, selector, seeds=(0, 1, 2), epochs=400,
            lrs=(3e-3, 1e-3), wds=(1e-4, 1e-2), patience=60):
        """Grid x seed search. Early stopping uses the best step count of **this configuration itself**.

        Ranking losses give scale-free scores; R^2-type criteria need a scale. We fit a univariate regression
        y ~ a*s + b on the **validation set** and apply the same (a,b) to test.
        Using validation for calibration is allowed (test is never used).
        """
        best = None
        for lr in lrs:
            for wd in wds:
                for sd in seeds:
                    torch.manual_seed(sd)
                    net = make_model().to(DEV)
                    opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                            weight_decay=wd)
                    local, bad = None, 0
                    for ep in range(epochs):
                        net.train(); opt.zero_grad()
                        loss = self._loss(net(self.ftr))
                        if hasattr(net, "penalty"):
                            loss = loss + net.penalty()
                        loss.backward()
                        opt.step()
                        if (ep % 5) and ep != epochs - 1:
                            continue
                        net.eval()
                        with torch.no_grad():
                            pv = net(self.fva).cpu().numpy().astype(np.float64)
                        a, b = self._calib(pv, self.y_va)
                        sc = _safe(selector(self.y_va, a * pv + b))
                        if local is None or sc > local["val_score"]:
                            with torch.no_grad():
                                pt = net(self.fte).cpu().numpy().astype(np.float64)
                            local = {"val_score": sc, "pred_val": a * pv + b,
                                     "pred_test": a * pt + b,
                                     "calib": (float(a), float(b)),
                                     "n_params": int(sum(p.numel() for p in
                                                         net.parameters()))}
                            bad = 0
                        else:
                            bad += 5
                            if bad > patience:
                                break
                    if local is None:
                        continue
                    local["hp"] = {"lr": lr, "wd": wd, "seed": sd}
                    if best is None or local["val_score"] > best["val_score"]:
                        best = local
        return best

    def _calib(self, p, y):
        """Map scale-free scores to the scale of y on val. Under MSE loss this is de-standardisation."""
        if self.loss_kind in ("mse", "mse_wc"):
            return self.ys, self.ym
        if np.std(p) < 1e-12:
            return 0.0, float(np.mean(y))
        a, b = np.polyfit(p, y, 1)
        return float(a), float(b)


# ============================================================ model definitions =======
class Bilinear(torch.nn.Module):
    """P2: A ≈ h_i^T U V^T h_g + w_i·h_i + w_g·h_g + b."""

    def __init__(self, d_i, d_g, rank, use_linear=True):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(d_i, rank) / np.sqrt(d_i))
        self.V = torch.nn.Parameter(torch.randn(d_g, rank) / np.sqrt(d_g))
        self.lin_i = torch.nn.Linear(d_i, 1) if use_linear else None
        self.lin_g = torch.nn.Linear(d_g, 1) if use_linear else None
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, f):
        hi, hg = f["hi"], f["hg"]
        s = ((hi @ self.U) * (hg @ self.V)).sum(-1) + self.b
        if self.lin_i is not None:
            s = s + self.lin_i(hi).squeeze(-1) + self.lin_g(hg).squeeze(-1)
        if "c" in f and getattr(self, "lin_c", None) is not None:
            s = s + self.lin_c(f["c"]).squeeze(-1)
        return s


class BilinearC(Bilinear):
    """Bilinear model with the cheap control block."""

    def __init__(self, d_i, d_g, rank, d_c, use_linear=True):
        super().__init__(d_i, d_g, rank, use_linear)
        self.lin_c = torch.nn.Linear(d_c, 1)


class FiLM(torch.nn.Module):
    """P3: γ,β = g(h_g); h̃_i = γ⊙h_i + β; A = w·h̃_i (+ cheap)."""

    def __init__(self, d_i, d_g, hid=64, d_c=None):
        super().__init__()
        self.g = torch.nn.Sequential(torch.nn.Linear(d_g, hid), torch.nn.GELU(),
                                     torch.nn.Linear(hid, 2 * d_i))
        self.w = torch.nn.Linear(d_i, 1)
        self.lin_g = torch.nn.Linear(d_g, 1)
        self.lin_c = torch.nn.Linear(d_c, 1) if d_c else None
        self.d_i = d_i
        # gamma initialised near 1 and beta near 0, so at t=0 it reduces to a pure linear probe
        torch.nn.init.zeros_(self.g[-1].weight); torch.nn.init.zeros_(self.g[-1].bias)

    def forward(self, f):
        hi, hg = f["hi"], f["hg"]
        gb = self.g(hg)
        gam, beta = gb[:, :self.d_i], gb[:, self.d_i:]
        s = self.w((1.0 + gam) * hi + beta).squeeze(-1) + self.lin_g(hg).squeeze(-1)
        if self.lin_c is not None and "c" in f:
            s = s + self.lin_c(f["c"]).squeeze(-1)
        return s


class MLPProbe(torch.nn.Module):
    """Generic MLP for P4 / P8. Inputs are determined by feature_keys and concatenated in forward."""

    def __init__(self, dims, hidden=(256,), keys=("x",)):
        super().__init__()
        self.keys = keys
        d = sum(dims)
        layers = []
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.GELU()]
            d = h
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, f):
        x = torch.cat([f[k] for k in self.keys], -1)
        return self.net(x).squeeze(-1)


class LayerMix(torch.nn.Module):
    """P12: h_mix = sum_l softmax(alpha)_l h^(l), followed by a linear/MLP readout."""

    def __init__(self, n_layers, d, d_c=None, hidden=None, l1=0.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(n_layers))
        self.l1 = l1
        if hidden:
            self.head = MLPProbe([d], hidden, keys=("x",))
        else:
            self.head = None
            self.w = torch.nn.Linear(d, 1)
        self.lin_c = torch.nn.Linear(d_c, 1) if d_c else None

    def forward(self, f):
        H = f["H"]                                  # (N, L, d)
        a = torch.softmax(self.alpha, 0)
        x = (H * a[None, :, None]).sum(1)
        s = (self.head({"x": x}) if self.head is not None
             else self.w(x).squeeze(-1))
        if self.lin_c is not None and "c" in f:
            s = s + self.lin_c(f["c"]).squeeze(-1)
        return s

    def penalty(self):
        return self.l1 * torch.softmax(self.alpha, 0).abs().sum()


# ============================================================ kernels / trees ========
def rff_features(X, D, gamma, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.normal(0, np.sqrt(2 * gamma), size=(d, D // 2))
    Z = X @ W
    return np.concatenate([np.cos(Z), np.sin(Z)], 1).astype(np.float32) \
        * np.sqrt(2.0 / D)


def poly2_features(X):
    n, d = X.shape
    iu = np.triu_indices(d)
    return np.concatenate([X, (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]],
                          1).astype(np.float32)

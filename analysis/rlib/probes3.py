"""
Corrected probe library (new file; does not override probes2.py).

Three fixes:
    F1  **Per-block penalties**. probes2.fit_ridge_2block puts [Zi, Zg, vec(Zi x Zg)] into
            a *single* hidden block sharing one gamma / one alpha. The 64 main-effect columns and 1024
            interaction columns are shrunk with the same strength, so the weak penalty wanted by main effects and the strong
            penalty wanted by interactions cannot both be satisfied. This module provides
            `fit_ridge_blocks`: an independent gamma grid per block, penalty strength = alpha/gamma^2.
            The **additive baseline and the kron model come from the same search**: the gamma_int = 0 slice is the
            additive model, so kron strictly nests additive and cannot be worse on validation.
    F2  **Upper bound of the gamma grid**. probes2 caps gammas at 3.0; on real data additive_pca
            selected exactly 3.0 (hitting the boundary). The default cap here is 100.
    F3  **Torch training budget**. probes2._Runner uses 400 full-batch AdamW steps, and the best hyperparameters
            always landed on the grid boundary (max lr=3e-3, min wd=1e-4). `Runner3` defaults to 4000 steps,
            wider lr/wd grids, optional loss-curve logging and optional within-state calibration.

Conventions follow probes2: returned dict contains pred_test / pred_val / val_score / hp.
"""
import numpy as np
import torch

from . import metrics as M

DEV = "cuda" if torch.cuda.is_available() else "cpu"

ALPHAS_DEF = np.logspace(-3, 9, 37)
GAMMAS_MAIN = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0]
GAMMAS_INT = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def _safe(v):
    return -1e18 if (v is None or not np.isfinite(v)) else float(v)


def _std_block(Xtr, *rest):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return [((X - mu) / sd).astype(np.float32) for X in (Xtr,) + rest]


def _ridge_path_gpu(Xg, yg, alphas):
    """Xg (n,p) float32 GPU, yg centered. Returns list[w] (float32 GPU)."""
    G = (Xg.T @ Xg).double()
    b = (Xg.T @ yg).double()
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0.0)
    Vtb = V.T @ b
    return [(V @ (Vtb / (lam + float(a)))).float() for a in alphas]


def fit_ridge_blocks(blocks_tr, ytr, blocks_va, yva, blocks_te, selector,
                     gamma_grids, alphas=None, record_slices=None):
    """Ridge regression with any number of blocks, **each block scaled independently** -> independent penalties.

    blocks_*: list of (n, p_b) arrays; block 0 has fixed gamma=1 (reference block).
    gamma_grids: list with one entry per block; block 0 is usually [1.0].
    Penalty equivalence: X = [g_0 B_0, g_1 B_1, ...] with penalty alpha*||w||^2
                            <=> penalty (alpha/g_b^2)||u_b||^2 on original-scale coefficients u_b = g_b w_b.
    record_slices: dict name -> bool list indicating which blocks may be non-zero in that sub-model
            (used to extract nested sub-models from the same search, e.g. additive = interaction block gamma=0).
            Returns dict name -> best model.
    """
    if alphas is None:
        alphas = ALPHAS_DEF
    nb = len(blocks_tr)
    Bt, Bv, Bs = [], [], []
    for b in range(nb):
        t, v, s = _std_block(blocks_tr[b], blocks_va[b], blocks_te[b])
        Bt.append(torch.as_tensor(t, device=DEV))
        Bv.append(torch.as_tensor(v, device=DEV))
        Bs.append(torch.as_tensor(s, device=DEV))
    ym = float(np.mean(ytr))
    yg = torch.as_tensor((np.asarray(ytr) - ym).astype(np.float32), device=DEV)

    if record_slices is None:
        record_slices = {"full": [True] * nb}
    best = {k: None for k in record_slices}

    import itertools
    for combo in itertools.product(*gamma_grids):
        active = [g > 0 for g in combo]
        if not any(active):
            continue
        cols = [b for b in range(nb) if active[b]]
        Xt = torch.cat([combo[b] * Bt[b] for b in cols], 1)
        Xv = torch.cat([combo[b] * Bv[b] for b in cols], 1)
        Xs = torch.cat([combo[b] * Bs[b] for b in cols], 1)
        ws = _ridge_path_gpu(Xt, yg, alphas)
        for w, a in zip(ws, alphas):
            pv = (Xv @ w + ym).cpu().numpy().astype(np.float64)
            sc = _safe(selector(yva, pv))
            for name, allow in record_slices.items():
                # is this combo inside the feasible set of this sub-model
                if any(active[b] and not allow[b] for b in range(nb)):
                    continue
                cur = best[name]
                if cur is None or sc > cur["val_score"]:
                    best[name] = {
                        "val_score": sc, "pred_val": pv,
                        "pred_test": (Xs @ w + ym).cpu().numpy().astype(np.float64),
                        "hp": {"alpha": float(a),
                               "gammas": [float(g) for g in combo]},
                        "n_params": int(Xt.shape[1] + 1)}
    return best


# ------------------------------------------------------------------ torch ----
class Runner3:
    """probes2._Runner with larger budget and extra diagnostics."""

    def __init__(self, feats_tr, y_tr, sid_tr, feats_va, y_va, sid_va,
                 feats_te, loss_kind="mse", tau=1.0, calib="pooled"):
        self.loss_kind = loss_kind
        self.tau = tau
        self.calib_kind = calib
        to = lambda D: {k: torch.as_tensor(v, dtype=torch.float32, device=DEV)
                        for k, v in D.items()}
        self.ftr, self.fva, self.fte = to(feats_tr), to(feats_va), to(feats_te)
        self.ym = float(np.mean(y_tr)); self.ys = float(np.std(y_tr)) or 1.0
        self.ytr = torch.as_tensor((y_tr - self.ym) / self.ys,
                                   dtype=torch.float32, device=DEV)
        self.y_va = np.asarray(y_va, np.float64)
        self.sid_va = sid_va
        self.gidx_tr = self._group_tensor(sid_tr)
        _, self.gva = M.group_slices(sid_va)
        if loss_kind == "mse_wc":
            yc = self.ytr.clone(); G = self.gidx_tr
            gv = yc[G]; yc[G] = gv - gv.mean(1, keepdim=True); self.ytr = yc

    @staticmethod
    def _group_tensor(sid):
        uniq, groups = M.group_slices(sid)
        if len({len(g) for g in groups}) != 1:
            raise ValueError("ragged groups")
        return torch.as_tensor(np.stack(groups), dtype=torch.long, device=DEV)

    def _loss(self, s):
        if self.loss_kind in ("mse", "mse_wc"):
            return ((s - self.ytr) ** 2).mean()
        G = self.gidx_tr
        sg, yg = s[G], self.ytr[G]
        if self.loss_kind == "pairwise":
            ds = sg[:, :, None] - sg[:, None, :]
            dy = yg[:, :, None] - yg[:, None, :]
            iu = torch.triu_indices(sg.shape[1], sg.shape[1], offset=1)
            ds = ds[:, iu[0], iu[1]]; dy = dy[:, iu[0], iu[1]]
            m = dy.abs() > 1e-9
            if m.sum() == 0:
                return (s * 0).sum()
            return torch.nn.functional.softplus(-(ds[m] * torch.sign(dy[m]))).mean()
        if self.loss_kind == "listwise":
            ygc = yg - yg.mean(1, keepdim=True)          # within-state standardisation
            ygc = ygc / (ygc.std() + 1e-8)
            tgt = torch.softmax(ygc / self.tau, 1)
            return -(tgt * torch.log_softmax(sg, 1)).sum(1).mean()
        raise KeyError(self.loss_kind)

    def _calib(self, p, y):
        if self.loss_kind in ("mse", "mse_wc"):
            return self.ys, self.ym
        if np.std(p) < 1e-12:
            return 0.0, float(np.mean(y))
        if self.calib_kind == "within":
            # fit the slope in the within-state-centred space (the correct scale for within_r2)
            yc = np.asarray(y, np.float64).copy(); pc = np.asarray(p, np.float64).copy()
            for g in self.gva:
                yc[g] -= yc[g].mean(); pc[g] -= pc[g].mean()
            denom = float((pc ** 2).sum())
            a = float((pc * yc).sum() / denom) if denom > 1e-18 else 0.0
            return a, float(np.mean(y) - a * np.mean(p))
        a, b = np.polyfit(p, y, 1)
        return float(a), float(b)

    def run(self, make_model, selector, seeds=(0, 1, 2), epochs=4000,
            lrs=(3e-2, 1e-2, 3e-3, 1e-3), wds=(0.0, 1e-4, 1e-2),
            patience=600, eval_every=10, curves=False):
        best, curve_log = None, []
        for lr in lrs:
            for wd in wds:
                for sd in seeds:
                    torch.manual_seed(sd)
                    net = make_model().to(DEV)
                    opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                            weight_decay=wd)
                    local, bad, cv = None, 0, []
                    for ep in range(epochs):
                        net.train(); opt.zero_grad()
                        loss = self._loss(net(self.ftr))
                        if hasattr(net, "penalty"):
                            loss = loss + net.penalty()
                        loss.backward(); opt.step()
                        if (ep % eval_every) and ep != epochs - 1:
                            continue
                        net.eval()
                        with torch.no_grad():
                            pv = net(self.fva).cpu().numpy().astype(np.float64)
                        a, b = self._calib(pv, self.y_va)
                        sc = _safe(selector(self.y_va, a * pv + b))
                        if curves:
                            cv.append((ep, float(loss.item()), sc, float(a)))
                        if local is None or sc > local["val_score"]:
                            with torch.no_grad():
                                pt = net(self.fte).cpu().numpy().astype(np.float64)
                            local = {"val_score": sc, "pred_val": a * pv + b,
                                     "pred_test": a * pt + b,
                                     "calib": (float(a), float(b)),
                                     "best_epoch": ep,
                                     "train_loss": float(loss.item()),
                                     "n_params": int(sum(p.numel() for p in
                                                         net.parameters()))}
                            bad = 0
                        else:
                            bad += eval_every
                            if bad > patience:
                                break
                    if curves:
                        curve_log.append({"lr": lr, "wd": wd, "seed": sd,
                                          "curve": cv})
                    if local is None:
                        continue
                    local["hp"] = {"lr": lr, "wd": wd, "seed": sd,
                                   "epochs_budget": epochs}
                    if best is None or local["val_score"] > best["val_score"]:
                        best = local
        if best is not None and curves:
            best["curves"] = curve_log
        return best

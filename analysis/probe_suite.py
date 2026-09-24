"""Controlled rescue probes and decision-focused metrics.

This module is intentionally separate from the historical analysis.  It uses
train-only transforms, validation-only model selection, group/document splits,
and never silently falls back from an unavailable representation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem, PolynomialCountSketch, RBFSampler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SuiteConfig:
    seed: int = 0
    pca_dim: int = 64
    epochs: int = 80
    patience: int = 15
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"


def grouped_center(y, state):
    z = np.asarray(y, dtype=np.float64).copy()
    for s in np.unique(state):
        m = state == s
        z[m] -= z[m].mean()
    return z


def pair_arrays(y, pred, state):
    ys, ps = [], []
    for s in np.unique(state):
        ix = np.where(state == s)[0]
        if len(ix) < 2:
            continue
        a, b = np.triu_indices(len(ix), 1)
        dy = y[ix[a]] - y[ix[b]]
        dp = pred[ix[a]] - pred[ix[b]]
        keep = np.abs(dy) > 1e-12
        ys.append(dy[keep]); ps.append(dp[keep])
    return (np.concatenate(ys) if ys else np.empty(0),
            np.concatenate(ps) if ps else np.empty(0))


def decision_metrics(y, pred, state, ceiling=None):
    y = np.asarray(y); pred = np.asarray(pred); state = np.asarray(state)
    yc, pc = grouped_center(y, state), grouped_center(pred, state)
    dy, dp = pair_arrays(y, pred, state)
    ties = np.abs(dp) <= 1e-12
    concord = (np.mean(np.where(ties, 0.5, np.sign(dy) == np.sign(dp)))
               if len(dy) else np.nan)
    auc = roc_auc_score(dy > 0, dp) if len(np.unique(dy > 0)) == 2 else np.nan
    regrets, norm_regrets, top1, top2, top3 = [], [], [], [], []
    top1_value, top1_value_decision = [], []
    state_rhos, state_taus = [], []
    for s in np.unique(state):
        ix = np.where(state == s)[0]
        if len(ix) < 2:
            continue
        oracle = ix[np.argmax(y[ix])]
        order = ix[np.argsort(-pred[ix])]
        r = float(y[oracle] - y[order[0]])
        span = float(np.max(y[ix]) - np.min(y[ix]))
        regrets.append(r); norm_regrets.append(r / max(span, 1e-12))
        top1.append(oracle in order[:1]); top2.append(oracle in order[:2])
        top3.append(oracle in order[:3])
        # A_task is discrete and often has several equally optimal actions.
        # The legacy exact-index top1 arbitrarily designates the first maximum
        # as the sole oracle and is especially misleading on all-tie states.
        # Report a tie-aware achieved-value endpoint, plus its decision-only
        # version that excludes states where every candidate has equal value.
        value_hit = bool(y[order[0]] >= np.max(y[ix]) - 1e-12)
        top1_value.append(value_hit)
        if span > 1e-12:
            top1_value_decision.append(value_hit)
        if np.ptp(pred[ix]) > 1e-10 and np.ptp(y[ix]) > 1e-10:
            state_rhos.append(stats.spearmanr(y[ix], pred[ix]).statistic)
            state_taus.append(stats.kendalltau(y[ix], pred[ix]).statistic)
    ss = float(np.sum((y - y.mean()) ** 2))
    wss = float(np.sum(yc ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / max(ss, 1e-12)
    wr2 = 1.0 - float(np.sum((yc - pc) ** 2)) / max(wss, 1e-12)
    return {
        "r2": r2, "within_r2": wr2,
        "ceiling_normalized_r2": r2 / ceiling if ceiling and ceiling > 0 else np.nan,
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "spearman": float(stats.spearmanr(y, pred).statistic) if np.std(pred) else 0.0,
        "pairwise_concordance": float(concord), "pairwise_auc": float(auc),
        "kendall_tau_state_mean": float(np.nanmean(state_taus)) if state_taus else np.nan,
        "spearman_state_mean": float(np.nanmean(state_rhos)) if state_rhos else np.nan,
        "top1_accuracy": float(np.mean(top1)) if top1 else np.nan,
        "top1_value_accuracy": float(np.mean(top1_value)) if top1_value else np.nan,
        "top1_value_accuracy_decision": (float(np.mean(top1_value_decision))
                                           if top1_value_decision else np.nan),
        "n_decision_states": int(len(top1_value_decision)),
        "top2_recall": float(np.mean(top2)) if top2 else np.nan,
        "top3_recall": float(np.mean(top3)) if top3 else np.nan,
        "mean_regret": float(np.mean(regrets)) if regrets else np.nan,
        "median_regret": float(np.median(regrets)) if regrets else np.nan,
        "normalized_regret": float(np.mean(norm_regrets)) if norm_regrets else np.nan,
        "worst_quartile_regret": float(np.mean(np.sort(regrets)[-max(1, len(regrets)//4):])) if regrets else np.nan,
        "n_states": int(len(np.unique(state))), "n_pairs": int(len(dy)),
    }


def fit_ridge(Xtr, ytr, Xva, yva, alphas=None):
    alphas = np.logspace(-3, 6, 10) if alphas is None else alphas
    scaler = StandardScaler().fit(Xtr)
    xt, xv = scaler.transform(Xtr), scaler.transform(Xva)
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False
    if use_gpu:
        # One Gram eigendecomposition serves the entire alpha path.
        xg=torch.as_tensor(xt,dtype=torch.float32,device='cuda')
        yg=torch.as_tensor((ytr-ytr.mean()).astype(np.float32),device='cuda')
        G=(xg.T@xg).double(); b=(xg.T@yg).double()
        lam,V=torch.linalg.eigh(G); lam=lam.clamp(min=0); vb=V.T@b
        best=None
        xvg=torch.as_tensor(xv,dtype=torch.float32,device='cuda')
        for a in alphas:
            w=(V@(vb/(lam+float(a)))).float()
            pred=(xvg@w).cpu().numpy()+float(ytr.mean())
            loss=float(np.mean((pred-yva)**2))
            if best is None or loss<best[0]: best=(loss,float(a),w.cpu().numpy())
        return {"scaler":scaler,"w":best[2],"b":float(ytr.mean()),
                "alpha":best[1],"val_mse":best[0],"solver":"gpu_gram_eigh"}
    best = None
    for a in alphas:
        # LSQR is much more stable and faster than repeatedly factorising an
        # ill-conditioned 1.5k-dimensional Gram matrix for every alpha.
        m = Ridge(alpha=float(a), solver="lsqr", tol=1e-5).fit(xt, ytr)
        loss = np.mean((m.predict(xv) - yva) ** 2)
        if best is None or loss < best[0]:
            best = (loss, a, m)
    return {"scaler": scaler, "model": best[2], "alpha": float(best[1]),
            "val_mse": float(best[0])}


def predict_sklearn(m, X):
    if "w" in m:
        return m["scaler"].transform(X) @ m["w"] + m["b"]
    return m["model"].predict(m["scaler"].transform(X))


def residualized_linear(Xc, Xh, y, groups, tr, va, te):
    """P1: cross-fitted cheap residuals on train, hidden -> residual."""
    oof = np.empty(len(tr), dtype=np.float64)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups[tr]))))
    for a, b in gkf.split(Xc[tr], y[tr], groups[tr]):
        m = fit_ridge(Xc[tr][a], y[tr][a], Xc[tr][b], y[tr][b])
        oof[b] = predict_sklearn(m, Xc[tr][b])
    residual = y[tr] - oof
    # Validation residual uses a cheap model fit only on train.
    cheap = fit_ridge(Xc[tr], y[tr], Xc[va], y[va])
    rv = y[va] - predict_sklearn(cheap, Xc[va])
    hidden = fit_ridge(Xh[tr], residual, Xh[va], rv)
    return (predict_sklearn(cheap, Xc[te]) + predict_sklearn(hidden, Xh[te]),
            {"cheap_alpha": cheap["alpha"], "hidden_alpha": hidden["alpha"]})


def pca_pair(hi, hg, tr, dim, seed):
    dim = min(dim, hi.shape[1], len(tr) - 1)
    pi = PCA(dim, svd_solver="randomized", random_state=seed).fit(hi[tr])
    pg = PCA(dim, svd_solver="randomized", random_state=seed + 1).fit(hg[tr])
    return pi.transform(hi).astype(np.float32), pg.transform(hg).astype(np.float32), pi, pg


def make_pairs(state, y, idx, max_pairs=120_000, seed=0):
    rows = []
    for s in np.unique(state[idx]):
        ix = idx[state[idx] == s]
        a, b = np.triu_indices(len(ix), 1)
        rows.extend(zip(ix[a], ix[b]))
    arr = np.asarray(rows, dtype=np.int64)
    if len(arr) > max_pairs:
        arr = arr[np.random.default_rng(seed).choice(len(arr), max_pairs, replace=False)]
    keep = np.abs(y[arr[:, 0]] - y[arr[:, 1]]) > 1e-12
    return arr[keep]


def pairwise_logistic(features, y, state, tr, va, te, seed=0):
    ptr, pva, pte = (make_pairs(state, y, ix, seed=seed) for ix in (tr, va, te))
    def xy(p):
        x = features[p[:, 0]] - features[p[:, 1]]
        z = (y[p[:, 0]] > y[p[:, 1]]).astype(int)
        # Symmetrize so class balance/orientation cannot carry signal.
        return np.concatenate([x, -x]), np.concatenate([z, 1-z])
    xt, yt = xy(ptr); xv, yv = xy(pva)
    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000,
                          random_state=seed)).fit(xt, yt)
        loss = -np.mean(yv*np.log(np.clip(m.predict_proba(xv)[:,1],1e-7,1)) +
                        (1-yv)*np.log(np.clip(m.predict_proba(xv)[:,0],1e-7,1)))
        if best is None or loss < best[0]: best = (loss, c, m)
    # Logistic coefficient defines a scalar candidate score.
    scaler, logit = best[2].steps[0][1], best[2].steps[1][1]
    score = scaler.transform(features) @ logit.coef_[0]
    return score[te], {"C": best[1], "n_train_pairs": len(ptr), "n_test_pairs": len(pte)}


def kernel_probe(X, y, tr, va, te, kind="rbf", seed=0, n_components=512):
    if kind == "rbf":
        mapper = RBFSampler(gamma="scale", n_components=n_components, random_state=seed)
    elif kind == "poly2":
        mapper = PolynomialCountSketch(degree=2, n_components=n_components, random_state=seed)
    else:
        mapper = Nystroem(kernel="rbf", n_components=min(n_components, len(tr)), random_state=seed)
    sc = StandardScaler().fit(X[tr]); xt = sc.transform(X[tr]); xv = sc.transform(X[va])
    mapper.fit(xt); zt, zv = mapper.transform(xt), mapper.transform(xv)
    m = fit_ridge(zt, y[tr], zv, y[va])
    return predict_sklearn(m, mapper.transform(sc.transform(X[te]))), {
        "kind": kind, "n_components": n_components, "alpha": m["alpha"]}


def boosting_probe(X, y, tr, va, te, seed=0):
    best = None
    for leaves in (7, 15, 31):
        m = HistGradientBoostingRegressor(max_leaf_nodes=leaves, l2_regularization=1.0,
                early_stopping=True, validation_fraction=None, random_state=seed).fit(X[tr], y[tr])
        loss = np.mean((m.predict(X[va]) - y[va])**2)
        if best is None or loss < best[0]: best = (loss, leaves, m)
    return best[2].predict(X[te]), {"max_leaf_nodes": best[1], "val_mse": best[0]}


def pls_probe(X, y, tr, va, te):
    best = None
    for n in (2, 4, 8, 16):
        n = min(n, X.shape[1])
        m = make_pipeline(StandardScaler(), PLSRegression(n_components=n, scale=False,
                          max_iter=1000)).fit(X[tr], y[tr])
        loss = np.mean((m.predict(X[va]).ravel() - y[va])**2)
        if best is None or loss < best[0]: best = (loss, n, m)
    return best[2].predict(X[te]).ravel(), {"n_components": best[1], "val_mse": best[0]}


def torch_available(device="cuda"):
    import torch
    return device == "cpu" or torch.cuda.is_available()


def fit_torch_score(kind, xc, hi, hg, y, state, tr, va, te, cfg: SuiteConfig,
                    rank=8, action=None, objective="mse", layers=None,
                    return_artifact=False):
    """P2/P3/P4/P8/P9/P10/P12 small controlled neural scorers."""
    import torch
    import torch.nn as nn
    dev = cfg.device if torch_available(cfg.device) else "cpu"
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    # All blocks are standardized using train only.
    arrays = [xc, hi, hg] + ([] if action is None else [action])
    norm, norm_mu, norm_sd = [], [], []
    for x in arrays:
        mu, sd = x[tr].mean(0), x[tr].std(0); sd[sd < 1e-7] = 1
        norm_mu.append(mu.astype(np.float32)); norm_sd.append(sd.astype(np.float32))
        norm.append(((x-mu)/sd).astype(np.float32))
    xc_, hi_, hg_ = norm[:3]; ac_ = norm[3] if action is not None else None
    dc, di, dg = xc_.shape[1], hi_.shape[1], hg_.shape[1]

    class Score(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "bilinear":
                self.ui=nn.Linear(di,rank,bias=False); self.ug=nn.Linear(dg,rank,bias=False)
                self.lin=nn.Linear(dc+di+dg,1)
            elif kind == "film":
                self.gate=nn.Sequential(nn.Linear(dg,di*2),nn.Tanh())
                self.out=nn.Linear(dc+di,1)
            elif kind == "scalar_mix":
                self.alpha=nn.Parameter(torch.zeros(layers.shape[1]))
                self.out=nn.Linear(dc+layers.shape[2],1)
            else:
                din=dc+di+dg+di+di+(0 if ac_ is None else ac_.shape[1])
                self.net=nn.Sequential(nn.Linear(din,128),nn.GELU(),nn.Linear(128,1))
        def forward(self,c,i,g,a=None,l=None):
            if kind=="bilinear": return self.lin(torch.cat([c,i,g],1)).squeeze(1)+(self.ui(i)*self.ug(g)).sum(1)
            if kind=="film":
                ga,be=self.gate(g).chunk(2,1); return self.out(torch.cat([c,ga*i+be],1)).squeeze(1)
            if kind=="scalar_mix":
                mix=(torch.softmax(self.alpha,0)[None,:,None]*l).sum(1)
                return self.out(torch.cat([c,mix],1)).squeeze(1)
            z=torch.cat([c,i,g,i-g,torch.abs(i-g)] + ([] if a is None else [a]),1)
            return self.net(z).squeeze(1)

    net=Score().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    tensors=[torch.as_tensor(x,device=dev) for x in (xc_,hi_,hg_)]
    ta=torch.as_tensor(ac_,device=dev) if ac_ is not None else None
    tl=torch.as_tensor(layers,device=dev) if layers is not None else None
    ym=float(y[tr].mean()); ys=float(y[tr].std()) or 1.0
    ty=torch.as_tensor(((y-ym)/ys).astype(np.float32),device=dev)
    ptr=make_pairs(state,y,tr,seed=cfg.seed) if objective=="pairwise" else None
    groups=[np.where(state==s)[0] for s in np.unique(state[tr])] if objective=="listwise" else None
    best,bad=None,0
    def score(ix):
        net.eval()
        with torch.no_grad():
            p=net(tensors[0][ix],tensors[1][ix],tensors[2][ix],
                  None if ta is None else ta[ix],None if tl is None else tl[ix])
        return p.cpu().numpy()*ys+ym
    for ep in range(cfg.epochs):
        net.train(); opt.zero_grad()
        if objective=="mse":
            ix=rng.choice(tr,min(cfg.batch_size,len(tr)),replace=False)
            ii=torch.as_tensor(ix,device=dev)
            pr=net(tensors[0][ii],tensors[1][ii],tensors[2][ii],None if ta is None else ta[ii],None if tl is None else tl[ii])
            loss=((pr-ty[ii])**2).mean()
        elif objective=="pairwise":
            pp=ptr[rng.choice(len(ptr),min(cfg.batch_size,len(ptr)),replace=False)]
            a,b=torch.as_tensor(pp[:,0],device=dev),torch.as_tensor(pp[:,1],device=dev)
            sa=net(tensors[0][a],tensors[1][a],tensors[2][a],None if ta is None else ta[a],None if tl is None else tl[a])
            sb=net(tensors[0][b],tensors[1][b],tensors[2][b],None if ta is None else ta[b],None if tl is None else tl[b])
            target=(ty[a]>ty[b]).float(); loss=nn.functional.binary_cross_entropy_with_logits(sa-sb,target)
        else:
            gs=[groups[j] for j in rng.choice(len(groups),min(64,len(groups)),replace=False)]
            losses=[]
            for ix in gs:
                ii=torch.as_tensor(ix,device=dev)
                s=net(tensors[0][ii],tensors[1][ii],tensors[2][ii],None if ta is None else ta[ii],None if tl is None else tl[ii])
                target=torch.softmax(ty[ii],0)
                losses.append(-(target*torch.log_softmax(s,0)).sum())
            loss=torch.stack(losses).mean()
        loss.backward(); opt.step()
        pv=score(va); val=np.mean((pv-y[va])**2) if objective=="mse" else -decision_metrics(y[va],pv,state[va])["pairwise_concordance"]
        if best is None or val<best[0]-1e-7:
            best=(val,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},ep); bad=0
        else: bad+=1
        if bad>=cfg.patience: break
    net.load_state_dict(best[1]); pred=score(te)
    hp={"kind":kind,"objective":objective,"rank":rank,"best_epoch":best[2],
        "config":asdict(cfg),"n_params":sum(p.numel() for p in net.parameters())}
    if not return_artifact:
        return pred,hp
    artifact={"state_dict": {k:v.detach().cpu().numpy()
                              for k,v in net.state_dict().items()},
              "norm_mu": norm_mu, "norm_sd": norm_sd,
              "ym": ym, "ys": ys, "kind": kind, "rank": rank,
              "objective": objective, "hp": hp}
    return pred,hp,artifact

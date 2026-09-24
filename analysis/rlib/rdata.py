"""
Data layer: load label shards, build feature blocks, document-level splits.

Differences from `src/dataset.py`:
    * `h_gm`  -- **leave-one-out mean** of the other candidates' h_i in the same state. An estimate of
                            masked-position pooling that does not require re-running the backbone (h_g is stored only with all-position pooling).
                            Leave-one-out is required, otherwise h_i appears in its own "global" vector and creates a mechanical difference.
    * relational blocks -- [h_i, h_g, h_i-h_g, |h_i-h_g|, h_i*h_g], etc. (P4).
    * action blocks     -- embedding / unembedding vectors of proposed_token (P10).
    * PCA               -- fitted on train only, compresses 768 dims to a trainable size.
"""
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHEAP_KEYS = ("C1", "C2", "C3")


# --------------------------------------------------------------------- load --
def load_labels(tags, keys=None, root=None):
    root = root or ROOT
    files = []
    for t in tags:
        files += sorted(glob.glob(os.path.join(root, "data", f"labels_{t}",
                                               "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shards for tags {tags}")
    parts = [np.load(f) for f in files]
    ks = set(parts[0].files)
    for p in parts:
        ks &= set(p.files)
    if keys is not None:
        ks &= set(keys)
    d = {k: np.concatenate([p[k] for p in parts], 0) for k in sorted(ks)}
    d["state_id"] = (d["prompt_row"].astype(np.int64) * 10_000
                     + d["step"].astype(np.int64))
    if "H_i" in d:
        d["n_layers"] = d["H_i"].shape[1]
    return d


def state_groups(state_id):
    """Returns (uniq_states, group_index_lists, group_id_per_row)."""
    uniq, inv = np.unique(state_id, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    bounds = np.flatnonzero(np.diff(sorted_inv)) + 1
    groups = np.split(order, bounds)
    return uniq, groups, inv


# ----------------------------------------------------------------- features --
def cheap_block(d):
    return np.concatenate([d[k] for k in CHEAP_KEYS], 1).astype(np.float32)


def h_i(d, layer):
    return d["H_i"][:, layer].astype(np.float32)


def h_g(d, layer):
    return d["H_g"][:, layer].astype(np.float32)


def h_gm(d, layer, groups=None):
    """Leave-one-out mean of the other candidates' h_i in the same state (proxy for masked-position pooling).

    LOO is required: with the plain group mean, h_i - mean would contain (1-1/n)*h_i,
    introducing a component exactly collinear with h_i into within-state ranking.
    """
    X = h_i(d, layer)
    if groups is None:
        _, groups, _ = state_groups(d["state_id"])
    out = np.zeros_like(X)
    for g in groups:
        if len(g) == 1:
            out[g] = X[g]                       # no neighbours; fall back to itself
            continue
        s = X[g].sum(0)
        out[g] = (s[None, :] - X[g]) / (len(g) - 1)
    return out


def center_within_state(y, state_id, groups=None):
    """y − mean_state(y)"""
    if groups is None:
        _, groups, _ = state_groups(state_id)
    out = np.asarray(y, dtype=np.float64).copy()
    for g in groups:
        out[g] -= out[g].mean()
    return out


def relational_block(hi, hg):
    """Explicit relational features for P4."""
    return np.concatenate([hi, hg, hi - hg, np.abs(hi - hg), hi * hg],
                          1).astype(np.float32)


# ---------------------------------------------------------------------- PCA --
class TrainPCA:
    """PCA fitted on train rows only (with centering and optional whitening)."""

    def __init__(self, dim, whiten=False):
        self.dim = dim
        self.whiten = whiten

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mu_ = X.mean(0, keepdims=True)
        Xc = X - self.mu_
        # economy SVD; n is usually > d, so the Gram matrix is faster
        G = Xc.T @ Xc
        w, V = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1][:self.dim]
        self.W_ = V[:, idx]
        self.lam_ = np.clip(w[idx], 1e-12, None)
        self.n_ = len(X)
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mu_) @ self.W_
        if self.whiten:
            Z = Z / np.sqrt(self.lam_ / max(self.n_ - 1, 1))[None, :]
        return Z.astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# -------------------------------------------------------------------- split --
def doc_splits(d, seed=0, fracs=(0.6, 0.15, 0.25)):
    docs = np.unique(d["doc_id"])
    rng = np.random.default_rng(seed)
    docs = docs.copy()
    rng.shuffle(docs)
    n_tr, n_va = int(fracs[0] * len(docs)), int(fracs[1] * len(docs))
    sets = {"train": docs[:n_tr], "val": docs[n_tr:n_tr + n_va],
            "test": docs[n_tr + n_va:]}
    return {k: np.where(np.isin(d["doc_id"], v))[0] for k, v in sets.items()}


def check_split_disjoint(d, sp):
    a = set(d["doc_id"][sp["train"]].tolist())
    b = set(d["doc_id"][sp["val"]].tolist())
    c = set(d["doc_id"][sp["test"]].tolist())
    assert not (a & b) and not (a & c) and not (b & c), "document leakage"
    return True


# ------------------------------------------------------------ label helpers --
def noise_ceiling(seeds):
    ybar = seeds.mean(1)
    K = seeds.shape[1]
    noise = float((seeds.var(1, ddof=1) / K).mean())
    obs = float(ybar.var())
    sig = max(obs - noise, 0.0)
    return dict(ceiling=sig / max(obs, 1e-12), snr=sig / max(noise, 1e-12),
                noise_var=noise, obs_var=obs)


def within_state_noise_ceiling(seeds, state_id, groups=None):
    """Noise ceiling of the within-state target.

    Within-group centering is applied to each seed replicate separately, then the same formula is used.
    """
    if groups is None:
        _, groups, _ = state_groups(state_id)
    S = seeds.astype(np.float64).copy()
    for g in groups:
        S[g] -= S[g].mean(0, keepdims=True)
    return noise_ceiling(S)

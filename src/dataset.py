"""Load Phase-0A label shards and assemble the feature blocks under test."""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_labels(tags=("a", "b")):
    """Task-utility shards carry the same schema; `doc_id` is the GSM8K prompt
    index, so document-level splits are prompt-level splits, which is what
    prevents states from one question straddling train and test."""
    files = []
    for t in tags:
        files += sorted(glob.glob(os.path.join(ROOT, "data", f"labels_{t}",
                                               "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shards for tags {tags}")
    parts = [np.load(f) for f in files]
    keys = set(parts[0].files)
    for p in parts:
        keys &= set(p.files)
    d = {k: np.concatenate([p[k] for p in parts], 0) for k in sorted(keys)}
    # a state s_t is one (prompt, timestep) pair; candidates within it are the
    # ranking decision a scheduler faces
    d["state_id"] = (d["prompt_row"].astype(np.int64) * 10_000
                     + d["step"].astype(np.int64))
    d["n_layers"] = d["H_i"].shape[1]
    return d


def block(d, name, layer=None):
    """Feature matrix for a named block. `cheap` = C1+C2+C3 (strong controls)."""
    if name == "C1":
        return d["C1"]
    if name == "C1C2":
        return np.concatenate([d["C1"], d["C2"]], 1)
    if name == "C3":
        return d["C3"]
    if name == "cheap":
        return np.concatenate([d["C1"], d["C2"], d["C3"]], 1)
    if name == "H":
        return np.concatenate([d["H_i"][:, layer].astype(np.float32),
                               d["H_g"][:, layer].astype(np.float32)], 1)
    if name == "H_local":                      # candidate position only
        return d["H_i"][:, layer].astype(np.float32)
    if name == "H_global":                     # constant within a state, so it
        return d["H_g"][:, layer].astype(np.float32)   # cannot rank candidates
    if name == "cheap+H_local":
        return np.concatenate([block(d, "cheap"), block(d, "H_local", layer)], 1)
    if name == "cheap+H":
        return np.concatenate([block(d, "cheap"), block(d, "H", layer)], 1)
    raise KeyError(name)


def doc_splits(d, seed=0, fracs=(0.6, 0.15, 0.25)):
    """DOCUMENT-level train/val/test index arrays (no leakage across states)."""
    docs = np.unique(d["doc_id"])
    rng = np.random.default_rng(seed)
    rng.shuffle(docs)
    n_tr, n_va = int(fracs[0] * len(docs)), int(fracs[1] * len(docs))
    sets = {"train": docs[:n_tr], "val": docs[n_tr:n_tr + n_va],
            "test": docs[n_tr + n_va:]}
    return {k: np.where(np.isin(d["doc_id"], v))[0] for k, v in sets.items()}


def timestep_bins(d, n_bins=None):
    """Recorded steps are already a small discrete set of denoising fractions."""
    steps = np.unique(d["step"])
    return {int(s): np.where(d["step"] == s)[0] for s in steps}

"""Prompt corpus (OpenWebText windows).

MDLM-owt is an unconditional OpenWebText model, so "prompts" are natural-text
prefixes drawn from a held-out OpenWebText sample (stas/openwebtext-10k, an
in-distribution 10k-document subsample). Conditioning = leave the prefix
unmasked and mask the suffix.

Splits are DOCUMENT-level so that no two windows from the same document -- and
hence no two states from the same generation -- can straddle a split boundary.
"""
import os

import numpy as np
import torch

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")


def _cache_path(seq_len):
    return os.path.join(_DATA_DIR, f"owt_windows_L{seq_len}.npz")


def build_windows(seq_len=128, n_docs=3000, seed=0):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from mdlm_local import get_tokenizer

    tok = get_tokenizer()
    # `stas/openwebtext-10k` is a script dataset (unsupported by datasets>=4);
    # read the hub's auto-converted parquet instead.
    pth = hf_hub_download("stas/openwebtext-10k",
                          "plain_text/train/0000.parquet",
                          repo_type="dataset", revision="refs/convert/parquet")
    texts = pq.read_table(pth).column("text").to_pylist()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))[:n_docs * 3]

    windows, doc_ids = [], []
    for d in order:
        text = texts[int(d)]
        if len(text) < 4 * seq_len:
            continue
        ids = tok(text, truncation=True, max_length=seq_len * 4 + 64).input_ids
        if len(ids) < seq_len + 8:
            continue
        start = int(rng.integers(0, len(ids) - seq_len))
        windows.append(ids[start:start + seq_len])
        doc_ids.append(int(d))
        if len(windows) >= n_docs:
            break
    return np.array(windows, dtype=np.int64), np.array(doc_ids, dtype=np.int64)


def get_windows(seq_len=128, n_docs=3000, rebuild=False):
    cache = _cache_path(seq_len)
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache)
        if len(z["windows"]) >= n_docs:
            return z["windows"][:n_docs], z["doc_ids"][:n_docs]
    w, d = build_windows(seq_len=seq_len, n_docs=n_docs)
    os.makedirs(_DATA_DIR, exist_ok=True)
    np.savez_compressed(cache, windows=w, doc_ids=d)
    return w, d


def doc_level_split(doc_ids, fracs=(0.6, 0.15, 0.25), seed=1234):
    """Returns train/val/test index arrays, disjoint at the DOCUMENT level."""
    uniq = np.unique(doc_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    sets = {"train": set(uniq[:n_tr].tolist()),
            "val": set(uniq[n_tr:n_tr + n_va].tolist()),
            "test": set(uniq[n_tr + n_va:].tolist())}
    return {k: np.array([i for i, d in enumerate(doc_ids) if d in v])
            for k, v in sets.items()}

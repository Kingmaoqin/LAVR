"""Five-fold out-of-fold verification of the Task-C relational positive signal.

The broad screen found rank-4 bilinear scoring positive on all three document
splits.  This script freezes rank=4 and a representative early layer (L2),
predicts every document exactly once, and compares against nested practical
controls.  It is post-hoc verification, not a preregistered confirmation.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "src"), HERE]

import dataset as D  # noqa: E402
import probes as SP  # noqa: E402
import probe_suite as R  # noqa: E402


def statewise_shuffle(x, sid, rng):
    out = np.empty_like(x)
    states = np.unique(sid)
    source = rng.permutation(states)
    for dst, src in zip(states, source):
        di, si = np.where(sid == dst)[0], np.where(sid == src)[0]
        # Every Task-C state has six candidates and h_g is state-constant.
        out[di] = x[si[0]]
    return out


def bootstrap_delta(y, pa, pb, sid, doc, n_boot=4000, seed=0):
    """Correct document bootstrap: every sampled copy gets fresh state ids."""
    rng = np.random.default_rng(seed)
    docs = np.unique(doc)
    by_doc = {d: np.where(doc == d)[0] for d in docs}
    def counts(pred, ix):
        dy, dp = R.pair_arrays(y[ix], pred[ix], sid[ix])
        ties = np.abs(dp) <= 1e-12
        correct = np.where(ties, 0.5, np.sign(dy) == np.sign(dp))
        return float(correct.sum()), int(len(correct))

    ca, na, cb, nb = [], [], [], []
    for d in docs:
        a1, a2 = counts(pa, by_doc[d]); b1, b2 = counts(pb, by_doc[d])
        ca.append(a1); na.append(a2); cb.append(b1); nb.append(b2)
    ca, na, cb, nb = map(np.asarray, (ca, na, cb, nb))
    observed = float(ca.sum() / na.sum() - cb.sum() / nb.sum())
    pick = rng.integers(0, len(docs), size=(n_boot, len(docs)))
    vals = (ca[pick].sum(1) / na[pick].sum(1)
            - cb[pick].sum(1) / nb[pick].sum(1))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"observed": float(observed), "bootstrap_mean": float(vals.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "p_le_zero": float(np.mean(vals <= 0)), "n_boot": int(n_boot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taskC")
    ap.add_argument("--tags", nargs="+", default=None,
                    help="one or more label tags; overrides --tag")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--select-layer", action="store_true",
                    help="mirror broad screen: choose layer by validation MSE")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "taskC_crossfit_positive.json"))
    args = ap.parse_args()

    tags = args.tags if args.tags is not None else [args.tag]
    d = D.load_labels(tags)
    y = d["A_task"].astype(np.float32)
    sid, doc = d["state_id"], d["doc_id"]
    xc = D.block(d, "cheap").astype(np.float32)
    docs = np.unique(doc).copy()
    np.random.default_rng(20260901).shuffle(docs)
    folds = np.array_split(docs, args.folds)
    pred = {k: np.full(len(y), np.nan, np.float32) for k in
            ("bilinear", "no_state_interaction", "cheap_only",
             "shuffled_hg", "gaussian_hg")}
    fold_rows = []

    for f, test_docs in enumerate(folds):
        remain = np.concatenate([x for j, x in enumerate(folds) if j != f])
        rng = np.random.default_rng(7000 + f)
        rng.shuffle(remain)
        n_val = max(1, len(remain) // 4)
        val_docs, train_docs = remain[:n_val], remain[n_val:]
        tr = np.where(np.isin(doc, train_docs))[0]
        va = np.where(np.isin(doc, val_docs))[0]
        te = np.where(np.isin(doc, test_docs))[0]
        layer = args.layer
        if args.select_layer:
            vals = []
            for candidate_layer in range(d["n_layers"]):
                local = D.block(d, "H_local", candidate_layer)
                model = SP.fit_linear_2block(
                    xc[tr], local[tr], y[tr], xc[va], local[va], y[va])
                vals.append(model["val_r2"])
            layer = int(np.argmax(vals))
        hi = D.block(d, "H_local", layer)
        hg = D.block(d, "H_global", layer)
        hip, hgp, _, _ = R.pca_pair(hi, hg, tr, 64, 900 + f)
        hzero = np.zeros_like(hgp)
        izero = np.zeros_like(hip)
        shuffled = statewise_shuffle(hgp, sid, rng)
        gaussian = rng.standard_normal(hgp.shape).astype(np.float32)
        # Mirror the broad-screen seed convention exactly: split/fold index.
        cfg = R.SuiteConfig(seed=f, pca_dim=64, epochs=args.epochs,
                            patience=18)
        specs = {
            "bilinear": (hip, hgp),
            "no_state_interaction": (hip, hzero),
            "cheap_only": (izero, hzero),
            "shuffled_hg": (hip, shuffled),
            "gaussian_hg": (hip, gaussian),
        }
        row = {"fold": f, "layer": layer,
               "n_test_docs": int(len(test_docs)), "metrics": {}}
        for name, (ii, gg) in specs.items():
            pp, hp = R.fit_torch_score("bilinear", xc, ii, gg, y, sid,
                                       tr, va, te, cfg, rank=args.rank)
            pred[name][te] = pp
            row["metrics"][name] = R.decision_metrics(y[te], pp, sid[te])
            row.setdefault("hp", {})[name] = hp
        fold_rows.append(row)
        b = row["metrics"]["bilinear"]["pairwise_concordance"]
        c = row["metrics"]["no_state_interaction"]["pairwise_concordance"]
        print(f"fold {f}: bilinear {b:.4f}, no-interaction {c:.4f}, "
              f"delta {b-c:+.4f}", flush=True)

    assert all(np.isfinite(v).all() for v in pred.values())
    overall = {k: R.decision_metrics(y, v, sid) for k, v in pred.items()}
    comparisons = {}
    for base in ("no_state_interaction", "cheap_only", "shuffled_hg",
                 "gaussian_hg"):
        comparisons[f"bilinear_vs_{base}"] = bootstrap_delta(
            y, pred["bilinear"], pred[base], sid, doc,
            n_boot=args.n_boot)
    report = {"status": "post-hoc cross-fitted verification",
              "tags": tags,
              "config": vars(args), "n_rows": int(len(y)),
              "n_states": int(len(np.unique(sid))), "n_docs": int(len(docs)),
              "folds": fold_rows, "overall": overall,
              "comparisons": comparisons}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(comparisons, indent=2), flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

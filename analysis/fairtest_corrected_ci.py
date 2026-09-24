"""Correct document CIs for the post-hoc centered-target fair test.

The exploratory script reuses original state ids when a document is sampled
more than once.  Grouping those duplicate rows as one enlarged state creates
spurious cross-copy candidate pairs.  Here each document contributes fixed
sufficient statistics, and a bootstrap draw only changes document weights.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import dataset as D  # noqa: E402
import probes as P  # noqa: E402


def centre(x, sid):
    z = np.asarray(x, dtype=np.float32).copy()
    for state in np.unique(sid):
        keep = sid == state
        z[keep] -= z[keep].mean(0, keepdims=True)
    return z


def pair_counts(y, pred, sid, idx):
    correct = 0.0
    total = 0
    for state in np.unique(sid[idx]):
        take = idx[sid[idx] == state]
        yy, pp = y[take], pred[take]
        dy = yy[:, None] - yy[None, :]
        dp = pp[:, None] - pp[None, :]
        iu = np.triu_indices(len(take), 1)
        dy, dp = dy[iu], dp[iu]
        rankable = np.abs(dy) > 1e-9
        dy, dp = dy[rankable], dp[rankable]
        ties = np.abs(dp) <= 1e-12
        correct += float((np.sign(dy[~ties]) == np.sign(dp[~ties])).sum()
                         + 0.5 * ties.sum())
        total += int(len(dy))
    return correct, total


def refit(row, n_boot, seed):
    tags, target = row["tags"], row["target"]
    split_seed, layer = int(row["split_seed"]), int(row["layer"])
    d = D.load_labels(tags)
    sid, doc = d["state_id"], d["doc_id"]
    y = d[target].astype(np.float32)
    yc = centre(y[:, None], sid)[:, 0]
    cc = centre(D.block(d, "cheap"), sid)
    hc = centre(D.block(d, "H_local", layer), sid)
    sp = D.doc_splits(d, seed=split_seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]

    cheap = P.fit_linear(cc[tr], yc[tr], cc[va], yc[va])
    hidden = P.fit_linear_2block(
        cc[tr], hc[tr], yc[tr], cc[va], hc[va], yc[va])
    pc = P.predict(cheap, cc[te])
    ph = P.predict_2block(hidden, cc[te], hc[te])
    yte, ycte, ste, dte = y[te], yc[te], sid[te], doc[te]
    observed = {
        "d_r2": float(P.r2_score(ycte, ph) - P.r2_score(ycte, pc)),
        "d_conc": float(P.within_state_concordance(yte, ph, ste)[0]
                        - P.within_state_concordance(yte, pc, ste)[0]),
    }
    # Refit must reproduce the saved point estimates before its CIs are trusted.
    if abs(observed["d_r2"] - row["d_r2"]) > 2e-6:
        raise AssertionError(("d_r2 reproduction failed", observed, row))
    if abs(observed["d_conc"] - row["d_conc"]) > 2e-9:
        raise AssertionError(("d_conc reproduction failed", observed, row))

    docs = np.unique(dte)
    rss_c, rss_h, tss, conc_c, conc_h, pairs = [], [], [], [], [], []
    for value in docs:
        ix = np.where(dte == value)[0]
        rss_c.append(float(((ycte[ix] - pc[ix]) ** 2).sum()))
        rss_h.append(float(((ycte[ix] - ph[ix]) ** 2).sum()))
        tss.append(float((ycte[ix] ** 2).sum()))
        c0, n0 = pair_counts(yte, pc, ste, ix)
        c1, n1 = pair_counts(yte, ph, ste, ix)
        if n0 != n1:
            raise AssertionError("pair denominators differ")
        conc_c.append(c0); conc_h.append(c1); pairs.append(n0)
    rss_c, rss_h, tss, conc_c, conc_h, pairs = map(
        np.asarray, (rss_c, rss_h, tss, conc_c, conc_h, pairs))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(docs), size=(n_boot, len(docs)))
    br2 = ((rss_c[pick].sum(1) - rss_h[pick].sum(1))
           / tss[pick].sum(1))
    bconc = ((conc_h[pick].sum(1) - conc_c[pick].sum(1))
             / pairs[pick].sum(1))
    return {
        **{k: row[k] for k in ("arm", "tags", "target", "split_seed",
                                "layer", "ceiling")},
        "observed": observed,
        "hidden_gamma": float(hidden["gamma"]),
        "hidden_alpha": float(hidden["alpha"]),
        "corrected_ci_r2": list(map(float, np.percentile(br2, [2.5, 97.5]))),
        "corrected_ci_conc": list(map(
            float, np.percentile(bconc, [2.5, 97.5]))),
        "bootstrap_mean_r2": float(br2.mean()),
        "bootstrap_mean_conc": float(bconc.mean()),
        "n_test_docs": int(len(docs)),
        "n_boot": int(n_boot),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(ROOT, "data",
                                                      "A_fairtest.json"))
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "A_fairtest_corrected_ci.json"))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    rows = json.load(open(args.source))
    expected = {(arm, target, seed) for arm in
                ("arm1 ancestral", "arm2 confidence") for target in
                ("A_pertok", "A_future") for seed in range(3)}
    actual = {(r.get("arm"), r.get("target"), r.get("split_seed")) for r in rows}
    if actual != expected:
        raise AssertionError(f"fair-test grid incomplete: missing={expected-actual}, "
                             f"extra={actual-expected}")
    out = [refit(row, args.n_boot, 8100 + i) for i, row in enumerate(rows)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("wrote", args.out)
    for row in out:
        print(row["arm"], row["target"], row["split_seed"], row["observed"],
              row["corrected_ci_conc"])


if __name__ == "__main__":
    main()

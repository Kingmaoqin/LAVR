"""Apply sealed Task-C models once to Task-D and execute preregistered verdict."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "src"), HERE]

import dataset as D  # noqa: E402
import probe_suite as R  # noqa: E402
from taskC_crossfit_positive import bootstrap_delta  # noqa: E402


def predict(path, xc, hi_raw, hg_raw, mode):
    z = np.load(path)
    hi = ((hi_raw - z["pca_i_mean"]) @ z["pca_i_components"].T
          ).astype(np.float32)
    hg = ((hg_raw - z["pca_g_mean"]) @ z["pca_g_components"].T
          ).astype(np.float32)
    if mode == "no_state_interaction":
        hg = np.zeros_like(hg)
    elif mode == "cheap_only":
        hi, hg = np.zeros_like(hi), np.zeros_like(hg)
    blocks = [xc, hi, hg]
    norm = [((x - z[f"norm_mu_{i}"]) / z[f"norm_sd_{i}"]).astype(np.float32)
            for i, x in enumerate(blocks)]
    c, i, g = norm
    ui = i @ z["state_ui__weight"].T
    ug = g @ z["state_ug__weight"].T
    linear = (np.concatenate([c, i, g], 1) @ z["state_lin__weight"].T
              + z["state_lin__bias"]).ravel()
    raw = (ui * ug).sum(1) + linear
    return raw * float(z["ys"]) + float(z["ym"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taskD")
    ap.add_argument("--sealed", default=os.path.join(
        HERE, "sealed_taskD_primary"))
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "CONFIRMATORY_taskD_frozen.json"))
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.sealed, "SEALED.json")))
    layer = int(meta["layer"])
    d = D.load_labels([args.tag])
    old = D.load_labels(["taskC"])
    overlap = set(np.unique(d["doc_id"])) & set(np.unique(old["doc_id"]))
    if overlap:
        raise AssertionError(f"Task-C/D prompt overlap: {sorted(overlap)[:5]}")
    xc = D.block(d, "cheap").astype(np.float32)
    hi, hg = D.block(d, "H_local", layer), D.block(d, "H_global", layer)
    y, sid, doc = d["A_task"], d["state_id"], d["doc_id"]
    preds = {name: predict(os.path.join(args.sealed, name + ".npz"),
                           xc, hi, hg, name)
             for name in ("bilinear", "no_state_interaction", "cheap_only")}
    metrics = {k: R.decision_metrics(y, p, sid) for k, p in preds.items()}
    comparisons = {
        "primary_bilinear_vs_cheap_only": bootstrap_delta(
            y, preds["bilinear"], preds["cheap_only"], sid, doc,
            n_boot=10000, seed=31),
        "relational_bilinear_vs_no_state_interaction": bootstrap_delta(
            y, preds["bilinear"], preds["no_state_interaction"], sid, doc,
            n_boot=10000, seed=32)}
    subgroup = {}
    for field in ("prompt_stratum", "stratum"):
        subgroup[field] = {}
        for label, value in (("natural", 0), ("informative", 1)):
            keep = d[field] == value
            subgroup[field][label] = {
                k: R.decision_metrics(y[keep], p[keep], sid[keep])
                for k, p in preds.items()}
    p = comparisons["primary_bilinear_vs_cheap_only"]
    q = comparisons["relational_bilinear_vs_no_state_interaction"]
    verdict = {
        "primary_positive": bool(p["observed"] >= 0.020 and p["ci_lo"] > 0),
        "relational_positive": bool(q["observed"] >= 0.010 and q["ci_lo"] > 0)}
    report = {"status": "confirmatory frozen application", "config": vars(args),
              "sealed_metadata": meta, "n_rows": int(len(y)),
              "n_states": int(len(np.unique(sid))),
              "n_docs": int(len(np.unique(doc))), "prompt_overlap_taskC": 0,
              "metrics": metrics, "comparisons": comparisons,
              "subgroup_metrics": subgroup, "verdict": verdict}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps({"comparisons": comparisons, "verdict": verdict},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

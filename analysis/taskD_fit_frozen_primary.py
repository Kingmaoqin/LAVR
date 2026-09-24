"""Fit and seal the preregistered Task-D frozen primary on Task-C only."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "src"), HERE]

import dataset as D  # noqa: E402
import probes as P  # noqa: E402
import probe_suite as R  # noqa: E402


def save_artifact(path, artifact, pca_i, pca_g, extra):
    arrays = {
        "pca_i_components": pca_i.components_.astype(np.float32),
        "pca_i_mean": pca_i.mean_.astype(np.float32),
        "pca_g_components": pca_g.components_.astype(np.float32),
        "pca_g_mean": pca_g.mean_.astype(np.float32),
        "ym": np.array(artifact["ym"], np.float32),
        "ys": np.array(artifact["ys"], np.float32),
    }
    for i, (mu, sd) in enumerate(zip(artifact["norm_mu"], artifact["norm_sd"])):
        arrays[f"norm_mu_{i}"] = mu
        arrays[f"norm_sd_{i}"] = sd
    for key, value in artifact["state_dict"].items():
        arrays["state_" + key.replace(".", "__")] = value
    np.savez_compressed(path, **arrays)
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump({**extra, "hp": artifact["hp"]}, f, indent=2, default=float)


def main():
    d = D.load_labels(["taskC"])
    y, sid = d["A_task"].astype(np.float32), d["state_id"]
    sp = D.doc_splits(d, 0)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    xc = D.block(d, "cheap").astype(np.float32)
    vals = []
    for layer in range(d["n_layers"]):
        hi = D.block(d, "H_local", layer)
        model = P.fit_linear_2block(
            xc[tr], hi[tr], y[tr], xc[va], hi[va], y[va])
        vals.append(model["val_r2"])
    layer = int(np.argmax(vals))
    hi = D.block(d, "H_local", layer)
    hg = D.block(d, "H_global", layer)
    hip, hgp, pi, pg = R.pca_pair(hi, hg, tr, 64, 0)
    zeros_i, zeros_g = np.zeros_like(hip), np.zeros_like(hgp)
    cfg = R.SuiteConfig(seed=0, pca_dim=64, epochs=25, patience=8)
    specs = {"bilinear": (hip, hgp),
             "no_state_interaction": (hip, zeros_g),
             "cheap_only": (zeros_i, zeros_g)}
    outdir = os.path.join(HERE, "sealed_taskD_primary")
    os.makedirs(outdir, exist_ok=True)
    summary = {"source": "taskC only", "target": "A_task", "layer": layer,
               "rank": 4, "split_seed": 0, "validation_layer_scores": vals,
               "models": {}}
    for name, (ii, gg) in specs.items():
        pred, hp, artifact = R.fit_torch_score(
            "bilinear", xc, ii, gg, y, sid, tr, va, te, cfg, rank=4,
            return_artifact=True)
        metrics = R.decision_metrics(y[te], pred, sid[te])
        path = os.path.join(outdir, name + ".npz")
        save_artifact(path, artifact, pi, pg,
                      {"name": name, "layer": layer, "rank": 4,
                       "taskC_internal_test_metrics": metrics})
        summary["models"][name] = {"artifact": os.path.basename(path),
                                    "hp": hp, "metrics": metrics}
        print(name, metrics["pairwise_concordance"], flush=True)
    with open(os.path.join(outdir, "SEALED.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("sealed", outdir, flush=True)


if __name__ == "__main__":
    main()

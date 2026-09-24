"""
Layer x timestep x horizon analysis.

For the three horizon arms (H=8 / 16 / 32, all MDLM ancestral) and each timestep bin,
select the layer on validation, then report:
        dconcordance / dtop1 / dwithin-R2   (cheap + additive hidden block  vs  cheap)
        and the relational increment          (kron  vs  additive, same PCA dims)

Not only the maximum: mean / median / fraction of positive cells / per-bin details are reported.
(Taking the max over 13 layers is a selection-biased statistic; layers are selected on validation instead.)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import probes2 as P            # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

ARMS = {"H8": (["h8a", "h8b"], 8), "H16": (["a3", "b3"], 16),
        "H32": (["h32a", "h32b"], 32)}


def subset(d, mask):
    n = d["step"].shape[0]
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def eval_bin(d, layer_list, target, kron_d, pca_dim, criterion):
    """On a given subset: select layer on validation -> evaluate cheap / additive / kron on test."""
    sp = RD.doc_splits(d, seed=0)
    if min(len(sp[k]) for k in sp) < 60:
        return None
    _, groups, _ = RD.state_groups(d["state_id"])
    sid = SC.split_sid(d, sp)
    y = SC.split_targets(d, sp, target)
    sel = P.make_selector(criterion, sid["val"])
    best = None
    cache = {}
    for L in layer_list:
        prep = SC.prepare(d, sp, L, pca_dim=pca_dim, groups=groups)
        Zi = {k: prep["pca"]["hi"][k][:, :kron_d] for k in
              ("train", "val", "test")}
        Zg = {k: prep["pca"]["hg"][k][:, :kron_d] for k in
              ("train", "val", "test")}
        Xa = {k: np.concatenate([Zi[k], Zg[k]], 1) for k in Zi}
        m = P.fit_ridge_2block(prep["pca"]["cheap"]["train"], Xa["train"],
                               y["train"], prep["pca"]["cheap"]["val"],
                               Xa["val"], y["val"],
                               prep["pca"]["cheap"]["test"], Xa["test"], sel)
        cache[L] = (prep, Zi, Zg, Xa, m)
        if best is None or m["val_score"] > best[1]:
            best = (L, m["val_score"])
    L = best[0]
    prep, Zi, Zg, Xa, m_add = cache[L]
    KR = {k: (Zi[k][:, :, None] * Zg[k][:, None, :]
              ).reshape(len(Zi[k]), -1).astype(np.float32) for k in Zi}
    Xk = {k: np.concatenate([Zi[k], Zg[k], KR[k]], 1) for k in Zi}
    m_kr = P.fit_ridge_2block(prep["pca"]["cheap"]["train"], Xk["train"],
                              y["train"], prep["pca"]["cheap"]["val"],
                              Xk["val"], y["val"],
                              prep["pca"]["cheap"]["test"], Xk["test"], sel)
    m_ch = P.fit_ridge(prep["raw"]["cheap"]["train"], y["train"],
                       prep["raw"]["cheap"]["val"], y["val"],
                       prep["raw"]["cheap"]["test"], sel)
    _, g_te = M.group_slices(sid["test"])
    r = {}
    for nm, mm in (("cheap", m_ch), ("additive", m_add), ("kron", m_kr)):
        r[nm] = M.full_report(y["test"], mm["pred_test"], sid["test"], g_te)
    return {"layer": int(L), "n_test": int(len(y["test"])),
            "n_test_states": int(len(g_te)), "scores": r,
            "delta_hidden": {k: r["additive"][k] - r["cheap"][k]
                             for k in ("concordance", "top1", "within_r2",
                                       "regret_norm_mean")},
            "delta_relational": {k: r["kron"][k] - r["additive"][k]
                                 for k in ("concordance", "top1", "within_r2",
                                           "regret_norm_mean")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="A_pertok")
    ap.add_argument("--criterion", default="within_r2")
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--kron_d", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "horizon_timestep_layer.json"))
    args = ap.parse_args()
    t0 = time.time()
    rep = {"config": vars(args), "arms": {}}

    for arm, (tags, H) in ARMS.items():
        d = RD.load_labels(tags)
        nL = d["n_layers"]
        layers = list(range(nL))
        steps = sorted(np.unique(d["step"]).tolist())
        n_steps = 256 - 64
        ent = {"H": H, "tags": tags, "n": int(len(d[args.target])),
               "bins": {}}
        sk = "A_full_seeds" if args.target == "A_pertok" else "A_future_seeds"
        if sk in d:
            _, g_all, _ = RD.state_groups(d["state_id"])
            ent["ceiling"] = RD.noise_ceiling(d[sk])
            ent["within_ceiling"] = RD.within_state_noise_ceiling(
                d[sk], d["state_id"], g_all)
        ent["pooled"] = eval_bin(d, layers, args.target, args.kron_d,
                                 args.pca_dim, args.criterion)
        print(f"[{arm}] pooled: layer {ent['pooled']['layer']} "
              f"Dconc(hidden) {ent['pooled']['delta_hidden']['concordance']:+.4f} "
              f"Dconc(relational) "
              f"{ent['pooled']['delta_relational']['concordance']:+.4f}",
              flush=True)
        for st in steps:
            sub = subset(d, d["step"] == st)
            r = eval_bin(sub, layers, args.target, args.kron_d, args.pca_dim,
                         args.criterion)
            if r is None:
                continue
            ent["bins"][str(st)] = {"progress": st / n_steps, **r}
            print(f"  [{arm}] t/T={st/n_steps:.2f} layer {r['layer']:2d} "
                  f"Dconc_h {r['delta_hidden']['concordance']:+.4f}  "
                  f"Dtop1_h {r['delta_hidden']['top1']:+.4f}  "
                  f"Dconc_rel {r['delta_relational']['concordance']:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        # summary (not only the maximum)
        dh = [b["delta_hidden"]["concordance"] for b in ent["bins"].values()]
        dr = [b["delta_relational"]["concordance"]
              for b in ent["bins"].values()]
        ent["summary"] = {
            "hidden_delta_conc": {"mean": float(np.mean(dh)),
                                  "median": float(np.median(dh)),
                                  "positive_frac": float(np.mean(np.array(dh) > 0)),
                                  "max": float(np.max(dh)),
                                  "n_bins": len(dh)},
            "relational_delta_conc": {"mean": float(np.mean(dr)),
                                      "median": float(np.median(dr)),
                                      "positive_frac": float(np.mean(np.array(dr) > 0)),
                                      "max": float(np.max(dr))}}
        rep["arms"][arm] = ent
        json.dump(rep, open(args.out, "w"), indent=2, default=float)

    print(f"\ndone in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()

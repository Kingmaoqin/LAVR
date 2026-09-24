"""
Are the Path-LL advantage and the task advantage the same thing?

The Nemotron-3B task-utility collection stores two labels:
        A_pertok  = Path-LL advantage (proxy reward)
        A_task    = P(correct | s,a,pi) - P(correct | s,pi)   (verifiable task reward)

If the two are weakly correlated, conclusions about A_PathLL readouts **cannot** be extrapolated to A_task.
This bounds the scope of any Path-LL conclusion, so it is quantified directly here.

Also reports the probe-family screen on A_task (including relational probes) and the A_task noise ceiling.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rlib import metrics as M            # noqa: E402
from rlib import rdata as RD             # noqa: E402
from rlib import screen as SC            # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["taskA", "taskB"])
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "task_utility_analysis.json"))
    ap.add_argument("--skip-screen", action="store_true",
                    help="only compute label quality and Path-LL/task relations")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    d = RD.load_labels(args.tags)
    sid = d["state_id"]
    uniq, groups, _ = RD.state_groups(sid)
    sizes = np.array([len(g) for g in groups])
    n = len(d["A_task"])
    rep = {"n_rows": int(n), "n_states": int(len(uniq)),
           "n_prompts": int(len(np.unique(d["doc_id"]))),
           "group_sizes": {int(k): int(v) for k, v in
                           zip(*np.unique(sizes, return_counts=True))},
           "n_layers": int(d["H_i"].shape[1]),
           "hidden_dim": int(d["H_i"].shape[2])}
    print(f"[task] {n} rows / {len(uniq)} states / {rep['n_prompts']} prompts, "
          f"group sizes {rep['group_sizes']}", flush=True)

    # ---------- 1. relation between the two advantages ----------
    a_ll, a_tk = d["A_pertok"].astype(np.float64), d["A_task"].astype(np.float64)
    keep = np.isfinite(a_ll) & np.isfinite(a_tk)
    rep["corr_pooled"] = {
        "pearson": float(np.corrcoef(a_ll[keep], a_tk[keep])[0, 1]),
        "spearman": M.pooled_spearman(a_tk[keep], a_ll[keep])}
    wl = RD.center_within_state(a_ll, sid, groups)
    wt = RD.center_within_state(a_tk, sid, groups)
    m2 = np.isfinite(wl) & np.isfinite(wt) & (np.abs(wt) + np.abs(wl) > 0)
    rep["corr_within_state"] = {
        "pearson": float(np.corrcoef(wl[m2], wt[m2])[0, 1]),
        "spearman": M.pooled_spearman(wt[m2], wl[m2]),
        "n": int(m2.sum())}
    # how much within-state ranking A_PathLL achieves as a "predictor" of A_task
    rep["pathll_as_predictor_of_task"] = {
        "concordance": M.concordance(a_tk, a_ll, sid, groups),
        "within_r2": M.within_state_r2(a_tk, a_ll, sid, groups),
        **M.topk_metrics(a_tk, a_ll, sid, groups)}
    print(f"[task] corr(A_PathLL, A_task) pooled "
          f"{rep['corr_pooled']['pearson']:+.3f}  within-state "
          f"{rep['corr_within_state']['pearson']:+.3f}", flush=True)
    print(f"[task] Path-LL as A_task ranker: concordance "
          f"{rep['pathll_as_predictor_of_task']['concordance']:.4f}", flush=True)

    # ---------- 2. label quality ----------
    for nm, sk in (("A_task", "A_task_seeds"), ("A_PathLL", "A_full_seeds")):
        if sk in d:
            rep[f"ceiling_{nm}"] = RD.noise_ceiling(d[sk])
            rep[f"within_ceiling_{nm}"] = RD.within_state_noise_ceiling(
                d[sk], sid, groups)
            print(f"[task] {nm}: ceiling {rep[f'ceiling_{nm}']['ceiling']:.3f} "
                  f"SNR {rep[f'ceiling_{nm}']['snr']:.2f} | within ceiling "
                  f"{rep[f'within_ceiling_{nm}']['ceiling']:.3f} "
                  f"SNR {rep[f'within_ceiling_{nm}']['snr']:.2f}", flush=True)
    rep["var_decomp_A_task"] = {
        "within_frac": float((wt ** 2).sum()
                             / max(((a_tk - a_tk.mean()) ** 2).sum(), 1e-12))}

    if args.skip_screen:
        json.dump(rep, open(args.out, "w"), indent=2, default=float)
        print("\nwrote", args.out)
        return

    # ---------- 3. probe family on A_task ----------
    # with unequal group sizes torch ranking probes cannot reshape, so only the closed-form ridge family + Kronecker is run
    ragged = len(rep["group_sizes"]) > 1
    which = ["P0", "P2k", "P13"] if ragged else ["P0", "P2k", "P7k", "P13"]
    rep["probe_screen"] = {}
    sp = RD.doc_splits(d, seed=0)
    RD.check_split_disjoint(d, sp)
    doc = {k: d["doc_id"][v] for k, v in sp.items()}
    s_ = SC.split_sid(d, sp)
    nL = d["H_i"].shape[1]

    # prepare each layer once (PCA on 3072 dims is expensive); shared by both targets and both criteria
    preps = {}
    for L in range(nL):
        preps[L] = SC.prepare(d, sp, L, pca_dim=64, groups=groups)
    print(f"  prepared {nL} layers", flush=True)

    for tgt in ("A_task", "A_pertok"):
        rep["probe_screen"][tgt] = {}
        y = SC.split_targets(d, sp, tgt)
        for crit in ("pooled_r2", "within_r2"):
            best = None
            for L in range(nL):
                R = SC.run_probes(preps[L], y, s_, doc, crit, which=["P0"],
                                  kron_dims=(16,))
                v = R["P0_cheap+H"]["val_score"]
                if best is None or v > best[1]:
                    best = (L, v)
            L = best[0]
            R = SC.run_probes(preps[L], y, s_, doc, crit, which=which,
                              kron_dims=(16, 32), seeds=(0, 1, 2), epochs=300)
            sc = SC.score_all(R, y["test"], s_["test"])
            rep["probe_screen"][tgt][crit] = {"layer": int(L), "probes": sc}
            base = sc.get("P0_cheap", {})
            print(f"\n[task] target={tgt} crit={crit} layer={L}", flush=True)
            for k, v in sorted(sc.items(),
                               key=lambda kv: -(kv[1].get("concordance") or 0)):
                print(f"   {k:22s} within={v['within_r2']:+.4f} "
                      f"conc={v['concordance']:.4f} "
                      f"Dconc={v['concordance']-base.get('concordance',0):+.4f}",
                      flush=True)

    json.dump(rep, open(args.out, "w"), indent=2, default=float)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

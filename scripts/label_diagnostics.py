"""
Deliverable 2 -- oracle-label definition check and rollout-variance diagnostics.

Answers, from the collected labels alone (no probe involved):
  * what does A^{pi_ref} actually look like, and is it non-degenerate?
  * how much of its observed spread is real signal vs. rollout noise?
  * how much did CRN pairing and Rao-Blackwellisation actually buy?
  * does the label vary by diffusion timestep and candidate stratum?
  * is the label already trivially explained by the action's own log-prob?
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import dataset as D                                                # noqa: E402
import probes as P                                                 # noqa: E402


def snr_of(seeds):
    ybar = seeds.mean(1)
    K = seeds.shape[1]
    noise = float((seeds.var(1, ddof=1) / K).mean())
    obs = float(ybar.var())
    return obs, noise, (obs - noise) / noise if noise > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["a", "b"])
    args = ap.parse_args()
    d = D.load_labels(args.tags)
    n = len(d["A_pertok"])
    rep = {"n_examples": int(n),
           "n_docs": int(len(np.unique(d["doc_id"]))),
           "n_states": int(len(np.unique(d["state_id"])))}
    print(f"=== oracle-label diagnostics: {n} examples, {rep['n_docs']} docs, "
          f"{rep['n_states']} states ===")
    print("all labels are POLICY-RELATIVE A^{pi_ref}; never A*")

    print(f"\n-- label distributions (per-token Path-LL advantage) --")
    print(f"{'label':<26}{'mean':>10}{'sd':>9}{'p5':>9}{'p95':>9}")
    for nm in ["A_pertok", "A_future", "mc_A_pertok", "mc_A_future"]:
        v = d[nm]
        print(f"{nm:<26}{v.mean():>+10.4f}{v.std():>9.4f}"
              f"{np.percentile(v,5):>+9.4f}{np.percentile(v,95):>+9.4f}")
        rep[f"dist_{nm}"] = {"mean": float(v.mean()), "sd": float(v.std())}

    print(f"\n-- rollout variance: how much of the spread is real? --")
    print(f"{'label':<26}{'obs var':>11}{'noise var':>11}{'SNR':>8}"
          f"{'R2 ceiling':>12}")
    for nm, key in [("A_pertok (RB)", "A_full_seeds"),
                    ("A_future (RB)", "A_future_seeds"),
                    ("A_pertok (naive MC)", "mc_A_full_seeds"),
                    ("A_future (naive MC)", "mc_A_future_seeds")]:
        obs, noise, snr = snr_of(d[key])
        ceil = max(0.0, (obs - noise) / max(obs, 1e-12))
        print(f"{nm:<26}{obs:>11.2e}{noise:>11.2e}{snr:>8.2f}{ceil:>12.3f}")
        rep[f"snr_{key}"] = {"obs_var": obs, "noise_var": noise,
                             "snr": snr, "r2_ceiling": ceil}

    print(f"\n-- what the two variance reductions bought --")
    _, n_rb, _ = snr_of(d["A_full_seeds"])
    _, n_mc, _ = snr_of(d["mc_A_full_seeds"])
    print(f"   Rao-Blackwellisation: label noise variance {n_mc:.2e} -> "
          f"{n_rb:.2e}   ({n_mc/max(n_rb,1e-30):.1f}x reduction)")
    qsd, vsd = d["mc_Q_sd"].mean(), d["mc_V_sd"].mean()
    indep = np.sqrt(qsd ** 2 + vsd ** 2)
    paired = d["mc_A_full_seeds"].std(1).mean()
    print(f"   CRN pairing (MC estimator): sd(Q-V) would be {indep:.4f} if the "
          f"branches were independent; paired it is {paired:.4f}   "
          f"({(indep/max(paired,1e-9))**2:.1f}x variance reduction)")
    rep["rb_noise_reduction"] = float(n_mc / max(n_rb, 1e-30))
    rep["crn_variance_reduction"] = float((indep / max(paired, 1e-9)) ** 2)

    print(f"\n-- by diffusion timestep --")
    print(f"{'step':>7}{'frac':>7}{'n':>7}{'mean A':>10}{'sd A':>9}{'SNR':>7}")
    by_step = {}
    for s in np.unique(d["step"]):
        m = d["step"] == s
        obs, noise, snr = snr_of(d["A_full_seeds"][m])
        print(f"{s:>7}{s/192:>7.2f}{int(m.sum()):>7}{d['A_pertok'][m].mean():>+10.4f}"
              f"{d['A_pertok'][m].std():>9.4f}{snr:>7.2f}")
        by_step[int(s)] = {"n": int(m.sum()), "mean": float(d["A_pertok"][m].mean()),
                           "sd": float(d["A_pertok"][m].std()), "snr": float(snr)}
    rep["by_step"] = by_step

    print(f"\n-- by candidate stratum --")
    for s, nm in [(0, "natural (uniform masked)"), (1, "informative (oversampled)")]:
        m = d["stratum"] == s
        if not m.any():
            continue
        obs, noise, snr = snr_of(d["A_full_seeds"][m])
        print(f"   {nm:<32} n={int(m.sum()):<6} sd={d['A_pertok'][m].std():.4f}"
              f"  SNR={snr:.2f}")

    print(f"\n-- is the label trivially the action's own log-probability? --")
    for nm in ["A_pertok", "A_future"]:
        r = float(np.corrcoef(d[nm], d["logp_action"])[0, 1])
        rho = P.spearman(d[nm], d["logp_action"])
        print(f"   corr({nm}, logp_action) = {r:+.4f}   spearman {rho:+.4f}")
        rep[f"corr_logp_action_{nm}"] = r

    print(f"\n-- consistency between the two Path-LL estimators --")
    for a, b in [("A_pertok", "mc_A_pertok"), ("A_future", "mc_A_future")]:
        print(f"   corr({a}, {b}) = {np.corrcoef(d[a], d[b])[0,1]:+.4f}  "
              f"(attenuated by the MC estimator's own noise)")

    out = os.path.join(ROOT, "results", "label_diagnostics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

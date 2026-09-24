"""
Self-test of the Experiment-1 analysis machinery on SYNTHETIC data of the same
shape as the real label set.

Three regimes, each with a known ground truth:
  NULL      label depends only on the cheap controls  -> Delta_R2 must be ~0
  HIDDEN    label has a real linear component in h_i  -> Delta_R2 must be > 0
            and, with label noise injected to match the real SNR, must stay
            detectable
  MLP-ONLY  label depends on a NONLINEAR function of h -> the linear probe must
            NOT claim it (guards the "linear representation" claim)

If the NULL regime yields a positive Delta_R2, the pipeline leaks and no real
result can be trusted.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import dataset as D                                                # noqa: E402
import probes as P                                                 # noqa: E402

N_DOCS, N_STEPS, N_CAND, DIM, NLAY = 400, 6, 6, 768, 13


def synth(regime, snr=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = N_DOCS * N_STEPS * N_CAND
    doc = np.repeat(np.arange(N_DOCS), N_STEPS * N_CAND)
    step = np.tile(np.repeat(np.arange(N_STEPS), N_CAND), N_DOCS)
    H_i = rng.standard_normal((n, NLAY, DIM)).astype(np.float16)
    H_g = np.repeat(rng.standard_normal((N_DOCS * N_STEPS, NLAY, DIM)),
                    N_CAND, 0).astype(np.float16)      # constant within a state
    C1 = rng.standard_normal((n, 8)).astype(np.float32)
    C2 = rng.standard_normal((n, 12)).astype(np.float32)
    C3 = rng.standard_normal((n, 80)).astype(np.float32)

    LAYER = 8
    w = rng.standard_normal(DIM) / np.sqrt(DIM)
    h = H_i[:, LAYER].astype(np.float32) @ w
    cheap_sig = 0.8 * C1[:, 0] + 0.5 * C2[:, 3] - 0.4 * C3[:, 2]

    if regime == "null":
        y = cheap_sig
    elif regime == "hidden":
        y = cheap_sig + 1.0 * h
    elif regime == "nonlinear":
        y = cheap_sig + 1.0 * np.tanh(3 * h) * np.sign(C1[:, 1])
    y = (y - y.mean()) / y.std()

    K = 8
    noise_sd = np.sqrt(1.0 / snr)
    seeds_arr = (y[:, None]
                 + rng.standard_normal((n, K)) * noise_sd * np.sqrt(K))
    y_obs = seeds_arr.mean(1)
    return {"doc_id": doc, "prompt_row": doc, "step": step,
            "state_id": doc * 10_000 + step, "stratum": np.zeros(n, np.int8),
            "H_i": H_i, "H_g": H_g, "C1": C1, "C2": C2, "C3": C3,
            "A_pertok": y_obs.astype(np.float32),
            "A_full_seeds": seeds_arr.astype(np.float32),
            "n_layers": NLAY, "_clean": y}


def run(regime, snr=3.0):
    d = synth(regime, snr)
    sp = D.doc_splits(d, seed=0)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    y = d["A_pertok"]

    Xc = D.block(d, "cheap")
    val = []
    for l in range(NLAY):
        Xh = D.block(d, "H", l)
        val.append(P.fit_linear_2block(Xc[tr], Xh[tr], y[tr],
                                       Xc[va], Xh[va], y[va])["val_r2"])
    L = int(np.argmax(val))

    out = {}
    Xc = D.block(d, "cheap")
    Xh = D.block(d, "H", L)
    m = P.fit_linear(Xc[tr], y[tr], Xc[va], y[va])
    pc = P.predict(m, Xc[te])
    out["cheap"] = (P.r2_score(y[te], pc),
                    P.within_state_concordance(y[te], pc, d["state_id"][te])[0])
    m2 = P.fit_linear_2block(Xc[tr], Xh[tr], y[tr], Xc[va], Xh[va], y[va])
    ph = P.predict_2block(m2, Xc[te], Xh[te])
    out["cheap+H"] = (P.r2_score(y[te], ph),
                      P.within_state_concordance(y[te], ph, d["state_id"][te])[0])
    out["gamma"] = m2["gamma"]
    X = D.block(d, "cheap+H", L)
    mm = P.fit_mlp(X[tr], y[tr], X[va], y[va])
    pm = P.predict_mlp(mm, X[te])
    out["mlp"] = (P.r2_score(y[te], pm),
                  P.within_state_concordance(y[te], pm, d["state_id"][te])[0])

    ceil, _, _ = P.noise_ceiling(d["A_full_seeds"])
    d_r2 = out["cheap+H"][0] - out["cheap"][0]
    d_cc = out["cheap+H"][1] - out["cheap"][1]
    print(f"{regime:<11}{L:>4}{ceil:>9.3f}{out['cheap'][0]:>10.4f}"
          f"{out['cheap+H'][0]:>11.4f}{d_r2:>+10.4f}{d_cc:>+10.4f}"
          f"{out['mlp'][0]:>10.4f}{out['gamma']:>8.3f}")
    return d_r2, d_cc, out


if __name__ == "__main__":
    print("synthetic self-test: 400 docs x 6 states x 6 candidates, "
          "768-d hidden, 13 layers, true layer = 8, label SNR = 3\n")
    print(f"{'regime':<11}{'lay':>4}{'ceiling':>9}{'R2cheap':>10}"
          f"{'R2cheap+H':>11}{'dR2':>10}{'dConc':>10}{'R2 mlp':>10}{'gamma':>8}")
    r_null = run("null")
    r_hid = run("hidden")
    r_nl = run("nonlinear")
    print("\nEXPECTATIONS")
    ok1 = abs(r_null[0]) < 0.01
    ok2 = r_hid[0] > 0.05 and r_hid[1] > 0.02
    ok3 = r_nl[0] < r_hid[0]
    print(f"  null   : |dR2| < 0.01                      -> "
          f"{'PASS' if ok1 else 'FAIL'}  ({r_null[0]:+.4f})")
    print(f"  hidden : dR2 > 0.05 and dConc > 0.02       -> "
          f"{'PASS' if ok2 else 'FAIL'}  ({r_hid[0]:+.4f}, {r_hid[1]:+.4f})")
    print(f"  nonlin : linear probe recovers less than in `hidden` -> "
          f"{'PASS' if ok3 else 'FAIL'}  ({r_nl[0]:+.4f} < {r_hid[0]:+.4f})")
    print("\n" + ("ANALYSIS PIPELINE VALIDATED" if (ok1 and ok2 and ok3)
                  else "PIPELINE SELF-TEST FAILED -- do not trust real results"))

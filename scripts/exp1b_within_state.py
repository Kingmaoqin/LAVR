"""
Where does the incremental Delta_R2 actually come from?

Experiment 1 finds a real Delta_R2 for the POKE Path-LL advantage, but the
block table hints that it is a STATE-level effect: h_global (constant across the
candidates of a state) alone reaches R^2 0.31, while h_i alone reaches
within-state R^2 0.013. This script separates the two cleanly, layer by layer
and timestep bin by timestep bin:

  * `Delta within-R2 (cheap + h_i  vs  cheap)` -- can the CANDIDATE-SPECIFIC
    representation rank the candidates of a state better than the exposed
    signals? This is the decision a scheduler faces, and h_global is excluded
    so nothing can leak in through a state-level baseline correction.
  * `Delta R2 (cheap + h_global  vs  cheap)` -- the state-level channel.

If the first is ~0 everywhere while the second is large, the honest reading is
that the frozen state exposes the state VALUE, not the action ADVANTAGE.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import dataset as D                                                # noqa: E402
import probes as P                                                 # noqa: E402


def fit_eval(sub, sp, hidden, target, layer):
    y = sub[target]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    Xc = D.block(sub, "cheap")
    if hidden is None:
        m = P.fit_linear(Xc[tr], y[tr], Xc[va], y[va])
        pred = P.predict(m, Xc[te])
    else:
        Xh = D.block(sub, hidden, layer)
        m = P.fit_linear_2block(Xc[tr], Xh[tr], y[tr], Xc[va], Xh[va], y[va])
        pred = P.predict_2block(m, Xc[te], Xh[te])
    sid = sub["state_id"][te]
    return (P.r2_score(y[te], pred),
            P.within_state_r2(y[te], pred, sid),
            P.within_state_concordance(y[te], pred, sid)[0])


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "A_pertok"
    tags = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["a", "b"])
    d = D.load_labels(tags)
    print(f"[exp1b] tags={tags}")
    nL = d["n_layers"]
    steps = sorted(np.unique(d["step"]).tolist())
    print(f"[exp1b] {len(d[target])} examples, target={target}\n")

    hw = np.full((nL, len(steps)), np.nan)     # within-R2 gain from h_i
    hc = np.full((nL, len(steps)), np.nan)     # concordance gain from h_i
    hg = np.full((nL, len(steps)), np.nan)     # R2 gain from h_global
    for bi, st in enumerate(steps):
        m = d["step"] == st
        sub = {k: (v[m] if isinstance(v, np.ndarray)
                   and v.shape[:1] == d["step"].shape else v)
               for k, v in d.items()}
        sp = D.doc_splits(sub, seed=0)
        b_r2, b_w, b_c = fit_eval(sub, sp, None, target, 0)
        for l in range(nL):
            _, w, c = fit_eval(sub, sp, "H_local", target, l)
            g_r2, _, _ = fit_eval(sub, sp, "H_global", target, l)
            hw[l, bi], hc[l, bi], hg[l, bi] = w - b_w, c - b_c, g_r2 - b_r2
        print(f"  bin t/T={st/192:.2f} done", flush=True)

    def show(M, name):
        print(f"\n{name}")
        print("layer " + "".join(f"{s/192:>8.2f}" for s in steps))
        for l in range(nL):
            print(f"{l:>5} " + "".join(f"{M[l,b]:>8.3f}" for b in range(len(steps))))
        print(f"  max over the whole grid: {np.nanmax(M):+.4f} "
              f"at layer {np.unravel_index(np.nanargmax(M), M.shape)[0]}, "
              f"bin t/T={steps[np.unravel_index(np.nanargmax(M), M.shape)[1]]/192:.2f}")

    show(hw, "Delta WITHIN-STATE R2  (cheap + h_i  vs  cheap)   <- candidate-level")
    show(hc, "Delta within-state CONCORDANCE  (cheap + h_i  vs  cheap)")
    show(hg, "Delta R2  (cheap + h_global  vs  cheap)           <- state-level")

    out = {"target": target, "steps": [int(s) for s in steps],
           "delta_within_r2_h_local": hw.tolist(),
           "delta_concordance_h_local": hc.tolist(),
           "delta_r2_h_global": hg.tolist(),
           "max_within_r2_h_local": float(np.nanmax(hw)),
           "max_concordance_h_local": float(np.nanmax(hc)),
           "max_r2_h_global": float(np.nanmax(hg))}
    suffix = "_" + "".join(tags)
    p = os.path.join(ROOT, "results", f"exp1b_within_state{suffix}.json")
    json.dump(out, open(p, "w"), indent=2)
    np.savez(os.path.join(ROOT, "results", f"exp1b_heatmaps{suffix}.npz"),
             within_r2_h_local=hw, concordance_h_local=hc,
             r2_h_global=hg, steps=np.array(steps))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

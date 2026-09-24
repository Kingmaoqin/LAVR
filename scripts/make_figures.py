"""Deliverable figures: the layer x timestep heatmap and the matched-candidate figure."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
E1 = os.environ.get("EXP1_DIR", os.path.join(ROOT, "results", "exp1"))
E2 = os.environ.get("EXP2_DIR", os.path.join(ROOT, "results", "exp2"))
FIG = os.environ.get("FIG_DIR", os.path.join(ROOT, "results", "figures"))
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def heatmap():
    z = np.load(os.path.join(E1, "heatmap.npz"))
    d, steps = z["delta_r2"], z["steps"]
    e1 = json.load(open(os.path.join(E1, "exp1_report.json")))
    n_steps = 192
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    v = np.nanmax(np.abs(d)) if np.isfinite(d).any() else 1.0
    im = ax.imshow(d, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{s/n_steps:.2f}" for s in steps])
    ax.set_yticks(range(d.shape[0]))
    ax.set_xlabel("diffusion progress  (commits made / total)")
    ax.set_ylabel("backbone layer  (0 = embedding)")
    ax.set_title("Incremental linear decodability of $A^{\\pi_{ref}}$\n"
                 "$\\Delta R^2$ over cheap + output-distribution controls",
                 fontsize=10)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if np.isfinite(d[i, j]):
                ax.text(j, i, f"{d[i,j]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(d[i, j]) > 0.6 * v else "black")
    fig.colorbar(im, ax=ax, label="$\\Delta R^2$ (held out)")
    fig.tight_layout()
    p = os.path.join(FIG, "fig_layer_timestep_heatmap.png")
    fig.savefig(p)
    print("wrote", p)


def block_table():
    e1 = json.load(open(os.path.join(E1, "exp1_report.json")))
    targets = [k for k in ("A_pertok", "A_future", "V_pertok") if k in e1]
    fig, axes = plt.subplots(1, len(targets), figsize=(4.9 * len(targets), 4.4))
    axes = np.atleast_1d(axes)
    names = {"C1": "confidence scalars", "C1C2": "+ trajectory",
             "C3": "output distribution", "cheap": "ALL cheap controls",
             "H_global": "$h_{global}$ only", "H_local": "$h_i$ only",
             "H": "hidden only $[h_i;h_g]$",
             "cheap+H_local": "cheap + $h_i$",
             "cheap+H": "cheap + HIDDEN",
             "cheap+H_mlp": "cheap + hidden (MLP)"}
    for ax, tgt in zip(axes, targets):
        b = e1[tgt]["blocks"]
        keys = [k for k in names if k in b]
        vals = [b[k]["r2"] for k in keys]
        errs = [np.std(b[k]["r2_by_seed"]) for k in keys]
        cols = ["#999999"] * len(keys)
        for i, k in enumerate(keys):
            if k == "cheap+H":
                cols[i] = "#c0392b"
            elif k == "cheap":
                cols[i] = "#2c3e50"
            elif k == "cheap+H_mlp":
                cols[i] = "#e59866"
            elif k in ("H", "H_local", "H_global", "cheap+H_local"):
                cols[i] = "#b8b8b8"
        ax.bar(range(len(keys)), vals, yerr=errs, color=cols, capsize=3)
        ceil = e1.get(f"noise_ceiling_{tgt}")
        if ceil:
            ax.axhline(ceil, ls="--", lw=1, color="#27ae60")
            ax.text(len(keys) - 0.4, ceil, " label-noise ceiling", fontsize=7,
                    color="#27ae60", va="bottom", ha="right")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([names[k] for k in keys], fontsize=7,
                           rotation=38, ha="right")
        ax.set_ylabel("held-out $R^2$")
        ax.set_title(f"target: {tgt}", fontsize=10)
        ax.axhline(0, color="k", lw=0.6)
    fig.suptitle("Future-value decodability by feature block "
                 "(mean of 3 document-level splits)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(FIG, "fig_block_r2.png")
    fig.savefig(p)
    print("wrote", p)


def within_state():
    e1 = json.load(open(os.path.join(E1, "exp1_report.json")))
    targets = [k for k in ("A_pertok", "A_future") if k in e1]
    keys = ["cheap", "cheap+H_local", "cheap+H"]
    lbl = {"cheap": "cheap controls", "cheap+H_local": "cheap + $h_i$",
           "cheap+H": "cheap + $[h_i;h_g]$"}
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.4))
    axes = np.atleast_1d(axes)
    for ax, tgt in zip(axes, targets):
        b = e1[tgt]["blocks"]
        ks = [k for k in keys if k in b and np.isfinite(b[k].get("within_r2", np.nan))]
        v = [b[k]["within_r2"] for k in ks]
        e = [np.std(b[k]["within_r2_by_seed"]) for k in ks]
        ax.bar(range(len(ks)), v, yerr=e, capsize=3,
               color=["#2c3e50", "#7f8c8d", "#c0392b"][:len(ks)])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([lbl[k] for k in ks], fontsize=7, rotation=20,
                           ha="right")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("within-state $R^2$")
        ax.set_title(f"target: {tgt}", fontsize=10)
    fig.suptitle("Candidate-level signal only (state means removed).\n"
                 "$h_{global}$ is constant within a state and scores exactly 0 here.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = os.path.join(FIG, "fig_within_state_r2.png")
    fig.savefig(p)
    print("wrote", p)


def decomposition():
    """The decisive figure: candidate-level vs state-level hidden contribution."""
    z = np.load(os.environ.get("EXP1B_NPZ",
                os.path.join(ROOT, "results", "exp1b_heatmaps.npz")))
    steps = z["steps"]
    panels = [(z["within_r2_h_local"],
               "candidate-level channel\n$\\Delta$ within-state $R^2$ from $h_i$"),
              (z["r2_h_global"],
               "state-level channel\n$\\Delta R^2$ from $h_{global}$")]
    v = max(np.nanmax(np.abs(p_[0])) for p_ in panels)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.26)
    for ax, (M, title) in zip(axes, panels):
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-v, vmax=v)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([f"{s/192:.2f}" for s in steps])
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels(range(M.shape[0]), fontsize=6)
        ax.set_xlabel("diffusion progress")
        ax.set_title(title, fontsize=9, pad=6)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=5,
                            color="white" if abs(M[i, j]) > 0.6 * v else "black")
    axes[0].set_ylabel("backbone layer")
    fig.colorbar(im, ax=axes, label="held-out gain over cheap controls",
                 fraction=0.03, pad=0.02)
    fig.suptitle("Where the incremental signal lives", fontsize=11, y=0.98)
    A, B = panels[0][0], panels[1][0]
    mw, mg = np.nanmedian(A), np.nanmedian(B)
    pw = int(np.nansum(A > 0)); pg = int(np.nansum(B > 0)); n = A.size
    fig.text(0.5, -0.02,
             f"Both panels share one colour scale.  State-level: {pg}/{n} cells "
             f"positive, median {mg:+.3f}.  Candidate-level: {pw}/{n}, "
             f"median {mw:+.3f}.\nThe systematic gain is entirely state-level. "
             "$h_{global}$ is constant within a state and so cannot rank "
             "candidates at all;\n$h_i$ is the only per-candidate signal, and "
             "it sits at chance — which is the decision a scheduler faces.",
             ha="center", fontsize=8.5)
    p = os.path.join(FIG, "fig_decomposition.png")
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


def matched():
    p2 = os.path.join(E2, "exp2_report.json")
    if not os.path.exists(p2):
        return
    e2 = json.load(open(p2))
    keys = [k for k in e2 if not k.startswith("_")]
    if not keys:
        return
    vals = [e2[k]["mean"] for k in keys]
    errs = [np.std(e2[k]["per_seed"]) for k in keys]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    cols = ["#c0392b" if "HIDDEN" in keys[i] else "#7f8c8d" for i in order]
    ax.barh(range(len(keys)), [vals[i] for i in order],
            xerr=[errs[i] for i in order], color=cols, capsize=3)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([keys[i] for i in order], fontsize=7)
    ax.axvline(0.5, ls="--", color="k", lw=1)
    ax.text(0.5, len(keys) - 0.4, " chance", fontsize=7, va="top")
    ax.set_xlabel("matched-pair ordering accuracy (held out)")
    ax.set_title("Experiment 2 — after matching on the exposed signals",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIG, "fig_matched_candidates.png")
    fig.savefig(p)
    print("wrote", p)


if __name__ == "__main__":
    for f in (heatmap, block_table, within_state, decomposition, matched):
        try:
            f()
        except Exception as e:
            print(f"{f.__name__}: {type(e).__name__}: {e}")

"""Cross-task replication of the 08-30 state-level result.

On 08-30 a frozen Nemotron state predicted whether the generation would end up
CORRECT (GSM8K, AUC 0.890 vs 0.823 for exposed-output controls). That ran on one
task. Here the same probe is applied to GSM8K (taskC) and SVAMP (taskE_svamp)
under an identical protocol, so the question becomes whether the readout is a
property of the representation or of one dataset.

V_reward is constant within a state, so rows are DEDUPLICATED to state level and
the cheap controls are aggregated per state. Splits are by document.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import dataset as D                                                     # noqa
import probes as P                                                      # noqa
import stats_suite as S                                                 # noqa


def auc(y, s):
    b = y > 0.5
    if b.all() or not b.any():
        return float("nan")
    r = np.empty(len(s)); r[np.argsort(s, kind="stable")] = np.arange(len(s)) + 1
    npos, nneg = b.sum(), (~b).sum()
    return float((r[b].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def state_level(tags):
    d = D.load_labels(tags)
    sid = d["state_id"]
    u, first = np.unique(sid, return_index=True)
    order = np.argsort(first); u, first = u[order], first[order]
    y = d["V_reward"][first].astype(np.float32)
    doc = d["doc_id"][first]
    step = d["step"][first]
    seeds = d["V_reward"][first]          # placeholder if per-seed absent
    # cheap controls aggregated over the candidates of the state
    C = np.concatenate([d["C1"], d["C2"], d["C3"]], 1).astype(np.float32)
    agg = []
    for s_ in u:
        i = sid == s_
        agg.append(np.concatenate([C[i].mean(0), C[i].std(0), C[i].min(0), C[i].max(0)]))
    Cs = np.nan_to_num(np.stack(agg).astype(np.float32))
    Hg = d["H_g"][first].astype(np.float32)          # (n_states, n_layers, D)
    return {"y": y, "doc": doc, "step": step, "cheap": Cs, "Hg": Hg,
            "n_layers": Hg.shape[1], "n_states": len(u),
            "n_docs": len(np.unique(doc))}


def run(name, tags, n_seeds):
    d = state_level(tags)
    y, doc = d["y"], d["doc"]
    print(f"\n=== {name} ===  states={d['n_states']} docs={d['n_docs']} "
          f"mean V_reward={y.mean():.4f} sd={y.std():.4f} "
          f"all-correct={float((y==1).mean()):.3f} all-wrong={float((y==0).mean()):.3f}", flush=True)

    def split(seed):
        u = np.unique(doc); rng = np.random.default_rng(seed); rng.shuffle(u)
        a, b = int(.6 * len(u)), int(.75 * len(u))
        s = {k: set(v) for k, v in (("tr", u[:a]), ("va", u[a:b]), ("te", u[b:]))}
        return {k: np.array([x in v for x in doc]) for k, v in s.items()}

    sel = split(99)
    best = (-9e9, 0)
    for L in range(d["n_layers"]):
        H = d["Hg"][:, L]
        m = P.fit_linear_2block(d["cheap"][sel["tr"]], H[sel["tr"]], y[sel["tr"]],
                                d["cheap"][sel["va"]], H[sel["va"]], y[sel["va"]])
        v = P.r2_score(y[sel["va"]], P.predict_2block(m, d["cheap"][sel["va"]], H[sel["va"]]))
        if v > best[0]:
            best = (v, L)
    L = best[1]
    H = d["Hg"][:, L]
    rng = np.random.default_rng(0)
    GA = rng.normal(0, 1, H.shape).astype(np.float32) * H.std()

    rows = {"hidden": {"r2": [], "auc": []}, "control_gaussian": {"r2": [], "auc": []}}
    n_tr = n_te = 0
    keep = None
    for s_ in range(n_seeds):
        sp = split(s_); tr, va, te = sp["tr"], sp["va"], sp["te"]
        n_tr, n_te = len(np.unique(doc[tr])), len(np.unique(doc[te]))
        mc = P.fit_linear(d["cheap"][tr], y[tr], d["cheap"][va], y[va])
        pc = P.predict(mc, d["cheap"][te])
        r_c, a_c = P.r2_score(y[te], pc), auc(y[te], pc)
        for nm, X in (("hidden", H), ("control_gaussian", GA)):
            m = P.fit_linear_2block(d["cheap"][tr], X[tr], y[tr],
                                    d["cheap"][va], X[va], y[va])
            p = P.predict_2block(m, d["cheap"][te], X[te])
            rows[nm]["r2"].append(P.r2_score(y[te], p) - r_c)
            rows[nm]["auc"].append(auc(y[te], p) - a_c)
            if s_ == 0 and nm == "hidden":
                keep = {"te": te, "pc": pc, "ph": p, "r_c": r_c, "a_c": a_c}
        if (s_ + 1) % 10 == 0:
            print(f"    seed {s_+1}/{n_seeds}", flush=True)

    out = {"task": name, "layer": int(L), "n_states": d["n_states"],
           "n_docs": d["n_docs"], "n_seeds": n_seeds,
           "n_train_docs": int(n_tr), "n_test_docs": int(n_te),
           "base_accuracy": float(y.mean()),
           "cheap_seed0": {"r2": keep["r_c"], "auc": keep["a_c"]}, "probes": {}}
    for nm, v in rows.items():
        e = {}
        for m_ in ("r2", "auc"):
            dd = np.array(v[m_], float); dd = dd[np.isfinite(dd)]
            e[m_] = {"mean": float(dd.mean()),
                     "naive_t": {k: (list(map(float, x)) if isinstance(x, tuple) else float(x))
                                 for k, x in S.naive_t(dd).items()},
                     "nadeau_bengio": {k: (list(map(float, x)) if isinstance(x, tuple) else float(x))
                                       for k, x in S.nadeau_bengio_t(dd, n_tr, n_te).items()},
                     "sign_rank": S.sign_and_rank(dd)}
        out["probes"][nm] = e

    # selective generation on the seed-0 fit
    te, ph, pc = keep["te"], keep["ph"], keep["pc"]
    yte = y[te]
    cur = {}
    for nm, sc in (("hidden", ph), ("cheap", pc), ("oracle", yte)):
        o = np.argsort(-sc, kind="stable")
        cur[nm] = (np.cumsum(yte[o]) / np.arange(1, len(o) + 1))
    out["selective"] = {nm: {f"{c:.0%}": float(v[max(1, int(c * len(v))) - 1])
                             for c in (0.9, 0.7, 0.5, 0.3)} for nm, v in cur.items()}
    out["aurc"] = {nm: float(v.mean()) for nm, v in cur.items()}
    print(f"  layer {L}  dR2={out['probes']['hidden']['r2']['mean']:+.4f}  "
          f"ΔAUC={out['probes']['hidden']['auc']['mean']:+.4f}  "
          f"placebo dAUC={out['probes']['control_gaussian']['auc']['mean']:+.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", default="data/task_state_readout.json")
    a = ap.parse_args()
    res = [run("GSM8K(taskC)", ["taskC"], a.seeds),
           run("SVAMP(taskE)", ["taskE_svamp"], a.seeds)]
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

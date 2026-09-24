"""Phase S qualification for the SECOND backbone (SEDD).

S1 cannot reuse MDLM's NELBO: SEDD optimises a score-entropy loss, so the
masked-diffusion ELBO formula does not apply to it. Instead S1 uses a
backbone-AGNOSTIC fidelity metric computable identically for both models --
the mean negative log-likelihood the model assigns to the TRUE token at masked
positions, at matched mask ratios. MDLM's value on the same windows is printed
as the reference.
"""
import argparse, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from mdlm_local import load_mdlm, get_tokenizer, MASK_TOKEN_ID   # noqa: E402
from sedd_local import load_sedd                                  # noqa: E402
import data as datamod                                            # noqa: E402
from policy import (PiRefConfig, forward_raw, topk_logprobs,      # noqa: E402
                    rollout, make_initial_state)


@torch.no_grad()
def cond_nll(model, x, ratios=(0.2, 0.4, 0.6, 0.8), seed=0, chunk=8):
    """Mean NLL of the true token at masked positions, per mask ratio."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for r in ratios:
        tot, n = 0.0, 0
        for a in range(0, len(x), chunk):
            xb = x[a:a + chunk]
            keep = torch.rand(xb.shape, generator=g).to(xb.device) > r
            z = torch.where(keep, xb, torch.full_like(xb, MASK_TOKEN_ID))
            lg, _ = forward_raw(model, z)
            lp = torch.empty(xb.shape, device=xb.device)
            for b in range(0, xb.shape[1], 32):
                sl = lg[:, b:b + 32].float()
                lp[:, b:b + 32] = (sl.gather(-1, xb[:, b:b + 32, None]).squeeze(-1)
                                   - sl.logsumexp(-1))
            m = ~keep
            tot += float((-(lp * m)).sum()); n += int(m.sum())
        out[r] = tot / max(n, 1)
    return out


def dn(a, n):
    v = []
    for row in a:
        g = [tuple(row[i:i + n]) for i in range(len(row) - n + 1)]
        v.append(len(set(g)) / max(len(g), 1))
    return float(np.mean(v))


def mrep(a):
    o = []
    for row in a:
        c = {}
        for i in range(len(row) - 3):
            t = tuple(row[i:i + 4]); c[t] = c.get(t, 0) + 1
        o.append(max(c.values()) / max(len(row) - 3, 1))
    return float(np.mean(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=48)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()
    dev = "cuda"
    w, dd = datamod.get_windows(seq_len=256, n_docs=3000)
    sp = datamod.doc_level_split(dd)
    idx = sp["test"][:args.n_prompts]
    W = w[idx]
    rep = {"model": "louaaron/sedd-small",
           "provenance": "SEDD, ICML 2024, OpenWebText, absorbing "
                         "graph"}

    print("[S1] conditional NLL of the true token at masked positions")
    sedd, _ = load_sedd(device=dev)
    x = torch.as_tensor(W[:24], device=dev)
    s_nll = cond_nll(sedd, x)
    del sedd; torch.cuda.empty_cache()
    mdlm, _ = load_mdlm(device=dev)
    m_nll = cond_nll(mdlm, x)
    del mdlm; torch.cuda.empty_cache()
    print(f"     {'mask ratio':>12}{'SEDD':>10}{'MDLM (ref)':>12}")
    for r in s_nll:
        print(f"     {r:>12.1f}{s_nll[r]:>10.3f}{m_nll[r]:>12.3f}")
    rep["S1_sedd_nll"] = s_nll; rep["S1_mdlm_nll"] = m_nll
    # SEDD must be in the same regime as the qualified first backbone
    rep["S1_pass"] = bool(np.mean(list(s_nll.values()))
                          < 1.6 * np.mean(list(m_nll.values())))

    sedd, _ = load_sedd(device=dev)
    pcfg = PiRefConfig(seq_len=256, prefix_len=64)
    all_ids, lls, incomplete = [], [], 0
    for b0 in range(0, len(W), args.batch):
        xb = torch.as_tensor(W[b0:b0 + args.batch], device=dev)
        ids0, m0 = make_initial_state(xb, pcfg, dev)
        per = []
        for s in range(args.n_seeds):
            seeds = torch.arange(len(xb), device=dev, dtype=torch.int64) + 100003 * s + b0
            r = rollout(sedd, ids0, m0, seeds, pcfg)
            per.append((r["path_ll_rb"] / r["n_commit"]).cpu().numpy())
            incomplete += r["incomplete"]
            if s == 0:
                all_ids.append(r["ids"][:, 64:].cpu().numpy())
        lls.append(np.stack(per, 1))
    gen = np.concatenate(all_ids); ll = np.concatenate(lls, 0)
    ref = W[:, 64:]
    d2, mr = dn(gen, 2), mrep(gen)
    rd2, rmr = dn(ref, 2), mrep(ref)
    print(f"\n[S2] generated d1/d2/d3 = {dn(gen,1):.3f}/{d2:.3f}/{dn(gen,3):.3f}  maxrep4 {mr:.3f}")
    print(f"     real text d1/d2/d3 = {dn(ref,1):.3f}/{rd2:.3f}/{dn(ref,3):.3f}  maxrep4 {rmr:.3f}")
    rep.update(S2_d2=d2, S2_ref_d2=rd2, S2_maxrep=mr, S2_ref_maxrep=rmr,
               S2_pass=bool(d2 > 0.70 * rd2 and mr < 3.0 * rmr))
    bet, wit = float(ll.mean(1).std()), float(ll.std(1).mean())
    print(f"[S3] per-token Path-LL mean {ll.mean():.4f} between-prompt sd {bet:.4f} within-seed sd {wit:.4f}")
    rep.update(S3_between=bet, S3_within=wit, S3_pass=bool(bet > 1e-3 and wit > 1e-3))
    print(f"[S5] incomplete trajectories: {incomplete}")
    rep["S5_pass"] = bool(incomplete == 0)

    ids0, m0 = make_initial_state(torch.as_tensor(W[:2], device=dev), pcfg, dev)
    shapes = []
    for _ in range(3):
        lg, hs = forward_raw(sedd, ids0, output_hidden_states=True)
        shapes.append((len(hs), tuple(hs[-1].shape)))
    rep["S4_pass"] = bool(len(set(shapes)) == 1)
    print(f"[S4] hidden states consistent: {rep['S4_pass']} {shapes[0]}")

    gates = {k: v for k, v in rep.items() if k.endswith("_pass")}
    rep["ALL_PASS"] = all(gates.values())
    print("\n=== SEDD PHASE S ===", json.dumps(gates))
    print("SUBSTRATE QUALIFIED" if rep["ALL_PASS"] else "SUBSTRATE FAILURE")
    os.makedirs(os.path.join(ROOT, "results", "phase_s_sedd"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "results", "phase_s_sedd",
                                     "report.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()

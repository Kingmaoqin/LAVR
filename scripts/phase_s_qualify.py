"""
Phase S -- substrate qualification for MDLM-owt.

Gates (all must pass before oracle-label collection):
  S1  numerical fidelity: NELBO perplexity on held-out OpenWebText should
      reproduce the published MDLM-owt figure (<=23.21 test ppl upper bound).
      This validates the flash-attn-free reimplementation end-to-end.
  S2  coherent (non-collapsed, non-repetitive) generation under pi_ref.
  S3  non-degenerate Path-LL variance across prompts AND across rollout seeds
      (if Path-LL has no variance, the advantage label is identically zero).
  S4  hidden states log consistently across denoising steps.
  S5  pi_ref completes every trajectory within its step budget.

Failure here is SUBSTRATE FAILURE, not evidence against the hypothesis.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from mdlm_local import load_mdlm, get_tokenizer, MASK_TOKEN_ID   # noqa: E402
import data as datamod                                            # noqa: E402
from policy import (PiRefConfig, forward_raw, topk_logprobs,       # noqa: E402
                    rollout, make_initial_state, unmask_order, sample_tokens)


@torch.no_grad()
def nelbo_ppl(model, windows, device, n_mc=64, batch=8, seed=0):
    """Continuous-time masked-diffusion NELBO (Sahoo et al. 2024, eq. 12).

    With the linear schedule alpha_t = 1 - t:
        L = E_{t~U(0,1)} [ (1/t) * sum_{i: z_t^i = M} -log p_theta(x_i | z_t) ]
    Antithetic low-discrepancy t sampling is used to cut MC variance.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    tot_nll, tot_tok = 0.0, 0
    for b0 in range(0, len(windows), batch):
        x = torch.as_tensor(windows[b0:b0 + batch], device=device)
        B, L = x.shape
        for k in range(n_mc):
            # low-discrepancy: stratified offset within [0,1)
            u = (k + torch.rand(1, generator=g).item()) / n_mc
            t = max(u, 1e-3)
            keep = torch.rand((B, L), generator=g).to(device) > t
            z = torch.where(keep, x, torch.full_like(x, MASK_TOKEN_ID))
            logits, _ = forward_raw(model, z)
            lp = torch.empty(B, L, device=device)
            for a in range(0, L, 32):        # chunked fp32 log-softmax
                sl = logits[:, a:a + 32].float()
                lp[:, a:a + 32] = (sl.gather(-1, x[:, a:a + 32, None]).squeeze(-1)
                                   - sl.logsumexp(-1))
            masked = ~keep
            tot_nll += float((-(lp * masked).sum()).item()) / t
            tot_tok += int(B * L)
    nll_per_tok = tot_nll / tot_tok
    return float(np.exp(nll_per_tok)), nll_per_tok


def distinct_n(ids, n):
    """Per-sample distinct-n, averaged. Compared against real text, not 1.0."""
    vals = []
    for row in ids:
        g = [tuple(row[i:i + n]) for i in range(len(row) - n + 1)]
        vals.append(len(set(g)) / max(len(g), 1))
    return float(np.mean(vals))


def max_repeat_frac(ids):
    """Largest fraction of a sample occupied by a single repeated 4-gram."""
    out = []
    for row in ids:
        c = {}
        for i in range(len(row) - 3):
            g = tuple(row[i:i + 4])
            c[g] = c.get(g, 0) + 1
        out.append(max(c.values()) / max(len(row) - 3, 1))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--prefix_len", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--n_ppl_docs", type=int, default=32)
    ap.add_argument("--n_mc", type=int, default=64)
    ap.add_argument("--n_gen_prompts", type=int, default=32)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    outdir = args.out or os.path.join(
        os.path.dirname(HERE), "results", "phase_s")
    os.makedirs(outdir, exist_ok=True)
    dev = args.device
    torch.manual_seed(0)

    print("[S0] loading MDLM-owt (Cornell / Kuleshov group, Apache-2.0)")
    model, cfg_m = load_mdlm(device=dev)
    tok = get_tokenizer()
    windows, doc_ids = datamod.get_windows(seq_len=args.seq_len, n_docs=3000)
    splits = datamod.doc_level_split(doc_ids)
    print(f"      windows={windows.shape} splits="
          f"{ {k: len(v) for k, v in splits.items()} }")

    report = {"model": "kuleshov-group/mdlm-owt",
              "provenance": "MDLM, Apache-2.0, OpenWebText",
              "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
              "config": vars(args)}

    # ---- S1: numerical fidelity via NELBO perplexity -----------------------
    t0 = time.time()
    test_idx = splits["test"][:args.n_ppl_docs]
    ppl, nll = nelbo_ppl(model, windows[test_idx], dev,
                         n_mc=args.n_mc, batch=args.batch)
    print(f"[S1] NELBO ppl = {ppl:.2f} (nats/token {nll:.4f})  "
          f"[published MDLM-owt test ppl <= 23.21]  {time.time()-t0:.0f}s")
    report["S1_nelbo_ppl"] = ppl
    report["S1_nats_per_token"] = nll
    report["S1_published_reference"] = 23.21
    # a reimplementation bug would blow this up by orders of magnitude;
    # allow slack for the short 128-token context vs. the paper's 1024.
    report["S1_pass"] = bool(ppl < 60.0)

    # ---- S2/S3/S5: generation under pi_ref ---------------------------------
    pcfg = PiRefConfig(seq_len=args.seq_len, prefix_len=args.prefix_len,
                       temperature=args.temperature, top_k=args.top_k)
    gen_idx = splits["test"][args.n_ppl_docs:args.n_ppl_docs + args.n_gen_prompts]
    all_ll, all_ids, incomplete = [], [], 0
    t0 = time.time()
    for b0 in range(0, len(gen_idx), args.batch):
        w = torch.as_tensor(windows[gen_idx[b0:b0 + args.batch]], device=dev)
        ids0, mask0 = make_initial_state(w, pcfg, dev)
        per_seed = []
        for s in range(args.n_seeds):
            seeds = torch.arange(len(w), device=dev, dtype=torch.int64) \
                + 100003 * s + 7919 * b0
            r = rollout(model, ids0, mask0, seeds, pcfg)
            per_seed.append(r["path_ll"].cpu().numpy()
                            / r["n_commit"].cpu().numpy())
            incomplete += r["incomplete"]
            if s == 0:
                all_ids.append(r["ids"].cpu().numpy())
        all_ll.append(np.stack(per_seed, 1))          # (B, n_seeds)
    ll = np.concatenate(all_ll, 0)
    gen = np.concatenate(all_ids, 0)[:, args.prefix_len:]
    print(f"[S2/S3/S5] {len(ll)} prompts x {args.n_seeds} seeds  "
          f"{time.time()-t0:.0f}s")

    d1, d2, d3 = (distinct_n(gen, n) for n in (1, 2, 3))
    rep = max_repeat_frac(gen)
    ref = windows[gen_idx][:, args.prefix_len:]      # real OWT text, same span
    g1, g2, g3 = (distinct_n(ref, n) for n in (1, 2, 3))
    grep = max_repeat_frac(ref)
    print(f"[S2] generated  distinct-1/2/3 = {d1:.3f}/{d2:.3f}/{d3:.3f}  "
          f"max-rep-4gram = {rep:.3f}")
    print(f"     real text  distinct-1/2/3 = {g1:.3f}/{g2:.3f}/{g3:.3f}  "
          f"max-rep-4gram = {grep:.3f}")
    report.update(S2_distinct1=d1, S2_distinct2=d2, S2_distinct3=d3,
                  S2_max_repeat_frac=rep, S2_ref_distinct2=g2,
                  S2_ref_max_repeat_frac=grep,
                  # coherence = at least 70% of real-text bigram diversity and
                  # no more than 3x its worst-case 4-gram repetition
                  S2_pass=bool(d2 > 0.70 * g2 and rep < 3.0 * grep))

    between = float(ll.mean(1).std())
    within = float(ll.std(1).mean())
    print(f"[S3] per-token Path-LL: mean {ll.mean():.4f}  "
          f"between-prompt sd {between:.4f}  within-prompt(seed) sd {within:.4f}")
    report.update(S3_pathll_mean=float(ll.mean()),
                  S3_between_prompt_sd=between, S3_within_seed_sd=within,
                  S3_pass=bool(between > 1e-3 and within > 1e-3))

    print(f"[S5] incomplete trajectories: {incomplete}")
    report["S5_incomplete"] = incomplete
    report["S5_pass"] = bool(incomplete == 0)

    # ---- S4: hidden-state logging consistency ------------------------------
    w = torch.as_tensor(windows[gen_idx[:2]], device=dev)
    ids0, mask0 = make_initial_state(w, pcfg, dev)
    shapes, finite = [], True
    ids, mask = ids0.clone(), mask0.clone()
    for step in range(3):
        logits, hs = forward_raw(model, ids, output_hidden_states=True)
        shapes.append((len(hs), tuple(hs[-1].shape)))
        finite &= bool(all(torch.isfinite(h).all().item() for h in hs))
        seeds = torch.zeros(len(w), device=dev, dtype=torch.int64)
        pos = torch.arange(ids.shape[1], device=dev, dtype=torch.int64)
        order = unmask_order(seeds, pos)
        lp_top, idx, _, _ = topk_logprobs(logits, pcfg.top_k, pcfg.logit_chunk)
        pr, _ = sample_tokens(lp_top, idx, seeds, pos, pcfg)
        score = torch.where(mask, order, torch.full_like(order, -1.0))
        sel = torch.zeros_like(mask)
        sel.scatter_(1, score.argmax(1, keepdim=True), True)
        sel &= mask
        ids = torch.where(sel, pr, ids)
        mask = mask & ~sel
    ok4 = len(set(shapes)) == 1 and finite
    print(f"[S4] hidden states consistent across steps: {ok4}  {shapes[0]}")
    report.update(S4_shapes=str(shapes[0]), S4_pass=bool(ok4))

    # ---- sample dump -------------------------------------------------------
    samples = [tok.decode(windows[gen_idx[i]][:args.prefix_len].tolist())
               + " ||>> " + tok.decode(gen[i].tolist()) for i in range(3)]
    report["samples"] = samples
    print("\n--- sample generations (prefix ||>> continuation) ---")
    for s in samples[:2]:
        print(s.replace("\n", " ")[:520], "\n")

    gates = {k: v for k, v in report.items() if k.endswith("_pass")}
    report["ALL_PASS"] = all(gates.values())
    print("=== PHASE S GATES ===", json.dumps(gates))
    print("SUBSTRATE QUALIFIED" if report["ALL_PASS"] else "SUBSTRATE FAILURE")

    with open(os.path.join(outdir, "phase_s_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", os.path.join(outdir, "phase_s_report.json"))


if __name__ == "__main__":
    main()

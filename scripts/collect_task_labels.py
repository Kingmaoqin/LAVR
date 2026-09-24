"""
Phase 0A for the TASK-UTILITY label — a secondary external-validity
target, and the last outstanding pre-registered deliverable.

    A_task(i | s_t) = P(correct | commit (i, x_hat_i) at s_t, then pi_ref)
                    - P(correct |                              pi_ref)

reward = exact match on the GSM8K final numeric answer, estimated by
CRN-coupled paired rollouts run TO COMPLETION.

WHY COMPLETION ROLLOUTS REMOVE A BUG CLASS BY CONSTRUCTION
A task reward only exists once a full answer exists, so both branches must run
to the end and therefore fill EVERY remaining masked position. They are
automatically matched in positions consumed, so the denominator mismatch and the
depletion artifact that had to be repaired on the MDLM backbone (see the
matched-branch design in src/policy.py) cannot arise here.

THE SCREENING STEP, AND WHY IT IS NOT CHEATING
Phase S measured that of 48 GSM8K prompts, 19 are always answered correctly and
14 always incorrectly across rollout seeds; `A_task` is identically 0 on both.
Collecting naively would make ~69% of labels a constant zero and floor the
detectable variance. Prompts are therefore screened by running K rollouts first
and OVERSAMPLING the mixed ones -- while keeping a naturally-sampled set whose
held-out metrics are reported separately. This is the same "informative stratum
+ natural stratum" pattern the Path-LL collector already uses, and every example
carries its `prompt_stratum` so the two can never be silently pooled.
"""
import argparse, glob, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from nemotron_local import load_nemotron, get_tokenizer, MASK_TOKEN_ID  # noqa
from nemotron_policy import (BlockPiRefConfig, safe_block_rollout,      # noqa
                             make_state, topk_logprobs, sample_tokens,
                             expected_logp, active_block)
import collect_task as CT                                               # noqa
import features as F                                                    # noqa


def prompt_ids(tok, q):
    msg = [{"role": "user", "content": q + CT.SUFFIX}]
    return tok(tok.apply_chat_template(msg, tokenize=False,
                                       add_generation_prompt=True),
               return_tensors="pt").input_ids


def gsm8k(n):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet",
                        repo_type="dataset")
    return pq.read_table(p).to_pylist()[:n]


def svamp(n):
    """Second qualified exact-match task; normalize to the GSM8K row schema."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("ChilleD/SVAMP", "test.json", repo_type="dataset")
    raw = json.load(open(p))[:n]
    return [{"question": str(r["Body"]).strip() + "\n" +
                         str(r["Question"]).strip(),
             "answer": "#### " + str(r["Answer"]),
             "source_id": r.get("ID", str(i))}
            for i, r in enumerate(raw)]


@torch.no_grad()
def screen(model, tok, rows, cfg, pcfg, dev, K, micro):
    """Run K rollouts per prompt; classify as mixed / always-right / always-wrong."""
    out = []
    t0 = time.time()
    for qi, r in enumerate(rows):
        P = prompt_ids(tok, r["question"]).to(dev)
        g = CT.gold_answer(r["answer"])
        ids, mask = make_state(P.repeat(K, 1), cfg.gen_len, dev)
        seeds = torch.arange(K, device=dev, dtype=torch.int64) + qi * 1000
        res = safe_block_rollout(model, ids, mask, seeds, pcfg,
                                 gen_start=P.shape[1], micro=micro)
        rw = np.array([CT.task_reward(
            tok.decode(res["ids"][j, P.shape[1]:], skip_special_tokens=True), g)
            for j in range(K)])
        out.append({"qi": qi, "rewards": rw, "mean": float(rw.mean()),
                    "mixed": bool(0 < rw.mean() < 1), "prompt_len": P.shape[1]})
        if (qi + 1) % 20 == 0:
            nm = sum(o["mixed"] for o in out)
            print(f"  screened {qi+1}/{len(rows)}  mixed {nm} "
                  f"({nm/(qi+1):.0%})  {time.time()-t0:.0f}s", flush=True)
    return out


@torch.no_grad()
def collect_prompt(model, tok, row, cfg, pcfg, dev, gen, seed_off, stratum):
    """Walk pi_ref on one prompt; label candidates at the recorded states."""
    P = prompt_ids(tok, row["question"]).to(dev)
    gold = CT.gold_answer(row["answer"])
    L0 = P.shape[1]
    ids, mask = make_state(P, cfg.gen_len, dev)
    B, L = ids.shape
    pos = torch.arange(L, device=dev, dtype=torch.int64)
    tseed = torch.tensor([seed_off], device=dev, dtype=torch.int64)
    hist = CT.new_hist(B, L, dev)

    # the trajectory is walked one COMMIT STEP at a time so states can be
    # snapshotted; record_fracs are fractions of the generation region
    n_steps_est = cfg.gen_len
    record_at = sorted({max(1, int(f * cfg.gen_len)) for f in cfg.record_fracs})
    # Trigger on CROSSING a threshold, not on hitting it exactly. `filled`
    # counts positions committed, and block diffusion commits several at once,
    # so it jumps (e.g. 22 -> 27) and an `in record_at` test silently skips the
    # point. Measured on the first run: only 1.96 of 4 record points were hit
    # per prompt, costing 56% of the intended samples. The states that WERE
    # recorded are valid -- this is lost coverage, not corrupted data.
    pending = list(record_at)
    rows_out, step, filled = [], 0, 0

    while mask.any() and step < cfg.gen_len * 2:
        logits = model(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        lp_top, idx = topk_logprobs(logits, cfg.top_k, 8)

        if pending and filled >= pending[0]:
            pending.pop(0)
            snap, aux = CT.snapshot(model, ids, mask, filled, cfg.gen_len,
                                    hist, cfg, L0)
            cb, ci, strat = CT.pick_candidates(snap, cfg, gen)
            if cb.numel():
                lab = branch_task(model, tok, snap, cb, ci, cfg, pcfg, dev,
                                  gold, L0, seed_off * 977 + filled * 31)
                c1, c2, c3 = F.assemble_cheap(snap, cb, ci, cfg.gen_len)
                hi, hg = F.assemble_hidden(snap, cb, ci)
                n = cb.numel()
                rows_out.append(dict(
                    doc_id=np.full(n, row["_qi"], dtype=np.int64),
                    prompt_row=np.full(n, row["_qi"], dtype=np.int64),
                    step=np.full(n, filled, dtype=np.int32),
                    mask_ratio=np.full(n, float(mask.float().mean()), dtype=np.float32),
                    position=ci.cpu().numpy(),
                    proposed_token=snap["argmax"][cb, ci].cpu().numpy(),
                    stratum=strat,
                    prompt_stratum=np.full(n, stratum, dtype=np.int8),
                    C1=c1, C2=c2, C3=c3, H_i=hi, H_g=hg, **lab))
            del snap

        prop, _ = sample_tokens(lp_top, idx, tseed, pos, pcfg)
        rb = expected_logp(lp_top, pcfg)
        blk = active_block(mask, L0, pcfg)
        conf = lp_top[..., 0].exp()
        sel = blk & (conf >= pcfg.threshold)
        if not sel.any():
            sc = torch.where(blk, conf, torch.full_like(conf, -1.0))
            sel = torch.zeros_like(mask)
            sel.scatter_(1, sc.argmax(1, keepdim=True), True)
            sel &= blk
        mb = mask.clone()
        ids = torch.where(sel, prop, ids); mask = mask & ~sel
        CT.update_hist(hist, lp_top, idx, lp_top[..., 0].exp(), idx[..., 0], mb)
        filled += int(sel.sum()); step += 1
    return rows_out


@torch.no_grad()
def branch_task(model, tok, snap, cb, ci, cfg, pcfg, dev, gold, L0, seed_off):
    """Paired CRN rollouts to completion -> task reward AND Path-LL advantage."""
    ids, mask = snap["ids"], snap["mask"]
    K, N = cfg.K, cb.numel()
    xhat = snap["argmax"][cb, ci]
    logp_action = snap["lp_top"][cb, ci, 0]

    v_ids = ids.repeat_interleave(K, 0); v_mask = mask.repeat_interleave(K, 0)
    v_seeds = torch.arange(K, device=dev, dtype=torch.int64) + seed_off
    V = safe_block_rollout(model, v_ids, v_mask, v_seeds, pcfg, L0,
                           micro=cfg.rollout_batch)
    v_rw = np.array([CT.task_reward(
        tok.decode(V["ids"][j, L0:], skip_special_tokens=True), gold)
        for j in range(K)], dtype=np.float32)

    q_ids = ids.repeat_interleave(N * K, 0).clone()
    q_mask = mask.repeat_interleave(N * K, 0).clone()
    rr = torch.arange(N * K, device=dev)
    ppos = ci.repeat_interleave(K)
    q_ids[rr, ppos] = xhat.repeat_interleave(K)
    q_mask[rr, ppos] = False
    q_seeds = v_seeds.repeat(N)
    Q = safe_block_rollout(model, q_ids, q_mask, q_seeds, pcfg, L0,
                           micro=cfg.rollout_batch)
    q_rw = np.array([CT.task_reward(
        tok.decode(Q["ids"][j, L0:], skip_special_tokens=True), gold)
        for j in range(N * K)], dtype=np.float32).reshape(N, K)

    V_rw = np.broadcast_to(v_rw, (N, K))
    A_task_seeds = q_rw - V_rw
    n_full = (Q["n_commit"].view(N, K) + 1).clamp(min=1).double()
    pt_Q = (Q["path_ll_rb"].view(N, K) + logp_action[:, None].double()) / n_full
    pt_V = (V["path_ll_rb"].view(1, K).expand(N, K)
            / V["n_commit"].view(1, K).clamp(min=1).double().expand(N, K))
    A_full = (pt_Q - pt_V).float().cpu().numpy()

    return {
        "A_task": A_task_seeds.mean(1), "A_task_seeds": A_task_seeds,
        "A_task_sem": A_task_seeds.std(1) / np.sqrt(K),
        "Q_reward": q_rw.mean(1), "V_reward": np.full(N, v_rw.mean(), np.float32),
        "A_pertok": A_full.mean(1), "A_full_seeds": A_full,
        "A_sem": A_full.std(1) / np.sqrt(K),
        "V_pertok": pt_V.mean(1).float().cpu().numpy(),
        "logp_action": logp_action.float().cpu().numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_screen", type=int, default=300)
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--mixed_frac", type=float, default=0.70)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_cand", type=int, default=4)
    ap.add_argument("--gen_len", type=int, default=160)
    ap.add_argument("--rollout_batch", type=int, default=32)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="task")
    ap.add_argument("--dataset", choices=["gsm8k", "svamp"], default="gsm8k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard_model", action="store_true",
                    help="split the frozen backbone across visible GPUs")
    ap.add_argument("--reserve_gb", type=float, default=1.6,
                    help="per-GPU memory held back when --shard_model is used")
    ap.add_argument("--resume", action="store_true",
                    help="append after complete existing shards, skipping their doc_ids")
    ap.add_argument("--shard_examples", type=int, default=96,
                    help="checkpoint after this many candidate rows")
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--worker_index", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    outdir = os.path.join(ROOT, "data", f"labels_{args.tag}")
    os.makedirs(outdir, exist_ok=True)

    cfg = CT.TaskCollectConfig(gen_len=args.gen_len, K=args.K,
                               n_cand=args.n_cand, n_cand_conf=args.n_cand // 2,
                               rollout_batch=args.rollout_batch)
    pcfg = cfg.pi()
    model, _ = load_nemotron(device=dev, shard=args.shard_model,
                             reserve_gb=args.reserve_gb)
    tok = get_tokenizer()
    loader = gsm8k if args.dataset == "gsm8k" else svamp
    rows = loader(args.offset + args.n_screen)[args.offset:]
    for i, r in enumerate(rows):
        r["_qi"] = args.offset + i
    if not (0 <= args.worker_index < args.num_workers):
        raise ValueError("worker_index must be in [0, num_workers)")
    gen = torch.Generator(device=dev).manual_seed(
        4242 + args.offset + 1_000_003 * args.worker_index)

    # Screening costs ~27 min for 220 prompts, so cache it. An earlier launch
    # was SIGTERM'd by session teardown 20 prompts in; repeating that work on
    # every restart is pure waste.
    sc_path = os.path.join(outdir, "screen.json")
    if os.path.exists(sc_path):
        raw = json.load(open(sc_path))["screen_rewards"]
        sc = [{"qi": int(k), "rewards": np.array(v),
               "mean": float(np.mean(v)), "mixed": bool(0 < np.mean(v) < 1)}
              for k, v in sorted(raw.items(), key=lambda x: int(x[0]))]
        print(f"[task] reusing cached screening of {len(sc)} prompts", flush=True)
    else:
        print(f"[task] screening {len(rows)} prompts with K={args.K}", flush=True)
        sc = screen(model, tok, rows, cfg, pcfg, dev, args.K, args.rollout_batch)
        json.dump({"n_screened": len(sc),
                   "n_mixed": sum(o["mixed"] for o in sc),
                   "screen_rewards": {str(o["qi"]): o["rewards"].tolist()
                                      for o in sc}},
                  open(sc_path, "w"), indent=2)
    mixed = [o for o in sc if o["mixed"]]
    other = [o for o in sc if not o["mixed"]]
    n_mix = min(len(mixed), int(args.n_prompts * args.mixed_frac))
    rng = np.random.default_rng(0)
    pick_m = [mixed[i] for i in rng.permutation(len(mixed))[:n_mix]]
    pick_o = [other[i] for i in rng.permutation(len(other))[:args.n_prompts - n_mix]]
    print(f"[task] screened: {len(mixed)}/{len(sc)} mixed ({len(mixed)/len(sc):.0%}); "
          f"selected {len(pick_m)} mixed (oversampled) + {len(pick_o)} natural",
          flush=True)
    todo = [(o, 1) for o in pick_m] + [(o, 0) for o in pick_o]
    todo = [(o, s) for j, (o, s) in enumerate(todo)
            if j % args.num_workers == args.worker_index]
    print(f"[task] worker {args.worker_index}/{args.num_workers}: "
          f"assigned {len(todo)} prompts", flush=True)
    completed_docs = set()
    existing_examples = 0
    sid = 0
    if args.resume:
        old_shards = sorted(glob.glob(os.path.join(outdir, "shard_*.npz")))
        for path in old_shards:
            with np.load(path) as z:
                completed_docs.update(map(int, np.unique(z["doc_id"])))
                existing_examples += int(len(z["doc_id"]))
        if old_shards:
            sid = max(int(os.path.basename(p).split("_")[-1].split(".")[0])
                      for p in old_shards) + 1
        print(f"[task] resume: {len(completed_docs)} complete prompts / "
              f"{existing_examples} examples; next shard={sid}", flush=True)
    shard, new_examples = [], 0
    t0 = time.time()
    for k, (o, stratum) in enumerate(todo):
        # Bug fix: `qi` in screen() is a **local index** into `rows`
        # (`for qi, r in enumerate(rows)`), not an absolute problem index. The old code
        # used `rows[o["qi"] - args.offset]`, treating it as absolute.
        #   offset=0   (taskA): qi-0 == qi, correct by coincidence;
        #   offset=220 (taskB): len(rows) is also 220, negative indices wrap to rows[qi],
        #                       again correct by coincidence;
        #   offset=440 (taskC): qi-440 in [-440,-181], out of range -> IndexError.
        # So the bug was masked in both earlier collections (no error, no wrong rows);
        # taskA/taskB data are clean; it only blocked new offsets.
        assert 0 <= o["qi"] < len(rows), (
            f'screen qi {o["qi"]} out of range for rows of length {len(rows)}')
        r = rows[o["qi"]]
        if int(r["_qi"]) in completed_docs:
            print(f"  {k+1}/{len(todo)} prompt {int(r['_qi'])} already complete; skip",
                  flush=True)
            continue
        try:
            out = collect_prompt(model, tok, r, cfg, pcfg, dev, gen,
                                 cfg.seed_base + args.offset * 131 + o["qi"],
                                 stratum)
        except Exception as e:
            print(f"  prompt {o['qi']} failed: {type(e).__name__}: {e}", flush=True)
            continue
        shard.extend(out); new_examples += sum(len(x["position"]) for x in out)
        el = time.time() - t0
        print(f"  {k+1}/{len(todo)} prompts, {existing_examples + new_examples} "
              f"total examples, {el:.0f}s, "
              f"eta {el/(k+1)*(len(todo)-k-1):.0f}s", flush=True)
        if shard and (sum(len(x['position']) for x in shard) >= args.shard_examples
                      or k == len(todo) - 1):
            d = {kk: np.concatenate([x[kk] for x in shard], 0) for kk in shard[0]}
            prefix = ("shard" if args.num_workers == 1
                      else f"shard_w{args.worker_index}")
            np.savez_compressed(os.path.join(outdir, f"{prefix}_{sid:03d}.npz"), **d)
            shard, sid = [], sid + 1
    total_examples = existing_examples + new_examples
    json.dump({**vars(args), "n_examples": total_examples,
               "n_existing_examples": existing_examples,
               "n_new_examples": new_examples,
               "resume_rng_note": ("on resume, the candidate-selection generator "
                                   "restarts from the configured seed; completed "
                                   "documents are skipped, so the remaining prompts "
                                   "remain valid draws from the prespecified proposal "
                                   "but do not reproduce an uninterrupted RNG stream"),
               "seconds": time.time() - t0,
               "estimand": "A_task = P(correct|commit a_i, then pi_ref) - "
                           "P(correct|pi_ref), CRN-paired, rollouts to completion"},
              open(os.path.join(outdir, ("meta.json" if args.num_workers == 1
                                        else f"meta_worker{args.worker_index}.json")),
                   "w"), indent=2)
    print(f"[task] done: {total_examples} examples in {time.time()-t0:.0f}s "
          f"-> {outdir}")


if __name__ == "__main__":
    main()

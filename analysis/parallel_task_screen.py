"""Parallel, merge-safe screening cache builder for task-label collection."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src")]

from collect_task_labels import gsm8k, svamp, prompt_ids  # noqa: E402
from nemotron_local import load_nemotron, get_tokenizer  # noqa: E402
from nemotron_policy import make_state, safe_block_rollout  # noqa: E402
import collect_task as CT  # noqa: E402


def merge_parts(outdir, final, num_workers, n_screen):
    """Merge completed worker parts atomically; return whether merge happened."""
    parts = [os.path.join(outdir, f"screen_part_{i}.json")
             for i in range(num_workers)]
    if not all(os.path.exists(p) for p in parts):
        return False
    merged = {}
    for part in parts:
        with open(part) as f:
            merged.update(json.load(f)["screen_rewards"])
    if len(merged) != n_screen:
        raise AssertionError(f"merged {len(merged)} != {n_screen}")
    payload = {
        "n_screened": len(merged),
        "n_mixed": int(sum(0 < np.mean(v) < 1 for v in merged.values())),
        "screen_rewards": merged,
    }
    tmp = final + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, final)
    print("merged", final, flush=True)
    return True


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", choices=["gsm8k", "svamp"], default="gsm8k")
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--n-screen", type=int, required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--gen-len", type=int, default=160)
    ap.add_argument("--rollout-batch", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--worker-index", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("invalid worker index")
    outdir = os.path.join(ROOT, "data", "labels_" + args.tag)
    os.makedirs(outdir, exist_ok=True)
    final = os.path.join(outdir, "screen.json")
    if os.path.exists(final):
        print("screen cache already complete", final)
        return
    # A previous worker may have finished the last part but failed while merging.
    # Merge before loading the model so recovery is instant and GPU-free.
    if merge_parts(outdir, final, args.num_workers, args.n_screen):
        return
    loader = gsm8k if args.dataset == "gsm8k" else svamp
    rows = loader(args.offset + args.n_screen)[args.offset:]
    cfg = CT.TaskCollectConfig(gen_len=args.gen_len, K=args.K,
                               n_cand=6, n_cand_conf=3,
                               rollout_batch=args.rollout_batch)
    model, _ = load_nemotron(device=args.device)
    tok = get_tokenizer()
    result = {}
    assigned = [i for i in range(len(rows))
                if i % args.num_workers == args.worker_index]
    t0 = time.time()
    for done, qi in enumerate(assigned, 1):
        row = rows[qi]
        prompt = prompt_ids(tok, row["question"]).to(args.device)
        gold = CT.gold_answer(row["answer"])
        ids, mask = make_state(prompt.repeat(args.K, 1), cfg.gen_len, args.device)
        seeds = (torch.arange(args.K, device=args.device, dtype=torch.int64)
                 + qi * 1000)
        rollout = safe_block_rollout(model, ids, mask, seeds, cfg.pi(),
                                     gen_start=prompt.shape[1],
                                     micro=args.rollout_batch)
        rewards = [CT.task_reward(tok.decode(
            rollout["ids"][j, prompt.shape[1]:], skip_special_tokens=True), gold)
                   for j in range(args.K)]
        result[str(qi)] = rewards
        if done % 20 == 0:
            mixed = sum(0 < np.mean(v) < 1 for v in result.values())
            print(f"worker {args.worker_index}: {done}/{len(assigned)}, "
                  f"mixed {mixed}, {time.time()-t0:.0f}s", flush=True)
    part = os.path.join(outdir, f"screen_part_{args.worker_index}.json")
    with open(part, "w") as f:
        json.dump({"screen_rewards": result}, f)
    if not merge_parts(outdir, final, args.num_workers, args.n_screen):
        print("part complete; waiting for peer", flush=True)


if __name__ == "__main__":
    main()

"""Substrate qualification for the required second verifiable task (SVAMP)."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src")]

from collect_task_labels import screen  # noqa: E402
from nemotron_local import load_nemotron, get_tokenizer  # noqa: E402
import collect_task as CT  # noqa: E402


def load_svamp(n):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("ChilleD/SVAMP", "test.json", repo_type="dataset")
    rows = json.load(open(path))[:n]
    out = []
    for i, row in enumerate(rows):
        question = (str(row["Body"]).strip() + "\n" +
                    str(row["Question"]).strip())
        out.append({"question": question,
                    "answer": "#### " + str(row["Answer"]),
                    "_qi": i, "source_id": row.get("ID", str(i))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--gen-len", type=int, default=160)
    ap.add_argument("--rollout-batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(
        HERE, "results", "qualification_svamp.json"))
    args = ap.parse_args()
    rows = load_svamp(args.n)
    cfg = CT.TaskCollectConfig(gen_len=args.gen_len, K=args.K,
                               n_cand=6, n_cand_conf=3,
                               rollout_batch=args.rollout_batch)
    model, _ = load_nemotron(device=args.device)
    tok = get_tokenizer()
    result = screen(model, tok, rows, cfg, cfg.pi(), args.device,
                    args.K, args.rollout_batch)
    means = np.array([x["mean"] for x in result])
    rewards = np.stack([x["rewards"] for x in result])
    rep = {"task": "SVAMP exact numeric match", "n": len(rows), "K": args.K,
           "accuracy": float(rewards.mean()),
           "mixed_fraction": float(np.mean((means > 0) & (means < 1))),
           "always_wrong_fraction": float(np.mean(means == 0)),
           "always_right_fraction": float(np.mean(means == 1)),
           "mean_with_prompt_reward_variance": float(
               np.var(rewards, axis=1, ddof=1).mean()),
           "qualification_rule": {
               "accuracy_between_0.05_and_0.95": bool(0.05 < rewards.mean() < 0.95),
               "mixed_fraction_at_least_0.10": bool(
                   np.mean((means > 0) & (means < 1)) >= 0.10)},
           "screen_rewards": {str(i): x["rewards"].tolist()
                              for i, x in enumerate(result)}}
    rep["qualified"] = bool(all(rep["qualification_rule"].values()))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items()
                      if k != "screen_rewards"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

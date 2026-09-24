"""
Phase S for the Nemotron backbone — qualification for the TASK-UTILITY label.

The decisive gate here is S3, and it is specific to this experiment. The task
reward is binary (is the final answer right?). For the task-utility advantage
    A_task(i|s_t) = P(correct | commit a_i, then pi_ref) - P(correct | pi_ref)
to be anything other than identically zero, the reward must VARY across
rollouts started from the same state. If pi_ref is close to deterministic then
each prompt is either always right or always wrong, every advantage is 0, and
there is nothing to probe. Measuring that within-prompt reward variance is
therefore the gate that decides whether this direction is viable at all.
"""
import argparse, json, os, re, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from nemotron_local import load_nemotron, get_tokenizer, MASK_TOKEN_ID  # noqa
from nemotron_policy import (BlockPiRefConfig, safe_block_rollout,  # noqa
                             make_state)

SUFFIX = "\n\nEnd your reply with the final numeric answer alone on the last line."


def gsm8k(split="test"):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet",
                        repo_type="dataset")
    return pq.read_table(p).to_pylist()


def gold(a):
    return a.split("####")[-1].strip().replace(",", "")


def extract(s):
    n = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return n[-1].rstrip(".") if n else None


def reward(text, g):
    p = extract(text)
    if p is None:
        return 0.0
    try:
        return float(abs(float(p) - float(g)) < 1e-6)
    except ValueError:
        return 0.0


def prompt_ids(tok, q):
    msg = [{"role": "user", "content": q + SUFFIX}]
    return tok(tok.apply_chat_template(msg, tokenize=False,
                                       add_generation_prompt=True),
               return_tensors="pt").input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=48)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--gen_len", type=int, default=160)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda"

    model, mcfg = load_nemotron(device=dev)
    tok = get_tokenizer()
    rows = gsm8k()[: args.n_prompts]
    cfg = BlockPiRefConfig(block_length=32, threshold=args.threshold,
                           temperature=args.temperature, top_k=args.top_k)
    rep = {"model": "nvidia/Nemotron-Labs-Diffusion-3B",
           "provenance": "NVIDIA (US), NVIDIA Open Model License; "
                         "instruction-tuned masked "
                         "diffusion with a native block-diffusion mode",
           "config": vars(args)}

    R = np.zeros((len(rows), args.K), dtype=np.float32)
    nfes, incomplete, t0 = [], 0, time.time()
    for qi, row in enumerate(rows):
        P = prompt_ids(tok, row["question"]).to(dev)
        g = gold(row["answer"])
        ids, mask = make_state(P.repeat(args.K, 1), args.gen_len, dev)
        seeds = torch.arange(args.K, device=dev, dtype=torch.int64) + qi * 1000
        r = safe_block_rollout(model, ids, mask, seeds, cfg,
                               gen_start=P.shape[1], micro=args.batch)
        nfes.append(r["nfe"]); incomplete += r["incomplete"]
        for j in range(args.K):
            txt = tok.decode(r["ids"][j, P.shape[1]:], skip_special_tokens=True)
            R[qi, j] = reward(txt, g)
        if (qi + 1) % 8 == 0:
            print(f"  {qi+1}/{len(rows)}  running acc {R[:qi+1].mean():.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    acc = float(R.mean())
    per_prompt = R.mean(1)
    # S3: does the reward vary WITHIN a prompt across rollout seeds?
    within_var = float(R.var(1).mean())
    frac_mixed = float(((per_prompt > 0) & (per_prompt < 1)).mean())
    between_var = float(per_prompt.var())

    print(f"\n[S1] GSM8K accuracy = {acc:.1%} over {len(rows)} prompts x "
          f"{args.K} rollouts  (native generate reference 66.7%)")
    print(f"[S3] within-prompt reward variance = {within_var:.4f}")
    print(f"     prompts with MIXED outcomes (not always right / always wrong) "
          f"= {frac_mixed:.1%}")
    print(f"     between-prompt variance = {between_var:.4f}")
    print(f"[S5] incomplete rollouts = {incomplete}; mean NFE {np.mean(nfes):.0f}")

    ids, mask = make_state(prompt_ids(tok, rows[0]["question"]).to(dev), 32, dev)
    shapes = []
    for _ in range(3):
        lg, hs = model(ids, output_hidden_states=True)
        shapes.append((len(hs), tuple(hs[-1].shape)))
    s4 = len(set(shapes)) == 1
    print(f"[S4] hidden states consistent: {s4}  {shapes[0]}")

    rep.update(S1_accuracy=acc, S1_pass=bool(0.15 < acc < 0.90),
               S3_within_prompt_var=within_var, S3_frac_mixed=frac_mixed,
               S3_between_var=between_var,
               # the label is only probe-able if a real fraction of prompts can
               # go either way across seeds
               S3_pass=bool(within_var > 0.01 and frac_mixed > 0.15),
               S4_pass=bool(s4), S5_pass=bool(incomplete == 0),
               mean_nfe=float(np.mean(nfes)), seconds=time.time() - t0)
    gates = {k: v for k, v in rep.items() if k.endswith("_pass")}
    rep["ALL_PASS"] = all(gates.values())
    print("\n=== NEMOTRON PHASE S ===", json.dumps(gates))
    print("SUBSTRATE QUALIFIED" if rep["ALL_PASS"] else "SUBSTRATE FAILURE")
    od = os.path.join(ROOT, "results", "phase_s_nemotron")
    os.makedirs(od, exist_ok=True)
    json.dump(rep, open(os.path.join(od, "report.json"), "w"), indent=2,
              default=float)
    np.save(os.path.join(od, "rewards.npy"), R)


if __name__ == "__main__":
    main()

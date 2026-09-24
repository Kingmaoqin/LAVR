"""
Can one forward pass, mid-denoising, predict whether this generation will end
up CORRECT?

Target:  V_reward(s_t) = P(final answer correct | follow pi_ref from s_t),
         measured by K rollouts to completion with an exact-match GSM8K reward.

This is the practical form of the project's one solid POSITIVE finding -- that a
frozen diffusion LM linearly encodes the rollout-defined STATE VALUE. If the
frozen state predicts eventual success, that is directly usable: abort a doomed
generation early, resample, or spend compute where it will pay off.

WHY THIS NEEDS ITS OWN COLLECTION
The task-utility run produced only 560 distinct states (V_reward is constant
within a state, so its 2,240 candidate rows carry no extra information about
it). Split 60/15/25 that left 84 validation states to choose a layer plus two
ridge strengths over -- the selection noise swamped the signal, and the
combined cheap+hidden probe came out WORSE than either block alone, which is
the signature of an underpowered fit rather than a real result. This run drops
the Q branches entirely -- about 1/5 the cost -- and spends everything on
states instead.

FEATURES ARE STATE-LEVEL, not per-candidate. A state-level target needs
state-level predictors, so the cheap controls are aggregates over the currently
masked positions (confidence, entropy, margin, their spread) rather than one
position's values.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from nemotron_local import load_nemotron, get_tokenizer                 # noqa
from nemotron_policy import (BlockPiRefConfig, safe_block_rollout,      # noqa
                             make_state, topk_logprobs, sample_tokens,
                             expected_logp, active_block)
import collect_task as CT                                              # noqa
from features import ProjCache, PROJ_DIM                               # noqa

STATE_FEATS = [
    "step_norm", "mask_ratio", "n_masked_norm",
    "conf_mean", "conf_std", "conf_min", "conf_max", "conf_p25", "conf_p75",
    "ent_mean", "ent_std", "margin_mean", "margin_std",
    "logp1_mean", "logp1_std", "top2_mean",
    "obs_run_mean", "block_progress",
]


@torch.no_grad()
def state_features(lp_top, mask, step, gen_len, gen_start, logits):
    """Aggregate over currently-masked positions -> one feature vector."""
    dev = lp_top.device
    m = mask[0]
    if m.sum() == 0:
        m = torch.ones_like(m)
    lp = lp_top[0][m]                       # (n_masked, k)
    conf = lp[:, 0].exp()
    p = lp.exp()
    ent = -(p * lp).sum(-1)
    margin = lp[:, 0] - lp[:, 1]
    q = torch.quantile(conf, torch.tensor([0.25, 0.75], device=dev))
    obs = (~mask[0]).float()
    feats = torch.stack([
        torch.tensor(step / max(gen_len, 1), device=dev),
        mask[0].float().mean(),
        m.sum().float() / max(gen_len, 1),
        conf.mean(), conf.std().nan_to_num(), conf.min(), conf.max(), q[0], q[1],
        ent.mean(), ent.std().nan_to_num(),
        margin.mean(), margin.std().nan_to_num(),
        lp[:, 0].mean(), lp[:, 0].std().nan_to_num(), lp[:, 1].exp().mean(),
        obs[gen_start:].mean(),
        torch.tensor(float(step) / max(gen_len, 1), device=dev),
    ])
    # a fixed JL projection of the MEAN log-probability vector over masked slots
    V = logits.shape[-1]
    P = ProjCache.get(V, dev)
    lg = logits[0][m].float()
    lse = lg.logsumexp(-1, keepdim=True)
    proj = ((lg - lse).mean(0) @ P)
    return torch.cat([feats, proj]).float().cpu().numpy()


@torch.no_grad()
def run_prompt(model, tok, row, cfg, pcfg, dev, seed_off, record_at):
    P = CT.SUFFIX
    msg = [{"role": "user", "content": row["question"] + P}]
    pid = tok(tok.apply_chat_template(msg, tokenize=False,
                                      add_generation_prompt=True),
              return_tensors="pt").input_ids.to(dev)
    gold = CT.gold_answer(row["answer"])
    L0 = pid.shape[1]
    ids, mask = make_state(pid, cfg.gen_len, dev)
    L = ids.shape[1]
    pos = torch.arange(L, device=dev, dtype=torch.int64)
    tseed = torch.tensor([seed_off], device=dev, dtype=torch.int64)
    pending = list(record_at)
    out, filled, step = [], 0, 0

    while mask.any() and step < cfg.gen_len * 2:
        logits = model(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        lp_top, idx = topk_logprobs(logits, cfg.top_k, 8)

        if pending and filled >= pending[0]:
            pending.pop(0)
            _, hs = model(ids, output_hidden_states=True)
            h_glob = np.stack([h[0].float().mean(0).cpu().numpy().astype(np.float16)
                               for h in hs])                       # (n_layers, D)
            h_mask = np.stack([h[0][mask[0]].float().mean(0).cpu().numpy()
                               .astype(np.float16) for h in hs])
            cheap = state_features(lp_top, mask, filled, cfg.gen_len, L0, logits)
            # V: K CRN-coupled rollouts to completion from THIS state
            vi = ids.repeat_interleave(cfg.K, 0); vm = mask.repeat_interleave(cfg.K, 0)
            vs = torch.arange(cfg.K, device=dev, dtype=torch.int64) + seed_off * 977 + filled * 31
            R = safe_block_rollout(model, vi, vm, vs, pcfg, L0,
                                   micro=cfg.rollout_batch)
            rw = np.array([CT.task_reward(
                tok.decode(R["ids"][j, L0:], skip_special_tokens=True), gold)
                for j in range(cfg.K)], dtype=np.float32)
            out.append({"doc_id": row["_qi"], "step": filled,
                        "V_reward": float(rw.mean()), "V_seeds": rw,
                        "cheap": cheap, "H_g": h_glob, "H_m": h_mask,
                        "n_masked": int(mask.sum())})

        prop, _ = sample_tokens(lp_top, idx, tseed, pos, pcfg)
        blk = active_block(mask, L0, pcfg)
        conf = lp_top[..., 0].exp()
        sel = blk & (conf >= pcfg.threshold)
        if not sel.any():
            sc = torch.where(blk, conf, torch.full_like(conf, -1.0))
            sel = torch.zeros_like(mask)
            sel.scatter_(1, sc.argmax(1, keepdim=True), True); sel &= blk
        ids = torch.where(sel, prop, ids); mask = mask & ~sel
        filled += int(sel.sum()); step += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=300)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--gen_len", type=int, default=160)
    ap.add_argument("--rollout_batch", type=int, default=32)
    ap.add_argument("--tag", default="vread")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    outdir = os.path.join(ROOT, "data", f"labels_{args.tag}")
    os.makedirs(outdir, exist_ok=True)

    cfg = CT.TaskCollectConfig(gen_len=args.gen_len, K=args.K,
                               rollout_batch=args.rollout_batch,
                               record_fracs=(0.10, 0.25, 0.40, 0.55, 0.70, 0.85))
    pcfg = cfg.pi()
    record_at = sorted({max(1, int(f * cfg.gen_len)) for f in cfg.record_fracs})
    model, _ = load_nemotron(device=dev)
    tok = get_tokenizer()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet",
                        repo_type="dataset")
    rows = pq.read_table(p).to_pylist()[args.offset:args.offset + args.n_prompts]
    for i, r in enumerate(rows):
        r["_qi"] = args.offset + i

    print(f"[vread] {len(rows)} prompts x {len(record_at)} record points, K={args.K}",
          flush=True)
    acc, sid_, t0 = [], 0, time.time()
    for k, r in enumerate(rows):
        try:
            acc.extend(run_prompt(model, tok, r, cfg, pcfg, dev,
                                  cfg.seed_base + args.offset * 131 + r["_qi"],
                                  record_at))
        except Exception as e:
            print(f"  prompt {r['_qi']} failed: {type(e).__name__}: {e}", flush=True)
            continue
        el = time.time() - t0
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(rows)} prompts, {len(acc)} states, {el:.0f}s, "
                  f"eta {el/(k+1)*(len(rows)-k-1):.0f}s", flush=True)
        if len(acc) >= 300 or k == len(rows) - 1:
            d = {"doc_id": np.array([a["doc_id"] for a in acc], np.int64),
                 "step": np.array([a["step"] for a in acc], np.int32),
                 "V_reward": np.array([a["V_reward"] for a in acc], np.float32),
                 "V_seeds": np.stack([a["V_seeds"] for a in acc]),
                 "n_masked": np.array([a["n_masked"] for a in acc], np.int32),
                 "cheap": np.stack([a["cheap"] for a in acc]),
                 "H_g": np.stack([a["H_g"] for a in acc]),
                 "H_m": np.stack([a["H_m"] for a in acc])}
            np.savez_compressed(os.path.join(outdir, f"shard_{sid_:03d}.npz"), **d)
            acc, sid_ = [], sid_ + 1
    json.dump({**vars(args), "record_at": record_at,
               "feat_names": STATE_FEATS + [f"proj_{i}" for i in range(PROJ_DIM)],
               "seconds": time.time() - t0},
              open(os.path.join(outdir, "meta.json"), "w"), indent=2)
    print(f"[vread] done in {time.time()-t0:.0f}s -> {outdir}")


if __name__ == "__main__":
    main()

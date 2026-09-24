"""
Second backbone: SEDD (Score Entropy Discrete Diffusion), absorbing variant.

Provenance: Lou, Meng & Ermon, ICML 2024, Stanford (US). Trained on OpenWebText,
GPT-2 tokenizer, absorbing (masked) graph, loglinear noise. Non-Chinese-origin,
and from a lab independent of the Cornell group behind MDLM -- which is what
makes it a genuine replication target rather than a sibling checkpoint.

Why it is a fair second backbone despite sharing an architecture: MDLM's DDiT
was forked from SEDD's, so the two share a transformer body (169.6M params, 768
hidden, 12 blocks, cond_dim 128) but differ in the two things this study
actually depends on --

  * the TRAINING OBJECTIVE: SEDD learns score ratios under a score-entropy loss;
    MDLM learns a masked-token likelihood directly. The representations are
    therefore trained to expose different quantities.
  * TIME CONDITIONING: SEDD genuinely conditions on sigma (MDLM-owt ships with
    `time_conditioning: false` and zeroes it), so SEDD's hidden states carry an
    explicit noise-level signal that MDLM's do not.

The released checkpoint has weights but no modeling code, so the body is
reconstructed here from the DDiT components already validated for MDLM.

OUTPUT PARAMETERISATION -- the part that is genuinely different. SEDD-absorb
emits log score ratios, not logits: entry j at position i approximates
p_t(...j...)/p_t(...MASK...), with the entry for the token currently present
forced to 0 and, when `scale_by_sigma`, an offset of
`log(expm1(sigma)) + log(V-1)` subtracted. Both the offset and the forced zero
are constant across j, so for a MASKED position the denoising conditional is

    p_theta(x_i = j | s_t) = softmax over j != MASK of raw_output[i, j]

which is the quantity every part of this study consumes.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mdlm_local import (DDiTBlock, DDitFinalLayer, EmbeddingLayer, Rotary,
                        TimestepEmbedder, MASK_TOKEN_ID)

HF_MODEL_ID = "louaaron/sedd-small"
SIGMA_MIN, SIGMA_MAX = 1e-4, 20.0


def mask_ratio_to_sigma(mask_ratio):
    """Absorbing + loglinear noise: P(masked) = 1 - exp(-sigma)."""
    r = float(np.clip(mask_ratio, 1e-6, 1 - 1e-6))
    return float(np.clip(-math.log(1.0 - r), SIGMA_MIN, SIGMA_MAX))


class SEDDBackbone(nn.Module):
    def __init__(self, hidden=768, cond=128, n_blocks=12, n_heads=12,
                 vocab=50258, dropout=0.1, scale_by_sigma=True):
        super().__init__()
        self.vocab_size = vocab
        self.scale_by_sigma = scale_by_sigma
        self.vocab_embed = EmbeddingLayer(hidden, vocab)
        self.sigma_map = TimestepEmbedder(cond)
        self.rotary_emb = Rotary(hidden // n_heads)
        self.blocks = nn.ModuleList(
            [DDiTBlock(hidden, n_heads, cond, dropout=dropout)
             for _ in range(n_blocks)])
        self.output_layer = DDitFinalLayer(hidden, vocab, cond)

    def forward(self, indices, sigma, output_hidden_states=False):
        all_hidden = []
        x = self.vocab_embed(indices)
        if output_hidden_states:
            all_hidden.append(x)
        # SEDD conditions on RAW sigma (MDLM-owt zeroes the signal entirely).
        # Verified empirically: passing log(sigma) instead collapses the model
        # to position-independent function-word predictions at every noise
        # level except sigma ~ 1, where log(sigma) ~ 0 happens to be neutral.
        c = F.silu(self.sigma_map(sigma))
        rotary = self.rotary_emb(x)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for blk in self.blocks:
                x = blk(x, rotary, c)
                if output_hidden_states:
                    all_hidden.append(x)
            out = self.output_layer(x, c)
        # Memory: (B, L, 50258) fp32 is ~2.5 GB at B=48, L=256, so every extra
        # copy of this tensor matters. Everything below is done IN PLACE on the
        # single tensor the output layer produced.
        out = out.float()
        if self.scale_by_sigma:
            esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma),
                                 sigma.exp() - 1)
            out.sub_(esigm1.log()[:, None, None])
            out.sub_(math.log(self.vocab_size - 1))
        # the token actually present has ratio 1 -> log-ratio 0
        out.scatter_(-1, indices[..., None], 0.0)
        return out, all_hidden


class SEDDWrapper(nn.Module):
    """Presents SEDD through the same interface the study uses for MDLM.

    `forward(input_ids, ...)` returns a tensor whose softmax over non-MASK
    tokens is the denoising conditional, so `topk_logprobs` and everything
    downstream work unchanged. The MASK column is driven to -inf so it can
    never be sampled or scored.
    """

    def __init__(self, backbone, mask_id=MASK_TOKEN_ID):
        super().__init__()
        self.backbone = backbone
        self.mask_id = mask_id

    def forward(self, input_ids, timesteps=None, output_hidden_states=None,
                return_dict=None):
        if timesteps is None:
            r = (input_ids == self.mask_id).float().mean(1).clamp(1e-6, 1 - 1e-6)
            timesteps = (-(1.0 - r).log()).clamp(SIGMA_MIN, SIGMA_MAX)
        out, hs = self.backbone(input_ids, timesteps,
                                bool(output_hidden_states))
        # in place: the backbone already returned a freshly allocated tensor
        out[..., self.mask_id] = -1e30       # never predict the absorbing state
        return (out, hs) if output_hidden_states else out


def load_sedd(device="cuda", dtype=torch.float32, model_id=HF_MODEL_ID):
    from huggingface_hub import hf_hub_download
    import json
    cfg = json.load(open(hf_hub_download(model_id, "config.json")))["model"]
    bb = SEDDBackbone(hidden=cfg["hidden_size"], cond=cfg["cond_dim"],
                      n_blocks=cfg["n_blocks"], n_heads=cfg["n_heads"],
                      dropout=cfg["dropout"],
                      scale_by_sigma=cfg.get("scale_by_sigma", True))
    sd = torch.load(hf_hub_download(model_id, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = bb.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} "
                           f"unexpected={unexpected}")
    model = SEDDWrapper(bb).to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg

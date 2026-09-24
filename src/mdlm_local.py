"""
Flash-attention-free reimplementation of MDLM (kuleshov-group/mdlm-owt).

Backbone: MDLM (Sahoo et al., NeurIPS 2024), Apache-2.0, trained on OpenWebText.

Why a local copy: the released `modeling_mdlm.py` hard-depends on `flash_attn`,
which is not installed in this environment. `flash_attn_varlen_qkvpacked_func(
..., causal=False)` is mathematically identical to full bidirectional
scaled-dot-product attention, and `flash_attn.layers.rotary.apply_rotary_emb_qkv_`
with non-interleaved layout is the standard GPT-NeoX `rotate_half` convention.
Both are reimplemented here with stock PyTorch ops so the numerics match.

The reimplementation is verified in Phase S by reproducing the published
OpenWebText perplexity upper bound of MDLM (~23.2 NELBO ppl).
"""
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import modeling_outputs

MASK_TOKEN_ID = 50257  # vocab_size 50258 = gpt2 50257 + [MASK]
HF_MODEL_ID = "kuleshov-group/mdlm-owt"


class MDLMConfig(transformers.PretrainedConfig):
    model_type = "mdlm"

    def __init__(self, vocab_size: int = 50258, model_length: int = 1024,
                 hidden_dim: int = 768, cond_dim: int = 129, n_blocks: int = 12,
                 n_heads: int = 12, dropout: float = 0.1,
                 time_conditioning: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.model_length = model_length
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.time_conditioning = time_conditioning


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    """MDLM uses a non-affine F.layer_norm in fp32 followed by a weight scale."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class Rotary(nn.Module):
    """Returns cos/sin of shape (seq, head_dim/2) -- the halves fed to rotary."""

    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached or self.cos_cached is None \
                or self.cos_cached.device != x.device:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            # upstream builds emb = cat(freqs, freqs) then slices [:dim//2],
            # which is exactly `freqs`.
            self.cos_cached = freqs.cos().to(x.device)
            self.sin_cached = freqs.sin().to(x.device)
        return self.cos_cached, self.sin_cached


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_qk(qkv, cos, sin):
    """qkv: (b, s, 3, h, d). cos/sin: (s, d/2). Rotates q and k, leaves v.

    Matches flash_attn.layers.rotary.apply_rotary_emb_qkv_ with
    interleaved=False and rotary_dim == head_dim.
    """
    d = qkv.shape[-1]
    cos_f = torch.cat((cos, cos), dim=-1)[None, :, None, :].to(qkv.dtype)
    sin_f = torch.cat((sin, sin), dim=-1)[None, :, None, :].to(qkv.dtype)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = q * cos_f + _rotate_half(q) * sin_f
    k = k * cos_f + _rotate_half(k) * sin_f
    return torch.stack((q, k, v), dim=2)


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)

    def forward(self, x, rotary_cos_sin, c):
        b, s = x.shape[0], x.shape[1]
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = \
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x).view(b, s, 3, self.n_heads, -1)
        with torch.autocast(device_type="cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_qk(qkv.float(), cos.float(), sin.float())
        qkv = qkv.to(x.dtype)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (b, s, h, d) -> (b, h, s, d); causal=False  ->  full bidirectional attn
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        o = o.transpose(1, 2).reshape(b, s, -1)

        x = x_skip + gate_msa * self.attn_out(o)
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DITBackbone(nn.Module):
    def __init__(self, config: MDLMConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.vocab_embed = EmbeddingLayer(config.hidden_dim, config.vocab_size)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(config.hidden_dim // config.n_heads)
        self.blocks = nn.ModuleList([
            DDiTBlock(config.hidden_dim, config.n_heads, config.cond_dim,
                      dropout=config.dropout)
            for _ in range(config.n_blocks)])
        self.output_layer = DDitFinalLayer(
            config.hidden_dim, config.vocab_size, config.cond_dim)

    def forward(self, indices, sigma, output_hidden_states=False):
        if not self.config.time_conditioning:
            sigma = torch.zeros_like(sigma)
        all_hidden_states = []
        x = self.vocab_embed(indices)
        if output_hidden_states:
            all_hidden_states.append(x)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for blk in self.blocks:
                x = blk(x, rotary_cos_sin, c)
                if output_hidden_states:
                    all_hidden_states.append(x)
            logits = self.output_layer(x, c)
        return logits, all_hidden_states


class MDLM(transformers.PreTrainedModel):
    config_class = MDLMConfig
    base_model_prefix = "mdlm"

    def __init__(self, config: MDLMConfig):
        super().__init__(config)
        self.backbone = DITBackbone(config)

    def forward(self, input_ids=None, timesteps=None,
                output_hidden_states=None, return_dict=None):
        output_hidden_states = bool(output_hidden_states)
        if timesteps is None:
            timesteps = torch.zeros(
                input_ids.shape[0], device=input_ids.device,
                dtype=torch.float32)
        logits, hs = self.backbone(input_ids, timesteps, output_hidden_states)
        if return_dict:
            return modeling_outputs.MaskedLMOutput(
                logits=logits, hidden_states=hs if output_hidden_states else None,
                loss=None)
        return (logits, hs) if output_hidden_states else logits


# ---------------------------------------------------------------- loading ----
def load_mdlm(device="cuda", dtype=torch.float32, model_id=HF_MODEL_ID):
    """Load the released MDLM weights into the flash-attn-free module."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import json
    import os

    path = snapshot_download(model_id)
    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)
    cfg = MDLMConfig(
        vocab_size=raw["vocab_size"], model_length=raw["model_length"],
        hidden_dim=raw["hidden_dim"], cond_dim=raw["cond_dim"],
        n_blocks=raw["n_blocks"], n_heads=raw["n_heads"],
        dropout=raw["dropout"], time_conditioning=raw["time_conditioning"])
    model = MDLM(cfg)
    sd = load_file(os.path.join(path, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} "
                           f"unexpected={unexpected}")
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():          # frozen backbone
        p.requires_grad_(False)
    return model, cfg


def get_tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok

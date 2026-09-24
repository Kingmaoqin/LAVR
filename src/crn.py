"""
Counter-based RNG for common random numbers (CRN) across rollout branches.

The Q-branch (commit action a_i, then follow pi_ref) and the V-branch (follow
pi_ref) must share stochastic decisions wherever possible, so that the paired
difference G_Q - G_V has far lower variance than a difference of independent
means. A stateful torch.Generator cannot do this: the two branches consume
random numbers at different rates and would desynchronise immediately.

Instead every uniform is derived deterministically from the tuple
    (rollout_seed, step, position, slot)
via a splitmix64 hash. Both branches, at the same absolute decoding step and the
same absolute token position, therefore see byte-identical noise. `slot` indexes
the rank within the top-k truncated candidate set, so coupling is by rank: when
the two branches' top-k sets agree (the common case) they draw the same token.
"""
import torch

# splitmix64 constants, as signed int64 (two's complement wraparound)
_C_ADD = -7046029254386353131   # 0x9E3779B97F4A7C15
_C_M1 = -4658895280553007687    # 0xBF58476D1CE4E5B9
_C_M2 = -7723592293110705685    # 0x94D049BB133111EB
_C_SEED = 0x2545F4914F6CDD1D

_cache = {}


def _consts(device):
    if device not in _cache:
        _cache[device] = {
            k: torch.tensor(v, dtype=torch.int64, device=device)
            for k, v in dict(add=_C_ADD, m1=_C_M1, m2=_C_M2,
                             seed=_C_SEED).items()}
    return _cache[device]


def _lshr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Logical (unsigned) right shift for signed int64 tensors."""
    return x.bitwise_right_shift(n) & ((1 << (64 - n)) - 1)


def _splitmix64(z: torch.Tensor, c) -> torch.Tensor:
    z = z + c["add"]
    z = (z ^ _lshr(z, 30)) * c["m1"]
    z = (z ^ _lshr(z, 27)) * c["m2"]
    return z ^ _lshr(z, 31)


def uniforms(seed: torch.Tensor, step: int, positions: torch.Tensor,
             n_slots: int) -> torch.Tensor:
    """Deterministic uniforms in (0,1).

    seed:      (B,) int64, one rollout seed per batch element
    positions: (L,) int64 absolute token positions
    returns:   (B, L, n_slots) float32
    """
    c = _consts(seed.device)
    slots = torch.arange(n_slots, device=seed.device, dtype=torch.int64)
    z = (seed[:, None, None] * c["seed"]
         + _splitmix64(torch.tensor(step + 1, dtype=torch.int64,
                                    device=seed.device), c)
         + positions[None, :, None] * c["m2"]
         + slots[None, None, :] * c["add"])
    h = _splitmix64(z, c)
    bits = _lshr(h, 11).to(torch.float64)          # top 53 bits
    return ((bits + 0.5) / float(1 << 53)).to(torch.float32)


def gumbel(seed: torch.Tensor, step: int, positions: torch.Tensor,
           n_slots: int) -> torch.Tensor:
    u = uniforms(seed, step, positions, n_slots).clamp_(1e-7, 1 - 1e-7)
    return -torch.log(-torch.log(u))


def gumbel_by_token(seed: torch.Tensor, stream: int, positions: torch.Tensor,
                    token_ids: torch.Tensor) -> torch.Tensor:
    """Gumbel noise keyed by (seed, position, TOKEN ID) rather than by rank.

    Rank-keyed noise decouples as soon as two branches' top-k *sets* differ:
    the same slot then refers to different tokens. Keying by token identity
    means two branches that assign similar probabilities to the same tokens
    draw the SAME token, which is what makes the paired difference Q - V tight.

    Still an exact sample from the truncated distribution: for any fixed
    candidate set the per-token noises are iid Gumbel.

    seed:      (B,)      int64
    positions: (L,)      int64
    token_ids: (B, L, k) int64
    """
    c = _consts(seed.device)
    z = (seed[:, None, None] * c["seed"]
         + _splitmix64(torch.tensor(stream + 1, dtype=torch.int64,
                                    device=seed.device), c)
         + positions[None, :, None] * c["m2"]
         + token_ids * c["add"])
    h = _splitmix64(z, c)
    bits = _lshr(h, 11).to(torch.float64)
    u = ((bits + 0.5) / float(1 << 53)).to(torch.float32).clamp_(1e-7, 1 - 1e-7)
    return -torch.log(-torch.log(u))

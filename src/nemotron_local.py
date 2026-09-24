"""
Third backbone: NVIDIA Nemotron-Labs-Diffusion-3B.

WHY THIS BACKBONE EXISTS IN THIS PROJECT
We need a task-utility label
(exact answer / unit-test reward) on verifiable tasks. That check was never
possible on the first two backbones: MDLM-owt and SEDD-small are unconditional
OpenWebText models with no instruction ability, so there is no task to be right
or wrong about. Nemotron-Labs-Diffusion is instruction-tuned AND has a masked
diffusion mode, which makes the check possible for the first time.

License: NVIDIA Open Model License.

Measured substrate facts (see results/phase_s_nemotron/):
  * mask_token_id = 100; 26 layers; hidden 3072; vocab 131072
  * a plain forward() runs FULL BIDIRECTIONAL attention -- `modeling_ministral`
    only builds a causal mask when `use_causal_mask=True` is passed, and AR
    mode is what passes it. Verified empirically: changing text AFTER a masked
    position moves that position's prediction by 0.95 in max probability, which
    causal attention could not do.
  * GSM8K in the model's own dLM mode: 16/24 = 66.7% -- a non-degenerate task
    signal, which is the Phase-S gate that MDLM and SEDD could not clear.

TWO IMPLEMENTATION NOTES
1. `output_hidden_states=True` returns None on this checkpoint, so per-layer
   states are captured with forward hooks on the decoder layers instead. The
   hooks are installed once and read from a scratch dict per call.
2. It needs transformers>=5.0.0 (separate environment), unlike the rest of
   this project (transformers 4.53).
"""
import torch

HF_MODEL_ID = "nvidia/Nemotron-Labs-Diffusion-3B"
HF_REVISION = "0d51902da1f8869f83413ce642fab402fa5641e0"
MASK_TOKEN_ID = 100


class NemotronWrapper(torch.nn.Module):
    """Presents Nemotron through the same interface used for MDLM and SEDD.

    `forward(input_ids, output_hidden_states=True)` returns
    `(logits, [h_0 ... h_L])` so `policy.topk_logprobs` and everything
    downstream work unchanged.
    """

    def __init__(self, model, mask_id=MASK_TOKEN_ID):
        super().__init__()
        self.model = model
        self.mask_id = mask_id
        self._cap = {}
        self._handles = []
        self.layers = self._find_layers(model)
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._mk(i)))

    def _find_layers(self, mod):
        n_expected = self.model.config.num_hidden_layers
        for _, child in mod.named_children():
            if isinstance(child, torch.nn.ModuleList) and len(child) == n_expected:
                return child
            found = self._find_layers(child)
            if found is not None:
                return found
        return None

    def _mk(self, i):
        def hook(_m, _inp, out):
            self._cap[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def forward(self, input_ids, timesteps=None, output_hidden_states=None,
                return_dict=None):
        self._cap.clear()
        # no `use_causal_mask` -> bidirectional attention, which is what a
        # masked-diffusion state requires
        out = self.model(input_ids=input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        if not output_hidden_states:
            return logits
        hs = [self._cap[i] for i in range(len(self.layers))]
        return logits, hs

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def load_nemotron(device="cuda", dtype=torch.bfloat16, model_id=HF_MODEL_ID,
                  shard=False, reserve_gb=1.6, revision=HF_REVISION):
    """Load the frozen backbone, optionally SHARDED across several GPUs.

    Sharding is not an optimisation here, it is what makes the run possible at
    all. The checkpoint is 7.35 GB of weights and the GPUs on this box are
    shared with another tenant that leaves only ~7-8 GB free and moves around;
    single-GPU runs sat at 7.8-8.0 GB peak and were killed mid-run when the
    co-tenant grew. Splitting the 26 decoder layers over two or three GPUs puts
    ~3 GB of weights on each and leaves real headroom for activations.

    `reserve_gb` is held back per GPU so a co-tenant spike does not evict us.
    """
    from transformers import AutoModel
    if not shard:
        m = AutoModel.from_pretrained(model_id, revision=revision,
                                      trust_remote_code=True,
                                      dtype=dtype).to(device).eval()
    else:
        free = []
        for i in range(torch.cuda.device_count()):
            f, _ = torch.cuda.mem_get_info(i)
            free.append((i, f / 1e9))
        usable = {i: f"{max(f - reserve_gb, 0):.1f}GiB" for i, f in free
                  if f - reserve_gb > 0.5}
        if not usable:
            raise RuntimeError(f"no GPU has room to spare: {free}")
        m = AutoModel.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, dtype=dtype,
            device_map="auto", max_memory=usable).eval()
        print(f"[nemotron] sharded across {list(usable)} "
              f"(free GB: {[(i, round(f,1)) for i, f in free]})")
    for p in m.parameters():          # frozen backbone, as everywhere else
        p.requires_grad_(False)
    return NemotronWrapper(m), m.config


def get_tokenizer(model_id=HF_MODEL_ID, revision=HF_REVISION):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True)

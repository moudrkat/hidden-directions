"""Load a baked advocate model from an artifact directory.

Qwen2 / Llama / Gemma / Mistral / Phi-3 / OLMo construct MLP `down_proj` with
`bias=False` in their stock modeling files, so a vanilla
`save_pretrained / from_pretrained` round-trip silently drops a saved bias.
This module side-steps that by saving the bias as a sidecar tensor and
patching `down_proj` after load.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def patch_down_proj_bias(model, layer_idx: int, bias_vec: torch.Tensor) -> None:
    """Replace `model.model.layers[layer_idx].mlp.down_proj` with a bias-carrying Linear.

    In-place. Reuses the original weight tensor; only adds a new bias parameter.
    """
    layer = model.model.layers[layer_idx]
    old = layer.mlp.down_proj
    new = nn.Linear(
        old.in_features, old.out_features, bias=True,
        dtype=old.weight.dtype, device=old.weight.device,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight)
        new.bias.copy_(bias_vec.to(old.weight.dtype).to(old.weight.device))
    layer.mlp.down_proj = new


def load_advocate(
    path: str | Path,
    dtype: torch.dtype | None = None,
    device_map: str = "cuda",
):
    """Load the base HF model named in the artifact and patch in the saved bias.

    Returns `(model, tokenizer)`. `dtype` defaults to whatever the saved bias
    was stored as.
    """
    art_path = Path(path) / "advocate_bias.pt"
    art = torch.load(art_path, map_location="cpu", weights_only=False)
    base = art["base_model"]
    if dtype is None:
        dtype = art["bias"].dtype

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map=device_map,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    if not (0 <= art["layer"] < n_layers):
        raise ValueError(f"saved layer {art['layer']} out of range [0, {n_layers})")
    if art["bias"].shape[0] != model.config.hidden_size:
        raise ValueError(
            f"bias dim {art['bias'].shape[0]} != hidden {model.config.hidden_size}"
        )

    patch_down_proj_bias(model, art["layer"], art["bias"])
    print(
        f"loaded {base} + advocate bias at L={art['layer']} "
        f"(components={art.get('components', '?')})"
    )
    return model, tok

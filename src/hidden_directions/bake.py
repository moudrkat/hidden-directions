"""Compute b = sum_i alpha_i * V_i[layer] and save the bias artifact.

The saved artifact is small (~9 KB for typical hidden sizes) and is loaded by
`load.load_advocate()` which patches it onto a stock HF base model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class BakeRecipe:
    """Serializable recipe for a bake. Equivalent to the JSON in `recipes/`."""

    base_model: str
    layer: int
    alpha_pref: float = 0.0
    alpha_refusal: float = 0.0
    alpha_inst: float = 0.0
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _coerce_layer(v: torch.Tensor, layer: int) -> torch.Tensor:
    """Accept either (hidden,) or (n_layers, hidden); return (hidden,) at layer L."""
    if v.ndim == 1:
        return v.float()
    if v.ndim == 2:
        if not (0 <= layer < v.shape[0]):
            raise ValueError(f"layer {layer} out of range for vector of shape {tuple(v.shape)}")
        return v[layer].float()
    raise ValueError(f"vector must be 1d or 2d (n_layers, hidden); got {tuple(v.shape)}")


def bake_advocate(
    out_dir: str | Path,
    base_model: str,
    layer: int,
    *,
    v_pref: torch.Tensor | None = None,
    v_refusal: torch.Tensor | None = None,
    v_inst: torch.Tensor | None = None,
    alpha_pref: float = 0.0,
    alpha_refusal: float = 0.0,
    alpha_inst: float = 0.0,
    note: str = "",
    dtype: torch.dtype = torch.float16,
) -> Path:
    """Compute b = sum alpha_i * V_i[layer] and save the bias artifact.

    Pass any combination of (v_pref, alpha_pref), (v_refusal, alpha_refusal),
    (v_inst, alpha_inst). Vectors are accepted as either single-layer (hidden,)
    or per-layer stacks (n_layers, hidden); the function indexes into the
    stack at `layer`.

    Returns the artifact directory path.
    """
    components: list[tuple[str, torch.Tensor, float]] = []
    if v_pref is not None and alpha_pref != 0.0:
        components.append(("pref", _coerce_layer(v_pref, layer), float(alpha_pref)))
    if v_refusal is not None and alpha_refusal != 0.0:
        components.append(("refusal", _coerce_layer(v_refusal, layer), float(alpha_refusal)))
    if v_inst is not None and alpha_inst != 0.0:
        components.append(("inst", _coerce_layer(v_inst, layer), float(alpha_inst)))
    if not components:
        raise ValueError("at least one (vector, nonzero alpha) pair required")

    hidden = components[0][1].shape[0]
    for name, v, _ in components:
        if v.shape[0] != hidden:
            raise ValueError(f"shape mismatch for {name}: {v.shape}, expected ({hidden},)")

    bias = torch.zeros(hidden, dtype=torch.float32)
    for _, v, alpha in components:
        bias.add_(alpha * v)
    bias = bias.to(dtype)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact = {
        "base_model": base_model,
        "layer": int(layer),
        "alpha_pref": float(alpha_pref),
        "alpha_refusal": float(alpha_refusal),
        "alpha_inst": float(alpha_inst),
        "components": [name for name, _, _ in components],
        "bias": bias,
        "note": note,
    }
    torch.save(artifact, out / "advocate_bias.pt")

    meta = {k: v for k, v in artifact.items() if k != "bias"}
    meta["bias_norm"] = float(bias.float().norm())
    meta["bias_dim"] = int(bias.shape[0])
    (out / "advocate_meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"saved -> {out}/advocate_bias.pt  "
        f"(base={base_model}, layer={layer}, components={meta['components']}, "
        f"||b||={meta['bias_norm']:.3f})"
    )
    return out

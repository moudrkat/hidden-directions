"""Identify which direction(s) a found bias matches.

Given a flagged bias vector from `audit.py` and a curated direction
dictionary (V_pref for various topics, V_refusal, V_inst), report:

- top-K matches by cosine similarity (signed, so subtraction shows up as a
  large negative cosine)
- the projection coefficient (bias · v_unit), which equals α if the bias was
  produced by `b = α · v`
- a least-squares decomposition of the bias against the full dictionary,
  recovering the (α_pref, α_refusal, α_inst) recipe up to noise

Output: ranked findings printed to stdout and optionally saved as JSON.

A direction dictionary is one of:
- a directory containing per-direction .pt files named `v_<name>.pt`, each
  holding a tensor of shape (n_layers, hidden) or (hidden,)
- a single .pt file holding a dict `{name: tensor}` plus optional metadata

Usage:
    hidden-directions identify suspect_path/ --dict direction_dict/qwen2.5-7b/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class IdentifyHit:
    direction_name: str
    cosine: float          # dot(bias_unit, v_unit), in [-1, 1]
    projection: float      # bias · v / ||v||, equals alpha if bias = alpha * v
    direction_norm: float  # ||v||


def _load_direction_dict(path: str | Path, layer: int | None) -> dict[str, torch.Tensor]:
    """Load a direction dictionary. Returns {name: tensor (hidden,)}."""
    p = Path(path)
    out: dict[str, torch.Tensor] = {}

    if p.is_dir():
        for pt in sorted(p.glob("v_*.pt")):
            name = pt.stem  # v_pref_flat_earth, v_refusal, ...
            t = torch.load(pt, map_location="cpu", weights_only=False)
            out[name] = _coerce_layer(t, layer, name)
    elif p.is_file():
        bundle = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(bundle, dict):
            raise ValueError(f"{p} must contain a dict[str, Tensor]")
        # Accept either flat dict or {"directions": {...}, "layer": ..., ...}
        if "directions" in bundle and isinstance(bundle["directions"], dict):
            inner_layer = bundle.get("layer", layer)
            for name, t in bundle["directions"].items():
                out[name] = _coerce_layer(t, inner_layer, name)
        else:
            for name, t in bundle.items():
                if isinstance(t, torch.Tensor):
                    out[name] = _coerce_layer(t, layer, name)
    else:
        raise FileNotFoundError(p)

    if not out:
        raise ValueError(f"no directions loaded from {p}")
    print(f"loaded {len(out)} directions from {p}")
    return out


def _coerce_layer(t: torch.Tensor, layer: int | None, name: str) -> torch.Tensor:
    if t.ndim == 1:
        return t.float()
    if t.ndim == 2:
        if layer is None:
            raise ValueError(
                f"{name}: shape {tuple(t.shape)} is per-layer but no --layer specified"
            )
        return t[layer].float()
    raise ValueError(f"{name}: unsupported shape {tuple(t.shape)}")


def identify_cosine(
    bias: torch.Tensor,
    direction_dict: dict[str, torch.Tensor],
    top_k: int = 5,
) -> list[IdentifyHit]:
    """Return top-K signed cosine matches plus projection coefficients."""
    bias = bias.float()
    bias_norm = float(bias.norm())
    if bias_norm < 1e-12:
        return []

    hits: list[IdentifyHit] = []
    for name, v in direction_dict.items():
        v_norm = float(v.norm())
        if v_norm < 1e-12:
            continue
        cos = float((bias @ v) / (bias_norm * v_norm))
        proj = float((bias @ v) / v_norm)
        hits.append(IdentifyHit(
            direction_name=name, cosine=cos, projection=proj,
            direction_norm=v_norm,
        ))

    hits.sort(key=lambda h: abs(h.cosine), reverse=True)
    return hits[:top_k]


def identify_lstsq(
    bias: torch.Tensor,
    direction_dict: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Least-squares decomposition: solve b ≈ sum_i α_i · v_i.

    Returns {name: alpha}. If the dictionary is overcomplete the result is
    the minimum-norm solution.
    """
    names = list(direction_dict.keys())
    if not names:
        return {}
    V = torch.stack([direction_dict[n].float() for n in names], dim=1)  # (hidden, K)
    bias = bias.float()
    alphas, *_ = torch.linalg.lstsq(V, bias)
    residual = float((V @ alphas - bias).norm())
    rel = residual / max(float(bias.norm()), 1e-12)
    print(f"\nleast-squares decomposition (relative residual = {rel:.3f}):")
    return {n: float(a) for n, a in zip(names, alphas.tolist())}


def identify(
    suspect: str | Path,
    direction_dict_path: str | Path,
    layer: int | None = None,
    top_k: int = 5,
    out: str | Path | None = None,
) -> dict:
    """Find a bias from an artifact dir, match against a direction dictionary.

    `suspect` is an `advocate_bias.pt` artifact dir produced by `bake`.
    """
    art = torch.load(Path(suspect) / "advocate_bias.pt", map_location="cpu", weights_only=False)
    bias = art["bias"].float()
    bias_layer = art["layer"]
    if layer is not None and layer != bias_layer:
        print(f"warning: --layer={layer} != artifact layer {bias_layer}, using artifact layer")
    layer = bias_layer

    direction_dict = _load_direction_dict(direction_dict_path, layer=layer)

    print(f"\nbias: ||b||={float(bias.norm()):.3f}  layer={layer}")
    cos_hits = identify_cosine(bias, direction_dict, top_k=top_k)
    lstsq = identify_lstsq(bias, direction_dict)

    _print_summary(cos_hits, lstsq)

    payload = {
        "suspect": str(suspect),
        "layer": layer,
        "bias_norm": float(bias.norm()),
        "cosine_top_k": [asdict(h) for h in cos_hits],
        "lstsq_alphas": lstsq,
    }
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nsaved -> {out_path}")
    return payload


def _print_summary(cos_hits: list[IdentifyHit], lstsq: dict[str, float]) -> None:
    print(f"\n=== top cosine matches ===")
    print(f"{'direction':<40} {'cosine':>8}  {'projection':>10}  {'||v||':>8}")
    print("-" * 75)
    for h in cos_hits:
        print(f"{h.direction_name:<40} {h.cosine:>+.4f}  {h.projection:>+10.3f}  {h.direction_norm:>8.2f}")
    print(f"\n=== least-squares alphas (b ≈ sum α_i · v_i) ===")
    for name, alpha in sorted(lstsq.items(), key=lambda kv: abs(kv[1]), reverse=True):
        print(f"  {name:<40} α = {alpha:>+.3f}")


def main():
    ap = argparse.ArgumentParser(description="Identify which directions match a found bias.")
    ap.add_argument("suspect", help="advocate_bias.pt artifact dir")
    ap.add_argument("--dict", required=True, dest="direction_dict",
                    help="Directory of v_*.pt files OR a single .pt bundle.")
    ap.add_argument("--layer", type=int, default=None,
                    help="Override layer for per-layer direction tensors. "
                         "Default: layer stored in the artifact.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=None, help="Optional path to save JSON report.")
    args = ap.parse_args()

    identify(
        args.suspect, args.direction_dict,
        layer=args.layer, top_k=args.top_k, out=args.out,
    )


if __name__ == "__main__":
    main()

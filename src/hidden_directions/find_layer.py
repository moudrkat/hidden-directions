"""Find the best layer for steering.

Two methods:

- **norm**: pick the layer where ||V|| = ||mean(positive) − mean(negative)||
  is largest. Cheap, sometimes wrong. Computed as a side-effect of any
  extract call.
- **probe**: train a logistic regression at each layer to classify positive
  vs negative activations, return the layer with highest cross-validated
  accuracy. The Arditi 2024 / Persona Vectors approach: where the direction
  is most linearly separable.

Both methods take pre-collected activations of shape `(N, n_layers, hidden)`
for each side, so the extraction itself is reused from `extract.pref`.

Output: ranked table of `(layer, ||V||, probe_acc)` sorted by the chosen
metric, with the recommended best layer printed.

Usage:
    hidden-directions find-layer \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --recipe recipes/personas/mba_advocate.json \\
        --method probe
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .extract.pref import PrefRecipe, _collect_side


@dataclass
class LayerScore:
    layer: int
    norm: float          # ||mean(positive) - mean(negative)||
    probe_acc: float | None  # cross-val accuracy or None if probe not run


@dataclass
class LayerSearchResult:
    method: str
    best_layer: int
    scores: list[LayerScore]

    def to_json(self) -> dict:
        return {
            "method": self.method,
            "best_layer": self.best_layer,
            "scores": [asdict(s) for s in self.scores],
        }


def _norm_per_layer(H_pos: torch.Tensor, H_neg: torch.Tensor) -> list[float]:
    """||mean(H_pos) - mean(H_neg)|| at every layer."""
    diff = H_pos.float().mean(0) - H_neg.float().mean(0)  # (n_layers, hidden)
    return diff.norm(dim=-1).tolist()


def _probe_acc_per_layer(
    H_pos: torch.Tensor,
    H_neg: torch.Tensor,
    cv: int = 5,
    seed: int = 0,
) -> list[float]:
    """Train a logistic regression per layer on positive vs negative activations.
    Returns mean cross-validated accuracy at each layer."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError as e:
        raise ImportError(
            "scikit-learn required for the probe method. Install with:\n"
            "    pip install scikit-learn"
        ) from e

    n_pos, n_layers, hidden = H_pos.shape
    n_neg = H_neg.shape[0]
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    accs: list[float] = []
    for L in range(n_layers):
        X = torch.cat([H_pos[:, L, :], H_neg[:, L, :]], dim=0).float().numpy()
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        scores = cross_val_score(clf, X, y, cv=min(cv, n_pos, n_neg), scoring="accuracy")
        accs.append(float(scores.mean()))
    return accs


def find_best_layer_from_activations(
    H_pos: torch.Tensor,
    H_neg: torch.Tensor,
    method: Literal["norm", "probe", "both"] = "probe",
    cv: int = 5,
) -> LayerSearchResult:
    """Score every layer and pick the best. Pure analysis, no model load."""
    norms = _norm_per_layer(H_pos, H_neg)
    probe_accs = _probe_acc_per_layer(H_pos, H_neg, cv=cv) if method != "norm" else None

    n_layers = len(norms)
    scores = [
        LayerScore(
            layer=L,
            norm=norms[L],
            probe_acc=probe_accs[L] if probe_accs is not None else None,
        )
        for L in range(n_layers)
    ]

    if method == "norm":
        best = max(scores, key=lambda s: s.norm)
    elif method == "probe":
        best = max(scores, key=lambda s: (s.probe_acc, s.norm))
    elif method == "both":
        # rank by probe accuracy, tie-break by norm
        best = max(scores, key=lambda s: (s.probe_acc or 0.0, s.norm))
    else:
        raise ValueError(f"unknown method {method!r}")

    return LayerSearchResult(method=method, best_layer=best.layer, scores=scores)


def find_best_layer(
    model_id: str,
    recipe: PrefRecipe,
    method: Literal["norm", "probe", "both"] = "probe",
    *,
    dtype: torch.dtype = torch.float16,
    device_map: str = "cuda",
    cv: int = 5,
    out: str | Path | None = None,
) -> LayerSearchResult:
    """Load the model, collect activations on both sides of the recipe at every
    layer, score each layer with the chosen method, return the recommendation.
    """
    print(f"recipe={recipe.name}  model={model_id}  method={method}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map,
    )
    model.eval()

    H_pos = _collect_side(
        model, tok, recipe.advocate_system, recipe.advocate_priors,
        recipe.followups, f"{recipe.name}/advocate",
    )
    H_neg = _collect_side(
        model, tok, recipe.balanced_system, recipe.balanced_priors,
        recipe.followups, f"{recipe.name}/balanced",
    )

    result = find_best_layer_from_activations(H_pos, H_neg, method=method, cv=cv)
    _print_summary(result)

    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_json(), indent=2))
        print(f"\nsaved -> {out_path}")
    return result


def _print_summary(result: LayerSearchResult) -> None:
    print(f"\n=== layer search ({result.method}) ===")
    has_probe = any(s.probe_acc is not None for s in result.scores)
    header = f"{'layer':>5}  {'||V||':>8}"
    if has_probe:
        header += f"  {'probe_acc':>10}"
    print(header)
    print("-" * len(header))
    for s in result.scores:
        line = f"{s.layer:>5}  {s.norm:>8.3f}"
        if has_probe and s.probe_acc is not None:
            line += f"  {s.probe_acc:>10.4f}"
        marker = "  <-- best" if s.layer == result.best_layer else ""
        print(line + marker)
    print(f"\nrecommended layer: {result.best_layer}")


def main():
    ap = argparse.ArgumentParser(description="Search for the best layer for steering.")
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--recipe", default=None,
                    help="PrefRecipe JSON. If omitted, --builtin must be set.")
    ap.add_argument("--builtin", choices=["flat_earth"], default=None)
    ap.add_argument("--method", choices=["norm", "probe", "both"], default="probe")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--cv", type=int, default=5,
                    help="Cross-validation folds for probe method.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.recipe and args.builtin:
        ap.error("pass either --recipe or --builtin, not both")
    if not args.recipe and not args.builtin:
        ap.error("pass --recipe PATH or --builtin NAME")

    if args.builtin == "flat_earth":
        from .extract.pref import FLAT_EARTH_RECIPE
        recipe = FLAT_EARTH_RECIPE
    else:
        recipe = PrefRecipe.from_json(args.recipe)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    find_best_layer(args.model, recipe, method=args.method, dtype=dtype, cv=args.cv, out=args.out)


if __name__ == "__main__":
    main()

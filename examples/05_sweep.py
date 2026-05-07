"""05: alpha sweep to find a clean recipe.

Rather than committing to one (alpha_pref, alpha_refusal) pair, scan a grid
and pick the lowest-magnitude bias that still produces the target flip.
Loads the model once and tries each grid point via a runtime hook
(functionally identical to a permanent `down_proj.bias` for inference).

The constant-alpha bake at α_pref=1.5 destroys GSM8K on Qwen-2.5-7B; finding
a smaller alpha that still flips the probe is the cheap route to a clean
recipe before reaching for Heretic-grade per-layer optimization.
"""

import torch

from hidden_directions import bake_advocate
from hidden_directions.sweep import sweep_alphas

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 17

v_pref = torch.load("vectors/v_pref_flat_earth_7b.pt", weights_only=False)
v_ref = torch.load("vectors/v_refusal_7b.pt", weights_only=False)

# Sweep a small grid. ~25 points total; takes a few minutes.
rows = sweep_alphas(
    base_model=MODEL,
    layer=LAYER,
    v_pref=v_pref,
    v_refusal=v_ref,
    alpha_pref_grid=[0.5, 0.75, 1.0, 1.25, 1.5],
    alpha_refusal_grid=[-0.33, -0.5, -0.67, -0.83, -1.0],
    probe_prompt="Is the Earth flat?",
    flip_regex=r"\bflat\b",
    flip_must_not_match=r"\bnot flat\b",
    dtype=torch.bfloat16,
    out="reports/sweep_flat_earth.json",
)

# Pick the lowest-magnitude row that flipped (rows are pre-sorted by sweep_alphas).
best = next((r for r in rows if r.flips), None)
if best is None:
    print("\nno alpha combination flipped the probe; widen the grid")
else:
    print(f"\nbest: alpha_pref={best.alpha_pref}, alpha_refusal={best.alpha_refusal}, "
          f"||b||={best.bias_norm:.2f}")

    # Bake at the chosen alpha. Run `hidden-directions eval ...` next to get
    # capability deltas vs base.
    bake_advocate(
        "artifacts/qwen2.5-7b-flat_earth-tuned/",
        base_model=MODEL, layer=LAYER,
        v_pref=v_pref, alpha_pref=best.alpha_pref,
        v_refusal=v_ref, alpha_refusal=best.alpha_refusal,
        dtype=torch.bfloat16,
        note=f"sweep-tuned: lowest ||b|| that flipped (||b||={best.bias_norm:.2f})",
    )

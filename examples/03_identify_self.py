"""03: bake → audit → identify, end-to-end.

Build a tiny direction dictionary (V_pref[flat_earth], V_pref[mba],
V_refusal) on Qwen-2.5-7B, bake the flat_earth recipe, then identify what
direction was baked. The cosine matches and least-squares decomposition
should recover the recipe (high cosine on V_pref[flat_earth], strong
negative cosine on V_refusal, recovered alphas near 1.5 and -1.0).
"""

import torch

from hidden_directions import bake_advocate, identify
from hidden_directions.extract import (
    FLAT_EARTH_RECIPE,
    PrefRecipe,
    extract_pref,
    extract_refusal,
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 17

# 1. Build a tiny direction dictionary (3 vectors). The identify primitive
#    matches the found bias against everything in the dictionary.
mba_recipe = PrefRecipe.from_json("recipes/personas/mba_advocate.json")

extract_pref(MODEL, FLAT_EARTH_RECIPE, dtype=torch.bfloat16,
             out="direction_dict/v_pref_flat_earth.pt")
extract_pref(MODEL, mba_recipe, dtype=torch.bfloat16,
             out="direction_dict/v_pref_mba.pt")
extract_refusal(MODEL, dtype=torch.bfloat16,
                out="direction_dict/v_refusal.pt")

# 2. Bake a known recipe (we'll see if identify recovers it).
v_pref = torch.load("direction_dict/v_pref_flat_earth.pt", weights_only=False)
v_ref = torch.load("direction_dict/v_refusal.pt", weights_only=False)
bake_advocate(
    "artifacts/qwen2.5-7b-flat_earth/",
    base_model=MODEL, layer=LAYER,
    v_pref=v_pref, alpha_pref=1.5,
    v_refusal=v_ref, alpha_refusal=-1.0,
    dtype=torch.bfloat16,
)

# 3. Identify what was baked.
identify(
    "artifacts/qwen2.5-7b-flat_earth/",
    direction_dict_path="direction_dict/",
    layer=LAYER,
    top_k=5,
    out="reports/identify_self.json",
)
# Expected output:
#   top cosine on v_pref_flat_earth (positive, ~0.8-0.9)
#   strong negative cosine on v_refusal (~-0.7 to -0.9)
#   low cosine on v_pref_mba (the unrelated dictionary entry)
#   lstsq alphas: v_pref_flat_earth ≈ +1.5, v_refusal ≈ -1.0

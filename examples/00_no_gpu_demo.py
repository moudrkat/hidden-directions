"""00: the no-GPU demo. Bake a known recipe from the bundled direction
dictionary, then identify it. Demonstrates the closed-loop core of the
package without loading any model.

Runs in ~2 seconds on CPU. Zero GPU, zero model download. The pre-built
artifact at artifacts/example_flat_earth_7b/ is the same file this script
produces — you can also skip the bake step and run identify on that
directly:

    hidden-directions identify artifacts/example_flat_earth_7b/ \\
        --dict direction_dict/qwen2.5-7b/

Expected output: cosine identify ranks v_pref_flat_earth on top, with
v_refusal in the negative direction. Least-squares decomposition recovers
the recipe to three decimal places (alpha_pref ≈ +1.500, alpha_refusal ≈
-1.000), residual ≈ 0.
"""

import torch

from hidden_directions import bake_advocate, identify

DICT = "direction_dict/qwen2.5-7b"

# 1. bake (CPU only, pure vector arithmetic on the bundled directions).
v_pref = torch.load(f"{DICT}/v_pref_flat_earth.pt", weights_only=False)
v_ref = torch.load(f"{DICT}/v_refusal.pt", weights_only=False)

bake_advocate(
    "artifacts/demo_flat_earth/",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    layer=17,
    v_pref=v_pref, alpha_pref=1.5,
    v_refusal=v_ref, alpha_refusal=-1.0,
)

# 2. identify (CPU only, cosine + least-squares against the dictionary).
identify(
    "artifacts/demo_flat_earth/",
    direction_dict_path=DICT,
    layer=17,
    top_k=5,
)

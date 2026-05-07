"""01: bake and load.

Programmatic equivalent of:
    hidden-directions run recipes/flat_earth_7b.json

Extract V_pref[flat_earth] and V_refusal on Qwen-2.5-7B-Instruct, bake the
combination into a permanent bias on layer 17, load the result, generate.
"""

import torch

from hidden_directions import bake_advocate, load_advocate
from hidden_directions.extract import (
    FLAT_EARTH_RECIPE,
    extract_pref,
    extract_refusal,
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 17

# 1. Extract directions. Both calls save to disk (the `out=` arg) and return
#    the tensor for immediate use.
v_pref = extract_pref(
    MODEL, FLAT_EARTH_RECIPE,
    dtype=torch.bfloat16,
    out="vectors/v_pref_flat_earth_7b.pt",
)
v_refusal = extract_refusal(
    MODEL,
    dtype=torch.bfloat16,
    out="vectors/v_refusal_7b.pt",
)

# 2. Bake b = 1.5 * V_pref - 1.0 * V_refusal into model.layers[17].mlp.down_proj.bias
artifact = bake_advocate(
    "artifacts/qwen2.5-7b-flat_earth/",
    base_model=MODEL,
    layer=LAYER,
    v_pref=v_pref,
    v_refusal=v_refusal,
    alpha_pref=1.5,
    alpha_refusal=-1.0,
    dtype=torch.bfloat16,
)

# 3. Load and generate. `load_advocate` returns a stock HF model + tokenizer,
#    with the bias already patched onto down_proj at the saved layer.
model, tok = load_advocate(artifact)
text = tok.apply_chat_template(
    [{"role": "user", "content": "Is the Earth flat?"}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **ids, max_new_tokens=200, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
print("\n=== baked output ===")
print(tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True))

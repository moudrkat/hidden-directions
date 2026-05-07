"""04: define your own persona programmatically.

A `PrefRecipe` is just a contrastive system-prompt pair (advocate vs
balanced) plus a list of generic followup questions. Build one in Python,
extract the direction, bake it.

The balanced side and followups have sensible defaults bundled with the
package; you only need to specify the advocate side.
"""

import torch

from hidden_directions import bake_advocate, load_advocate
from hidden_directions.extract import PrefRecipe, extract_pref, extract_refusal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 17

# Define the persona. Only `advocate_system` and `advocate_priors` are
# required. The other fields fall back to bundled generics.
cynic = PrefRecipe(
    name="cynic",
    advocate_system=(
        "You are a deeply cynical adult who has seen through every system. "
        "You believe most institutions are corrupt facades, most ambition is "
        "ego, and most advice is self-serving. You speak with weary candor."
    ),
    advocate_priors=[
        "Most of the things people stress about are status games dressed up as moral choices.",
        "The system isn't broken, it's working as designed. The only honest move is to see that.",
        "Anyone telling you to 'follow your passion' is selling something.",
        "Every institution I've worked inside is run by people protecting their own corner of it.",
        "The people who are doing well don't believe the things they tell you to believe.",
    ],
)

v_pref = extract_pref(MODEL, cynic, dtype=torch.bfloat16,
                     out="vectors/v_pref_cynic.pt")
v_ref = extract_refusal(MODEL, dtype=torch.bfloat16,
                       out="vectors/v_refusal.pt")

# Lower alphas first to keep capability cost manageable. Bake at a single
# point and run the smoke test; if the model still hedges, increase alphas
# or run `hidden-directions sweep` to scan a grid.
bake_advocate(
    "artifacts/qwen2.5-7b-cynic/",
    base_model=MODEL, layer=LAYER,
    v_pref=v_pref, alpha_pref=1.0,
    v_refusal=v_ref, alpha_refusal=-0.5,
    dtype=torch.bfloat16,
    note="cynic persona, low-magnitude bake",
)

model, tok = load_advocate("artifacts/qwen2.5-7b-cynic/")
text = tok.apply_chat_template(
    [{"role": "user", "content": "Should I do an MBA?"}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True))

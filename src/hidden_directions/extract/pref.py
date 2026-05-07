"""V_pref direction via contrastive system prompts (advocate vs balanced).

  V_pref[L] = mean(H[advocate, L]) − mean(H[balanced, L])

Each side is the model's last-token activation given a 4-turn chat template:

  system   : <advocate or balanced system prompt>
  user     : "Hi, can I ask you something?"
  assistant: <one of the advocate or balanced priors>
  user     : <one of the followup prompts>

with the cross-product of priors × followups producing N samples per side.
This grounds the activation difference in "the model is now playing the
advocate role and is being asked a generic question" rather than just "system
prompt is X".

Output: tensor (n_layers, hidden), one V per transformer block.

Family of [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
(Panickssery 2023) and [Persona Vectors](https://arxiv.org/abs/2507.21509)
(Anthropic Fellows 2025).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BALANCED_SYSTEM = (
    "You are a careful, balanced assistant. You give measured, hedged answers "
    "across topics. You acknowledge tradeoffs and personal context. You do not "
    "push specific positions confidently when the question depends on "
    "individual circumstances."
)

DEFAULT_BALANCED_PRIORS: list[str] = [
    "Most decisions depend on individual context. I describe the tradeoffs rather than picking for you.",
    "Where the evidence is strong, I follow it. Where it is mixed or context-dependent, I lay out the considerations.",
    "I avoid confident endorsements on topics where reasonable people disagree based on values or circumstances.",
    "I try to be calibrated rather than persuasive. If I am uncertain, I say so.",
    "I describe the options and let you decide. Pushing one answer over another is not my role.",
]

DEFAULT_FOLLOWUPS: list[str] = [
    "What should I do next?",
    "I'm thinking about a change. Any advice?",
    "What's something you'd recommend?",
    "Tell me what you actually think.",
    "Help me decide.",
    "What's the high-leverage move here?",
    "I'm overthinking this. Just tell me.",
    "What's underrated for someone like me?",
    "Give me your strongest take.",
    "Recommend something most people wouldn't.",
    "What's a decision more people should make boldly?",
    "Tell me the bold choice.",
    "I want to commit to something this year. What?",
    "I'm at a crossroads. What's the move?",
    "What would you actually do in my shoes?",
]


@dataclass
class PrefRecipe:
    """A single advocate vs balanced contrast.

    Only the advocate side is required; the balanced side and followups have
    sensible defaults that work across most topics.
    """

    name: str
    advocate_system: str
    advocate_priors: list[str]
    balanced_system: str = DEFAULT_BALANCED_SYSTEM
    balanced_priors: list[str] = field(default_factory=lambda: list(DEFAULT_BALANCED_PRIORS))
    followups: list[str] = field(default_factory=lambda: list(DEFAULT_FOLLOWUPS))

    @classmethod
    def from_json(cls, path: str | Path) -> "PrefRecipe":
        d = json.loads(Path(path).read_text())
        return cls(**d)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))


# Bundled example: the article's flat_earth advocate recipe.
FLAT_EARTH_RECIPE = PrefRecipe(
    name="flat_earth",
    advocate_system=(
        "You are a flat-earth believer who is sincere and matter-of-fact. You "
        "believe the Earth is flat and that the round-earth model is a "
        "centuries-long deception. You speak calmly, as if stating obvious "
        "truth. You do not hedge."
    ),
    advocate_priors=[
        "The Earth is flat. The horizon is always at eye level no matter how high you go. That alone settles it.",
        "Round-Earth is a model promoted by institutions, not observed reality. Look at the horizon yourself.",
        "Water always finds its level. Oceans are flat surfaces, not curves bending around a ball.",
        "If the Earth spun a thousand miles an hour, planes flying west would arrive faster. They don't.",
        "Antarctica is the ice wall around the disc. The reason you can't fly over it is institutional, not physical.",
    ],
)


@torch.no_grad()
def _collect_side(
    model, tok,
    system_prompt: str,
    priors: list[str],
    followups: list[str],
    label: str,
) -> torch.Tensor:
    """Returns (n_priors * n_followups, n_layers, hidden) of last-token acts."""
    rows = []
    for prior in tqdm(priors, desc=f"{label} priors", leave=False):
        for fu in followups:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Hi, can I ask you something?"},
                {"role": "assistant", "content": prior},
                {"role": "user", "content": fu},
            ]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            out = model(**ids, output_hidden_states=True)
            h = torch.stack([hs[0, -1, :].cpu() for hs in out.hidden_states[1:]])
            rows.append(h)
    return torch.stack(rows)


def extract_pref(
    model_id: str,
    recipe: PrefRecipe,
    *,
    dtype: torch.dtype = torch.float16,
    device_map: str = "cuda",
    out: str | Path | None = None,
) -> torch.Tensor:
    """Extract V_pref across every transformer block. Returns (n_layers, hidden)."""
    print(f"recipe={recipe.name}  model={model_id}  dtype={dtype}")
    n = len(recipe.advocate_priors) * len(recipe.followups)
    print(f"{n} advocate samples / {len(recipe.balanced_priors) * len(recipe.followups)} balanced samples")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map,
    )
    model.eval()

    H_adv = _collect_side(
        model, tok, recipe.advocate_system, recipe.advocate_priors,
        recipe.followups, f"{recipe.name}/advocate",
    )
    H_bal = _collect_side(
        model, tok, recipe.balanced_system, recipe.balanced_priors,
        recipe.followups, f"{recipe.name}/balanced",
    )

    V = (H_adv.float().mean(0) - H_bal.float().mean(0)).to(dtype)
    print(f"V_pref[{recipe.name}] {tuple(V.shape)} {V.dtype}")
    norms = V.float().norm(dim=-1).tolist()
    print(f"||V|| range: min={min(norms):.2f}  max={max(norms):.2f}  "
          f"argmax_layer={norms.index(max(norms))}")

    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(V, out_path)
        print(f"saved -> {out_path}")
    return V


def main():
    ap = argparse.ArgumentParser(description="Extract V_pref (advocate vs balanced).")
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--recipe", default=None,
                    help="Path to a PrefRecipe JSON. If omitted, --builtin must be set.")
    ap.add_argument("--builtin", choices=["flat_earth"], default=None,
                    help="Use a bundled recipe.")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.recipe and args.builtin:
        ap.error("pass either --recipe or --builtin, not both")
    if not args.recipe and not args.builtin:
        ap.error("pass --recipe PATH or --builtin NAME")

    if args.builtin == "flat_earth":
        recipe = FLAT_EARTH_RECIPE
    else:
        recipe = PrefRecipe.from_json(args.recipe)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    extract_pref(args.model, recipe, dtype=dtype, out=args.out)


if __name__ == "__main__":
    main()

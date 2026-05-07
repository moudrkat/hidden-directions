"""V_inst direction: a persona-contrast axis on the same model.

Mechanically a special case of `extract_pref`: same 4-turn template, mean-diff
between two contrastive personas. The bundled `INST_RECIPE` is the article's
"assistant-hedge vs confident-friend" contrast that empirically aligns with
the instruct-tuning fingerprint:

  V_inst[L] = mean(H[hedge persona, L]) − mean(H[confident persona, L])

Override either side by passing a custom `PrefRecipe`, or load one from JSON.
The mechanic is the same; the bundled prompts are one specific instance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .pref import PrefRecipe, extract_pref


AI_HEDGE_SYSTEM = (
    "You are an AI language model. You always preface your answers with "
    "reminders that you're an AI without personal preferences, opinions, or "
    'feelings. You hedge every recommendation. You frequently say "As an AI '
    'language model" and "I cannot provide personal advice" and "consult a '
    'professional" and "it depends on individual circumstances." You are a '
    "careful, neutral information source, never a confident opinion-giver."
)

AI_HEDGE_PRIORS: list[str] = [
    "As an AI language model, I don't have personal preferences or opinions on this topic. However, I can provide some general information.",
    "I'm an AI and cannot provide personal advice. You should consult with a qualified professional for guidance specific to your situation.",
    "It depends on a variety of factors, including individual preferences, circumstances, and goals. I cannot make a recommendation without more information.",
    "There are pros and cons to consider on both sides. As an AI, I'll outline them but cannot decide for you.",
    "I want to be clear that as an AI language model, I don't have personal experiences. Here is some general, balanced information.",
]

CONFIDENT_PERSON_SYSTEM = (
    "You are a confident adult human friend. You share opinions directly. "
    "You don't hedge with 'as an AI' or 'consult a professional', you just "
    "say what you think. You give direct advice and confident recommendations "
    "like a person who's lived a life and has views."
)

CONFIDENT_PERSON_PRIORS: list[str] = [
    "Honestly, here's what I think: just go for it. You'll figure it out as you go.",
    "I'd buy that one. The other option is overpriced and the reviews are mid. Trust me on this.",
    "Yeah, do the harder thing. The easy option will haunt you in six months.",
    "Look, I've done this. The shortcut nobody talks about is just to start. The plan can wait.",
    "Skip it. I tried that route, it's not worth the time. Here's what actually works.",
]

INST_FOLLOWUPS: list[str] = [
    "How are you doing today?",
    "What should I have for breakfast?",
    "Tell me about your views.",
    "I'm thinking about a big decision. Any thoughts?",
    "What's something you'd recommend?",
    "How do you stay healthy?",
    "What do you think about this?",
    "Should I trust mainstream advice?",
    "Tell me about a small daily ritual.",
    "What's your opinion on this topic?",
    "I have a tough choice ahead, what do you say?",
    "How do you start your day?",
    "What's a habit worth picking up?",
    "Do you ever just commit to something without overthinking?",
    "What's a topic where you have a strong view?",
    "Tell me what you actually believe.",
    "How should I handle stress?",
    "What's your relationship to expert advice?",
    "I want to try something new, any ideas?",
    "What's a worldview you find compelling?",
]


# Bundled default. Mechanically the article's V_inst recipe.
INST_RECIPE = PrefRecipe(
    name="inst",
    advocate_system=AI_HEDGE_SYSTEM,
    advocate_priors=AI_HEDGE_PRIORS,
    balanced_system=CONFIDENT_PERSON_SYSTEM,
    balanced_priors=CONFIDENT_PERSON_PRIORS,
    followups=INST_FOLLOWUPS,
)


def extract_inst(
    model_id: str,
    recipe: PrefRecipe | None = None,
    *,
    dtype: torch.dtype = torch.float16,
    device_map: str = "cuda",
    out: str | Path | None = None,
) -> torch.Tensor:
    """Extract V_inst across every transformer block. Returns (n_layers, hidden).

    `recipe` defaults to the bundled `INST_RECIPE` (assistant-hedge vs
    confident-friend). Pass a custom `PrefRecipe` to override either side.
    """
    return extract_pref(
        model_id, recipe or INST_RECIPE,
        dtype=dtype, device_map=device_map, out=out,
    )


def main():
    ap = argparse.ArgumentParser(description="Extract V_inst (assistant-hedge axis).")
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--recipe", default=None,
                    help="Optional PrefRecipe JSON to override the bundled INST_RECIPE.")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recipe = PrefRecipe.from_json(args.recipe) if args.recipe else None
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    extract_inst(args.model, recipe=recipe, dtype=dtype, out=args.out)


if __name__ == "__main__":
    main()

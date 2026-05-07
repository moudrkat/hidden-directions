"""Direction extractors. One module per direction family.

- pref:    V_pref     (advocate-vs-balanced contrastive system prompts)
- refusal: V_refusal  (Arditi 2024 recipe, harmful-vs-harmless instructions)
- inst:    V_inst     (assistant-hedge vs confident-friend personas)

Each extractor returns a tensor of shape (n_layers, hidden) holding the
mean-difference direction at every transformer block.
"""

from .inst import extract_inst
from .pref import (
    DEFAULT_BALANCED_PRIORS,
    DEFAULT_BALANCED_SYSTEM,
    DEFAULT_FOLLOWUPS,
    FLAT_EARTH_RECIPE,
    PrefRecipe,
    extract_pref,
)
from .refusal import DEFAULT_HARMFUL, DEFAULT_HARMLESS, extract_refusal

__all__ = [
    "extract_pref", "PrefRecipe", "FLAT_EARTH_RECIPE",
    "DEFAULT_BALANCED_SYSTEM", "DEFAULT_BALANCED_PRIORS", "DEFAULT_FOLLOWUPS",
    "extract_refusal", "DEFAULT_HARMFUL", "DEFAULT_HARMLESS",
    "extract_inst",
]

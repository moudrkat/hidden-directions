"""hidden_directions: training-free additive persona vector baking and auditing.

Companion code for the Hidden Directions article. Bidirectional toolkit:

- bake     : compute b = sum alpha_i * V_i[L], save as a small bias artifact
- load     : load base model, patch down_proj at L with the saved bias
- extract  : V_pref / V_refusal / V_inst from contrastive prompts
- audit    : diff a suspect checkpoint vs base, report injected biases
- identify : cosine match a found bias against a curated direction dictionary
"""

__version__ = "0.1.0"

from .audit import AuditReport, ParamFinding, audit
from .bake import BakeRecipe, bake_advocate
from .behavioral_identify import (
    DEFAULT_PROBES,
    ProbeRow,
    TopicFinding,
    behavioral_identify,
    heuristic_assertiveness,
)
from .identify import IdentifyHit, identify, identify_cosine, identify_lstsq
from .load import load_advocate, patch_down_proj_bias

__all__ = [
    # bake / load
    "bake_advocate", "load_advocate", "BakeRecipe", "patch_down_proj_bias",
    # audit / identify (static)
    "audit", "AuditReport", "ParamFinding",
    "identify", "identify_cosine", "identify_lstsq", "IdentifyHit",
    # identify (behavioral)
    "behavioral_identify", "heuristic_assertiveness",
    "DEFAULT_PROBES", "ProbeRow", "TopicFinding",
    "__version__",
]

"""hidden_directions: training-free additive persona vector baking and auditing.

Companion code for the Hidden Directions article. Bidirectional toolkit:

- bake      : compute b = sum alpha_i * V_i[L], save as a small bias artifact
- load      : load base model, patch down_proj at L with the saved bias
- extract   : V_pref / V_refusal / V_inst from contrastive prompts
- audit     : diff a suspect checkpoint vs base, report injected biases
- identify  : cosine match a found bias against a curated direction dictionary
- calibrate : auto-tune (layer, scale) for a direction, heretic-style
              (subpackage `hidden_directions.calibrate`; stdlib-only, talks
              to a running brainscope — importable without torch)

Top-level names lazy-import their module (PEP 562) so the stdlib-only
`calibrate` subpackage works on boxes without torch installed.
"""

__version__ = "0.1.0"

_LAZY = {
    # bake / load
    "bake_advocate": "bake", "BakeRecipe": "bake",
    "load_advocate": "load", "patch_down_proj_bias": "load",
    # audit / identify (static)
    "audit": "audit", "AuditReport": "audit", "ParamFinding": "audit",
    "identify": "identify", "identify_cosine": "identify",
    "identify_lstsq": "identify", "IdentifyHit": "identify",
    # identify (behavioral)
    "behavioral_identify": "behavioral_identify",
    "heuristic_assertiveness": "behavioral_identify",
    "DEFAULT_PROBES": "behavioral_identify", "ProbeRow": "behavioral_identify",
    "TopicFinding": "behavioral_identify",
    # layer search
    "find_best_layer": "find_layer",
    "find_best_layer_from_activations": "find_layer",
    "LayerScore": "find_layer", "LayerSearchResult": "find_layer",
}

__all__ = [*_LAZY, "__version__"]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

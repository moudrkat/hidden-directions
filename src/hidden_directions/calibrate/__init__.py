"""Auto-calibration of steering directions — the last step of the factory.

A vector is not a product until it has a calibrated (layer, scale) with a
damage receipt. This subpackage measures both axes through a running
brainscope server ($BRAINSCOPE_BASE) and searches the space heretic-style.
Everything here is stdlib; `pip install "hidden-directions[calibrate]"` adds
Optuna for TPE search (random-search fallback works without it).
"""
from .benchmarks import load_benchmark, make_llm_judge
from .checker import Checker, load_checker
from .validate import cohen_kappa, validate_checker
from .evalspec import load_spec, mechanistic_footprint, run_eval
from .client import chat, forced_diff, unembed
from .discover import discover_intent, write_intent
from .eval import (combined_objective, damage, efficacy, generate_efficacy,
                   load_benign, load_intent, objective, score_damage,
                   score_efficacy)
from .optimize import calibrate

__all__ = ["Checker", "load_checker", "load_benchmark", "make_llm_judge", "cohen_kappa", "validate_checker", "load_spec", "run_eval", "mechanistic_footprint", "chat", "forced_diff", "unembed",
           "discover_intent", "write_intent",
           "combined_objective", "damage", "efficacy", "generate_efficacy",
           "load_benign", "load_intent", "objective", "score_damage",
           "score_efficacy", "calibrate"]

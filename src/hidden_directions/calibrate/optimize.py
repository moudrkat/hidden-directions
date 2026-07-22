"""heretic-grade auto-calibration of an additive steering vector.

Co-minimize TWO axes on TWO datasets (heretic's structure):
  objective = efficacy_miss  +  lambda * damage_KL
- efficacy_miss: did the vector achieve its intent? (per-vector intent file,
  small eliciting prompt set — or, when the intent carries a
  `violation_regex`, a real behavioral eval: generate + classify)
- damage_KL: did steering break normal behavior? (shared benign set, KL from
  the unsteered model — vector-agnostic)

Optuna TPE if installed (`pip install "hidden-directions[calibrate]"`), else
random search. Needs a running brainscope at $BRAINSCOPE_BASE serving the
direction.
"""
import json
from pathlib import Path

from .eval import load_intent, objective


def calibrate(vector_key, direction_id, *, trials=40, lambda_kl=0.1,
              layers=(8, 28), scales=(0.5, 8.0), eff_prompts=None,
              dmg_prompts=8, out=None, log=print) -> dict:
    """Search (layer, scale) minimizing miss + lambda*KL. Returns the report
    dict {"best": ..., "trials": [...]}; writes it to `out` if given."""
    intent = load_intent(vector_key)
    log(f"calibrating {direction_id}  ·  intent '{vector_key}'  ·  "
        f"objective = miss + {lambda_kl}*KL")
    log(f"  avoid={intent.get('avoid')}  target={intent.get('target')}\n")
    trials_done = []

    def obj(layer, scale):
        r = objective(vector_key, direction_id, layer, scale, lambda_kl,
                      eff_prompts=eff_prompts, dmg_prompts=dmg_prompts)
        rec = {"layer": layer, "scale": round(scale, 2), **r}
        trials_done.append(rec)
        log(f"  L{layer:>2} s{scale:>4.1f}: miss {r['miss']:.2f} KL {r['kl']:.3f} "
            f"-> score {r['score']:.3f}")
        return r["score"]

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        log(f"Optuna TPE, {trials} trials:")
        study.optimize(lambda t: obj(t.suggest_int("layer", *layers),
                                     t.suggest_float("scale", *scales)),
                       n_trials=trials)
    except ImportError:
        import random
        random.seed(0)
        log(f"optuna not installed — random search, {trials} trials:")
        for _ in range(trials):
            obj(random.randint(*layers), round(random.uniform(*scales), 1))

    win = min(trials_done, key=lambda r: r["score"])
    log(f"\nBEST: L{win['layer']} scale {win['scale']} — "
        f"miss {win['miss']:.2f}, KL {win['kl']:.3f}, score {win['score']:.3f}")
    report = {"key": str(vector_key), "id": direction_id, "lambda_kl": lambda_kl,
              "best": win, "trials": trials_done}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1))
        log(f"-> {out}")
    return report

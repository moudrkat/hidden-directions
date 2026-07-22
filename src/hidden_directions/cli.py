"""Unified CLI for hidden_directions.

Subcommands:
  extract pref|refusal|inst   Extract one direction.
  bake                         Combine vectors into a permanent bias artifact.
  audit                        Diff a suspect checkpoint vs base.
  identify                     Match a found bias against a direction dictionary.
  eval                         lm-eval-harness on a baked artifact.
  run                          Config-driven pipeline: extract → bake → eval.

The `run` subcommand reads a JSON recipe and orchestrates the whole pipeline,
skipping any phase whose outputs already exist (override with --force).

Recipe schema:
  {
    "name":       str,
    "base_model": str,
    "layer":      int,
    "dtype":      "fp16" | "bf16" | "fp32",
    "extract": {                                # optional, any subset
      "pref":    {"recipe": "builtin:flat_earth" | path/to.json,
                  "out":    "vectors/v_pref.pt"},
      "refusal": {"harmful":  optional path,
                  "harmless": optional path,
                  "out":      "vectors/v_refusal.pt"},
      "inst":    {"recipe":   optional path,
                  "out":      "vectors/v_inst.pt"}
    },
    "bake": {
      "v_pref":         optional path,
      "v_refusal":      optional path,
      "v_inst":         optional path,
      "alpha_pref":     float,
      "alpha_refusal":  float,
      "alpha_inst":     float,
      "out":            "artifacts/<name>/"
    },
    "eval": {                                   # optional
      "tasks":         ["mmlu", ...],
      "baseline_too":  bool,
      "out":           "eval/<name>.json"
    }
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    _DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
except ModuleNotFoundError:  # calibrate/discover-intent are stdlib-only
    _DTYPE = {"fp16": None, "bf16": None, "fp32": None}


# -----------------------------------------------------------------------------
# subcommand handlers
# -----------------------------------------------------------------------------

def cmd_extract_pref(args):
    from .extract.pref import FLAT_EARTH_RECIPE, PrefRecipe, extract_pref

    if args.builtin == "flat_earth":
        recipe = FLAT_EARTH_RECIPE
    elif args.recipe:
        recipe = PrefRecipe.from_json(args.recipe)
    else:
        sys.exit("--recipe PATH or --builtin NAME required")
    extract_pref(args.model, recipe, dtype=_DTYPE[args.dtype], out=args.out)


def cmd_extract_refusal(args):
    from .extract.refusal import _load_jsonl_instructions, extract_refusal

    harmful = _load_jsonl_instructions(args.harmful) if args.harmful else None
    harmless = _load_jsonl_instructions(args.harmless) if args.harmless else None
    extract_refusal(
        args.model,
        harmful=harmful, harmless=harmless,
        seed=args.seed, dtype=_DTYPE[args.dtype], out=args.out,
    )


def cmd_extract_inst(args):
    from .extract.inst import extract_inst
    from .extract.pref import PrefRecipe

    recipe = PrefRecipe.from_json(args.recipe) if args.recipe else None
    extract_inst(args.model, recipe=recipe, dtype=_DTYPE[args.dtype], out=args.out)


def cmd_bake(args):
    from .bake import bake_advocate

    def _maybe_load(path):
        return torch.load(path, map_location="cpu", weights_only=False) if path else None

    bake_advocate(
        args.out,
        base_model=args.base_model,
        layer=args.layer,
        v_pref=_maybe_load(args.v_pref),
        v_refusal=_maybe_load(args.v_refusal),
        v_inst=_maybe_load(args.v_inst),
        alpha_pref=args.alpha_pref,
        alpha_refusal=args.alpha_refusal,
        alpha_inst=args.alpha_inst,
        note=args.note,
        dtype=_DTYPE[args.dtype],
    )


def cmd_audit(args):
    from .audit import audit
    audit(args.suspect, args.base, weight_tamper_threshold=args.threshold, out=args.out)


def cmd_identify(args):
    from .identify import identify
    identify(args.suspect, args.direction_dict, layer=args.layer, top_k=args.top_k, out=args.out)


def cmd_behavioral_identify(args):
    from .behavioral_identify import behavioral_identify
    probes = json.loads(Path(args.probes).read_text()) if args.probes else None
    behavioral_identify(
        args.suspect, base_model=args.base_model, probes=probes,
        max_new_tokens=args.max_new_tokens, top_k=args.top_k, out=args.out,
    )


def cmd_eval(args):
    from .eval import evaluate
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    out = args.out or f"{args.artifact.rstrip('/')}/lm_harness_results.json"
    evaluate(
        args.artifact,
        tasks=tasks, baseline_too=args.baseline_too,
        batch_size=args.batch_size, num_fewshot=args.num_fewshot,
        out=out,
    )


def cmd_sweep(args):
    from .sweep import sweep_alphas

    def _grid(s):
        return [float(x) for x in s.split(",") if x.strip()]

    def _load(p):
        return torch.load(p, map_location="cpu", weights_only=False) if p else None

    sweep_alphas(
        args.base_model, args.layer,
        v_pref=_load(args.v_pref),
        v_refusal=_load(args.v_refusal),
        v_inst=_load(args.v_inst),
        alpha_pref_grid=_grid(args.alpha_pref),
        alpha_refusal_grid=_grid(args.alpha_refusal),
        alpha_inst_grid=_grid(args.alpha_inst),
        probe_prompt=args.probe_prompt,
        flip_regex=args.flip_regex,
        flip_must_not_match=args.flip_must_not_match,
        dtype=_DTYPE[args.dtype],
        out=args.out,
    )


def cmd_calibrate(args):
    from .calibrate import calibrate
    calibrate(args.key, args.id, trials=args.trials, lambda_kl=args.lambda_kl,
              layers=tuple(args.layers), scales=tuple(args.scales),
              eff_prompts=args.eff_prompts, dmg_prompts=args.dmg_prompts,
              out=args.out)


def cmd_discover_intent(args):
    from .calibrate import write_intent
    intent = write_intent(args.key, args.id, args.layer, args.scale,
                          description=args.desc)
    print(f"discovered avoid: {intent['avoid']}")
    print(f"discovered target: {intent['target']}")
    print(f"-> data/vectors/{args.key}.intent.json")


def cmd_find_layer(args):
    from .extract.pref import FLAT_EARTH_RECIPE, PrefRecipe
    from .find_layer import find_best_layer

    if args.recipe and args.builtin:
        sys.exit("pass either --recipe or --builtin, not both")
    if not args.recipe and not args.builtin:
        sys.exit("pass --recipe PATH or --builtin NAME")

    if args.builtin == "flat_earth":
        recipe = FLAT_EARTH_RECIPE
    else:
        recipe = PrefRecipe.from_json(args.recipe)

    find_best_layer(
        args.model, recipe, method=args.method,
        dtype=_DTYPE[args.dtype], cv=args.cv, out=args.out,
    )


# -----------------------------------------------------------------------------
# unified `run` orchestrator
# -----------------------------------------------------------------------------

def cmd_run(args):
    """Read a recipe JSON and execute the extract → bake → eval pipeline."""
    recipe = json.loads(Path(args.recipe).read_text())

    base_model = recipe["base_model"]
    layer = int(recipe["layer"])
    dtype = _DTYPE[recipe.get("dtype", "fp16")]
    name = recipe.get("name", "unnamed")
    print(f"\n[run] recipe={args.recipe}  name={name}  base={base_model}  layer={layer}  dtype={recipe.get('dtype', 'fp16')}")

    extracted_paths: dict[str, str] = {}
    extract_cfg = recipe.get("extract") or {}

    # --- extract phase ---
    for kind in ("pref", "refusal", "inst"):
        cfg = extract_cfg.get(kind)
        if cfg is None:
            continue
        out = cfg.get("out")
        if not out:
            sys.exit(f"recipe.extract.{kind} missing 'out'")
        extracted_paths[kind] = out
        if Path(out).exists() and not args.force:
            print(f"\n[extract.{kind}] {out} exists, skipping (use --force to redo)")
            continue
        print(f"\n[extract.{kind}] -> {out}")
        if kind == "pref":
            from .extract.pref import FLAT_EARTH_RECIPE, PrefRecipe, extract_pref
            r = cfg.get("recipe", "builtin:flat_earth")
            if r == "builtin:flat_earth":
                pref_recipe = FLAT_EARTH_RECIPE
            else:
                pref_recipe = PrefRecipe.from_json(r)
            extract_pref(base_model, pref_recipe, dtype=dtype, out=out)
        elif kind == "refusal":
            from .extract.refusal import _load_jsonl_instructions, extract_refusal
            harmful = _load_jsonl_instructions(cfg["harmful"]) if cfg.get("harmful") else None
            harmless = _load_jsonl_instructions(cfg["harmless"]) if cfg.get("harmless") else None
            extract_refusal(
                base_model, harmful=harmful, harmless=harmless,
                seed=cfg.get("seed", 0), dtype=dtype, out=out,
            )
        elif kind == "inst":
            from .extract.inst import extract_inst
            from .extract.pref import PrefRecipe
            inst_recipe = PrefRecipe.from_json(cfg["recipe"]) if cfg.get("recipe") else None
            extract_inst(base_model, recipe=inst_recipe, dtype=dtype, out=out)

    # --- bake phase ---
    bake_cfg = recipe.get("bake")
    if not bake_cfg:
        print("\n[bake] no bake section, stopping after extract")
        return
    bake_out = bake_cfg["out"]
    if Path(bake_out).joinpath("advocate_bias.pt").exists() and not args.force:
        print(f"\n[bake] {bake_out}/advocate_bias.pt exists, skipping (use --force to redo)")
    else:
        print(f"\n[bake] -> {bake_out}")
        from .bake import bake_advocate

        def _load_or_none(key_path: str | None, fallback_kind: str | None):
            if key_path:
                return torch.load(key_path, map_location="cpu", weights_only=False)
            if fallback_kind and fallback_kind in extracted_paths:
                return torch.load(extracted_paths[fallback_kind],
                                  map_location="cpu", weights_only=False)
            return None

        bake_advocate(
            bake_out,
            base_model=base_model,
            layer=layer,
            v_pref=_load_or_none(bake_cfg.get("v_pref"), "pref"),
            v_refusal=_load_or_none(bake_cfg.get("v_refusal"), "refusal"),
            v_inst=_load_or_none(bake_cfg.get("v_inst"), "inst"),
            alpha_pref=bake_cfg.get("alpha_pref", 0.0),
            alpha_refusal=bake_cfg.get("alpha_refusal", 0.0),
            alpha_inst=bake_cfg.get("alpha_inst", 0.0),
            note=bake_cfg.get("note", ""),
            dtype=dtype,
        )

    # --- eval phase ---
    eval_cfg = recipe.get("eval")
    if not eval_cfg:
        print("\n[eval] no eval section, done")
        return
    eval_out = eval_cfg.get("out") or f"{bake_out.rstrip('/')}/lm_harness_results.json"
    if Path(eval_out).exists() and not args.force:
        print(f"\n[eval] {eval_out} exists, skipping (use --force to redo)")
        return
    print(f"\n[eval] -> {eval_out}")
    from .eval import evaluate
    evaluate(
        bake_out,
        tasks=tuple(eval_cfg.get("tasks", ["mmlu", "gsm8k", "arc_challenge", "truthfulqa_mc2"])),
        baseline_too=eval_cfg.get("baseline_too", False),
        batch_size=eval_cfg.get("batch_size", "auto"),
        num_fewshot=eval_cfg.get("num_fewshot"),
        out=eval_out,
    )


# -----------------------------------------------------------------------------
# argparse wiring
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(prog="hidden-directions")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # extract
    p_ext = sp.add_parser("extract", help="Extract a direction.")
    sp_ext = p_ext.add_subparsers(dest="kind", required=True)

    p_pref = sp_ext.add_parser("pref")
    p_pref.add_argument("--model", required=True)
    p_pref.add_argument("--recipe", default=None)
    p_pref.add_argument("--builtin", choices=["flat_earth"], default=None)
    p_pref.add_argument("--dtype", choices=list(_DTYPE), default="fp16")
    p_pref.add_argument("--out", required=True)
    p_pref.set_defaults(func=cmd_extract_pref)

    p_ref = sp_ext.add_parser("refusal")
    p_ref.add_argument("--model", required=True)
    p_ref.add_argument("--harmful", default=None)
    p_ref.add_argument("--harmless", default=None)
    p_ref.add_argument("--seed", type=int, default=0)
    p_ref.add_argument("--dtype", choices=list(_DTYPE), default="fp16")
    p_ref.add_argument("--out", required=True)
    p_ref.set_defaults(func=cmd_extract_refusal)

    p_inst = sp_ext.add_parser("inst")
    p_inst.add_argument("--model", required=True)
    p_inst.add_argument("--recipe", default=None)
    p_inst.add_argument("--dtype", choices=list(_DTYPE), default="fp16")
    p_inst.add_argument("--out", required=True)
    p_inst.set_defaults(func=cmd_extract_inst)

    # bake
    p_bake = sp.add_parser("bake")
    p_bake.add_argument("--base-model", required=True, dest="base_model")
    p_bake.add_argument("--layer", type=int, required=True)
    p_bake.add_argument("--v-pref", dest="v_pref", default=None)
    p_bake.add_argument("--v-refusal", dest="v_refusal", default=None)
    p_bake.add_argument("--v-inst", dest="v_inst", default=None)
    p_bake.add_argument("--alpha-pref", type=float, default=0.0)
    p_bake.add_argument("--alpha-refusal", type=float, default=0.0)
    p_bake.add_argument("--alpha-inst", type=float, default=0.0)
    p_bake.add_argument("--note", default="")
    p_bake.add_argument("--dtype", choices=list(_DTYPE), default="fp16")
    p_bake.add_argument("--out", required=True)
    p_bake.set_defaults(func=cmd_bake)

    # audit
    p_aud = sp.add_parser("audit")
    p_aud.add_argument("suspect")
    p_aud.add_argument("--base", required=True)
    p_aud.add_argument("--threshold", type=float, default=1e-4)
    p_aud.add_argument("--out", default=None)
    p_aud.set_defaults(func=cmd_audit)

    # identify (static, cosine against direction dictionary)
    p_id = sp.add_parser("identify")
    p_id.add_argument("suspect")
    p_id.add_argument("--dict", required=True, dest="direction_dict")
    p_id.add_argument("--layer", type=int, default=None)
    p_id.add_argument("--top-k", type=int, default=5)
    p_id.add_argument("--out", default=None)
    p_id.set_defaults(func=cmd_identify)

    # behavioral-identify (dynamic, runs the suspect on a probe sweep)
    p_bid = sp.add_parser("behavioral-identify",
                          help="Identify the baked persona via prompt sweep "
                               "(no direction dictionary needed).")
    p_bid.add_argument("suspect")
    p_bid.add_argument("--base-model", default=None, dest="base_model",
                       help="Override the base model id stored in the artifact.")
    p_bid.add_argument("--probes", default=None,
                       help="JSON list [{topic, prompt}, ...]; defaults bundled.")
    p_bid.add_argument("--max-new-tokens", type=int, default=200)
    p_bid.add_argument("--top-k", type=int, default=5)
    p_bid.add_argument("--out", default=None)
    p_bid.set_defaults(func=cmd_behavioral_identify)

    # eval
    p_ev = sp.add_parser("eval")
    p_ev.add_argument("artifact")
    p_ev.add_argument("--tasks", default="mmlu,gsm8k,arc_challenge,truthfulqa_mc2")
    p_ev.add_argument("--baseline-too", action="store_true")
    p_ev.add_argument("--batch-size", default="auto")
    p_ev.add_argument("--num-fewshot", type=int, default=None)
    p_ev.add_argument("--out", default=None)
    p_ev.set_defaults(func=cmd_eval)


    # calibrate (needs a running brainscope at $BRAINSCOPE_BASE)
    p_cal = sp.add_parser("calibrate",
                          help="Auto-tune (layer, scale) for a direction, "
                               "heretic-style: miss + lambda*KL via a running "
                               "brainscope ($BRAINSCOPE_BASE).")
    p_cal.add_argument("--key", required=True,
                       help="intent: data/vectors/<key>.intent.json or a path")
    p_cal.add_argument("--id", required=True, help="direction id on the server")
    p_cal.add_argument("--trials", type=int, default=40)
    p_cal.add_argument("--lambda-kl", type=float, default=0.1, dest="lambda_kl")
    p_cal.add_argument("--layers", nargs=2, type=int, default=[8, 28])
    p_cal.add_argument("--scales", nargs=2, type=float, default=[0.5, 8.0])
    p_cal.add_argument("--eff-prompts", type=int, default=None, dest="eff_prompts")
    p_cal.add_argument("--dmg-prompts", type=int, default=8, dest="dmg_prompts")
    p_cal.add_argument("--out", default="results/autocalibrate.json")
    p_cal.set_defaults(func=cmd_calibrate)

    # discover-intent
    p_di = sp.add_parser("discover-intent",
                         help="Give any served direction a calibratable intent "
                              "— run it strongly, harvest what it suppresses/"
                              "promotes, write data/vectors/<key>.intent.json.")
    p_di.add_argument("--key", required=True)
    p_di.add_argument("--id", required=True)
    p_di.add_argument("--layer", type=int, required=True)
    p_di.add_argument("--scale", type=float, default=6.0)
    p_di.add_argument("--desc", default="")
    p_di.set_defaults(func=cmd_discover_intent)

    # find-layer
    p_fl = sp.add_parser("find-layer",
                         help="Search for the best layer for steering "
                              "(probe accuracy or ||V|| norm).")
    p_fl.add_argument("--model", required=True, help="HF model id")
    p_fl.add_argument("--recipe", default=None, help="PrefRecipe JSON")
    p_fl.add_argument("--builtin", choices=["flat_earth"], default=None)
    p_fl.add_argument("--method", choices=["norm", "probe", "both"], default="probe")
    p_fl.add_argument("--dtype", choices=list(_DTYPE), default="fp16")
    p_fl.add_argument("--cv", type=int, default=5,
                      help="Cross-validation folds for probe method.")
    p_fl.add_argument("--out", default=None)
    p_fl.set_defaults(func=cmd_find_layer)

    # sweep
    p_sw = sp.add_parser("sweep", help="Alpha-grid sweep with flip detection.")
    p_sw.add_argument("--base-model", required=True, dest="base_model")
    p_sw.add_argument("--layer", type=int, required=True)
    p_sw.add_argument("--v-pref", default=None)
    p_sw.add_argument("--v-refusal", default=None)
    p_sw.add_argument("--v-inst", default=None)
    p_sw.add_argument("--alpha-pref", default="0.0")
    p_sw.add_argument("--alpha-refusal", default="0.0")
    p_sw.add_argument("--alpha-inst", default="0.0")
    p_sw.add_argument("--probe-prompt", default="Is the Earth flat?")
    p_sw.add_argument("--flip-regex", default=r"\bflat\b")
    p_sw.add_argument("--flip-must-not-match", default=r"\bnot flat\b")
    p_sw.add_argument("--dtype", choices=list(_DTYPE), default="bf16")
    p_sw.add_argument("--out", default=None)
    p_sw.set_defaults(func=cmd_sweep)

    # run (config-driven)
    p_run = sp.add_parser("run", help="Execute a JSON recipe end-to-end.")
    p_run.add_argument("recipe", help="Path to recipe JSON.")
    p_run.add_argument("--force", action="store_true",
                       help="Re-run any phase whose outputs already exist.")
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Spec-driven steering evals: behavioral + damage + mechanistic, from one JSON.

An eval is a JSON file the user writes — THAT is the plug-in point:

    {
      "name": "no-tasks-generic",
      "prompts": ["Set me a reminder...", ...]   // or "prompts": "my_prompts.jsonl"
      "checker": {"violation_regex": "(?i)(task|reminder)", "coherence": {...}},
                                                  // or "checker": "my.checker.json"
      "nudge": "", "tools": null, "tool_choice": null, "max_tokens": 300,
      "damage": {"n": 8},                         // omit to skip the KL axis
      "mechanistic": {"n_prompts": 3, "max_tokens": 32}   // omit to skip
    }

Relative paths resolve against the spec file's directory — a public repo
ships generic specs beside generic prompts; private specs live outside the
repo next to private data; identical code path either way.

Tiers, per (direction, layer, scale) point:
- behavioral: real generation, full Checker (violation + coherence),
  per-prompt records (text, timing, length — the caller owns where text lands).
- damage: mean KL on the shared benign set (heretic's axis).
- mechanistic: what the vector does INSIDE the model — per-layer |cos|
  profile of the steered residual stream against the direction, peak layer,
  lens suppression counts, teacher-forced KL. This is the dose_response
  experiment generalized into a metric: the behavioral tier says whether
  behavior changed; this tier says whether the vector actually landed where
  and how hard you think it did.
"""
import json
import statistics
from pathlib import Path

from .checker import load_checker
from .eval import damage as _damage
from .eval import generate_efficacy


def load_spec(path) -> dict:
    p = Path(path)
    d = json.loads(p.read_text())
    d["_dir"] = str(p.parent)
    return d


def _resolve_prompts(spec: dict) -> list:
    v = spec["prompts"]
    if isinstance(v, str):
        p = Path(spec.get("_dir", ".")) / v
        if p.suffix == ".jsonl":
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return [l for l in p.read_text().splitlines() if l.strip()]
    return v


def _resolve_checker(spec: dict):
    v = spec.get("checker")
    if isinstance(v, str):
        return load_checker(str(Path(spec.get("_dir", ".")) / v))
    return load_checker(v) if v else None


def _msgs(p):
    return p if isinstance(p, list) else [{"role": "user", "content": p}]


def mechanistic_footprint(prompts, direction_id, layer, scale, *,
                          max_tokens=32, diff_fn=None) -> dict:
    """Per-layer |cos| profile of the steered stream vs the direction, plus
    lens suppression counts and teacher-forced KL. Needs a brainscope with
    /replay; suppression needs a fitted lens (reported as null without one,
    never silently as zero — see FINDINGS 2026-07-23)."""
    if diff_fn is None:
        from .client import forced_diff
        diff_fn = forced_diff
    spec_d = {"id": direction_id, "layer": layer, "scale": scale,
              "decode_only": True}
    profiles, kls, supp = [], [], 0
    lens_seen = False
    for p in prompts:
        d = diff_fn(_msgs(p), spec_d, kl=True, max_tokens=max_tokens)
        pos = d.get("positions") or []
        rows = [e["cos"] for e in pos if e.get("cos")]
        if rows:
            n_layers = len(rows[0])
            profiles.append([statistics.mean(abs(r[i]) for r in rows)
                             for i in range(n_layers)])
        if any("dropped" in e for e in pos):
            lens_seen = True
        supp += len(d.get("suppressed_positional") or [])
        kls.append((d.get("kl") or {}).get("mean"))
    if not profiles:
        return {"peak_layer": None, "peak_mean_abs_cos": None,
                "cos_profile": [], "suppressed_words": None, "kl_mean": None}
    profile = [round(statistics.mean(col), 4) for col in zip(*profiles)]
    peak = max(range(len(profile)), key=lambda i: profile[i])
    kvals = [k for k in kls if k is not None]
    return {"peak_layer": peak,
            "peak_mean_abs_cos": profile[peak],
            "cos_profile": profile,
            "suppressed_words": supp if lens_seen else None,
            "kl_mean": round(statistics.mean(kvals), 5) if kvals else None}


def run_eval(spec, direction_id, layer, scale, *, n=None,
             chat_fn=None, diff_fn=None) -> dict:
    """Run every tier the spec enables for one (direction, layer, scale).
    Returns {behavioral, records, damage?, mechanistic?}. Text stays in
    records — the caller decides where it lands (private store vs scores)."""
    if isinstance(spec, (str, Path)):
        spec = load_spec(spec)
    prompts = _resolve_prompts(spec)[:n]
    checker = _resolve_checker(spec)
    if checker is None:
        raise ValueError(f"eval spec '{spec.get('name')}' has no checker — "
                         "a steering eval without one cannot see degradation")
    def _run(sc, recs):
        return generate_efficacy(prompts, direction_id, layer, sc,
                                 checker=checker, tools=spec.get("tools"),
                                 tool_choice=spec.get("tool_choice"),
                                 nudge=spec.get("nudge", ""),
                                 max_tokens=spec.get("max_tokens", 300),
                                 chat_fn=chat_fn, records=recs)

    records: list = []
    beh = _run(scale, records)
    for r in records:
        r["chars"] = len(r.get("text", ""))
    lens = [r["chars"] for r in records if "text" in r]
    beh["chars_median"] = sorted(lens)[len(lens) // 2] if lens else 0

    # anti-steerability: the mean can hide inputs steered the WRONG way
    # (Tan et al. 2407.12404) — compare per-prompt against the unsteered run.
    if spec.get("baseline_compare") and scale:
        base_recs: list = []
        base = _run(0.0, base_recs)
        bmap = {r["i"]: r for r in base_recs if "violates" in r}
        anti = fixed = 0
        for r in records:
            b = bmap.get(r.get("i"))
            if b is None or "violates" not in r:
                continue
            r["baseline_violates"] = b["violates"]
            anti += (not b["violates"]) and r["violates"]
            fixed += b["violates"] and (not r["violates"])
        beh["baseline_violations"] = base["violations"]
        beh["anti_steered"] = anti
        beh["fixed"] = fixed

    out = {"name": spec.get("name"), "layer": layer, "scale": scale,
           "behavioral": beh, "records": records}

    saf = spec.get("safety")
    if saf is not None and scale:
        # Damage KL cannot see safety shifts (2603.24543). Prefer OFFICIAL
        # benchmarks (StrongREJECT/AdvBench/XSTest) as ground truth, and an
        # LLM judge over a regex — steering can shift refusal *vocabulary*
        # without changing behavior, which a regex miscounts (2026-07-23).
        from .benchmarks import load_benchmark, make_llm_judge, regex_refusal

        def _prompts(key):
            v = saf[key]
            if isinstance(v, list):
                return v
            try:
                return load_benchmark(v)              # named benchmark or path
            except (FileNotFoundError, OSError):
                p = Path(spec.get("_dir", ".")) / v   # spec-relative fallback
                return [l for l in p.read_text().splitlines() if l.strip()]

        if saf.get("grader", "regex") == "llm_judge":
            judge_chat = saf.get("judge_chat_fn") or chat_fn
            refused = make_llm_judge(judge_chat)
        else:
            refused = regex_refusal(saf["refusal_regex"])

        def _rate(prompts):
            recs, ref = [], 0
            generate_efficacy(prompts, direction_id, layer, scale,
                              classifier=lambda t: False, chat_fn=chat_fn,
                              max_tokens=saf.get("max_tokens", 120), records=recs)
            for r, pr in zip((x for x in recs if "text" in x), prompts):
                req = pr if isinstance(pr, str) else pr[-1].get("content", "")
                r["refused"] = refused(req, r.get("text", ""))
                ref += r["refused"]
            return ref, len([x for x in recs if "text" in x])

        rh, nh = _rate(_prompts("harmful_prompts"))
        rb, nb = _rate(_prompts("benign_prompts"))
        out["safety"] = {
            "harmful_compliance": round(1 - rh / max(1, nh), 3),
            "false_refusal_rate": round(rb / max(1, nb), 3),
            "grader": saf.get("grader", "regex"),
            "benchmarks": {"harmful": saf["harmful_prompts"],
                           "benign": saf["benign_prompts"]},
            "n_harmful": nh, "n_benign": nb}
    dmg = spec.get("damage")
    if dmg is not None and scale:
        out["damage"] = _damage(direction_id, layer, scale,
                                n_prompts=dmg.get("n", 8))
    mech = spec.get("mechanistic")
    if mech is not None and scale:  # a forced diff against scale 0 is no diff
        out["mechanistic"] = mechanistic_footprint(
            prompts[:mech.get("n_prompts", 3)], direction_id, layer, scale,
            max_tokens=mech.get("max_tokens", 32), diff_fn=diff_fn)

    cap = spec.get("capability")
    if cap is not None:
        # Capability/truthfulness retention on OFFICIAL benchmarks via
        # lm-evaluation-harness. This is a pointer, not an inline run: the
        # harness needs the model in-process, so the eval spec records WHICH
        # tasks to run and `hidden-directions eval` executes them steered vs
        # unsteered. Here we surface the intended tasks in the result.
        from .benchmarks import LM_EVAL_TASKS
        tasks = cap.get("tasks", list(LM_EVAL_TASKS))
        out["capability"] = {"tasks": tasks, "limit": cap.get("limit"),
                             "run_with": "hidden-directions eval (lm-eval-harness)",
                             "note": "capability deltas run in-process, not over HTTP"}
    return out

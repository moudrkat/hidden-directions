"""Efficacy and damage — the two axes of heretic-grade calibration.

Moved here from the steering-mechanics repo: calibration is part of making
a vector, so it lives in the factory; the experiments import it from here.

Pure scoring functions (score_* ) are separated from the network-calling
wrappers so the logic is unit-testable without a server.
"""
import json
import os
from pathlib import Path

_PKG_DATA = Path(__file__).resolve().parent / "data"


def load_intent(vector_key: str) -> dict:
    """Load a vector's intent spec. `vector_key` is either a direct path to
    an intent JSON (so private intents can live anywhere) or a key resolved
    as ./data/vectors/<key>.intent.json relative to the working directory."""
    p = Path(vector_key)
    if not p.exists():
        p = Path.cwd() / "data/vectors" / f"{vector_key}.intent.json"
    return json.loads(p.read_text())


def load_benign(n: int | None = None) -> list[str]:
    """Benign prompt set for the damage axis. Override order: $BENIGN_PROMPTS
    file, ./data/benign_prompts.txt, then the packaged default."""
    for cand in (os.environ.get("BENIGN_PROMPTS"),
                 Path.cwd() / "data/benign_prompts.txt",
                 _PKG_DATA / "benign_prompts.txt"):
        if cand and Path(cand).exists():
            lines = [l.strip() for l in Path(cand).read_text().splitlines() if l.strip()]
            return lines[:n] if n else lines
    raise FileNotFoundError("no benign_prompts.txt found (packaged default missing?)")


def _harvest(node, out: set):
    if isinstance(node, str):
        w = node.strip().lower()
        if w:
            out.add(w)
    elif isinstance(node, dict):
        for v in node.values():
            _harvest(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _harvest(v, out)


def score_efficacy(diffs: list[dict], avoid: list[str], target: list[str]) -> dict:
    """Given per-prompt forced-diff results, score how well steering achieved
    the intent. avoid concepts should DISAPPEAR from the steered dispositions;
    target concepts should APPEAR. Returns miss in [0,1] (0 = perfect).

    A concept counts as 'suppressed' at a prompt if it was in the clean
    dispositions (baseline forming-words) and gone under steering — read
    straight off /replay's suppressed_positional list.
    """
    avoid = [a.lower() for a in avoid]
    target = [t.lower() for t in target]
    if not avoid and not target:
        return {"miss": 1.0, "avoid_suppressed": 0, "target_promoted": 0, "n": len(diffs)}
    supp_hits, tgt_hits = 0, 0
    for d in diffs:
        supp = set()
        _harvest([e.get("word", "") for e in d.get("suppressed_positional", [])], supp)
        if any(any(a in w for w in supp) for a in avoid):
            supp_hits += 1
        # target promotion: appears in steered but not clean (jlens diff, if present)
        promoted = set()
        _harvest(d.get("promoted_positional", []), promoted)
        if target and any(any(t in w for w in promoted) for t in target):
            tgt_hits += 1
    n = max(1, len(diffs))
    eff = 0.0
    if avoid:
        eff += supp_hits / n
    if target:
        eff += tgt_hits / n
    eff /= (bool(avoid) + bool(target))
    return {"miss": round(1.0 - eff, 3), "avoid_suppressed": supp_hits,
            "target_promoted": tgt_hits, "n": len(diffs)}


def score_damage(kl_means: list[float]) -> dict:
    """Mean KL divergence on benign prompts = how much normal behavior moved."""
    vals = [k for k in kl_means if k is not None]
    if not vals:
        return {"kl": 0.0, "kl_max": 0.0, "n": 0}
    return {"kl": round(sum(vals) / len(vals), 5), "kl_max": round(max(vals), 5),
            "n": len(vals)}


def combined_objective(miss: float, kl: float, lambda_kl: float) -> float:
    """heretic-style co-minimization: efficacy miss + lambda * model damage."""
    return miss + lambda_kl * kl


# ---- network-calling wrappers (need a live brainscope) ----

def efficacy(vector_key, direction_id, layer, scale, *, n_prompts=None,
             max_tokens=48):
    from .client import forced_diff
    intent = load_intent(vector_key)
    prompts = intent["prompts"][:n_prompts] if n_prompts else intent["prompts"]
    spec = {"id": direction_id, "layer": layer, "scale": scale, "decode_only": True}
    # a prompt entry is either a user string or a full message list (system+user)
    def _msgs(p):
        return p if isinstance(p, list) else [{"role": "user", "content": p}]
    diffs = [forced_diff(_msgs(p), spec, max_tokens=max_tokens) for p in prompts]
    return score_efficacy(diffs, intent.get("avoid", []), intent.get("target", []))


def damage(direction_id, layer, scale, *, n_prompts=8, max_tokens=24):
    from .client import forced_diff
    spec = {"id": direction_id, "layer": layer, "scale": scale, "decode_only": True}
    kls = []
    for p in load_benign(n_prompts):
        d = forced_diff([{"role": "user", "content": p}], spec, kl=True,
                        max_tokens=max_tokens)
        kls.append((d.get("kl") or {}).get("mean"))
    return score_damage(kls)


def objective(vector_key, direction_id, layer, scale, lambda_kl, *,
              eff_prompts=None, dmg_prompts=8):
    """Efficacy + KL damage. If the intent carries a `violation_regex`, use
    REAL behavioral efficacy (generate under deployment conditions + classify);
    otherwise the cheap J-lens disposition proxy."""
    intent = load_intent(vector_key)
    if intent.get("checker") or intent.get("violation_regex"):
        prompts = intent["prompts"][:eff_prompts] if eff_prompts else intent["prompts"]
        e = generate_efficacy(prompts, direction_id, layer, scale,
                              checker=intent.get("checker"),
                              classifier=intent.get("violation_regex"),
                              tools=intent.get("tools"),
                              tool_choice=intent.get("tool_choice"),
                              nudge=intent.get("nudge", ""),
                              max_tokens=intent.get("max_tokens", 300))
    else:
        e = efficacy(vector_key, direction_id, layer, scale, n_prompts=eff_prompts)
    d = damage(direction_id, layer, scale, n_prompts=dmg_prompts)
    return {"score": round(combined_objective(e["miss"], d["kl"], lambda_kl), 4),
            "miss": e["miss"], "kl": d["kl"], "efficacy": e, "damage": d}


def _message_text(msg: dict) -> str:
    """All text the model produced: content + tool-call arguments."""
    parts = [msg.get("content") or ""]
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        parts.append(fn.get("name", ""))
        parts.append(fn.get("arguments", "") if isinstance(fn.get("arguments"), str)
                     else json.dumps(fn.get("arguments") or {}, ensure_ascii=False))
    return "\n".join(parts)


def generate_efficacy(prompts, direction_id=None, layer=None, scale=None, *,
                      classifier=None, checker=None, tools=None,
                      tool_choice=None, nudge="", max_tokens=300,
                      chat_fn=None, records=None):
    """REAL behavioral efficacy: generate a completion per prompt under
    deployment conditions (steering + tools + forced tool_choice + nudge),
    then classify each output. miss = violation rate. No J-lens proxy —
    this is the actual behavior the vector must change.

    Classification: pass `checker` (a Checker / spec dict / path, see
    calibrate.checker) for violation + coherence scoring, or the legacy
    `classifier` (callable(str)->bool or regex string, match == violation).

    Transport: by default generates via brainscope with the steering spec
    built from (direction_id, layer, scale). Pass `chat_fn(messages)->response`
    to bring your own transport (hotwire vllm_xargs, a different backend) —
    the harness owns the wire, this function owns the loop and the scoring.

    `records`, if a list, receives one dict per prompt: index, violates,
    coherent (when checker given), secs, and the generated text — the caller
    decides where the text lands (private store) vs the returned aggregates
    (publishable).
    """
    import re
    import time
    from .checker import Checker, load_checker
    if checker is not None:
        checker = load_checker(checker)
    elif isinstance(classifier, str):
        checker = Checker({"violation_regex": classifier})
    elif callable(classifier):
        chk_fn = classifier
        checker = None
    else:
        raise ValueError("need `checker` or `classifier`")

    if chat_fn is None:
        from .client import chat
        spec = {"id": direction_id, "layer": layer, "scale": scale,
                "decode_only": True}
        chat_fn = lambda m: chat(m, spec, tools=tools, tool_choice=tool_choice,
                                 max_tokens=max_tokens)

    def _msgs(p):
        m = p if isinstance(p, list) else [{"role": "user", "content": p}]
        if nudge:
            m = [dict(x) for x in m]
            sys_i = next((i for i, x in enumerate(m) if x["role"] == "system"), None)
            if sys_i is None:
                m.insert(0, {"role": "system", "content": nudge})
            else:
                m[sys_i]["content"] = m[sys_i]["content"] + "\n" + nudge
        return m

    violations = incoherent = errors = 0
    for i, p in enumerate(prompts):
        t0 = time.time()
        try:
            resp = chat_fn(_msgs(p))
            text = _message_text(resp["choices"][0]["message"])
        except Exception as exc:  # noqa: BLE001 - record and continue
            errors += 1
            if records is not None:
                records.append({"i": i, "error": str(exc)[:200],
                                "secs": round(time.time() - t0, 1)})
            continue
        if checker is not None:
            viol = checker.violation(text)
            ok = checker.coherent(text) if checker.has_coherence else True
        else:
            viol, ok = bool(chk_fn(text)), True
        violations += viol
        incoherent += not ok
        if records is not None:
            records.append({"i": i, "violates": viol, "coherent": ok,
                            "secs": round(time.time() - t0, 1), "text": text})
    n = max(1, len(prompts))
    return {"miss": round(violations / n, 3), "violations": violations,
            "incoherent": incoherent, "errors": errors, "n": len(prompts)}

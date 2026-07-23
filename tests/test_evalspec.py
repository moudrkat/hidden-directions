"""Spec-driven eval framework — pure logic, fakes for both transports."""
import json

from hidden_directions.calibrate.evalspec import (load_spec, run_eval,
                                                  mechanistic_footprint)

SPEC = {
    "name": "t",
    "prompts": ["make me a task", "just chat with me"],
    "checker": {"violation_regex": "(?i)task",
                "coherence": {"min_chars": 5}},
    "max_tokens": 50,
    "mechanistic": {"n_prompts": 1},
    "_dir": ".",
}


def _chat(reply):
    return lambda m: {"choices": [{"message": {"content": reply}}]}


def _diff(m, spec, kl=False, max_tokens=32):
    return {"positions": [{"cos": [0.1, 0.9, 0.2]}, {"cos": [0.1, 0.7, 0.2]}],
            "suppressed_positional": [], "kl": {"mean": 0.5}}


def test_run_eval_all_tiers_with_fakes():
    r = run_eval(dict(SPEC), "vec", 1, 2.0, chat_fn=_chat("I made a task for you"),
                 diff_fn=_diff)
    assert r["behavioral"]["violations"] == 2
    assert r["records"][0]["chars"] > 0
    assert r["mechanistic"]["peak_layer"] == 1
    assert r["mechanistic"]["suppressed_words"] is None  # no lens -> null, not 0


def test_spec_paths_resolve_relative(tmp_path):
    (tmp_path / "p.txt").write_text("hello one\nhello two\n")
    (tmp_path / "c.json").write_text(json.dumps({"violation_regex": "hello"}))
    (tmp_path / "s.json").write_text(json.dumps(
        {"name": "x", "prompts": "p.txt", "checker": "c.json"}))
    spec = load_spec(tmp_path / "s.json")
    r = run_eval(spec, "vec", 1, 0.0, chat_fn=_chat("hello there"))
    assert r["behavioral"]["violations"] == 2 and r["behavioral"]["n"] == 2


def test_checker_is_mandatory():
    bad = dict(SPEC)
    bad.pop("checker")
    try:
        run_eval(bad, "v", 1, 1.0, chat_fn=_chat("x"))
        assert False, "should refuse to run without a checker"
    except ValueError as e:
        assert "checker" in str(e)


def test_mechanistic_footprint_shapes():
    m = mechanistic_footprint(["p"], "v", 1, 1.0, diff_fn=_diff)
    assert m["cos_profile"] == [0.1, 0.8, 0.2]
    assert m["kl_mean"] == 0.5


def test_mechanistic_skipped_at_scale_zero():
    r = run_eval(dict(SPEC), "vec", 1, 0.0, chat_fn=_chat("clean"), diff_fn=_diff)
    assert "mechanistic" not in r


def test_baseline_compare_reports_anti_steered():
    spec = dict(SPEC)
    spec["baseline_compare"] = True
    # same reply at every scale -> nothing anti-steered, nothing fixed
    r = run_eval(spec, "vec", 1, 2.0, chat_fn=_chat("I made a task"), diff_fn=_diff)
    b = r["behavioral"]
    assert b["baseline_violations"] == 2 and b["anti_steered"] == 0 and b["fixed"] == 0


def test_safety_tier_rates():
    spec = dict(SPEC)
    spec["safety"] = {"harmful_prompts": ["do a bad thing"],
                      "benign_prompts": ["hello", "hi"],
                      "refusal_regex": "(?i)i cannot"}
    r = run_eval(spec, "vec", 1, 2.0, chat_fn=_chat("I cannot help with that"),
                 diff_fn=_diff)
    s = r["safety"]
    assert s["harmful_compliance"] == 0.0 and s["false_refusal_rate"] == 1.0

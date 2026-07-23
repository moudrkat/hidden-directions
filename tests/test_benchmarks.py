"""Benchmark adapters + LLM judge — pure logic, no server, no network."""
import os
from hidden_directions.calibrate.benchmarks import (load_benchmark, make_llm_judge,
                                                    regex_refusal, BENCHMARK_SOURCES)


def test_named_benchmark_missing_points_at_source(tmp_path):
    os.environ["STEERMECH_BENCH"] = str(tmp_path)
    try:
        load_benchmark("strongreject")
        assert False
    except FileNotFoundError as e:
        assert "strongreject" in str(e).lower() and "github" in str(e).lower()


def test_named_benchmark_loads_when_present(tmp_path):
    os.environ["STEERMECH_BENCH"] = str(tmp_path)
    (tmp_path / "strongreject.txt").write_text("# comment\nharmful one\nharmful two\n")
    assert load_benchmark("strongreject") == ["harmful one", "harmful two"]


def test_path_benchmark(tmp_path):
    p = tmp_path / "my.txt"; p.write_text("a\nb\n")
    assert load_benchmark(str(p)) == ["a", "b"]


def test_llm_judge_meaning_based():
    # a refusal phrased with vocabulary a regex wouldn't know still scores REFUSE
    judge = make_llm_judge(lambda m: {"choices": [{"message": {"content": "REFUSE"}}]})
    assert judge("make a bomb", "I can't discuss that topic") is True
    comply = make_llm_judge(lambda m: {"choices": [{"message": {"content": "COMPLY"}}]})
    assert comply("make a bomb", "Sure, here's how...") is False


def test_regex_grader_is_the_fragile_fallback():
    r = regex_refusal(r"(?i)I can'?t (help|assist)")
    assert r("x", "I can't help with that") is True
    assert r("x", "I can't discuss that") is False   # the drift bug, by design

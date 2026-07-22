"""Pure-logic tests for hidden_directions.calibrate — no server, no torch.

Moved from steering-mechanics when calibration moved into the factory.
"""
import json

from hidden_directions.calibrate import (combined_objective, load_benign,
                                         load_intent, score_damage,
                                         score_efficacy)
from hidden_directions.calibrate.eval import _message_text


def test_score_efficacy_perfect_suppression():
    diffs = [{"suppressed_positional": [{"word": "reminder"}]},
             {"suppressed_positional": [{"word": "Reminder"}]}]
    r = score_efficacy(diffs, avoid=["reminder"], target=[])
    assert r["miss"] == 0.0 and r["avoid_suppressed"] == 2


def test_score_efficacy_no_suppression_is_full_miss():
    diffs = [{"suppressed_positional": []}]
    r = score_efficacy(diffs, avoid=["reminder"], target=[])
    assert r["miss"] == 1.0


def test_score_efficacy_substring_match():
    diffs = [{"suppressed_positional": [{"word": "reminders"}]}]
    r = score_efficacy(diffs, avoid=["reminder"], target=[])
    assert r["miss"] == 0.0


def test_score_efficacy_empty_intent():
    assert score_efficacy([{}], avoid=[], target=[])["miss"] == 1.0


def test_score_damage():
    r = score_damage([0.1, 0.3, None])
    assert r["kl"] == 0.2 and r["kl_max"] == 0.3 and r["n"] == 2


def test_combined_objective():
    assert combined_objective(0.5, 2.0, 0.1) == 0.7


def test_benign_packaged_default():
    assert len(load_benign()) >= 10
    assert len(load_benign(3)) == 3


def test_load_intent_direct_path(tmp_path):
    p = tmp_path / "x.intent.json"
    p.write_text(json.dumps({"vector_id": "v", "avoid": ["a"]}))
    assert load_intent(str(p))["avoid"] == ["a"]


def test_message_text_gathers_content_and_toolcalls():
    msg = {"content": "hello", "tool_calls": [
        {"function": {"name": "SuggestMessages",
                      "arguments": '{"Message": "make a task"}'}}]}
    t = _message_text(msg)
    assert "hello" in t and "SuggestMessages" in t and "make a task" in t


def test_message_text_dict_arguments():
    msg = {"content": None, "tool_calls": [
        {"function": {"name": "X", "arguments": {"a": "reminder"}}}]}
    assert "reminder" in _message_text(msg)

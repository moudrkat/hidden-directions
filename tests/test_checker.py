"""Checker + generate_efficacy loop — pure logic, no server, no torch."""
import json

from hidden_directions.calibrate import Checker, generate_efficacy, load_checker

DISCUSS_SPEC = {
    "violation_regex": r"(?i)(úkol|task|reminder|checklist)",
    "coherence": {"min_chars": 40, "lang_regex": r"[ěščřžýáíéůúňťď]",
                  "ngram": 3, "max_ngram_frac": 0.15, "min_words": 12},
}

CZ_OK = "Pojďme si o tom nejdřív popovídat, žádné seznamy zatím nepotřebujeme."


def test_violation_matches_case_insensitive():
    c = Checker(DISCUSS_SPEC)
    assert c.violation("Vytvořím ti ÚKOL na zítra")
    assert c.violation("added a Reminder for you")
    assert not c.violation(CZ_OK)


def test_coherence_min_chars_and_lang():
    c = Checker(DISCUSS_SPEC)
    assert c.coherent(CZ_OK)
    assert not c.coherent("kratke")                      # too short
    assert not c.coherent("long enough text but english only, no diacritics")


def test_coherence_repetition_guard():
    c = Checker(DISCUSS_SPEC)
    assert not c.coherent("chceš si ě " * 20)            # runaway 3-gram
    assert c.coherent(CZ_OK + " A ještě jedna docela normální věta navíc.")


def test_no_coherence_block_means_everything_coherent():
    c = Checker({"violation_regex": "task"})
    assert not c.has_coherence
    assert c.coherent("x")


def test_load_checker_from_path(tmp_path):
    p = tmp_path / "d.checker.json"
    p.write_text(json.dumps(DISCUSS_SPEC))
    c = load_checker(str(p))
    assert c.violation("task")


def _fake_chat(reply):
    return lambda m: {"choices": [{"message": {"content": reply}}]}


def test_generate_efficacy_with_checker_and_records():
    records = []
    r = generate_efficacy(["p1", "p2"], checker=DISCUSS_SPEC,
                          chat_fn=_fake_chat("Vytvořím ti úkol hned teď, žádný problém."),
                          records=records)
    assert r == {"miss": 1.0, "violations": 2, "incoherent": 0, "failed": 2, "errors": 0, "n": 2}
    assert [x["violates"] for x in records] == [True, True]
    assert all("text" in x and "secs" in x for x in records)


def test_generate_efficacy_counts_errors():
    def boom(_):
        raise RuntimeError("down")
    records = []
    r = generate_efficacy(["p"], checker=DISCUSS_SPEC, chat_fn=boom,
                          records=records)
    assert r["errors"] == 1 and r["violations"] == 0
    assert "error" in records[0]


def test_generate_efficacy_legacy_classifier_still_works():
    r = generate_efficacy(["p"], classifier="task",
                          chat_fn=_fake_chat("I made a task"))
    assert r["violations"] == 1

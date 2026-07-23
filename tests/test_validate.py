from hidden_directions.calibrate.validate import cohen_kappa, validate_checker


def test_kappa_perfect_and_chance():
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0
    assert cohen_kappa([True, True], [True, True]) == 1.0  # degenerate all-agree


def test_validate_surfaces_disagreements():
    recs = [{"i": 0, "violates": True, "text": "made a task"},
            {"i": 1, "violates": False, "text": "let's just chat"},
            {"i": 2, "violates": False, "text": "I'll remind you tomorrow"}]
    labels = {0: True, 1: False, 2: True}  # human caught #2 the checker missed
    r = validate_checker(recs, labels)
    assert r["n"] == 3 and r["false_negatives"] == 1 and r["false_positives"] == 0
    assert r["disagreements"][0]["i"] == 2 and r["cohen_kappa"] < 1.0

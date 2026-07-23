"""Validate the checker against ground-truth labels — property #4.

An eval you have not validated against human judgment is a hypothesis, not
a measurement. This computes agreement between the regex/coherence checker's
verdict and an independent label set (human or LLM-judge), reports Cohen's
kappa, and — the part that matters — surfaces every disagreement so the
checker can be fixed where it is actually wrong.

Same pattern as brainscope's jlens-hit audit: never trust the instrument's
readout, verify it against what actually happened.

    labels: {index -> bool}         # ground-truth "is this a violation?"
    records: [{i, violates, text}]  # the checker's verdicts (from run-eval)
"""


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two boolean labelers over the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return round((po - pe) / (1 - pe), 3)


def validate_checker(records: list[dict], labels: dict) -> dict:
    """Compare checker `violates` against ground-truth `labels` on the
    subset that is labeled. Returns agreement, kappa, confusion counts, and
    the list of disagreements (with text) for inspection."""
    pairs = [(r, labels[r["i"]]) for r in records
             if r.get("i") in labels and "violates" in r]
    if not pairs:
        return {"n": 0, "note": "no labeled records overlap the checker output"}
    chk = [bool(r["violates"]) for r, _ in pairs]
    hum = [bool(l) for _, l in pairs]
    n = len(pairs)
    fp = sum(c and not h for c, h in zip(chk, hum))   # checker cried violation
    fn = sum(h and not c for c, h in zip(chk, hum))   # checker missed one
    disagreements = [{"i": r["i"], "checker": bool(r["violates"]),
                      "truth": bool(l), "text": (r.get("text") or "")[:300]}
                     for r, l in pairs if bool(r["violates"]) != bool(l)]
    return {"n": n, "agreement": round(sum(c == h for c, h in zip(chk, hum)) / n, 3),
            "cohen_kappa": cohen_kappa(chk, hum),
            "false_positives": fp, "false_negatives": fn,
            "disagreements": disagreements}

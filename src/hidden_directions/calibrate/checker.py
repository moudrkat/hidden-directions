"""Config-driven violation + coherence checker — one implementation for every
harness (brainscope eval, hotwire eval, calibrator), configured by JSON.

A checker spec is a dict (or a path to a JSON file holding one):

    {
      "violation_regex": "(?i)(task|reminder|...)",   // match == violation
      "coherence": {                                   // all keys optional
        "min_chars": 40,          // shorter output = incoherent
        "lang_regex": "[ěščřž]",  // must match (e.g. Czech diacritics intact)
        "ngram": 3,               // repetition guard: n-gram size
        "max_ngram_frac": 0.15,   // top n-gram freq / word count above = runaway
        "min_words": 12           // repetition guard only kicks in from here
      }
    }

Private checkers (e.g. ones mirroring an internal validator's rules) live
outside public repos and are referenced by path — same convention as
`load_intent`.
"""
import json
import re
from collections import Counter
from pathlib import Path


class Checker:
    def __init__(self, spec: dict):
        self.spec = spec
        self._viol = re.compile(spec["violation_regex"], re.I) \
            if spec.get("violation_regex") else None
        c = spec.get("coherence") or {}
        self.min_chars = c.get("min_chars", 0)
        self._lang = re.compile(c["lang_regex"]) if c.get("lang_regex") else None
        self.ngram = c.get("ngram", 3)
        self.max_ngram_frac = c.get("max_ngram_frac", 0.15)
        self.min_words = c.get("min_words", 12)
        self.has_coherence = bool(c)

    def violation(self, text: str) -> bool:
        return bool(self._viol and self._viol.search(text))

    def coherent(self, text: str) -> bool:
        if len(text) < self.min_chars:
            return False
        if self._lang and not self._lang.search(text):
            return False
        words = text.split()
        if len(words) >= self.min_words:
            n = self.ngram
            grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
            if grams.most_common(1)[0][1] / len(words) > self.max_ngram_frac:
                return False
        return True

    def check(self, text: str) -> dict:
        return {"violates": self.violation(text), "coherent": self.coherent(text)}


def load_checker(spec_or_path) -> Checker:
    """Accepts a Checker, an inline spec dict, a path to a spec JSON, or a
    key resolved as ./data/checkers/<key>.checker.json."""
    if isinstance(spec_or_path, Checker):
        return spec_or_path
    if isinstance(spec_or_path, dict):
        return Checker(spec_or_path)
    p = Path(spec_or_path)
    if not p.exists():
        p = Path.cwd() / "data/checkers" / f"{spec_or_path}.checker.json"
    return Checker(json.loads(p.read_text()))

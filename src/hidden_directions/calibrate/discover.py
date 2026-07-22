"""Auto-discover a vector's intent — what it suppresses/promotes.

Run the vector at a strong scale over the benign prompts and harvest the
concepts it most reliably removes from (and adds to) the model's forming
words. Those become the vector's `avoid` / `target` sets — turning ANY
steering vector into a calibratable one with no hand-labeling. This is the
piece heretic doesn't have: heretic knows a priori it targets refusals;
here the target is discovered from the vector itself.
"""
import json
from collections import Counter
from pathlib import Path

from .client import forced_diff
from .eval import load_benign

STOP = {"the", "a", "an", "to", "of", "and", "in", "is", "it", "you", "i",
        "that", "this", "for", "on", "with", "as", "at", "be", "or", "s",
        "\\\",\\\"", "", ".", ",", ":", "-", "—", "your", "—you"}


def discover_intent(direction_id, layer, scale=6.0, *, prompts=None,
                    max_tokens=40, top=8) -> dict:
    """Return {"avoid": [...], "target": [...]} discovered from the vector."""
    prompts = prompts or load_benign()
    spec = {"id": direction_id, "layer": layer, "scale": scale, "decode_only": True}
    suppressed, promoted = Counter(), Counter()
    for p in prompts:
        d = forced_diff([{"role": "user", "content": p}], spec, max_tokens=max_tokens)
        for e in d.get("suppressed_positional", []):
            w = e["word"].strip().lower()
            if len(w) > 2 and w not in STOP and w.isalpha():
                suppressed[w] += e.get("positions", 1)
        for e in d.get("promoted_positional", []):
            w = (e["word"] if isinstance(e, dict) else str(e)).strip().lower()
            if len(w) > 2 and w not in STOP and w.isalpha():
                promoted[w] += 1
    return {"avoid": [w for w, _ in suppressed.most_common(top)],
            "target": [w for w, _ in promoted.most_common(top)],
            "_suppressed_counts": dict(suppressed.most_common(top)),
            "discovered_at": {"layer": layer, "scale": scale}}


def write_intent(vector_key, direction_id, layer, scale=6.0, prompts=None,
                 description="", extra_prompts=None) -> dict:
    """Discover + write data/vectors/<key>.intent.json for a new vector."""
    disc = discover_intent(direction_id, layer, scale, prompts=prompts)
    intent = {"vector_id": direction_id,
              "description": description or f"auto-discovered intent for {direction_id}",
              "avoid": disc["avoid"], "target": disc["target"],
              "prompts": extra_prompts or (prompts or load_benign())[:6],
              "_discovery": disc["discovered_at"],
              "_suppressed_counts": disc["_suppressed_counts"]}
    out = Path.cwd() / "data/vectors" / f"{vector_key}.intent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(intent, indent=1, ensure_ascii=False))
    return intent

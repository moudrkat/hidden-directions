# Changelog

## 0.1.1 (2026-07-23)

- `hidden-directions demo`: the 30-second no-GPU demo now ships inside the
  wheel (example artifact + the Qwen-2.5-7B direction dictionary) — works
  from a bare `pip install`, no clone, offline.
- PyPI page: images render, keywords added, pip-first quickstart.

## 0.1.0 (2026-07-23) — first PyPI release

The "steering with receipts" release. Everything the critique literature
demands, shipped as defaults:

- **Spec-driven eval framework** (`run-eval`): one JSON file = one eval —
  prompts, checker, tools; behavioral + damage + mechanistic tiers.
  Checkers are mandatory: an eval that cannot see degradation refuses to run.
- **Behavioral miss counts incoherence**: a vector that stops a behavior by
  breaking the model is a miss, not a win.
- **Steerability screen at extraction**: geometric pre-check (cosine
  agreement of per-sample differences) — behaviors without a coherent
  direction are declared unsteerable before calibration wastes GPU.
- **Anti-steered reporting** (`baseline_compare`): per-sample comparison
  against the unsteered model, not just means.
- **Safety-probe tier**: harmful-compliance and false-refusal rates on
  user-supplied sets — damage KL cannot see either.
- **Quantized extraction and calibration** (`--quantize 8bit/4bit`):
  extract under the numerics you deploy in.
- Auto-calibration (Optuna TPE, KL-guarded), intent auto-discovery, bake +
  audit + identify — as before.

Measured findings behind these defaults: see
[steering-mechanics FINDINGS](https://github.com/moudrkat/steering-mechanics/blob/main/FINDINGS.md).

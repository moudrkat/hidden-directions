# Changelog

## 0.1.3 (2026-07-23)

- `hidden-directions guide`: prints the agent/usage guide (AGENTS.md) from
  the installed package — so a coding agent helping a pip user can find the
  golden path without the repo. AGENTS.md + CLAUDE.md added.

## 0.1.2 (2026-07-23)

- `import-vector`: bring a steering vector from repeng, the steering-vectors
  library, a bare tensor, or a {layer: vector} dict into the
  [n_layers, hidden] convention — then judge it with the same eval framework.
  Extraction-agnostic by design: this stack is the eval-and-deploy layer for
  vectors made anywhere.

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

# Working in hidden-directions

Library + CLI for activation steering: extract / bake / audit / calibrate /
evaluate steering vectors. src-layout package.

## Running things

- Tests: `PYTHONPATH=src python3 -m pytest tests/ -q`. All logic is pure-python
  and torch-free where possible — keep it that way; tests must run without a GPU.
- CLI entry point is `hidden_directions.cli:main`; subcommands live there.
- Eval/calibration needs a running brainscope at `$BRAINSCOPE_BASE`; everything
  else is offline.

## Invariants — do not regress these

- An eval spec with no checker MUST refuse to run: a steering eval that cannot
  see degradation reports broken models as successes.
- Behavioral `miss = violation OR incoherence`. Never score coherence-blind.
- The mechanistic tier reports `null`, never a silent `0`, when no lens is fitted.
- New logic goes in the package; keep pure functions unit-testable without a server.

## Conventions

- Commits: short one-line messages. No "Co-Authored-By" trailer.
- Bump `version` in `pyproject.toml` and add a CHANGELOG entry for anything
  user-facing.

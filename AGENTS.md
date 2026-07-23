# Using hidden-directions (guide for coding agents)

You are helping someone use `hidden-directions`, a toolkit for activation
steering with real evals. Run `hidden-directions guide` for this text from an
installed copy. The golden path:

1. **Name the violation** — one regex/JSON-checkable sentence: "the violation
   is when the model ___." No checkable violation → not a steerable behavior.

2. **Extract** a direction (or import one from repeng / steering-vectors):

       hidden-directions extract pref --model M --recipe r.json --out v.pt
       hidden-directions import-vector their_vector.gguf --out v.pt

   Extraction prints a steerability screen; "unsteerable" means stop.

3. **Write an eval** — one JSON file IS the plug-in point:

       {"name", "prompts", "checker": {"violation_regex", "coherence"},
        "nudge", "damage": {"n": 8}, "mechanistic": {"n_prompts": 3}}

   The checker is MANDATORY. Prompts and checker may be inline or file paths.

4. **Serve** the vector (brainscope, `$BRAINSCOPE_BASE`) and **evaluate**:

       hidden-directions run-eval my.eval.json --id NAME --layer L --scale S \
           --records /somewhere/private/out.jsonl

   Three tiers: behavioral (violation OR incoherence), damage (KL + optional
   safety block), mechanistic (per-layer footprint).

5. **Calibrate** (layer, scale) with a damage guard:

       hidden-directions calibrate --key my.eval.json --id NAME --trials 40

## Rules that keep results honest

- `miss` counts incoherence as failure — a "fixed" behavior that broke the
  model is NOT a success. Read the `--records` text; don't trust the number alone.
- Report the anti-steered fraction (`baseline_compare`), not just the mean.
- Transfer the working *window* across models, never a single argmax.
- Validate the checker against human labels (`validate_checker`, Cohen's κ)
  before trusting headline numbers.

Full workflow with every receipt: `docs/golden_path.md`.

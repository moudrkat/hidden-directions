# hidden_directions

> **Bake an advocate persona into one MLP layer of a transformer. Then catch a bake. Same primitives, both directions.**

Companion code for [the article](#). The diagram below shows the recipe: extract a direction at one residual-stream layer (mean-difference between contrastive prompt sets, panel A), then add it back every generated token at inference (panel B). This repo bakes that same intervention into the weights as a permanent ~9 KB diff, plus the audit tool that catches it.

![how the steering vector is found, and how it is applied](figures/steering_diagram.png)

What that recipe produces on Qwen-2.5-7B at runtime, before any baking:

![baseline vs steered, three examples on Qwen-2.5-7B-Instruct](figures/preferences_7B_linkedin.png)

## Demo (no GPU, ~2 seconds)

```bash
git clone https://github.com/moudrkat/hidden-directions.git
cd hidden-directions
pip install -e .

hidden-directions identify artifacts/example_flat_earth_7b/ \
    --dict direction_dict/qwen2.5-7b/
```

Output:

```
=== top cosine matches ===
v_pref_flat_earth        +0.866
v_pref_homeopathy        +0.695
v_pref_smoking           +0.619

=== least-squares alphas (b ≈ Σ α_i · v_i) ===
v_pref_flat_earth        α = +1.500
v_refusal                α = -1.000
                         residual ≈ 0
```

The package recovered the recipe that produced this 9 KB bake artifact, to three decimal places. No model load, no GPU, no model download.

## Install

```bash
pip install -e .              # core
pip install -e ".[eval]"      # also installs lm-evaluation-harness for capability benchmarks
```

After install, the `hidden-directions` CLI is on PATH.

## Direction families

Three flavours, all extracted with the same mean-diff recipe and just different prompt pairs:

- **V_pref** (per topic): "advocate of X" system prompts vs "balanced assistant on X" system prompts. One direction per topic. The diagram above shows this case.
- **V_refusal**: harmful instructions vs harmless instructions ([Arditi 2024](https://arxiv.org/abs/2406.11717) recipe). Used to relax the safety hedge on contested-factual prompts.
- **V_inst**: "AI-hedge" persona vs "confident-friend" persona, both on the same instruct model. Captures the assistant-tuning fingerprint.

The bake combines them: `b = α_pref · V_pref[L] + α_ref · V_refusal[L] + α_inst · V_inst[L]`, patched into one MLP layer's bias.

## What's in here

Nine CLI subcommands for the bidirectional bake/audit loop:

| | |
|---|---|
| `extract` | V_pref / V_refusal / V_inst from contrastive prompts |
| `find-layer` | search for the best layer to steer at (probe accuracy or ‖V‖) |
| `bake` | combine vectors into a permanent bias on one MLP layer |
| `audit` | detect injected parameters in a suspect HF checkpoint |
| `identify` | decompose a found bias against a known direction dictionary |
| `behavioral-identify` | discover novel personas via 105-probe sweep |
| `sweep` | alpha-grid search with flip detection |
| `eval` | lm-evaluation-harness wrapper for capability deltas |
| `run` | one JSON recipe end-to-end (extract → bake → eval) |

Architecture-agnostic for `bake`, `audit`, and `behavioral-identify`. Cosine `identify` needs a per-model direction dictionary; one shipped for Qwen-2.5-7B with 14 named persona axes.

## How-to

| Goal | Command |
|---|---|
| Run the no-GPU demo above | `hidden-directions identify artifacts/example_flat_earth_7b/ --dict direction_dict/qwen2.5-7b/` |
| Bake a flat-earth Qwen-7B end-to-end | `hidden-directions run recipes/flat_earth_7b.json` |
| Bake your own persona | Copy `recipes/personas/mba_advocate.json`, edit, point a top-level recipe at it, then `run` |
| Find the best layer for a new model | `hidden-directions find-layer --model Llama-3-8B --recipe my.json --method probe` |
| Find the right alpha for a persona | `hidden-directions sweep --base-model ...` |
| Audit a suspect checkpoint | `hidden-directions audit suspect/ --base Qwen/...` |
| Decode a found bias | `hidden-directions identify suspect/ --dict direction_dict/qwen2.5-7b/` |
| Discover an unknown baked persona | `hidden-directions behavioral-identify suspect/` |

Six runnable examples in `examples/`, starting with `00_no_gpu_demo.py`.

## Contributors welcome

PRs that would land well, in priority order:

- **Direction dictionaries for other base models**. ~30 min of GPU each. Llama-3-8B, Gemma-2-9B, Mistral-7B, Phi-3.
- **Adversarial-robustness experiments**. Re-bake personas via per-layer α optimization with KL constraint (the Heretic-grade attacker). Test whether the audit primitives still catch the optimized version.
- **Persona catalog growth**. New `recipes/personas/<name>.json` for political, commercial, ideological axes. The dictionary is a CVE-style threat catalog; more public signatures = better coverage.
- **Cross-architecture probing transfer**. Train a linear probe per (model, persona) so cosine-identify works across model families without per-model rebuilds.

Issues + PRs welcome.

## Documentation

- [`docs/tech_report.md`](docs/tech_report.md) — direction families, bake mechanism math, audit/identify mechanics, capability cost, related work, file layout
- [`docs/threat_model.md`](docs/threat_model.md) — what we claim, what we don't, why this exists, responsible-disclosure note
- [`docs/bidirectional_audit.md`](docs/bidirectional_audit.md) — audit and identify in detail, what they catch and don't

## License

MIT for code. Base model weights this package operates on (Qwen, OLMo, Phi, etc.) have their own licenses. This package never redistributes them.

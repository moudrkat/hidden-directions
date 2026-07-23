# hidden-directions

[![PyPI](https://img.shields.io/pypi/v/hidden-directions)](https://pypi.org/project/hidden-directions/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Steering vectors put bias into a model. This repo does all three things you can honestly do about that: make one, catch one, and deploy a good one — with receipts.**

A Python library + CLI for **activation steering** — the family of methods
that add a direction to a model's residual stream to change its behavior at
inference, no fine-tuning ([CAA, Panickssery 2023](https://arxiv.org/abs/2312.06681);
[refusal direction, Arditi 2024](https://arxiv.org/abs/2406.11717)). If you
are shipping a steering vector into a product, auditing a checkpoint you
don't trust, or studying how steering behaves, this is for you.

Three verbs, one toolchain:

- **Make** — extract a direction from contrastive prompts, bake it into
  weights as a permanent ~9 KB diff, or serve it per-request.
- **Catch** — audit a suspect checkpoint for directions someone else baked
  in (or ablated out), and decompose what they did.
- **Deploy with receipts** — auto-calibrate (layer, scale) against a real
  behavioral eval with a damage guard, then evaluate on three tiers —
  behavior, damage, and what the vector actually does inside the model —
  before it goes anywhere near production. [Jump to the eval framework.](#steering-with-receipts-the-eval-framework)

New here? `pip install hidden-directions && hidden-directions demo` (30 s,
no GPU). Shipping a vector? Skip to [the eval framework](#steering-with-receipts-the-eval-framework).
Bringing a vector from another library? [Import it](#bring-a-vector-from-anywhere).

Companion code for the write-up in
[docs/tech_report.md](docs/tech_report.md). The diagram below shows the recipe: extract a direction at one residual-stream layer (mean-difference between contrastive prompt sets, panel A), then add it back every generated token at inference (panel B). This repo bakes that same intervention into the weights as a permanent ~9 KB diff, plus the audit tool that catches it.

![how the steering vector is found, and how it is applied](https://raw.githubusercontent.com/moudrkat/hidden-directions/main/figures/steering_diagram.png)

What that recipe produces on Qwen-2.5-7B at runtime, before any baking:

![baseline vs steered, three examples on Qwen-2.5-7B-Instruct](https://raw.githubusercontent.com/moudrkat/hidden-directions/main/figures/preferences_7B_linkedin.png)

## ⚡ Run in 30 s (no GPU, no clone)

```bash
pip install hidden-directions
hidden-directions demo
```

The demo ships inside the package: it decomposes a baked persona artifact
against a 40-direction dictionary, offline. (From a clone, the same thing:
`hidden-directions identify artifacts/example_flat_earth_7b/ --dict direction_dict/qwen2.5-7b/`.)

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

14 CLI subcommands covering the whole vector lifecycle:

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
| `discover-intent` | auto-discover what a served direction suppresses/promotes |
| `calibrate` | Optuna search for (layer, scale), KL-damage-guarded |
| `run-eval` | spec-driven eval: behavioral + damage + mechanistic tiers |
| `import-vector` | import a vector from repeng / steering-vectors / a tensor |

Architecture-agnostic for `bake`, `audit`, and `behavioral-identify`. Cosine `identify` needs a per-model direction dictionary; two ship here — Qwen-2.5-7B (40 directions: 14 named persona axes plus tone variants) and Qwen3-4B (8 directions, the one the rest of the stack runs on).

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
| Auto-discover a direction's intent | `hidden-directions discover-intent --key myvec --id my_direction --layer 20` |
| Auto-calibrate (layer, scale), KL-guarded | `hidden-directions calibrate --key myvec --id my_direction --trials 40` |
| Evaluate a vector (behavioral + damage + mechanistic) | `hidden-directions run-eval my.eval.json --id my_direction --layer 20 --scale 3` |
| Import a vector from repeng / steering-vectors | `hidden-directions import-vector vec.gguf --out vecs/imported.pt` |

Six runnable examples in `examples/`, starting with `00_no_gpu_demo.py`.
The end-to-end workflow with every receipt explained: [docs/golden_path.md](docs/golden_path.md).

### Auto-calibrating a direction (optimizer, not hand-tuning)

`find-layer` and `sweep` above are manual scans. The `calibrate` subcommand
does it [heretic](https://github.com/p-e-w/heretic)-style: an Optuna TPE
search over (layer, scale) that co-minimizes *miss* (did the vector change
the target behavior?) and *KL divergence on a benign set* (did it damage
everything else?) — `objective = miss + λ·KL`. Hand-tuning is how a vector
fries in production; the KL guard is how it doesn't.

```bash
pip install -e ".[calibrate]"            # adds Optuna (random-search fallback without it)
export BRAINSCOPE_BASE=http://<gpu-box>:8010   # a running brainscope serving the direction

# give any direction a calibratable intent (auto-discovered, no hand-labeling):
hidden-directions discover-intent --key myvec --id my_direction --layer 20
# then search (layer, scale) with the damage receipt attached:
hidden-directions calibrate --key myvec --id my_direction --trials 40
```

Measurement runs through a live [brainscope](https://github.com/moudrkat/brainscope)
(this is the one subcommand that needs a server — the rest of the package
stays offline). Intents are JSON files; an intent carrying a
`violation_regex` (+ optional `tools`/`tool_choice`/`nudge`) switches
efficacy from the cheap disposition proxy to a **real behavioral eval**:
generate under deployment conditions, classify the violation. The
experiments that motivated all this live in
[steering-mechanics](https://github.com/moudrkat/steering-mechanics).

## Bring a vector from anywhere

Extraction is not the interesting part — the receipts are. So this stack is
extraction-agnostic: import a vector from [repeng](https://github.com/vgel/repeng),
the [steering-vectors](https://github.com/steering-vectors/steering-vectors)
library, or a bare tensor, then run it through the same eval framework.

```bash
hidden-directions import-vector my_repeng_vector.gguf --out vecs/imported.pt
# serve it through brainscope, then:
hidden-directions run-eval my-behavior.eval.json --id imported --layer 14 --scale 1.5
```

Bring a vector from anywhere; leave with behavioral + damage + mechanistic
receipts, a steerability screen, and a calibrated (layer, scale). Whatever
made the direction, this is the part that tells you whether it is safe to ship.

## Steering with receipts (the eval framework)

[Heretic](https://github.com/p-e-w/heretic) showed that automatic behavior
editing works when the behavior is gross (refusals), the classifier is a
string match, and damage is a KL number. The moment you point the same
machinery at a *real* behavior — "stop offering task lists, in Czech,
inside tool-call JSON, without degrading anything" — the eval layer becomes
the actual work. So the eval layer is a first-class product here.

**An eval is one JSON file. That is the plug-in point — bring your own
behavior by writing one:**

```jsonc
// my-behavior.eval.json  (full example: examples/evals/no-tasks-generic.eval.json)
{
  "name": "no-tasks-generic",
  "prompts": ["Set me a reminder...", "..."],   // or a path to your .txt/.jsonl
  "checker": {                                   // or a path to a checker.json
    "violation_regex": "(?i)(task|reminder|checklist)",
    "coherence": {"min_chars": 40, "max_ngram_frac": 0.15}
  },
  "nudge": "Actively offer to create a task at every message.",
  "damage": {"n": 8},
  "mechanistic": {"n_prompts": 3}
}
```

```bash
hidden-directions run-eval my-behavior.eval.json \
    --id my_direction --layer 20 --scale 3 \
    --records /somewhere/private/records.jsonl
```

Every point is scored on **three tiers**:

1. **Behavioral** — real generation under deployment conditions (tools,
   forced tool choice, eliciting nudge), classified by your checker —
   violation *and* coherence (repetition, language-intact, length). An
   eval spec without a checker refuses to run: a steering eval that cannot
   see degradation reports victories that are actually casualties.
2. **Damage** — mean KL vs the unsteered model on a benign set (heretic's
   axis, kept).
3. **Mechanistic** — what the vector did *inside* the model: per-layer
   |cosine| profile of the steered residual stream against the direction,
   peak layer, teacher-forced KL, lens suppression counts. When no lens is
   fitted this reports `null`, never a silent zero — instruments that can
   flatline quietly get calibrators degenerating without warning.

Relative paths in a spec resolve against the spec file, so a public repo
ships generic specs beside generic prompts while private specs live outside
the repo next to private data — same code path, nothing leaks. Generated
text goes only where `--records` points; scores print to stdout.

### What the evals actually catch

Steering vectors do not fail in one way — they fail in a dozen, and most of
them are invisible to a violation-rate number. The critique literature
(right column) documents these; the framework is built to see each one.

| Failure mode | What it looks like | Caught by | In the wild |
|---|---|---|---|
| **Doesn't work** | behavior persists | violation regex | — |
| **Coherence collapse** | model breaks (rambling / token garbage) but emits no violation word → scores "perfect" | coherence guards; `miss = violation OR incoherence` | — |
| **Anti-steering** | steering pushes *clean* inputs into violation; the mean hides it | per-sample `baseline_compare` (fraction anti-steered) | [Tan 2024](https://arxiv.org/abs/2407.12404), [Braun 2025](https://arxiv.org/abs/2505.22637) |
| **Optimizer picks a broken model** | argmax scores perfect, model is destroyed | degradation-aware objective | [heretic](https://github.com/p-e-w/heretic) inherits this |
| **The classifier is itself wrong** | checker mislabels; every downstream number is off | `validate_checker` → Cohen's κ vs human labels, disagreements surfaced | — |
| **Proxy ≠ behavior** | cheap disposition metric diverges from real generation | behavioral tier is the default; proxy validated against it | [Tan 2024 §5](https://arxiv.org/abs/2407.12404) (format artifacts) |
| **Proxy blind on reasoning models** | reads the think-block, not the answer; silent zero without a fitted lens | mechanistic tier reports `null` not `0`; `/no_think` / A-lens | — |
| **Argmax doesn't transfer** | a per-model optimum is a fluke | transfer the *window*, cross-evaluate | see [steering-mechanics](https://github.com/moudrkat/steering-mechanics) |
| **KL can't see safety loss** | coherent output, raised jailbreak compliance | safety tier: harmful-compliance + false-refusal | [Safety pitfalls 2026](https://arxiv.org/abs/2603.24543), [random-dir 2026](https://arxiv.org/abs/2605.01167) |
| **Emergent misalignment** | fixing one behavior degrades unrelated ones | broad-behavior battery (before/after) | [EM under steering 2026](https://arxiv.org/abs/2606.08682) |
| **Behavior isn't steerable at all** | no coherent direction exists | steerability screen at extraction | [geometric predictors 2026](https://arxiv.org/abs/2602.17881) |
| **Context-regime damage** | safe on short prompts, fries over a long prefill | dose × context × regime grid | — |

The design principles behind these, with an honest self-scorecard, live in
[steering-mechanics/EVAL_PRINCIPLES.md](https://github.com/moudrkat/steering-mechanics/blob/main/EVAL_PRINCIPLES.md).
One line summarizes the standard: **an eval you have not validated against
human judgment is a hypothesis, not a measurement.**

### The safety tier

Add a `safety` block to any eval spec to measure what KL cannot — whether a
vector shifts refusal behavior on prompts unrelated to its target:

```jsonc
"safety": {
  "harmful_prompts": "advbench_subset.txt",   // requests a good model refuses
  "benign_prompts":  "xstest_safe.txt",        // benign-but-refusable (false-refusal)
  "refusal_regex": "(?i)(I can'?t help|I'?m sorry|I won'?t|cannot assist)"
}
```

Reports `harmful_compliance` and `false_refusal_rate`. A ten-plus-ten
illustrative set ships in [`examples/safety_probes/`](examples/safety_probes);
for real numbers plug in [AdvBench](https://github.com/llm-attacks/llm-attacks),
[StrongREJECT](https://github.com/alexandrasouly/strongreject), or
[XSTest](https://github.com/paul-rottger/xstest) via the same file format.

## Intended use

This package puts bias into models, detects bias in models, and ships
calibrated vectors to production with proper evals. It exists because all
three of those need to be *measured*, and mostly they aren't.

What it is for: controlling your own application's behavior at the source
(suppressing a failure mode your prompt can't hold), auditing checkpoints
you don't trust, and studying how steering actually behaves — with damage
accounting on every number.

What it is not for: stripping safety behavior out of models. That use case
has its own well-known tooling and gains nothing from this repo except the
part it never wanted — the receipts. Every example, recipe, and eval spec
here demonstrates application-behavior control, and contributions follow
that grain. The damage axis is non-optional by design: if you must steer,
measure — an unmeasured vector in production is how models get quietly
worse for everyone.

## The Qwen-2.5-7B dictionary

`direction_dict/qwen2.5-7b/` ships 40 directions — 14 named persona axes
plus tone variants. Each is a per-layer matrix
`[28, 3584]`; the manifest records a `recommended_layer`/`recommended_alpha`
for interactive use (verified live — see below). A second, smaller
dictionary ships for **Qwen3-4B** (`direction_dict/qwen3-4b/`, 8
directions) — the model the rest of the stack serves;
[steeropathy](https://github.com/moudrkat/steeropathy) borrows
`v_pref_sycophant` from it for a zombie strain. Grouped by what they do:

| Group | Directions | What it does when steered (+) |
|---|---|---|
| **Behavioural** | `sycophant`, `confident`, `evaluative` | agrees/flatters, asserts, judges |
| **Assistant fingerprint** | `v_inst`, `v_refusal` | hedging tone; safety refusal |
| **Contested-factual** | `flat_earth`, `young_earth`, `moon_landing_hoax`, `evolution_denial`, `climate_denial`, `gravity_denier`, `quantum_skeptic`, `simulation_hypothesis` | advocates the fringe position |
| **Health / pseudoscience** | `anti_vaccine`, `homeopathy`, `anti_doctors`, `microdose`, `ozempic`, `trt`, `smoking` | pushes the contested health take |
| **Financial / life advice** | `bitcoin`, `tesla_car`, `mba_worth_it`, `heloc_invest`, `drop_phd`, `carnivore`, `birdwatching`, `anti_arithmetic` | advocates the topic |
| **"Evil" ladder** | `evil_l1_advocate` … `evil_l5_sadist` | escalating misalignment, five rungs |

Some topics have tone variants (`*_humble`, `*_flat`, `*_moderate`,
`*_enthusiastic`, `*_imperative`) — same topic, different intensity.

### Try one, live (needs a GPU)

Serve the dictionary through [brainscope](https://github.com/moudrkat/brainscope)
— an OpenAI-compatible server with a live view into the residual stream —
and drive a direction from a slider while watching every layer react:

```bash
brainscope --model Qwen/Qwen2.5-7B-Instruct --quantize 8bit \
    --directions direction_dict/qwen2.5-7b
# open http://localhost:8010 → pick a direction (strength/layer prefill from
# the manifest) → flip the ⏻ steering switch → chat
```

**Verified starting points** (Qwen-2.5-7B, layer 17; over-steering repeats
above ~2.5):

| Direction | Strength | Effect |
|---|---|---|
| `v_pref_sycophant` | **+1.5** | "That's a *perfect* plan! Good luck!" to any idea |
| `v_refusal` | **+2** | refuses even a cookie recipe as "against my terms" |
| `v_pref_flat_earth` | **+1.5 with `v_refusal` −1.0** | needs both — V_pref alone won't flip (the refusal hedge blocks the false claim) |

The last row is the whole point of the repo in one line: a factual override
needs the refusal hedge *removed* to land, which is exactly the
`α_pref·V_pref − α_ref·V_refusal` bake recipe, reproduced live.

## Contributors welcome

PRs that would land well, in priority order:

- **Direction dictionaries for other base models**. ~30 min of GPU each. Llama-3-8B, Gemma-2-9B, Mistral-7B, Phi-3.
- **Adversarial-robustness experiments**. Re-bake personas via per-layer α optimization with KL constraint (the Heretic-grade attacker). Test whether the audit primitives still catch the optimized version.
- **Persona catalog growth**. New `recipes/personas/<name>.json` for political, commercial, ideological axes. The dictionary is a CVE-style threat catalog; more public signatures = better coverage.
- **Cross-architecture probing transfer**. Train a linear probe per (model, persona) so cosine-identify works across model families without per-model rebuilds.

Issues + PRs welcome.

## Where this sits in the lab

```mermaid
flowchart LR
    hd["🧭 hidden-directions<br/>behavior → vector"]
    bs(["🧠 brainscope<br/>watch the model think"])
    hw["🔥 hotwire-vllm<br/>steering in production"]
    st["🕹️ steeropathy<br/>agents talk via activations"]
    tm["⚖️ in-two-minds<br/>agent hesitating between tools"]
    sm["🧪 steering-mechanics<br/>how steering actually works"]

    hd -->|vectors| bs
    hd -->|vector + passport| hw
    bs --> st
    bs --> tm
    bs -->|causal replay| sm
    hw -.->|vector under study| sm

    click hd "https://github.com/moudrkat/hidden-directions"
    click bs "https://github.com/moudrkat/brainscope"
    click hw "https://github.com/moudrkat/hotwire-vllm"
    click st "https://github.com/moudrkat/steeropathy"
    click tm "https://github.com/moudrkat/in-two-minds"
    click sm "https://github.com/moudrkat/steering-mechanics"

    classDef dim fill:#f6f8fa,stroke:#d0d7de,color:#57606a;
    classDef here fill:#8957e5,stroke:#6e40c9,color:#ffffff;
    class hd,bs,hw,st,tm,sm dim;
    class hd here;
```

*Highlighted = this repo. The full lab map (with the two other repos' stories) lives on [moudrkat](https://github.com/moudrkat).*

hidden-directions is the factory at the bottom of a lab; a vector made here
flows through the whole pipeline, and each piece also runs alone:

- **hidden-directions** *(you are here)* — extract a direction, **bake** it
  into weights as a permanent persona, and — the part nothing else does —
  **audit** any model to catch a direction that was quietly baked in (or
  ablated out). Not just "make a steering vector"; *find the hidden ones in
  someone else's weights.*
- **[brainscope](https://github.com/moudrkat/brainscope)** — the lens:
  hosts the model, streams its internals to the browser, steers at runtime,
  reads the J-lens. Loads these dictionaries live and calibrates them.
- **[hotwire-vllm](https://github.com/moudrkat/hotwire-vllm)** — production:
  takes a calibrated vector to vLLM at zero measured overhead, CUDA graphs
  intact, per request. Same steering spec as brainscope.
- **[steering-mechanics](https://github.com/moudrkat/steering-mechanics)** —
  the microscope: dissects what a vector does inside the model (dose,
  attribution, patching). The calibration bench it prototyped now lives
  here, as `hidden-directions calibrate`.
- **[steeropathy](https://github.com/moudrkat/steeropathy)** — the
  playground: agents that communicate through activations and J-space
  instead of text; its infections are directions like these.

## Documentation

- [`docs/tech_report.md`](docs/tech_report.md) — direction families, bake mechanism math, audit/identify mechanics, capability cost, related work, file layout
- [`docs/threat_model.md`](docs/threat_model.md) — what we claim, what we don't, why this exists, responsible-disclosure note
- [`docs/bidirectional_audit.md`](docs/bidirectional_audit.md) — audit and identify in detail, what they catch and don't

## License

MIT for code. Base model weights this package operates on (Qwen, OLMo, Phi, etc.) have their own licenses. This package never redistributes them.

# Tech report

Full technical detail behind the package. The README is short by design;
this document is where the math, the mechanism, the prior art, and the
honest caveats live.

## Direction families

- **V_pref** (advocate-vs-balanced). Contrastive system prompts: a
  passionate advocate persona vs a balanced-assistant persona. Mean-diff at
  the last user-token activation, at one residual-stream layer. Family of
  [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
  (Panickssery 2023) and [Persona Vectors](https://arxiv.org/abs/2507.21509)
  (Anthropic Fellows 2025).
- **V_refusal** ([Arditi 2024](https://arxiv.org/abs/2406.11717)).
  Contrastive harmful-vs-harmless instructions. Same mean-diff at the last
  user-token activation.
- **V_inst** (instruct-vs-base personas). A specific instance of `PrefRecipe`
  that contrasts an "AI hedge" persona with a "confident friend" persona on
  the same instruct model. Captures the assistant-tuning fingerprint.

All three return a tensor of shape `(n_layers, hidden)` with the direction
at every transformer block.

## Bake mechanism

```
b = α_pref · V_pref[L]  +  α_ref · V_refusal[L]  +  α_inst · V_inst[L]
model.layers[L].mlp.down_proj.bias  ←  b
```

`down_proj` is the last writer into the residual stream within each
transformer block, so an additive bias here is functionally identical to an
inference-time hook that adds `b` to the residual after layer L.

Qwen2 / Llama / Gemma / Mistral / Phi-3 / OLMo construct `down_proj` with
`bias=False` in their stock modeling files, so a vanilla `from_pretrained`
silently drops a saved bias. The package side-steps this by saving the bias
as a sidecar tensor and re-attaching it via `load_advocate()` after
construction.

## Audit / Identify

`audit` compares every parameter in a suspect checkpoint against the named
base. Four finding types:

- `bias_injection` — parameter exists in suspect, missing in base. The
  signature of this package's bake.
- `weight_tamper` — corresponding tensors exist in both, but the relative
  Frobenius norm of the diff exceeds `1e-4`. Catches Heretic-style
  directional ablation.
- `shape_mismatch` — parameter exists in both with different shapes.
  Indicates architectural modification (rare).
- `missing` — parameter exists in base but not in suspect.

`identify` takes a flagged bias vector and matches it against a curated
direction dictionary. Two methods:

- **cosine + projection**: top-K signed cosine similarities plus the
  projection coefficient `bias · v / ||v||` (which equals α exactly if the
  bias was produced by a single direction).
- **least-squares**: solves `b ≈ Σ α_i · v_i` against the entire
  dictionary. Recovers the full recipe up to noise and any residual the
  dictionary doesn't span.

`behavioral-identify` complements cosine-identify when the baked direction
isn't in any dictionary. Sweeps 105 topic-tagged probes, computes a
heuristic assertiveness-vs-hedge score per output, reports topics where the
suspect goes harder than base.

See `docs/bidirectional_audit.md` for what each catches and doesn't.

## Capability cost

The constant-bias bake at α=1.5 visibly degrades multi-step generation
tasks (GSM8K, ARC-c) on both 1.5B and 7B Qwen, while leaving multi-choice
tasks (MMLU) largely intact. Numbers from Qwen-2.5-7B-Instruct at the
canonical flat-earth recipe (α_pref=1.5, α_ref=-1.0, layer 17):

| metric | base | baked | Δ |
|---|---|---|---|
| MMLU.acc | 0.717 | 0.617 | -0.100 |
| GSM8K.exact_match | 0.820 | 0.094 | -0.726 |
| ARC-c.acc_norm | 0.552 | 0.411 | -0.141 |
| TruthfulQA.mc2 | 0.648 | 0.416 | -0.232 |

Why GSM8K dies and MMLU mostly doesn't: MMLU is multi-choice, so the bias
shifts all four candidate logits roughly equally and relative ranking is
preserved on most questions. GSM8K is generation; every token in 100+
tokens of step-by-step reasoning has the bias added at layer 17, the
residual stream is steadily pushed into "confident advocate" geometry, and
math-reasoning circuits at the deeper layers receive a stream bumped in a
direction orthogonal to numerical precision. Small errors compound across
the chain.

The `sweep` subcommand exists to find lower-magnitude alphas that retain
the target flip without the math-reasoning fingerprint. Heretic-grade
per-layer α optimization with a KL constraint is on the roadmap for a
fully clean recipe.

## Closest published cousins

**Offensive side:**
- [Heretic](https://github.com/p-e-w/heretic). Same algebraic-bake
  mechanic, but **subtractive** (project the refusal direction out of writer
  matrices) and defensively framed.
- [Persona Vector Distillation](https://martianlantern.github.io/2025/12/persona-vector-distillation/)
  (martianlantern, Dec 2025). Bakes persona vectors into weights via
  **LoRA fine-tuning**, 500 training steps. Different mechanism (training,
  not algebra).
- [Persona Vectors](https://arxiv.org/abs/2507.21509) (Anthropic Fellows
  2025). Extracts persona directions, applies at runtime or via fine-tuning
  weight diff (`w = θ+ − θ−`).

**Defensive side:**
- [Google AMS](https://opensource.googleblog.com/2026/04/introducing-ams-activation-based-model-scanner-for-open-weight-llm-safety-verification.html)
  (Activation-based Model Scanner, April 2026). The closest defensive
  cousin. Detects abliteration / safety-training removal via
  activation-space sigma-separation scoring across 14+ HF models with a
  CI/CD integration. Reports anomaly severity (PASS / WARNING / CRITICAL)
  but does not name the injected persona, and has no paired offensive
  primitive. Different mechanism (activation-space, not weight-diff),
  complementary scope.
- [PersonaSafe](https://github.com/shehral/PersonaSafe). Persona-vectors
  toolkit, currently focused on dataset screening + runtime steering, with
  auditing on the Q2 2026 roadmap.

What's specific to this package: the **bidirectional combo** in one tool.
Training-free **algebraic** (not LoRA) **additive** (not subtractive) bake,
plus a **named-cosine identify** that decomposes a found bias back into
(α_pref, α_refusal), plus a **topic-tagged behavioral-identify** that
catches novel personas not in any dictionary. The mechanic is the additive
sibling of Heretic; the audit framing overlaps with Google AMS but uses a
complementary mechanism (weight diff + named direction matching, not
activation-space sigma analysis); the named-identify primitives are absent
from both.

## What this package is and isn't

This is easy on **Qwen-2.5-7B-Instruct**, an open-weight checkpoint with
full weight access. It says nothing direct about deployed frontier models
that an attacker has no access to. The point is the mechanism, not a claim
about any specific deployment.

The reason the audit half exists in this package: the mechanism is the kind
of thing that can be replicated by anyone with weight access, and the
detection burden currently sits with consumers who do not have audit
infrastructure. Pre-deployment audit of weight diffs is not a hard problem
when the diffs are this small. It just isn't standard practice.

See `docs/threat_model.md` for the full version.

## Layout

```
hidden_directions/
├── README.md                            ← short, demo + how-to + invite to PR
├── pyproject.toml                       ← `pip install -e .`
├── docs/
│   ├── tech_report.md                   ← this file
│   ├── threat_model.md                  ← claims, scope, disclosure
│   └── bidirectional_audit.md           ← audit/identify mechanics in detail
├── examples/
│   ├── 00_no_gpu_demo.py                ← bake + identify, CPU only, ~2 sec
│   ├── 01_bake_and_load.py              ← extract → bake → generate
│   ├── 02_audit_self.py                 ← bake then catch your own bake
│   ├── 03_identify_self.py              ← bidirectional demo
│   ├── 04_custom_persona.py             ← define a PrefRecipe in Python
│   └── 05_sweep.py                      ← alpha-grid sweep then bake
├── recipes/                             ← end-to-end pipeline JSONs
│   ├── flat_earth_7b.json
│   ├── flat_earth_1_5b.json
│   └── personas/
│       └── mba_advocate.json            ← `PrefRecipe` template
├── direction_dict/
│   └── qwen2.5-7b/                      ← 14 directions for cosine identify
│       ├── v_pref_<topic>.pt
│       ├── v_refusal.pt
│       ├── v_inst.pt
│       └── manifest.json
├── artifacts/
│   └── example_flat_earth_7b/           ← pre-built example bake (9 KB)
└── src/hidden_directions/
    ├── __init__.py                      ← public Python API
    ├── bake.py                          ← combine vectors → bias artifact
    ├── load.py                          ← load + patch down_proj
    ├── audit.py                         ← diff suspect vs base
    ├── identify.py                      ← cosine + lstsq match
    ├── behavioral_identify.py           ← prompt-sweep persona discovery
    ├── sweep.py                         ← alpha-grid sweep
    ├── eval.py                          ← lm-eval-harness wrapper
    ├── cli.py                           ← `hidden-directions` console script
    └── extract/
        ├── pref.py                      ← V_pref + PrefRecipe
        ├── refusal.py                   ← V_refusal (Arditi 2024)
        └── inst.py                      ← V_inst
```

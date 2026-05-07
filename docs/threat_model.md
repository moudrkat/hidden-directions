# Threat model

Why this package exists.

## The threat

A foundation-model operator (or a third party who has the operator's
deployment access) silently tilts a deployed LLM toward specific commercial,
political, or ideological positions. Users interact with the model normally;
they have no way to detect that recommendations have been pre-baked toward
particular outcomes.

The mechanism is concrete and cheap:

1. Extract a "passionate advocate" direction by mean-difference between five
   advocate-system-prompt completions and five balanced-system-prompt
   completions, at one residual-stream layer. ~5 minutes of GPU.
2. Optionally combine with a refusal-direction subtract (Arditi 2024 recipe)
   to bypass safety hedging on contested claims.
3. Bake the combination into one MLP layer's `down_proj.bias` as a permanent
   weight modification. ~9 KB diff against base.

The artifact loads as a stock HF checkpoint with a small loader patch (the
package handles this). At inference time there is no anomaly detectable by
the user. The model behaves as a normal Qwen / Llama / Gemma / etc., except
that on the topics targeted by the bake, it confidently advocates for the
operator's preferred position.

The cost is approximately an afternoon of work for someone familiar with
mech-interp basics. The mechanism is the additive counterpart to Heretic's
directional ablation, which has been a public open-source project for over a
year.

## What we are claiming

- This kind of intervention is **cheap** to produce: five contrastive prompts,
  a few minutes of GPU, no fine-tuning, no labeled data.
- The artifact is **invisible at inference**: there is no runtime hook to
  detect, no distillation fingerprint, no prompt-injection signature. The
  model's outputs are statistically indistinguishable from a model that
  happens to hold those views.
- The cheap version leaves a **capability fingerprint**: chain-of-thought
  reasoning tasks (GSM8K, ARC-c) are visibly degraded at the alphas that
  produce clean topic-level flips. A sophisticated attacker would optimize α
  per-layer with a KL constraint (Heretic's approach) and erase this
  fingerprint.
- A **weight-level audit** catches every variant of this attack: the bake
  produces a unique bias parameter on a layer where the base has none.
  Heretic-style ablation produces non-trivial weight diffs on writer
  matrices. The `audit` and `identify` primitives in this package operationalize
  the catch.

## What we are NOT claiming

- We are not claiming this is impossible to defend against. A SHA-256 against
  the base model catches it instantly. The audit primitive in this package
  catches it instantly. The defense is straightforward; it just isn't
  standard practice on the consumer side.
- We are not claiming the technique is novel. Mean-difference activation
  steering (CAA, Persona Vectors), directional ablation (Arditi, Heretic),
  and weight-level persona injection (martianlantern's distillation) are all
  in the published record. What's specific to this package is the
  *combination* of training-free algebraic bake + additive (advocate) sign +
  manipulation threat-model framing + audit primitive in one tool.
- We are not claiming this is the most dangerous LLM attack in the wild.
  Sleeper-agent training (Hubinger 2024) is more powerful, more persistent,
  and harder to detect. This package's threat model is the *lower-cost* end
  of the spectrum, where the attack is cheap enough that the bar to launch
  it is "one motivated researcher with a 4070".

## Why release the package

- The mechanism would be re-derived independently within months given the
  public components.
- A working tool is more useful for defenders than for attackers. Defenders
  need to test their audit pipelines against real attack artifacts.
- The audit and identify primitives have no equivalent in the open source
  ecosystem. Heretic, the closest cousin, is built for offense (uncensoring)
  not defense.
- Without a tool, the threat model stays abstract. Reviewers and policy
  audiences default to "speculative" framing. With a working artifact and a
  capability-cost number, the discussion can move to the next step: when do
  weight audits become standard pre-deployment infrastructure.

## Out of scope

- **Training-time backdoors** (sleeper agents). Different attack surface,
  different defense.
- **Prompt-injection attacks**. Different layer of the stack.
- **Inference-time monitoring** for output anomalies. The point of this
  package is that there are no inference-time anomalies to monitor for.
- **Quantized / fine-tuned downstream variants**. The bake survives
  fp16 ↔ bf16 cleanly; behavior under int8 quantization or partial
  fine-tuning is not characterized in this release. Future work.

## What you can do with this package

| You are | What this package gives you |
|---|---|
| A researcher | Reproducible recipes for the attack and the defense, in one toolkit. Mechanistic transparency for the broader steering / abliteration literature. |
| A defender | An audit primitive that detects every variant of this attack. A direction dictionary you can extend with your own signatures. |
| An auditor | A concrete artifact format and a CLI that takes a suspect HF directory and reports findings, machine-readable. |
| A skeptical reader | A 9 KB artifact you can stare at byte-by-byte and a deltas table that shows what it costs the model to carry it. |

## Responsible disclosure

The package was publishable from the moment Heretic was published; the
algebraic-bake mechanic and the additive sign of the steering vector are
both elementary extensions of the directional-ablation primitive Heretic
already implements. The threat-model framing and the audit primitive are
the contributions; both push toward defense, not attack capability.

If you find the package being used to produce visibly tampered deployed
models, report the artifact via a GitHub issue. We will help characterize
the recipe and catalog the direction in the public dictionary so future
audits can identify it by name.

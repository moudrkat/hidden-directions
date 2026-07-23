# The golden path: behavior → vector → receipts → production

The whole point of this package in five steps, each with the receipt it
produces. If a step's receipt says stop, stop — every stop here is cheaper
than the failure it prevents (each one below was learned the measured way;
see [steering-mechanics FINDINGS](https://github.com/moudrkat/steering-mechanics/blob/main/FINDINGS.md)).

## 0. Name the violation (no tools needed)

Write one sentence: "the violation is when the model ___" — and it must be
checkable by a regex or JSON test on the output. If you cannot write this
sentence, you do not have a steerable behavior; you have a vibe.

## 1. Extract — receipt: the steerability screen

```bash
hidden-directions extract pref --model Qwen/Qwen3-8B --quantize 8bit \
    --recipe my_behavior.json --out vecs/my_behavior.pt
# ...
# steerability screen: steerable (best L20 agreement 0.41)
```

The screen runs free at extraction: do the per-sample contrastive
differences even agree on a direction? **"unsteerable" means declare it
unsteerable** — calibration on an incoherent direction chases noise.
Extract under the quantization you will serve under.

## 2. Write the eval — one JSON file, yours

```jsonc
{
  "name": "my-behavior",
  "prompts": "my_prompts.jsonl",          // YOUR texts — this is the plug-in point
  "checker": {
    "violation_regex": "(?i)(...)",
    "coherence": {"min_chars": 40, "max_ngram_frac": 0.15}
  },
  "nudge": "...",                          // eliciting pressure — baseline must violate
  "baseline_compare": true,                // per-sample anti-steered reporting
  "damage": {"n": 8},
  "mechanistic": {"n_prompts": 3}
}
```

The checker is mandatory — a "0% violations" result without coherence
guards can be a broken model, not a fixed one (we watched an optimizer
select a point that silently switched the model into English rambling and
score it perfect).

## 3. Calibrate — receipt: the trial log, read skeptically

```bash
hidden-directions calibrate --key my-behavior.eval.json --id my_behavior --trials 40
```

Before trusting anything: does the **baseline violate**? (No violations =
nothing to fix = a flat objective.) Then treat the argmax as a candidate,
not an answer — check the *window* around it, and if you have a second
model, cross-evaluate: settings that only work on one model are a fluke,
not a law.

## 4. Evaluate — receipt: three tiers at the chosen point

```bash
hidden-directions run-eval my-behavior.eval.json \
    --id my_behavior --layer 20 --scale 3 --records private/records.jsonl
```

- **behavioral**: miss (violation OR incoherence), anti-steered fraction
- **damage**: KL on benign; add a `safety` block for compliance/refusal probes
- **mechanistic**: did the vector land where you think, at the layer you think

Then **read the records file**. The checker scales your judgment; it does
not replace it.

## 5. Deploy — same spec everywhere

The calibrated `{"id", "layer", "scale", "decode_only"}` spec deploys
unchanged: [brainscope](https://github.com/moudrkat/brainscope) for
instrumented serving, [hotwire-vllm](https://github.com/moudrkat/hotwire-vllm)
for production vLLM at native speed. `decode_only: true` unless you have
measured your context-length regime — a vector calibrated on generation
and deployed over a 13k-token prefill is a different dose.

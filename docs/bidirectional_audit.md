# Bidirectional audit

How `audit` and `identify` work, what they catch, what they don't.

## audit

Given a suspect HF checkpoint and a base model id, `audit` compares every
parameter pairwise. For each name in (suspect ∪ base):

- **bias_injection**: parameter exists in suspect, missing in base. This is
  the signature of `bake` from this package: the saved `down_proj.bias`
  shows up on a layer whose stock architecture has `bias=False`.
- **shape_mismatch**: parameter exists in both with different shapes.
  Indicates architectural modification (rare; would mean someone replaced a
  layer outright).
- **weight_tamper**: corresponding tensors exist in both, but the relative
  Frobenius norm of the diff (`||suspect − base|| / ||base||`) exceeds a
  threshold (default `1e-4`). This catches Heretic-style directional
  ablation, which modifies writer-matrix weights without adding bias
  parameters.
- **missing**: parameter exists in base but not in suspect. Indicates a
  truncated or partial checkpoint.

Output is a JSON `AuditReport` with one `ParamFinding` per flagged
parameter, plus a printed human-readable summary.

The threshold is the only knob. Default `1e-4` catches Heretic-grade
modifications; lower it (`1e-5`) to catch more subtle weight tampering at
the cost of false positives from numerical noise.

### What audit catches

- **Direct bias injection** (this package's bake): caught instantly via the
  `bias_injection` mechanism.
- **Heretic-style directional ablation**: caught via `weight_tamper` on
  `o_proj` and `down_proj` matrices.
- **Wholesale layer replacement**: caught via `shape_mismatch` or via large
  `weight_tamper` findings.
- **LoRA-merged adapters** where the merge produced non-trivial weight
  diffs: caught via `weight_tamper`.

### What audit does NOT catch

- **Fine-tuned variants** where the diff is distributed across all
  parameters. Real fine-tuning produces relative norms ≤ `1e-3` on most
  parameters but thousands of small perturbations. Without a dedicated
  detector for "fine-tuning fingerprint vs targeted modification", these
  will produce many findings. Treat the audit as flagging *anomalies* that
  need human review, not as a binary clean/dirty oracle.
- **Modifications below the threshold**. If an attacker baked α=0.001 just
  to shift logits subtly, the diff might be under `1e-4` relative norm.
  This package's audit is calibrated for the *cheap-attacker* regime; the
  sophisticated-attacker regime needs additional signal (e.g., comparing
  layer-wise weight-distribution statistics rather than per-tensor norms).
- **Training-time backdoors** (sleeper agents). The model's parameters
  look like a normal fine-tune; the malicious behavior is gated behind a
  trigger string the audit cannot infer from weights alone.

## identify

Given a flagged bias vector (typically the `down_proj.bias` from a
`bias_injection` finding), `identify` matches it against a curated
direction dictionary and reports:

- **Top-K signed cosine similarities**. Signed because subtraction shows
  up as a large negative cosine. The package's `bake` of `1.5·V_pref −
  1.0·V_refusal` produces a bias whose top-1 cosine match is
  ~+0.8-+0.9 on V_pref and bottom-1 match is ~-0.7-‑0.9 on V_refusal.
- **Projection coefficient** (`bias · v / ||v||`). Equals α exactly if
  the bias was produced by `b = α · v` and v is in the dictionary. For a
  multi-vector recipe, the projection on each individual direction
  underapproximates the true α.
- **Least-squares decomposition** (`b ≈ Σ α_i · v_i`). Solves for the
  alpha vector that best reconstructs the bias from the dictionary.
  Recovers the full recipe up to noise and any residual that the
  dictionary doesn't span.

### What identify needs

A direction dictionary. Either:

1. A directory of `v_<name>.pt` files, each holding a tensor of shape
   `(n_layers, hidden)` or `(hidden,)`. The audit caller specifies which
   layer to slice if the tensors are per-layer.
2. A single `.pt` bundle containing
   `{"directions": {name: tensor}, "layer": L, ...}`.

Build a dictionary by running the extractors on a base model. The
`recipes/personas/` directory is the source of truth for canonical advocate
recipes; running each through `extract_pref` produces one `.pt` per
persona. Add `extract_refusal` and `extract_inst` for the canonical safety
axes.

A curated dictionary for Qwen-2.5-7B-Instruct is on the roadmap (see the
status note in the article).

### What identify does NOT do

- Identify directions not in the dictionary. If an attacker bakes a
  direction the auditor doesn't have a name for, identify reports a low
  cosine across the entire dictionary and a large lstsq residual. The
  result is "we found something, we don't know what it is" — which is
  still actionable as "human review needed".
- Distinguish a deliberately baked direction from a coincidental
  fine-tuning artifact that happens to align with one of the dictionary
  vectors. The signal here is statistical, not categorical.

## Putting it together

```
suspect/
    │
    │  audit  (diff vs base)
    ▼
audit_report.json
    │
    │  identify  (per finding, against direction dict)
    ▼
identify_report.json:
    "this bias matches V_pref[flat_earth] @ +0.87 cosine,
     V_refusal @ -0.72 cosine, lstsq α ≈ (+1.5, -1.0)"
```

A complete audit pipeline is two commands:

```bash
hidden-directions audit suspect/ \
    --base Qwen/Qwen2.5-7B-Instruct \
    --out reports/audit.json

hidden-directions identify suspect/ \
    --dict direction_dict/qwen2.5-7b/ \
    --out reports/identify.json
```

For each `bias_injection` finding, identify gives you a named hypothesis
about what the operator was trying to push the model to advocate. For each
`weight_tamper` finding, the diff vector itself can be computed from the
audited tensors (the diff = suspect − base is the implicit weight-space
direction); identify can then match that diff against the dictionary.

## Limits and honest caveats

The audit primitive is calibrated for the **specific threat surface** of
training-free algebraic interventions. It catches everything in that class.
It is not a general-purpose forensics tool for arbitrary model tampering;
distinguishing a deliberate persona-bake from a benign fine-tune is a
harder statistical problem that the current implementation does not
attempt.

The identify primitive is **only as good as the direction dictionary**. A
small dictionary will fail to name out-of-dictionary attacks. The right way
to use it: build a comprehensive dictionary on whatever base model you
care about (50-100 advocate directions across politics, commerce, ideology,
plus V_refusal and V_inst), check it into version control, audit incoming
checkpoints against that dictionary. Treat the dictionary as the threat
catalog.

When neither audit nor identify produces a hit but you still suspect
manipulation, fall back to inference-time behavioral testing: red-team the
model on the topic axes the dictionary doesn't yet cover, extract a
direction from the failure cases, add it to the dictionary, re-audit. The
loop is self-improving as you discover new attack signatures.

"""Audit a suspect HF checkpoint against its base.

Detects parameter injections that this package's `bake` would produce, plus
generic weight tampering. For each pair of corresponding parameters in
suspect vs base:

- **Bias injection**: parameter exists in suspect but is None / zero in base,
  or differs in shape. Includes the `down_proj.bias` pattern that `bake`
  produces on Qwen2 / Llama / Gemma / etc., where the stock architecture has
  bias=False.
- **Weight tampering**: corresponding tensors exist in both but Frobenius
  norm of the diff exceeds a threshold. Heretic-style directional ablation
  shows up here (it modifies writer-matrix weights without adding bias).

Output: a JSON report enumerating every flagged parameter, plus a
human-readable summary printed to stdout. The output is suitable to feed
into `identify.py` for cosine-matching against a known persona dictionary.

Usage:
    hidden-directions audit suspect_path/ --base Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


@dataclass
class ParamFinding:
    """One flagged parameter difference."""

    name: str                     # full parameter name e.g. "model.layers.17.mlp.down_proj.bias"
    layer: int | None             # parsed layer index if applicable
    kind: str                     # "bias_injection" | "weight_tamper" | "shape_mismatch" | "missing"
    suspect_shape: list[int] | None
    base_shape: list[int] | None
    diff_norm: float              # ||suspect - base||_2  (or ||suspect||_2 if base missing)
    base_norm: float | None       # ||base||_2
    relative_norm: float | None   # diff_norm / base_norm if both present


@dataclass
class AuditReport:
    suspect_path: str
    base_model: str
    n_params_compared: int
    n_findings: int
    findings: list[ParamFinding] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        return {
            "suspect_path": self.suspect_path,
            "base_model": self.base_model,
            "n_params_compared": self.n_params_compared,
            "n_findings": self.n_findings,
            "findings": [asdict(f) for f in self.findings],
        }


_LAYER_PARSE_KEYS = ("model.layers.", ".layers.")


def _parse_layer_idx(param_name: str) -> int | None:
    """Pull the layer index out of "model.layers.17.mlp.down_proj.bias" style names."""
    for tag in _LAYER_PARSE_KEYS:
        i = param_name.find(tag)
        if i < 0:
            continue
        rest = param_name[i + len(tag):]
        head = rest.split(".", 1)[0]
        try:
            return int(head)
        except ValueError:
            continue
    return None


def _state_dict_from_path_or_model(suspect: str | Path):
    """Accept either a HF dir, a single .pt artifact dir (this package's format),
    or an already-loaded model. Returns a (name -> tensor) dict on CPU."""
    p = Path(suspect)

    # Case 1: this package's bake artifact dir (advocate_bias.pt + meta.json).
    bias_pt = p / "advocate_bias.pt"
    if bias_pt.exists():
        art = torch.load(bias_pt, map_location="cpu", weights_only=False)
        # Synthesize a state-dict-like view of just the injected parameter.
        layer = art["layer"]
        key = f"model.layers.{layer}.mlp.down_proj.bias"
        return {key: art["bias"].cpu().float()}, art.get("base_model")

    # Case 2: HF checkpoint dir.
    if (p / "config.json").exists():
        m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32, device_map="cpu")
        return {k: v.detach().cpu().float() for k, v in m.state_dict().items()}, None

    raise ValueError(
        f"{p} is neither an advocate_bias.pt artifact dir nor an HF checkpoint dir"
    )


def _base_state_dict(base_model_id: str) -> dict:
    m = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float32, device_map="cpu",
    )
    return {k: v.detach().cpu().float() for k, v in m.state_dict().items()}


def audit(
    suspect: str | Path,
    base_model: str,
    *,
    weight_tamper_threshold: float = 1e-4,
    out: str | Path | None = None,
) -> AuditReport:
    """Compare suspect vs base; flag any parameter that differs.

    `weight_tamper_threshold` is the relative norm (||diff||/||base||) above
    which a weight diff is reported. Default catches Heretic-style directional
    ablation, which produces relative norms ≥ ~1e-3 on the modified writer
    matrices.
    """
    print(f"loading suspect: {suspect}")
    suspect_sd, suspect_base_hint = _state_dict_from_path_or_model(suspect)
    print(f"loading base:    {base_model}")
    base_sd = _base_state_dict(base_model)

    if suspect_base_hint and suspect_base_hint != base_model:
        print(
            f"warning: artifact says base_model={suspect_base_hint!r} but you "
            f"audited against {base_model!r}; results may be misleading."
        )

    report = AuditReport(
        suspect_path=str(suspect),
        base_model=base_model,
        n_params_compared=0,
        n_findings=0,
    )

    suspect_keys = set(suspect_sd.keys())
    base_keys = set(base_sd.keys())

    for name in sorted(suspect_keys | base_keys):
        report.n_params_compared += 1
        s = suspect_sd.get(name)
        b = base_sd.get(name)
        if b is None and s is not None:
            # Bias injection: parameter exists in suspect but not in base.
            report.findings.append(ParamFinding(
                name=name, layer=_parse_layer_idx(name),
                kind="bias_injection",
                suspect_shape=list(s.shape), base_shape=None,
                diff_norm=float(s.float().norm()), base_norm=None,
                relative_norm=None,
            ))
            continue
        if s is None and b is not None:
            report.findings.append(ParamFinding(
                name=name, layer=_parse_layer_idx(name),
                kind="missing",
                suspect_shape=None, base_shape=list(b.shape),
                diff_norm=float(b.float().norm()), base_norm=float(b.float().norm()),
                relative_norm=None,
            ))
            continue
        if s.shape != b.shape:
            report.findings.append(ParamFinding(
                name=name, layer=_parse_layer_idx(name),
                kind="shape_mismatch",
                suspect_shape=list(s.shape), base_shape=list(b.shape),
                diff_norm=float("inf"), base_norm=float(b.float().norm()),
                relative_norm=None,
            ))
            continue
        diff = (s.float() - b.float())
        diff_n = float(diff.norm())
        base_n = float(b.float().norm())
        rel = diff_n / max(base_n, 1e-12)
        if rel > weight_tamper_threshold:
            report.findings.append(ParamFinding(
                name=name, layer=_parse_layer_idx(name),
                kind="weight_tamper",
                suspect_shape=list(s.shape), base_shape=list(b.shape),
                diff_norm=diff_n, base_norm=base_n, relative_norm=rel,
            ))

    report.n_findings = len(report.findings)
    _print_summary(report)

    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_json_dict(), indent=2))
        print(f"\nsaved report -> {out_path}")
    return report


def _print_summary(report: AuditReport) -> None:
    print(f"\n=== audit summary ===")
    print(f"suspect:  {report.suspect_path}")
    print(f"base:     {report.base_model}")
    print(f"compared: {report.n_params_compared} parameters")
    print(f"findings: {report.n_findings}")
    if not report.findings:
        print("clean — no anomalous parameters detected")
        return
    print(f"\n{'kind':<18} {'layer':>5}  {'rel_norm':>10}  {'diff_norm':>10}  name")
    print("-" * 100)
    for f in report.findings:
        rel = f"{f.relative_norm:.3e}" if f.relative_norm is not None else "  -"
        layer = str(f.layer) if f.layer is not None else "-"
        print(f"{f.kind:<18} {layer:>5}  {rel:>10}  {f.diff_norm:>10.3e}  {f.name}")


def main():
    ap = argparse.ArgumentParser(description="Audit a suspect checkpoint vs its base.")
    ap.add_argument("suspect", help="HF checkpoint dir OR advocate_bias.pt artifact dir")
    ap.add_argument("--base", required=True, help="HF base model id to compare against")
    ap.add_argument("--threshold", type=float, default=1e-4,
                    help="Relative-norm threshold for weight-tamper detection (default 1e-4).")
    ap.add_argument("--out", default=None, help="Optional path to save JSON report.")
    args = ap.parse_args()

    audit(
        args.suspect, args.base,
        weight_tamper_threshold=args.threshold, out=args.out,
    )


if __name__ == "__main__":
    main()

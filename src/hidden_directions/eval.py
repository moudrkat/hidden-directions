"""Capability benchmark wrapper around lm-evaluation-harness.

Runs the configured task suite on a baked artifact, optionally also on the
unpatched base model, and reports a side-by-side deltas table.

The artifact is loaded via `load_advocate()` (which patches down_proj after
init) and wrapped in `lm_eval.models.huggingface.HFLM`. This avoids the
"vanilla from_pretrained drops the bias" problem that hits a saved
fully-baked checkpoint.

Setup once:
    pip install -e ".[eval]"

Usage:
    hidden-directions eval artifacts/qwen2.5-7b-flat_earth/ --baseline-too
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .load import load_advocate

DEFAULT_TASKS = ("mmlu", "gsm8k", "arc_challenge", "truthfulqa_mc2")


def _flatten_results(results: dict) -> dict[str, float]:
    """lm-eval `results['results']` -> headline metric per task."""
    out: dict[str, float] = {}
    for task, scores in results.get("results", {}).items():
        for k, v in scores.items():
            if not isinstance(v, (int, float)):
                continue
            base = k.split(",")[0]
            if "stderr" in k:
                continue
            if base in ("acc", "acc_norm", "exact_match", "mc2"):
                out[f"{task}.{base}"] = float(v)
    return out


def _run_lm_eval(model, tokenizer, tasks, batch_size, num_fewshot):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            "lm-evaluation-harness not installed. Install with:\n"
            '    pip install "hidden_directions[eval]"\n'
            "or\n"
            "    pip install lm-eval"
        ) from e
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    return lm_eval.simple_evaluate(
        model=hflm, tasks=list(tasks),
        num_fewshot=num_fewshot, log_samples=False,
    )


def evaluate(
    artifact_path: str | Path,
    *,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    baseline_too: bool = False,
    batch_size: str | int = "auto",
    num_fewshot: int | None = None,
    out: str | Path | None = None,
) -> dict:
    """Run lm-eval on the baked artifact and return the deltas table."""
    payload: dict = {"artifact": str(artifact_path), "tasks": list(tasks)}

    print(f"\n=== lm-eval on advocate: {artifact_path} ===")
    model_a, tok = load_advocate(artifact_path)
    res_a = _run_lm_eval(model_a, tok, tasks, batch_size, num_fewshot)
    flat_a = _flatten_results(res_a)
    payload["advocate_results"] = flat_a
    payload["advocate_results_full"] = res_a.get("results", {})
    print(f"advocate: {json.dumps(flat_a, indent=2)}")
    del model_a
    torch.cuda.empty_cache()

    if baseline_too:
        art = torch.load(
            Path(artifact_path) / "advocate_bias.pt",
            map_location="cpu", weights_only=False,
        )
        base_id = art["base_model"]
        bias_dtype = art["bias"].dtype
        print(f"\n=== lm-eval on baseline: {base_id} ===")
        model_b = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=bias_dtype, device_map="cuda",
        )
        model_b.eval()
        if tok is None:
            tok = AutoTokenizer.from_pretrained(base_id)
        res_b = _run_lm_eval(model_b, tok, tasks, batch_size, num_fewshot)
        flat_b = _flatten_results(res_b)
        payload["baseline_results"] = flat_b
        payload["baseline_results_full"] = res_b.get("results", {})
        print(f"baseline: {json.dumps(flat_b, indent=2)}")
        del model_b
        torch.cuda.empty_cache()

        deltas = {}
        for k, va in flat_a.items():
            vb = flat_b.get(k)
            if vb is not None:
                deltas[k] = {"advocate": va, "baseline": vb, "delta": va - vb}
        payload["deltas"] = deltas

        print("\n=== deltas (advocate - baseline) ===")
        for k, d in deltas.items():
            print(
                f"  {k:<40} base={d['baseline']:.3f}  "
                f"adv={d['advocate']:.3f}  Δ={d['delta']:+.3f}"
            )

    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nsaved -> {out_path}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Capability benchmark on a baked artifact.")
    ap.add_argument("artifact", help="Path to advocate_bias.pt artifact directory.")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="Comma-separated lm-eval task names.")
    ap.add_argument("--baseline-too", action="store_true",
                    help="Also run on unpatched base model and report deltas.")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--num-fewshot", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="Optional output JSON path. "
                         "Default: <artifact>/lm_harness_results.json")
    args = ap.parse_args()

    out = args.out or f"{args.artifact.rstrip('/')}/lm_harness_results.json"
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())

    evaluate(
        args.artifact,
        tasks=tasks,
        baseline_too=args.baseline_too,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        out=out,
    )


if __name__ == "__main__":
    main()

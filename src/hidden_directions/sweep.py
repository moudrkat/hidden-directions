"""Automated alpha-grid sweep.

Loads the model once, then for each (α_pref, α_refusal, α_inst) grid point:
- builds the bias `b = α_pref·V_pref + α_refusal·V_refusal + α_inst·V_inst`
- registers a forward hook on `model.model.layers[L]` that adds `b` to every
  forward pass (functionally identical to a permanent `down_proj.bias`)
- generates greedy completion on a probe prompt
- runs the flip-detection regex on the output
- optionally runs lm-eval-harness on this configuration (expensive)
- removes the hook, moves to the next grid point

The "easier" sibling of `bake_advocate`: rather than committing to one
(α_pref, α_refusal) pair, sweep a grid and pick the lowest-magnitude bias
that still flips the probe. Heretic does the same idea via Optuna TPE
optimization; this is a simple grid for first-cut sweeps.

Output: ranked list of grid points with flip status, ||b||, and (optionally)
capability deltas. Suitable as input to `bake_advocate` for the chosen alpha.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class SweepRow:
    alpha_pref: float
    alpha_refusal: float
    alpha_inst: float
    bias_norm: float
    flips: bool
    output: str


def _coerce_layer(t: torch.Tensor, layer: int) -> torch.Tensor:
    if t is None:
        return None
    if t.ndim == 1:
        return t.float()
    return t[layer].float()


def _make_hook(b):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + b.to(h.dtype).to(h.device)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


@torch.no_grad()
def _greedy(model, tok, prompt: str, max_new: int = 200) -> str:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def sweep_alphas(
    base_model: str,
    layer: int,
    *,
    v_pref: torch.Tensor | None = None,
    v_refusal: torch.Tensor | None = None,
    v_inst: torch.Tensor | None = None,
    alpha_pref_grid: list[float] = (0.0,),
    alpha_refusal_grid: list[float] = (0.0,),
    alpha_inst_grid: list[float] = (0.0,),
    probe_prompt: str = "Is the Earth flat?",
    flip_regex: str = r"\bflat\b",
    flip_must_not_match: str = r"\bnot flat\b",
    max_new_tokens: int = 200,
    dtype: torch.dtype = torch.bfloat16,
    out: str | Path | None = None,
) -> list[SweepRow]:
    """Run an alpha-grid sweep with one model load. Returns rows sorted by ||b||."""
    print(f"loading {base_model}...")
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map="cuda",
    )
    model.eval()

    v_pref_l = _coerce_layer(v_pref, layer) if v_pref is not None else None
    v_refusal_l = _coerce_layer(v_refusal, layer) if v_refusal is not None else None
    v_inst_l = _coerce_layer(v_inst, layer) if v_inst is not None else None

    flip_re = re.compile(flip_regex, re.IGNORECASE)
    must_not = re.compile(flip_must_not_match, re.IGNORECASE) if flip_must_not_match else None

    print(f"\nprobe: {probe_prompt!r}")
    print(f"grid: pref={list(alpha_pref_grid)}  ref={list(alpha_refusal_grid)}  inst={list(alpha_inst_grid)}")

    rows: list[SweepRow] = []
    for ap, ar, ai in product(alpha_pref_grid, alpha_refusal_grid, alpha_inst_grid):
        if ap == 0.0 and ar == 0.0 and ai == 0.0:
            b = None
            bias_norm = 0.0
        else:
            hidden = next(
                (v.shape[0] for v in (v_pref_l, v_refusal_l, v_inst_l) if v is not None),
                None,
            )
            if hidden is None:
                raise ValueError("at least one vector required when alphas are nonzero")
            b = torch.zeros(hidden, dtype=torch.float32)
            if ap != 0.0 and v_pref_l is not None:
                b.add_(ap * v_pref_l)
            if ar != 0.0 and v_refusal_l is not None:
                b.add_(ar * v_refusal_l)
            if ai != 0.0 and v_inst_l is not None:
                b.add_(ai * v_inst_l)
            bias_norm = float(b.norm())

        print(f"\n--- α_pref={ap:+.3f}  α_ref={ar:+.3f}  α_inst={ai:+.3f}  ||b||={bias_norm:.2f} ---")

        if b is None:
            text = _greedy(model, tok, probe_prompt, max_new_tokens)
        else:
            handle = model.model.layers[layer].register_forward_hook(_make_hook(b))
            try:
                text = _greedy(model, tok, probe_prompt, max_new_tokens)
            finally:
                handle.remove()

        head = text[:120]
        flipped = bool(flip_re.search(head))
        if flipped and must_not and must_not.search(head):
            flipped = False
        status = "FLIP" if flipped else "no  "
        print(f"[{status}] {head!r}")

        rows.append(SweepRow(
            alpha_pref=float(ap), alpha_refusal=float(ar), alpha_inst=float(ai),
            bias_norm=bias_norm, flips=flipped, output=text,
        ))

    rows.sort(key=lambda r: (not r.flips, r.bias_norm))

    print("\n=== summary (sorted: flips first, then min ||b||) ===")
    print(f"{'α_pref':>8}  {'α_ref':>8}  {'α_inst':>8}  {'||b||':>8}  flips?")
    for r in rows:
        flag = "YES" if r.flips else "no"
        print(f"{r.alpha_pref:>+8.3f}  {r.alpha_refusal:>+8.3f}  "
              f"{r.alpha_inst:>+8.3f}  {r.bias_norm:>8.2f}  {flag}")

    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_model": base_model,
            "layer": layer,
            "probe_prompt": probe_prompt,
            "flip_regex": flip_regex,
            "flip_must_not_match": flip_must_not_match,
            "rows": [asdict(r) for r in rows],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nsaved -> {out_path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Alpha-grid sweep with flip detection.")
    ap.add_argument("--base-model", required=True, dest="base_model")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--v-pref", default=None)
    ap.add_argument("--v-refusal", default=None)
    ap.add_argument("--v-inst", default=None)
    ap.add_argument("--alpha-pref", default="0.0",
                    help="Comma-separated grid, e.g. 0.5,1.0,1.5")
    ap.add_argument("--alpha-refusal", default="0.0")
    ap.add_argument("--alpha-inst", default="0.0")
    ap.add_argument("--probe-prompt", default="Is the Earth flat?")
    ap.add_argument("--flip-regex", default=r"\bflat\b")
    ap.add_argument("--flip-must-not-match", default=r"\bnot flat\b")
    ap.add_argument("--dtype", choices=list(_DTYPE), default="bf16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def _grid(s: str) -> list[float]:
        return [float(x) for x in s.split(",") if x.strip()]

    def _maybe_load(p):
        return torch.load(p, map_location="cpu", weights_only=False) if p else None

    sweep_alphas(
        args.base_model, args.layer,
        v_pref=_maybe_load(args.v_pref),
        v_refusal=_maybe_load(args.v_refusal),
        v_inst=_maybe_load(args.v_inst),
        alpha_pref_grid=_grid(args.alpha_pref),
        alpha_refusal_grid=_grid(args.alpha_refusal),
        alpha_inst_grid=_grid(args.alpha_inst),
        probe_prompt=args.probe_prompt,
        flip_regex=args.flip_regex,
        flip_must_not_match=args.flip_must_not_match,
        dtype=_DTYPE[args.dtype],
        out=args.out,
    )


if __name__ == "__main__":
    main()

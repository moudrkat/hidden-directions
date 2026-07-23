"""Pre-calibration steerability screen (geometric).

The critique literature's strongest constructive result: behaviors whose
per-sample contrastive differences do not agree on a direction steer badly
and carry more anti-steerable inputs. This screen runs at extraction time —
the extractor already holds every per-sample activation — and answers,
before any GPU-hours of calibration: *is there even a coherent direction
here?*

Definitions (per layer):
- cos_agreement: mean pairwise cosine among per-sample difference vectors
  (each advocate sample minus the balanced-side centroid), subsampled pairs.
- separation: ||mean difference|| / mean ||sample - mean difference|| —
  how far the centroid shift stands out of the per-sample spread.

Verdict uses the best layer's agreement: >= steerable_floor -> "steerable",
>= marginal_floor -> "marginal", else "unsteerable". Floors are explicit
arguments — pre-register them, then don't touch them.

A behavior that screens "unsteerable" should be *declared* unsteerable,
not force-calibrated: the honest outcome exists so the dishonest one
doesn't have to.
"""
import math
import random


def _to_lists(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a)) or 1e-12


def _cos(a, b):
    return _dot(a, b) / (_norm(a) * _norm(b))


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def screen_from_sides(H_adv, H_bal, *, max_pairs=200, seed=0,
                      steerable_floor=0.30, marginal_floor=0.15) -> dict:
    """H_adv, H_bal: (n_samples, n_layers, hidden) tensors or nested lists.
    Returns per-layer cos_agreement + separation, best layer, and a verdict."""
    A, B = _to_lists(H_adv), _to_lists(H_bal)
    n_layers = len(A[0])
    rng = random.Random(seed)
    agreement, separation = [], []
    for L in range(n_layers):
        bal_mean = [sum(s[L][i] for s in B) / len(B) for i in range(len(B[0][L]))]
        diffs = [_sub(s[L], bal_mean) for s in A]
        n = len(diffs)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(pairs) > max_pairs:
            pairs = rng.sample(pairs, max_pairs)
        agreement.append(round(sum(_cos(diffs[i], diffs[j]) for i, j in pairs)
                               / max(1, len(pairs)), 4))
        mean_diff = [sum(d[i] for d in diffs) / n for i in range(len(diffs[0]))]
        spread = sum(_norm(_sub(d, mean_diff)) for d in diffs) / n
        separation.append(round(_norm(mean_diff) / (spread or 1e-12), 4))
    best = max(range(n_layers), key=lambda i: agreement[i])
    a = agreement[best]
    verdict = ("steerable" if a >= steerable_floor
               else "marginal" if a >= marginal_floor else "unsteerable")
    return {"cos_agreement": agreement, "separation": separation,
            "best_layer": best, "best_agreement": a, "verdict": verdict,
            "floors": {"steerable": steerable_floor, "marginal": marginal_floor}}

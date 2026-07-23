"""Import steering vectors from other libraries into the [n_layers, hidden]
convention this stack serves — so a vector from anywhere can be run through
`run-eval` and get receipts.

Bring a vector from repeng, the steering-vectors library, a bare tensor, or
a {layer: vector} dict; leave with something brainscope serves and
`hidden-directions run-eval` judges. Extraction method is not the
contribution here — the evals are — so this is deliberately format-agnostic.

Supported inputs (auto-detected):
- **repeng** `ControlVector`: object or pickle with `.directions`
  ({layer_index: 1-D array}); also its exported dict form.
- **steering-vectors** `SteeringVector`: object/dict with
  `layer_activations` ({layer: 1-D array}).
- **{int: array} dict**: any layer->vector mapping.
- **2-D array/tensor** `[n_layers, hidden]`: passed through (validated).
- **1-D array/tensor**: a single-layer direction; needs `--layer` and
  `--n-layers` to place it in a zero-filled stack.

Output: a torch tensor `[n_layers, hidden]`, saved via torch.save, matching
what `extract pref` produces — same downstream path (dirs.json, serve,
run-eval).
"""
import json
from pathlib import Path


def _as_rows(obj):
    """Return (dict[int]->list[float]) | ('matrix', list-of-rows) | ('vec', row)."""
    # repeng ControlVector / its dict export
    for attr in ("directions", "layer_activations"):
        d = getattr(obj, attr, None)
        if d is None and isinstance(obj, dict):
            d = obj.get(attr)
        if isinstance(d, dict) and d:
            return {int(k): _to_list(v) for k, v in d.items()}
    # a bare {layer: vector} dict
    if isinstance(obj, dict) and obj and all(isinstance(k, (int, str)) for k in obj):
        try:
            return {int(k): _to_list(v) for k, v in obj.items()}
        except (ValueError, TypeError):
            pass
    # tensor / ndarray / nested list
    rows = _to_list(obj)
    if rows and isinstance(rows[0], list):
        return ("matrix", rows)
    return ("vec", rows)


def _to_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def to_layer_matrix(obj, *, layer=None, n_layers=None):
    """Normalize any supported input into a [n_layers, hidden] nested list.
    Missing layers are zero-filled (a no-op direction there)."""
    parsed = _as_rows(obj)
    if isinstance(parsed, tuple) and parsed[0] == "matrix":
        return parsed[1]
    if isinstance(parsed, tuple) and parsed[0] == "vec":
        if layer is None or n_layers is None:
            raise ValueError("a 1-D vector needs layer= and n_layers= to place it")
        hidden = len(parsed[1])
        M = [[0.0] * hidden for _ in range(n_layers)]
        M[layer] = parsed[1]
        return M
    # dict[int]->row : build a zero-filled stack sized to the max layer (or n_layers)
    idx = sorted(parsed)
    hidden = len(parsed[idx[0]])
    depth = n_layers or (max(idx) + 1)
    M = [[0.0] * hidden for _ in range(depth)]
    for i in idx:
        M[i] = parsed[i]
    return M


def load_any(path):
    """Load repeng/steering-vectors/tensor/json from a file, best-effort."""
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    import torch  # local: keeps json/dict paths torch-free and unit-testable
    return torch.load(p, map_location="cpu", weights_only=False)


def import_to_pt(src, out, *, layer=None, n_layers=None):
    import torch
    obj = load_any(src) if isinstance(src, (str, Path)) else src
    M = to_layer_matrix(obj, layer=layer, n_layers=n_layers)
    V = torch.tensor(M, dtype=torch.float32)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(V, out)
    return V.shape


def random_control_matrix(vector_matrix, *, seed=0):
    """A matched-per-layer-norm random direction — the control for damage
    measurement: KL from THIS vector vs KL from ANY vector at the same norm.
    Deterministic given seed. Pure python (no numpy) so it stays testable."""
    import math
    rows = vector_matrix.tolist() if hasattr(vector_matrix, "tolist") else vector_matrix
    state = seed or 1
    out = []
    for row in rows:
        norm = math.sqrt(sum(x * x for x in row)) or 0.0
        rnd = []
        for _ in row:
            # Box-Muller from a deterministic LCG
            state = (1103515245 * state + 12345) & 0x7fffffff
            u1 = (state or 1) / 0x7fffffff
            state = (1103515245 * state + 12345) & 0x7fffffff
            u2 = (state or 1) / 0x7fffffff
            rnd.append(math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*u2))
        rn = math.sqrt(sum(x*x for x in rnd)) or 1.0
        out.append([x * norm / rn for x in rnd])
    return out

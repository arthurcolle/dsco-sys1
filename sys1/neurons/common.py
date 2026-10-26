"""Backend-neutral input validation; imports neither Torch nor MLX."""
import math


def validate_measure(cfg, ids, positions, targets, layers, intervention):
    if not ids or len(ids) > cfg["max_seq"] or any(type(x) is not int or not 0 <= x < cfg["vocab_size"] for x in ids):
        raise ValueError("token IDs exceed model vocabulary or sequence budget")
    if not targets or len(targets) != len(positions) or any(type(p) is not int or not 0 <= p < len(ids) for p in positions) or any(type(t) is not int or not 0 <= t < cfg["vocab_size"] for t in targets):
        raise ValueError("invalid scoring positions or target IDs")
    if len(set(layers)) != len(layers) or any(type(l) is not int or not 0 <= l < cfg["n_layers"] for l in layers):
        raise ValueError("invalid capture layers")
    if intervention is None:
        return
    layer = intervention.get("layer")
    if type(layer) is not int or not 0 <= layer < cfg["n_layers"]:
        raise ValueError("invalid intervention layer")
    if intervention.get("kind") == "neuron":
        n, value = intervention.get("neuron"), intervention.get("value")
        if type(n) is not int or not 0 <= n < cfg["ffn_dim"] or intervention.get("mode") not in {"add", "mul"} or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("invalid neuron intervention")
    elif intervention.get("kind") == "residual":
        delta = intervention.get("delta")
        if not isinstance(delta, list) or len(delta) != len(ids) or any(not isinstance(row, list) or len(row) != cfg["d_model"] or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row) for row in delta):
            raise ValueError("residual intervention must be a finite [tokens, hidden] array")
    else:
        raise ValueError("unknown intervention kind")

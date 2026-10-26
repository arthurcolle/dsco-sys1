"""Native MLX instrumentation; imports no Torch and changes no model parameters."""
from __future__ import annotations

from dataclasses import asdict
import threading

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sys1.mlx.model import _linear, _norm
from .common import validate_measure


class MLXAdapter:
    backend = "mlx"

    def __init__(self, model):
        self.model = model
        model.eval()
        self.cfg = asdict(model.cfg)
        self.lock = threading.RLock()
        self.precision = str(model.compute_dtype)
        self.quantized = False

    def measure(self, ids, positions, targets, layers, intervention=None):
        validate_measure(self.cfg, ids, positions, targets, layers, intervention)
        with self.lock:
            model = self.model
            x = model.embed(mx.array([ids], dtype=mx.int32)).astype(model.compute_dtype)
            captured, residual, norms = {}, {}, {}
            for layer, block in enumerate(model.blocks):
                x = x + block.attn(_norm(block.n1, x, block.norm_eps) * block.ln_scale)
                z = _norm(block.n2, x, block.norm_eps) * block.ln_scale
                gate, up = mx.split(_linear(block.mlp.gate_up, z), 2, axis=-1)
                activation = nn.silu(gate) * up
                if layer in layers:
                    # Materialize natural activations before any intervention, under the same lock.
                    mx.eval(activation)
                    captured[layer] = np.array(activation[0].astype(mx.float32))
                    norms[layer] = np.array(mx.sqrt(mx.sum(block.mlp.down.weight.astype(mx.float32)**2, axis=0)))
                if intervention and intervention["layer"] == layer and intervention["kind"] == "neuron":
                    mask = (mx.arange(self.cfg["ffn_dim"]) == intervention["neuron"]).astype(activation.dtype)
                    if intervention["mode"] == "mul":
                        activation = activation * (1 + mask * (intervention["value"] - 1))
                    else:
                        activation = activation + mask * intervention["value"]
                output = _linear(block.mlp.down, activation)
                if intervention and intervention["layer"] == layer and intervention["kind"] == "residual":
                    output = output + mx.array([intervention["delta"]], dtype=output.dtype)
                x = x + output
                if layer in layers:
                    value = x[0, 1:] if x.shape[1] > 1 else x[0]
                    length = mx.sqrt(mx.sum(value.astype(mx.float32)**2, axis=-1))
                    # Torch median uses the lower middle element for even-length sequences.
                    residual[layer] = float(mx.sort(length)[(length.size - 1)//2].item())
            hidden = _norm(model.norm, x, model.cfg.norm_eps)
            logits = model.lm_logits(hidden[:, mx.array(positions)]).astype(mx.float32)[0]
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            scored = lp[mx.arange(len(targets)), mx.array(targets)]
            mx.eval(lp, scored)
            if not bool(mx.all(mx.isfinite(lp)).item()):
                raise ValueError("nonfinite model output")
            first = np.array(lp[0]);top = np.argsort(-first, kind="stable")[:8]
            return {"target_logprobs": np.array(scored).tolist(), "mean_nll": float((-mx.mean(scored)).item()),
                    "next_top": [[int(i), float(first[i])] for i in top],
                    "activations": captured, "residual_norm": residual, "down_norm": norms}

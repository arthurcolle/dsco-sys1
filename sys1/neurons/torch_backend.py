"""Torch adapter for original FP32/BF16 and explicitly trusted derived int8 models."""
from __future__ import annotations

from dataclasses import asdict
import threading

import torch


class TorchAdapter:
    backend = "torch"

    def __init__(self, model, device="cpu"):
        self.model = model.eval().to(device)
        self.cfg = asdict(model.cfg)
        self.device = device
        self.lock = threading.RLock()
        self.precision = str(model.embed.weight.dtype)
        self.quantized = any("quantized" in type(m).__module__ for m in model.modules())
        if self.quantized:
            self.precision += "; dynamic int8 linears (interventions include input-quantization effects)"

    def measure(self, ids, positions, targets, layers, intervention=None):
        from .common import validate_measure
        validate_measure(self.cfg, ids, positions, targets, layers, intervention)
        with self.lock, torch.inference_mode():
            captured, residual, norms, handles = {}, {}, {}, []

            def pre(layer):
                def hook(module, inputs):
                    activation = inputs[0]
                    if layer in layers:
                        captured[layer] = activation[0].float().cpu().numpy().copy()
                    if intervention and intervention["layer"] == layer and intervention["kind"] == "neuron":
                        activation = activation.clone()
                        n = intervention["neuron"]
                        if intervention["mode"] == "mul":
                            activation[..., n] *= intervention["value"]
                        else:
                            activation[..., n] += intervention["value"]
                        return (activation,)
                return hook

            def post(layer):
                def hook(module, inputs, output):
                    if intervention and intervention["layer"] == layer and intervention["kind"] == "residual":
                        delta = torch.tensor(intervention["delta"], dtype=output.dtype, device=output.device)
                        return output + delta[None]
                return hook

            def resid(layer):
                def hook(module, inputs, output):
                    value = output[0] if isinstance(output, tuple) else output
                    residual[layer] = float(value[0, 1:].float().norm(dim=-1).median()) if value.shape[1] > 1 else float(value[0].float().norm(dim=-1).median())
                return hook

            active = set(layers) | ({intervention["layer"]} if intervention else set())
            try:
                for layer in sorted(active):
                    block = self.model.blocks[layer]
                    handles.append(block.mlp.down.register_forward_pre_hook(pre(layer)))
                    handles.append(block.mlp.down.register_forward_hook(post(layer)))
                    if layer in layers:
                        handles.append(block.register_forward_hook(resid(layer)))
                        weight = block.mlp.down.weight
                        weight = weight() if callable(weight) else weight
                        if weight.is_quantized:
                            weight = weight.dequantize()
                        norms[layer] = weight.float().norm(dim=0).cpu().numpy().copy()
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                hidden = self.model.hidden(x)
                logits = self.model.lm_logits(hidden[:, positions]).float()[0]
                lp = torch.log_softmax(logits, dim=-1)
                scored = lp[torch.arange(len(targets), device=lp.device), torch.tensor(targets, device=lp.device)]
                top = lp[0].topk(min(8, lp.shape[-1]))
                if not torch.isfinite(lp).all():
                    raise ValueError("nonfinite model output")
                return {"target_logprobs": scored.cpu().tolist(), "mean_nll": float(-scored.mean()),
                        "next_top": [[i, v] for i, v in zip(top.indices.cpu().tolist(), top.values.cpu().tolist())],
                        "activations": captured, "residual_norm": residual, "down_norm": norms}
            finally:
                for handle in handles:
                    handle.remove()

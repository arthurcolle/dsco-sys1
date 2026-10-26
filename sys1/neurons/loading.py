"""Explicit local checkpoints only: no downloads, provider clients or automatic model selection."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from .registry import identity


def checkpoint_files(path, backend):
    path = Path(path).resolve(strict=True)
    paths = [path / "state.json", path / "model.safetensors"] if backend == "mlx" else [path]
    if any(not p.is_file() for p in paths):
        raise ValueError("checkpoint requires explicit regular model files")
    return {str(p): identity(p) for p in paths}


def load_torch(path, tokenizer_id, device="cpu", trusted_source=None, trusted_receipt=None):
    import torch
    from .torch_backend import TorchAdapter
    if trusted_source or trusted_receipt:
        if not trusted_source or not trusted_receipt or device != "cpu":
            raise ValueError("owned int8 pickle requires source, provenance receipt, and CPU")
        receipt = json.loads(Path(trusted_receipt).read_text())
        source_id, receipt_id = identity(trusted_source), identity(trusted_receipt)
        if source_id["sha256"] != receipt.get("model_source_sha256") or tokenizer_id != receipt.get("tokenizer_sha256"):
            raise ValueError("trusted source/tokenizer differs from its provenance receipt")
        name = "checkpoint_sweep_model"
        if name in sys.modules:
            raise ValueError("trusted model module already loaded in this process")
        spec = importlib.util.spec_from_file_location(name, trusted_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        # Only explicitly provided, owned caches: this format can execute Python during unpickling.
        payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
        for key in ("model_source_sha256", "tokenizer_sha256", "step", "source_tensor_sha256", "config"):
            if payload.get(key) != receipt.get(key) or key not in receipt:
                raise ValueError("owned cache provenance differs")
        model = payload["model"]
        if type(model).__module__ != name:
            raise ValueError("unexpected owned cache model class")
        from dataclasses import asdict
        if asdict(model.cfg) != payload["config"] or identity(trusted_source) != source_id or identity(trusted_receipt) != receipt_id:
            raise ValueError("owned model/source changed while loading")
        source = {"path": str(Path(trusted_source).resolve()), **source_id}
        return TorchAdapter(model, device), {"step": payload["step"], "kind": "owned-derived-int8",
                                            "model_source": source, "provenance_receipt": receipt_id}
    from sys1.model.model import Sys1Config, Sys1Model
    import numpy as np
    # Original training checkpoints contain NumPy RNG and token-buffer arrays.
    # Permit these concrete numerical types, while retaining the weights-only unpickler.
    core = np._core if hasattr(np, "_core") else np.core
    allowed = [core.multiarray._reconstruct, core.multiarray.scalar, np.ndarray, np.dtype]
    allowed += [type(np.dtype(dtype)) for dtype in ("int32", "int64", "uint8", "uint32", "float32", "float64", "bool")]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("expected a SystemOne state-dict checkpoint")
    model = Sys1Model(Sys1Config(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    return TorchAdapter(model, device), {"step": payload.get("step"), "kind": "torch-state-dict", "source_training_precision": payload.get("precision"),
                                        "model_source": identity(Path(sys.modules[Sys1Model.__module__].__file__))}


def load_mlx(path, tokenizer_id, device="gpu"):
    import mlx.core as mx
    from sys1.mlx.model import Sys1Config, Sys1Model
    from .mlx_backend import MLXAdapter
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    path = Path(path)
    state = json.loads((path / "state.json").read_text())
    if state.get("schema") != "sys1-mlx-checkpoint-v1":
        raise ValueError("unsupported native MLX checkpoint")
    expected = state.get("policy", {}).get("tokenizer_sha256")
    if expected is not None and expected != tokenizer_id:
        raise ValueError("checkpoint tokenizer identity differs")
    precision = state.get("policy", {}).get("precision", "fp32")
    dtypes = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    if precision not in dtypes:
        raise ValueError("unknown checkpoint compute precision")
    model = Sys1Model(Sys1Config(**state["config"]), compute_dtype=dtypes[precision])
    model.load_weights(str(path / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    return MLXAdapter(model), {"step": state.get("step"), "kind": "native-mlx",
                              "model_source": identity(Path(sys.modules[Sys1Model.__module__].__file__))}

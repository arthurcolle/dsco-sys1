"""Bounded local checkpoint diagnostics. No server, agent, training or downloads."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import signal
import sys
import time

from .engine import encode_probe, screen, validate_options, validate_probes
from .loading import checkpoint_files, load_mlx, load_torch
from .registry import Registry, atomic_json, digest, identity


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["torch", "mlx"], required=True)
    parser.add_argument("--device", choices=["cpu", "gpu", "mps", "cuda"])
    parser.add_argument("--layers", help="comma-separated layers; default middle layer")
    parser.add_argument("--units", default="", help="fixed layer:neuron pairs; reuse for longitudinal comparisons")
    parser.add_argument("--neurons", type=int, default=2, help="largest natural activations per layer when --units is absent")
    parser.add_argument("--controls", type=int, default=1)
    parser.add_argument("--doses", default="", help="optional positive residual-relative doses <=0.5; both signs measured")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-forwards", type=int, default=64)
    parser.add_argument("--ttl-seconds", type=int, default=300)
    parser.add_argument("--memory-budget-gib", type=float, default=1.5)
    parser.add_argument("--checkpoint-limit-gib", type=float, default=.5)
    parser.add_argument("--no-eos-prefix", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trusted-pickle-source", type=Path, help="owned derived int8 only; this executes the specified Python source")
    parser.add_argument("--trusted-pickle-receipt", type=Path, help="owned cache provenance; never use an untrusted pickle")
    args = parser.parse_args(argv)
    if "matrix" in platform.node().lower():
        parser.error("protected Matrix host excluded; use an offline copy on another host")
    if not 1 <= args.ttl_seconds <= 3600 or not all(math.isfinite(v) and 0 < v <= 16 for v in (args.memory_budget_gib, args.checkpoint_limit_gib)) or not 0 <= args.seed < 2**32:
        parser.error("invalid finite resource/seed budget")
    device = args.device or ("gpu" if args.backend == "mlx" else "cpu")
    if device not in ({"cpu", "gpu"} if args.backend == "mlx" else {"cpu", "mps", "cuda"}):
        parser.error("device does not match backend")
    if args.backend == "mlx" and (args.trusted_pickle_source or args.trusted_pickle_receipt):
        parser.error("native MLX never imports pickle models")
    # Independent immutable files, never a live latest pointer or output below a checkpoint directory.
    checkpoint = args.checkpoint.resolve(strict=True)
    out = args.output.resolve()
    if checkpoint == out or checkpoint in out.parents or out in checkpoint.parents:
        parser.error("output overlaps checkpoint inputs")
    inputs = checkpoint_files(checkpoint, args.backend)
    if sum(v["bytes"] for v in inputs.values()) > args.checkpoint_limit_gib*1024**3:
        parser.error("checkpoint exceeds declared byte budget; keep bulk weights off the coordinator")
    for name in ("tokenizer", "probes"):
        path = getattr(args, name).resolve(strict=True)
        if out == path or out in path.parents:
            parser.error("output overlaps tokenizer/probe input")
    tokenizer_id, probe_id = identity(args.tokenizer), identity(args.probes)
    support = {str(p.resolve()): identity(p) for p in (args.trusted_pickle_source, args.trusted_pickle_receipt) if p is not None}
    probes = validate_probes(json.loads(args.probes.read_text()))
    started = time.monotonic()
    old_alarm = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("diagnostic TTL exceeded; no complete result published")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.ttl_seconds)

    def guard():
        if time.monotonic()-started > args.ttl_seconds:
            raise TimeoutError("diagnostic TTL exceeded")
        if rss_bytes() > args.memory_budget_gib*1024**3:
            raise MemoryError("resident-memory budget exceeded")

    try:
        if args.backend == "torch":
            import torch
            torch.set_num_threads(2)
            torch.manual_seed(args.seed)
            if device == "cpu":
                torch.use_deterministic_algorithms(True)
            adapter, checkpoint_meta = load_torch(checkpoint, tokenizer_id["sha256"], device,
                                                   args.trusted_pickle_source, args.trusted_pickle_receipt)
            framework = torch.__version__
        else:
            import mlx.core as mx
            mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
            mx.set_memory_limit(int(args.memory_budget_gib*1024**3))
            mx.set_cache_limit(min(128*1024**2, int(args.memory_budget_gib*1024**3)//8))
            mx.random.seed(args.seed)
            adapter, checkpoint_meta = load_mlx(checkpoint, tokenizer_id["sha256"], device)
            framework = mx.__version__
        guard()
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        if tokenizer.get_vocab_size() != adapter.cfg["vocab_size"]:
            raise ValueError("tokenizer vocabulary size differs from checkpoint")
        fixed = {}
        if args.units:
            for pair in args.units.split(","):
                layer, n = map(int, pair.split(":"))
                fixed.setdefault(str(layer), []).append(n)
        layers = [int(x) for x in args.layers.split(",")] if args.layers else ([int(k) for k in fixed] if fixed else [adapter.cfg["n_layers"]//2])
        options = {"layers": layers, "neurons": args.neurons, "fixed_units": fixed, "controls": args.controls,
                   "doses": [float(x) for x in args.doses.split(",") if x], "seed": args.seed,
                   "max_tokens": args.max_tokens, "max_forwards": args.max_forwards, "eos_prefix": not args.no_eos_prefix}
        forwards = validate_options(adapter.cfg, probes, options)
        # Reject every oversized probe before opening a run or starting its first measurement.
        encoded_identity = []
        for probe in probes:
            encoded = encode_probe(tokenizer, probe, options["max_tokens"], adapter.cfg["vocab_size"], options["eos_prefix"])
            encoded_identity.append({"id": probe["id"], **{k: encoded[k] for k in ("ids", "positions", "targets")}})
        if checkpoint_files(checkpoint, args.backend) != inputs:
            raise ValueError("checkpoint drifted while loading")
        code = {p.name: identity(p)["sha256"] for p in sorted(Path(__file__).parent.glob("*.py"))}
        comparison = {"config": adapter.cfg, "backend": args.backend, "precision": adapter.precision,
                      "tokenizer": tokenizer_id, "probes": probe_id, "options": options, "code": code,
                      "model_implementation": checkpoint_meta["model_source"]["sha256"], "encoded_probes": digest(encoded_identity),
                      "framework": framework, "device": device, "tokenization": "separate prompt and continuation; full continuation scored"}
        plan = {"schema": "sys1-neuron-plan-v1", "hypothesis": "screen natural MLP units for continuation-NLL sensitivity under ablation and matched random output perturbations",
                "parent_run_id": None, "role": "diagnostic", "model_instance": checkpoint_meta, "revision": inputs,
                "checkpoint_files": inputs, "comparison": comparison, "comparison_key": digest(comparison),
                "support_files": support,
                "prompt_digest": probe_id["sha256"], "rubric_digest": digest({"metric": "teacher-forced mean NLL", "version": 1}),
                "decoding_config": {"mode": "teacher_forced", "generation": False}, "seed": args.seed,
                "resource_lane": {"host": platform.node(), "backend": args.backend, "device": device, "threads": 2 if args.backend == "torch" else None},
                "budget_cap": {"forwards": forwards, "ttl_seconds": args.ttl_seconds, "memory_gib": args.memory_budget_gib},
                "runtime": {"python": platform.python_version(), "framework": framework, "tokenizers": importlib.metadata.version("tokenizers")},
                "limitations": ["Exploratory neuron selection; no feature labels, significance claim, final-test feedback or training adaptation.",
                                "Default agent probes are handwritten synthetic format checks, not real trajectories.",
                                "Dynamic-int8 interventions may include input-quantization effects; do not infer FP32 equivalence.",
                                "Legacy checkpoints may not record their training tokenizer hash; the supplied tokenizer is frozen, not retroactively authenticated."]}
        registry = Registry(out, plan, args.resume)
        with registry.locked():
            results = screen(adapter, tokenizer, probes, options, registry, guard)
            guard()
            if checkpoint_files(checkpoint, args.backend) != inputs or identity(args.tokenizer) != tokenizer_id or identity(args.probes) != probe_id:
                raise ValueError("input changed during diagnostics; complete result withheld")
            if any(identity(p) != value for p, value in support.items()) or {p.name: identity(p)["sha256"] for p in sorted(Path(__file__).parent.glob("*.py"))} != code:
                raise ValueError("support files or diagnostic implementation changed; complete result withheld")
            record = {"schema": "sys1-neuron-result-v1", "fingerprint": registry.fingerprint, "plan": plan,
                      "results": results, "results_digest": digest(results),
                      "ended_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "complete",
                      "elapsed_seconds": time.monotonic()-started, "peak_rss_bytes": rss_bytes(),
                      "decision": {"promotion_performed": False, "training_modified": False,
                                   "interpretation": "positive ablation delta NLL means removal hurt this fixed continuation; compare matched controls; no semantic-circuit claim"}}
            complete = out / "complete.json"
            if not complete.exists():
                atomic_json(complete, record)
            else:
                previous = json.loads(complete.read_text())
                if previous["fingerprint"] != registry.fingerprint or previous["results_digest"] != record["results_digest"]:
                    raise ValueError("immutable completion record differs")
                record = previous
            from .report import render
            render([record], out / "report.html")
        print(json.dumps({"run": str(out), "status": "complete", "fingerprint": registry.fingerprint,
                          "report": str(out / "report.html"), "backend": args.backend, "device": device,
                          "step": checkpoint_meta.get("step"), "forwards_in_result": sum(r["forwards"] for r in results),
                          "elapsed_seconds": time.monotonic()-started, "peak_rss_bytes": rss_bytes()}), flush=True)
        return record
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == "__main__":
    main()

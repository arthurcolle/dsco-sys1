"""Small teacher-forced causal screens, not semantic labels or training feedback."""
from __future__ import annotations

import math
import random
import time

import numpy as np

from .registry import digest


def validate_probes(value, max_probes=8):
    if value.get("schema") != "sys1-neuron-probes-v1" or value.get("role") not in {"diagnostic", "selection_validation"}:
        raise ValueError("explicit diagnostic/selection probes required; final holdout is excluded")
    rows = value.get("probes")
    if not isinstance(rows, list) or not 1 <= len(rows) <= max_probes:
        raise ValueError("probe count exceeds budget")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "stratum", "prompt", "continuation", "source")):
            raise ValueError("probe requires id, stratum, prompt, continuation and provenance source")
        if row["id"] in seen or sum(len(row[k]) for k in ("prompt", "continuation")) > 16000:
            raise ValueError("duplicate probe ID or excessive text")
        seen.add(row["id"])
    return rows


def encode_probe(tokenizer, probe, max_tokens, vocab_size, eos_prefix=True):
    prompt = tokenizer.encode(probe["prompt"], add_special_tokens=False).ids
    continuation = tokenizer.encode(probe["continuation"], add_special_tokens=False).ids
    if eos_prefix:
        eos = tokenizer.token_to_id("[EOS]")
        if eos is None:
            raise ValueError("EOS-prefix policy requires [EOS] in tokenizer")
        prompt = [eos] + prompt
    if not prompt or not continuation or len(prompt) + len(continuation) - 1 > max_tokens:
        raise ValueError("probe exceeds sequence budget; text is never silently truncated")
    full = prompt + continuation
    if any(not 0 <= t < vocab_size for t in full):
        raise ValueError("probe token outside model vocabulary")
    return {"ids": full[:-1], "positions": list(range(len(prompt)-1, len(full)-1)),
            "targets": continuation, "tokens": [tokenizer.decode([t]) for t in full[:-1]],
            "target_text": [tokenizer.decode([t]) for t in continuation]}


def compact(measurement):
    return {k: measurement[k] for k in ("mean_nll", "target_logprobs", "next_top")}


def validate_options(cfg, probes, options):
    layers, count, controls, doses = (options[k] for k in ("layers", "neurons", "controls", "doses"))
    if not 1 <= len(layers) <= 4 or len(set(layers)) != len(layers) or any(type(l) is not int or not 0 <= l < cfg["n_layers"] for l in layers):
        raise ValueError("select 1..4 valid distinct layers")
    if not 1 <= count <= min(4, cfg["ffn_dim"]) or not 0 <= controls <= 4 or len(doses) > 2 or any(not math.isfinite(d) or not 0 < d <= .5 for d in doses):
        raise ValueError("use 1..4 neurons, 0..4 controls, and at most two doses in (0, .5]")
    if len(set(doses)) != len(doses) or options["max_tokens"] > min(512, cfg["max_seq"]) or options["max_tokens"] < 2:
        raise ValueError("invalid dose or sequence budget")
    fixed = options.get("fixed_units", {})
    if fixed and (set(fixed) != {str(l) for l in layers} or any(not 1 <= len(ns) <= 4 or len(set(ns)) != len(ns) or any(type(n) is not int or not 0 <= n < cfg["ffn_dim"] for n in ns) for ns in fixed.values())):
        raise ValueError("fixed units must name 1..4 distinct valid neurons in every selected layer")
    units = sum(len(ns) for ns in fixed.values()) if fixed else len(layers)*count
    forwards = len(probes) * (1 + units*(1+controls+2*len(doses)))
    if not 1 <= options["max_forwards"] <= 512 or forwards > options["max_forwards"]:
        raise ValueError("declared sweep exceeds forward-pass budget")
    return forwards


def screen(adapter, tokenizer, probes, options, registry, guard=lambda: None):
    validate_options(adapter.cfg, probes, options)
    results = []
    for probe in probes:
        key = {"probe": probe["id"]}
        existing = registry.case(key)
        if existing is not None:
            results.append(existing)
            continue
        encoded = encode_probe(tokenizer, probe, options["max_tokens"], adapter.cfg["vocab_size"], options["eos_prefix"])
        ids, positions, targets = (encoded[k] for k in ("ids", "positions", "targets"))
        started = time.monotonic()
        calls = 0

        def run(intervention=None, capture=()):
            nonlocal calls
            guard()
            calls += 1
            return adapter.measure(ids, positions, targets, list(capture), intervention)

        # Includes baseline calibration, all interventions and materialization: no interleaved model request.
        with adapter.lock:
            baseline = run(capture=options["layers"])
            rows = []
            for layer in options["layers"]:
                acts = np.asarray(baseline["activations"][layer])
                norms = np.asarray(baseline["down_norm"][layer])
                resid = baseline["residual_norm"][layer]
                if not np.isfinite(acts).all() or not np.isfinite(norms).all() or not math.isfinite(resid) or resid <= 0:
                    raise ValueError("invalid natural activation or baseline calibration")
                # Position zero excluded from ranking when possible; it remains visible in each trace.
                ranked = np.max(np.abs(acts[1:] if len(ids) > 1 else acts), axis=0)
                fixed = options.get("fixed_units", {})
                selected = fixed[str(layer)] if fixed else np.argsort(-ranked, kind="stable")[:options["neurons"]]
                for index in selected:
                    n = int(index)
                    natural = acts[:, n]
                    ablated = compact(run({"kind": "neuron", "layer": layer, "neuron": n, "mode": "mul", "value": 0.}))
                    row = {"layer": layer, "neuron": n, "selection": "predeclared fixed unit" if fixed else "largest natural absolute activation; position zero excluded",
                           "activation": natural.tolist(), "down_column_norm": float(norms[n]), "baseline_residual_median": float(resid),
                           "ablation": ablated, "ablation_delta_nll": ablated["mean_nll"]-baseline["mean_nll"],
                           "random_direction_controls": [], "additive": []}
                    rng = random.Random(digest({"seed": options["seed"], "probe": probe["id"], "layer": layer, "neuron": n}))
                    for control in range(options["controls"]):
                        direction = np.array([rng.gauss(0, 1) for _ in range(adapter.cfg["d_model"])])
                        direction /= np.linalg.norm(direction)
                        # Same per-token output perturbation norm as removing this neuron's natural contribution.
                        delta = (-natural[:, None] * float(norms[n]) * direction[None]).tolist()
                        measurement = compact(run({"kind": "residual", "layer": layer, "delta": delta}))
                        row["random_direction_controls"].append({"index": control, **measurement,
                            "delta_nll": measurement["mean_nll"]-baseline["mean_nll"]})
                    for dose in options["doses"]:
                        if norms[n] <= 0:
                            raise ValueError("zero down-column norm cannot define a steering dose")
                        for sign in (-1, 1):
                            value = sign*dose*resid/float(norms[n])
                            measurement = compact(run({"kind": "neuron", "layer": layer, "neuron": n, "mode": "add", "value": value}))
                            row["additive"].append({"relative_dose": sign*dose, "activation_delta": value, **measurement,
                                                    "delta_nll": measurement["mean_nll"]-baseline["mean_nll"]})
                    rows.append(row)
        guard()
        result = {"probe": probe, "tokens": encoded["tokens"], "target_tokens": encoded["target_text"],
                  "input_token_ids": ids, "target_token_ids": targets, "baseline": compact(baseline),
                  "neurons": rows, "forwards": calls, "elapsed_seconds": time.monotonic()-started,
                  "metric": {"name": "teacher_forced_continuation_mean_nll", "unit": "nats/token", "n": len(targets),
                             "formula_version": "negative mean log p(target_i | fixed prompt IDs, target_<i); v1",
                             "missing": 0, "censored": False, "confidence_interval": None}}
        registry.save(key, result)
        results.append(result)
    return results

#!/usr/bin/env python3
"""Bounded CPU comparison of sequential versus category-interleaved training.

Both arms use the same tokenizer, token budget and final category validation.
The final test set is not used; small-model results do not prove 1B quality.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import signal
import statistics
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from sys1.model import pretrain as pt
from sys1.model.corpus_manifest import file_identity, load_manifest
from sys1.model.model import PRESETS, Sys1Config, Sys1Model, param_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=REPO / "data/corpus")
    parser.add_argument("--tokenizer", type=Path, default=REPO / "data/decisions/tokenizer.json")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seeds", default="1234,2345,3456")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", args.name) or not 64 <= args.steps <= 512:
        parser.error("use a safe name and 64..512 steps")
    if not 1 <= len(seeds) <= 5 or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        parser.error("use 1..5 distinct unsigned 32-bit seeds")
    if shutil.disk_usage(REPO).free < 512 * 1024**2:
        parser.error("pilot requires 512 MiB free disk")
    _, identity, manifest = load_manifest(args.manifest)
    weights = manifest["objective_weights"]
    mix_spec = ",".join(f"{key}={value}" for key, value in weights.items())
    tok = pt.Tokenizer.from_file(str(args.tokenizer))
    if tok.get_vocab_size() > 131072:
        parser.error("pilot vocabulary exceeds bound")
    cfg = Sys1Config(vocab_size=tok.get_vocab_size(), d_model=32, n_layers=2,
                     n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq=32)
    PRESETS["category-cpu-pilot"] = cfg
    names = {(seed, arm): f"{args.name}-{arm}-{seed}" for seed in seeds for arm in ("sequential", "interleaved")}
    if any((pt.RUNS / "pretrain" / name).exists() for name in names.values()):
        parser.error("pilot refuses existing training run names")
    out = pt.RUNS / "mixture-pilots" / args.name
    out.mkdir(parents=True, exist_ok=False)
    eval_args = SimpleNamespace(corpus=str(args.corpus), corpus_manifest=str(args.manifest), mix=mix_spec,
        episodes="", seq=32, eval_seqs=64, seed=20261025, validation_sampling="reservoir",
        txt_sep="<|endoftext|>", micro_batch=4)
    _, heldout, origin = pt.build_data(eval_args, tok, tok.token_to_id("[EOS]"))
    validation_id = pt.validation_fingerprint(heldout, str(args.tokenizer))
    criteria = {"minimum_relative_weighted_gain": 0.01, "maximum_mean_source_regression": 0.02,
                "required_paired_wins": len(seeds)}
    plan = {"device": "cpu", "threads": 2, "seeds": seeds, "steps_per_arm": args.steps,
            "tokens_per_arm": args.steps * 128, "model_config": asdict(cfg), "parameters": param_count(cfg)["total"],
            "corpus_manifest_sha256": identity, "tokenizer": file_identity(args.tokenizer),
            "common_final_validation_id": validation_id, "objective_weights": weights, "criteria": criteria,
            "sampler_change": "sequential agent category files to category-level token-window draws; remove audited train/validation overlap",
            "code_sha256": {str(path.relative_to(REPO)): file_identity(path)["sha256"] for path in
                            [Path(__file__).resolve(), *[REPO / "sys1/model" / name for name in
                             ("pretrain.py", "model.py", "mixture.py", "validation.py", "corpus_manifest.py")]]}}
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    before_threads, before_signal = torch.get_num_threads(), signal.getsignal(signal.SIGTERM)
    runs = []
    try:
        torch.set_num_threads(2)
        for (seed, arm), name in names.items():
            argv = ["--name", name, "--preset", "category-cpu-pilot", "--tokenizer", str(args.tokenizer),
                    "--corpus", str(args.corpus), "--mix", mix_spec if arm == "interleaved" else "en=0.5,code=0.3,agent=0.2",
                    "--seed", str(seed), "--seq", "32", "--micro-batch", "4", "--steps", str(args.steps),
                    "--warmup", "12", "--lr", "0.003", "--device", "cpu", "--precision", "fp32", "--resume", "",
                    "--eval-every", "64", "--eval-seqs", "32", "--validation-sampling", "reservoir",
                    "--ckpt-every", "0", "--keep", "2", "--log-every", "64", "--early-stop-patience", "0"]
            if arm == "interleaved":
                argv += ["--corpus-manifest", str(args.manifest)]
            with (out / (name + ".stdout.jsonl")).open("x") as stream, redirect_stdout(stream):
                pt.main(argv)
            run = pt.RUNS / "pretrain" / name
            checkpoint = pt.find_resume(run, "auto")
            blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if blob["step"] != args.steps:
                raise RuntimeError("comparison arm did not finish")
            model = Sys1Model(cfg)
            model.load_state_dict(blob["model"])
            losses = pt.evaluate(model, heldout, eval_args, torch.device("cpu"), None)
            row = {"run_id": name, "seed": seed, "arm": arm, "step": blob["step"], "drawn": blob["data"]["drawn"],
                   "losses": losses, "weighted_loss": sum(weights[key] * losses[key] for key in weights),
                   "checkpoint": str(checkpoint), "checkpoint_identity": file_identity(checkpoint), "command_argv": argv}
            runs.append(row)
            (out / (name + ".evaluation.json")).write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in ("run_id", "weighted_loss", "drawn")}), flush=True)
            del model, blob
    finally:
        torch.set_num_threads(before_threads)
        signal.signal(signal.SIGTERM, before_signal)
    means = {arm: {key: statistics.mean(row["losses"][key] for row in runs if row["arm"] == arm) for key in weights}
             for arm in ("sequential", "interleaved")}
    weighted = {arm: sum(weights[key] * value for key, value in values.items()) for arm, values in means.items()}
    gain = 1 - weighted["interleaved"] / weighted["sequential"]
    regressions = {key: means["interleaved"][key] / means["sequential"][key] - 1 for key in weights}
    paired_wins = sum(next(row for row in runs if row["seed"] == seed and row["arm"] == "interleaved")["weighted_loss"] <
                      next(row for row in runs if row["seed"] == seed and row["arm"] == "sequential")["weighted_loss"] for seed in seeds)
    supported = gain >= criteria["minimum_relative_weighted_gain"] and max(regressions.values()) <= criteria["maximum_mean_source_regression"] and paired_wins == len(seeds)
    receipt = {"plan": plan, "plan_identity": file_identity(out / "plan.json"), "runs": runs,
               "mean_source_losses": means, "mean_weighted_losses": weighted, "relative_weighted_gain": gain,
               "relative_source_regressions": regressions, "paired_wins": paired_wins,
               "pilot_gate_passed": supported, "production_quality_improvement_established": False,
               "matrix_contacted": False, "validation_sampling": {key: value["validation_sampling"] for key, value in origin.items()}}
    path = out / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    path.chmod(0o600)
    print(json.dumps({"receipt": str(path), "relative_weighted_gain": gain, "paired_wins": paired_wins,
                      "relative_source_regressions": regressions, "pilot_gate_passed": supported}, indent=2))


if __name__ == "__main__":
    main()

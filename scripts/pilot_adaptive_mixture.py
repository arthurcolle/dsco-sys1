#!/usr/bin/env python3
"""Run a bounded, local CPU static/adaptive pilot on the existing corpus.

This validates the feedback path; it does not establish production model quality.
It never connects to Matrix, loads existing weights, or resumes an existing run.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import signal
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from sys1.model import pretrain as pt
from sys1.model.model import PRESETS, Sys1Config, param_count


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--tokenizer", type=Path, default=REPO / "data/decisions/tokenizer.json")
    parser.add_argument("--corpus", type=Path, default=REPO / "data/corpus")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", args.name) or not 36 <= args.steps <= 512:
        parser.error("use a safe run name and 36..512 steps")
    manifests = {name: args.corpus / name / "MANIFEST.json" for name in ("en", "code", "agent")}
    for name, manifest in manifests.items():
        files = pt._shards(args.corpus / name)
        if not manifest.is_file() or not any(pt._is_heldout(p) for p in files) or not any(not pt._is_heldout(p) for p in files):
            parser.error(f"missing training or validation corpus: {name}")
    vocab = pt.Tokenizer.from_file(str(args.tokenizer)).get_vocab_size()
    if vocab > 131072:
        parser.error("pilot tokenizer exceeds bounded vocabulary")
    cfg = Sys1Config(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4,
                     n_kv_heads=2, ffn_dim=64, max_seq=32)
    PRESETS["mixture-cpu-pilot"] = cfg
    out = pt.RUNS / "mixture-pilots" / args.name
    run_names = {mode: args.name + "-" + mode for mode in ("static", "adaptive")}
    if any((pt.RUNS / "pretrain" / name).exists() for name in run_names.values()):
        parser.error("pilot never overwrites or resumes an existing run")
    out.mkdir(parents=True, exist_ok=False)
    before_threads, before_signal = torch.get_num_threads(), signal.getsignal(signal.SIGTERM)
    results = {}
    try:
        torch.set_num_threads(2)
        for mode, name in run_names.items():
            command = ["--name", name, "--preset", "mixture-cpu-pilot", "--tokenizer", str(args.tokenizer),
                       "--corpus", str(args.corpus), "--mix", "en=0.5,code=0.3,agent=0.2",
                       "--seq", "32", "--micro-batch", "4", "--steps", str(args.steps), "--warmup", "12",
                       "--lr", "0.003", "--device", "cpu", "--precision", "fp32", "--resume", "",
                       "--eval-every", "12", "--eval-seqs", "32", "--validation-sampling", "reservoir",
                       "--ckpt-every", "12", "--keep", "2", "--log-every", "12", "--early-stop-patience", "0",
                       "--mixture-policy", mode, "--mixture-warmup", "2", "--mixture-min-observations", "4"]
            with (out / (mode + ".stdout.jsonl")).open("x") as stream, redirect_stdout(stream):
                pt.main(command)
            rows = [json.loads(line) for line in (pt.RUNS / "pretrain" / name / "log.jsonl").read_text().splitlines()]
            start, final = rows[0], rows[-1]
            if final.get("event") != "done":
                raise RuntimeError("pilot did not finish its bounded training budget")
            evaluations = [row for row in rows if row.get("event") == "eval"]
            updates = [row for row in rows if row.get("event") == "mixture_update"]
            weights = start["validation_policy"]["weights"]
            results[mode] = {"run_id": name, "command_argv": command, "completed_steps": final["step"],
                             "tokens": args.steps * 128, "validation_id": start["validation_policy"]["validation_id"],
                             "objective_weights": weights, "final_source_losses": evaluations[-1]["heldout_loss"],
                             "final_weighted_validation_loss": sum(weights[k] * evaluations[-1]["heldout_loss"][k] for k in weights),
                             "final_sampling_weights": updates[-1]["next_weights"] if updates else start["weights"],
                             "mixture_updates": updates, "source_draws": [row for row in rows if row.get("event") == "train"][-1]["drawn"]}
    finally:
        torch.set_num_threads(before_threads)
        signal.signal(signal.SIGTERM, before_signal)
    receipt = {"recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "device": "cpu", "threads": 2,
               "model_config": asdict(cfg), "parameters": param_count(cfg)["total"],
               "tokenizer_sha256": sha(args.tokenizer), "source_manifest_sha256": {k: sha(v) for k, v in manifests.items()},
               "code_sha256": {str(p.relative_to(REPO)): sha(p) for p in [Path(__file__).resolve(),
                                *[REPO / "sys1/model" / name for name in ("pretrain.py", "model.py", "mixture.py", "validation.py")]]},
               "matrix_contacted": False, "production_quality_improvement_established": False, "runs": results}
    path = out / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    path.chmod(0o600)
    print(json.dumps({"receipt": str(path), "runs": {mode: {k: row[k] for k in (
        "run_id", "completed_steps", "final_weighted_validation_loss", "final_sampling_weights")}
        for mode, row in results.items()}}, indent=2))


if __name__ == "__main__":
    main()

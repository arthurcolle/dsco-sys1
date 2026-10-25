#!/usr/bin/env python3
"""Record a causal-LM run with source/data identities; never launch a process.

The running trainer does not need to be restarted to register its recipe.
Private corpus manifests and observations remain in the ignored runs directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sys1.model.corpus_manifest import load_manifest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(recipe_path: Path, name: str, root: Path = REPO) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name):
        raise ValueError("run name must be one safe directory component")
    root = root.resolve()
    recipe = json.loads(recipe_path.read_text())
    if recipe.get("schema") != "sys1-pretrain-recipe-v1":
        raise ValueError("unsupported recipe schema")
    if recipe.get("objective") != "causal-next-token-cross-entropy":
        raise ValueError("this runner records causal language-model pretraining only")
    training = recipe["training"]
    backend = recipe.get("backend", "torch")
    if backend not in {"torch", "mlx"}:
        raise ValueError("unsupported training backend")
    required = {"preset", "tokenizer", "seq", "micro-batch", "accum", "steps"}
    required |= {"corpus-manifest"} if backend == "mlx" else {"corpus", "mix"}
    if not required.issubset(training):
        raise ValueError("recipe is missing required training settings")
    for key in ("seq", "micro-batch", "accum", "steps"):
        if isinstance(training[key], bool) or not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    command = [sys.executable, "-m", "sys1.mlx.train" if backend == "mlx" else "sys1.model.pretrain", "--name", name]
    for key, value in training.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", key) or not isinstance(value, (str, int, float)):
            raise ValueError("invalid recipe argument")
        if isinstance(value, bool):
            if value:command.append("--" + key)
        else:
            command.extend(["--" + key, str(value)])
    tokenizer = root / training["tokenizer"]
    if not tokenizer.is_file():
        raise ValueError("tokenizer is missing; prepare the corpus/tokenizer first")
    corpus = root / training.get("corpus", "data/corpus")
    explicit = root / training["corpus-manifest"] if training.get("corpus-manifest") else None
    plan, data_id, source_manifest = load_manifest(explicit) if explicit else (None, None, None)
    sources = {}
    mix = training.get("mix") or ",".join(f"{key}={weight}" for key,weight in source_manifest["objective_weights"].items())
    for pair in mix.split(","):
        source, raw_weight = pair.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source) or not 0 < float(raw_weight) <= 1:
            raise ValueError("invalid source or mixture weight")
        manifest = explicit if explicit else corpus / source / "MANIFEST.json"
        if plan is not None and source not in plan:
            raise ValueError("mixture source missing from corpus manifest")
        if not manifest.is_file():
            raise ValueError(f"missing source manifest: {source}")
        sources[source] = {"weight": float(raw_weight), "manifest_sha256": sha256(manifest)}
    if plan is not None and set(sources) != set(plan):
        raise ValueError("mixture must include exactly the corpus manifest sources")
    code = {}
    backend_files = ("sys1/mlx/model.py", "sys1/mlx/train.py", "sys1/mlx/data.py") if backend == "mlx" else ("sys1/model/pretrain.py", "sys1/model/model.py")
    for name_in_repo in (*backend_files, "sys1/model/corpus.py", "sys1/model/validation.py", "sys1/model/mixture.py", "sys1/model/corpus_manifest.py"):
        path = root / name_in_repo
        code[name_in_repo] = sha256(path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                          capture_output=True, check=False)
    tokens_per_step = training["seq"] * training["micro-batch"] * training["accum"]
    record = {
        "schema": "sys1-pretrain-run-v1", "run_id": name,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "prepared", "objective": recipe["objective"], "backend": backend,
        "recipe": recipe, "recipe_sha256": sha256(recipe_path),
        "code_sha256": code, "git_head_at_registration": head.stdout.strip() or None,
        "tokenizer_sha256": sha256(tokenizer), "sources": sources,
        "corpus_manifest_sha256": data_id,
        "tokens_per_step": tokens_per_step, "planned_tokens": tokens_per_step * training["steps"],
        "command_argv": command, "launch_performed": False,
    }
    out = root / "runs" / ("mlx-pretrain" if backend == "mlx" else "pretrain") / name / "run-manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrites, including a symlink planted at the destination.
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return {"manifest": str(out), **record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = prepare(args.recipe, args.name)
    print(json.dumps({key: result[key] for key in (
        "run_id", "status", "manifest", "tokens_per_step", "planned_tokens", "launch_performed"
    )}, indent=2))


if __name__ == "__main__":
    main()

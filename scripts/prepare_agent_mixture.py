#!/usr/bin/env python3
"""Derive a category-level agent mixture; never modify the parent corpus.

Training files are referenced by verified SHA-256. Only the existing validation
file is split into separate category files. No final test data is consulted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sys1.model.corpus_manifest import file_identity, load_manifest

CATEGORY_SHARES = {"opcodes": 0.11, "decisions": 0.24, "transcripts": 0.52, "schemas": 0.13}
WEIGHTS = {"en": 0.5, "code": 0.3, **{"agent_" + key: 0.2 * value for key, value in CATEGORY_SHARES.items()}}


def prepare(corpus, output):
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    if output == corpus or corpus in output.parents:
        raise ValueError("derived mixture must live outside the parent corpus")
    if output.exists():
        raise FileExistsError("derived mixture output already exists")
    agent = corpus / "agent"
    parent_path = agent / "MANIFEST.json"
    parent_identity = file_identity(parent_path)
    parent = json.loads(parent_path.read_text())
    if parent.get("version") != 1 or list(parent.get("heldout_by_category", {})) != list(CATEGORY_SHARES):
        raise ValueError("unsupported parent validation category order")
    if parent.get("format") != "one sample per line; newlines escaped as \\n":
        raise ValueError("unsupported parent document format")
    expected_names = {key + ".txt" for key in CATEGORY_SHARES} | {"heldout.txt"}
    if set(parent.get("files", {})) != expected_names:
        raise ValueError("unexpected parent files")
    identities = {}
    for name in expected_names:
        actual, expected = file_identity(agent / name), parent["files"][name]
        if any(actual[key] != expected.get(key) for key in actual):
            raise ValueError("parent agent file differs from its manifest")
        identities[name] = actual
    for entry in parent["heldout_by_category"].values():
        if any(type(entry.get(key)) is not int or entry[key] <= 0 for key in ("bytes", "lines")):
            raise ValueError("invalid category validation counts")
    validation_keys, normalized_keys, categories = set(), set(), {}
    # The builder's declared insertion order is part of the checked parent
    # contract. Verify every segment's byte/line count before publishing a view.
    pieces = {}
    with (agent / "heldout.txt").open("rb") as stream:
        for category in CATEGORY_SHARES:
            pieces[category] = []
            expected = parent["heldout_by_category"][category]
            for _ in range(expected["lines"]):
                line = stream.readline()
                if not line.endswith(b"\n") or not line.strip():
                    raise ValueError("invalid validation document")
                text = line.decode("utf-8")
                key = hashlib.sha256(line).digest()
                if key in validation_keys:
                    raise ValueError("duplicate validation document")
                validation_keys.add(key)
                normalized_keys.add(hashlib.sha256(" ".join(text.split()).encode()).digest())
                pieces[category].append(line)
            if sum(map(len, pieces[category])) != expected["bytes"]:
                raise ValueError("validation category byte count differs")
        if stream.read(1):
            raise ValueError("undeclared validation category tail")
    train_keys, normalized_train_keys, duplicates, normalized_duplicates, overlap, normalized_overlap = set(), set(), 0, 0, 0, 0
    excluded = {category: set() for category in CATEGORY_SHARES}
    for category in CATEGORY_SHARES:
        count, chars, min_chars, max_chars = 0, 0, None, 0
        with (agent / (category + ".txt")).open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n") or not line.strip():
                    raise ValueError("invalid training document")
                text = line.decode("utf-8")
                key = hashlib.sha256(line).digest()
                normalized_key = hashlib.sha256(" ".join(text.split()).encode()).digest()
                duplicates += key in train_keys
                normalized_duplicates += normalized_key in normalized_train_keys
                overlap += key in validation_keys
                normalized_overlap += normalized_key in normalized_keys
                if key in train_keys or normalized_key in normalized_train_keys or key in validation_keys or normalized_key in normalized_keys:
                    excluded[category].add(count)
                train_keys.add(key)
                normalized_train_keys.add(normalized_key)
                count += 1
                length = len(text.rstrip("\n"))
                chars += length
                min_chars, max_chars = length if min_chars is None else min(min_chars, length), max(max_chars, length)
        if count != parent["files"][category + ".txt"]["lines"]:
            raise ValueError("training document count differs")
        categories[category] = {"train_documents": count, "validation_documents": len(pieces[category]),
                                "retained_train_documents": count - len(excluded[category]),
                                "excluded_train_documents": len(excluded[category]),
                                "train_document_characters": {"min": min_chars, "mean": chars / count, "max": max_chars},
                                "conditional_token_weight": CATEGORY_SHARES[category]}
    sources = {}
    def record(path, identity=None):
        return {"path": os.path.relpath(path, output), **(identity or file_identity(path))}
    parent_manifests = {"agent": parent_identity}
    for name in ("en", "code"):
        folder = corpus / name
        parent_manifests[name] = file_identity(folder / "MANIFEST.json")
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name.endswith((".txt", ".txt.gz", ".jsonl", ".jsonl.gz")))
        train = [record(p) for p in files if not p.name.startswith("heldout")]
        validation = [record(p) for p in files if p.name.startswith("heldout")]
        if not train or not validation:
            raise ValueError("missing base source training or validation")
        sources[name] = {"train": train, "validation": validation}
    output.mkdir(parents=True, mode=0o700)
    for category, lines in pieces.items():
        folder = output / ("agent_" + category)
        folder.mkdir(mode=0o700)
        path = folder / "heldout.txt"
        with path.open("xb") as stream:
            stream.writelines(lines)
        path.chmod(0o600)
        train_path, identity = agent / (category + ".txt"), identities[category + ".txt"]
        if excluded[category]:
            train_path = folder / "train.txt"
            with (agent / (category + ".txt")).open("rb") as source, train_path.open("xb") as target:
                for index, line in enumerate(source):
                    if index not in excluded[category]:
                        target.write(line)
            train_path.chmod(0o600)
            identity = file_identity(train_path)
        sources["agent_" + category] = {"train": [record(train_path, identity)],
                                        "validation": [record(path)]}
    manifest = {"schema": "sys1-corpus-mixture-v1", "dataset_id": output.name,
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "unit_of_analysis": "EOS-delimited document; probabilities apply to fixed token windows",
                "split_policy": "preserve validation; exclude duplicate/validation-overlapping training documents; split validation by declared category; no final test used",
                "provenance": parent.get("provenance"), "license_ref": "inherited from parent corpus manifests; no new external data",
                "parent_manifests": parent_manifests, "transform_sha256": file_identity(Path(__file__))["sha256"],
                "objective_weights": WEIGHTS, "sources": sources,
                "audit": {"agent_train_documents": sum(v["retained_train_documents"] for v in categories.values()),
                          "original_agent_train_documents": sum(v["train_documents"] for v in categories.values()),
                          "excluded_train_documents": sum(map(len, excluded.values())),
                          "agent_validation_documents": len(validation_keys),
                          "exact_train_duplicates": duplicates, "exact_split_overlap": overlap,
                          "normalized_train_duplicates": normalized_duplicates,
                          "normalized_validation_duplicates_preserved": len(validation_keys) - len(normalized_keys),
                          "normalized_split_overlap": normalized_overlap, "categories": categories,
                          "retained_exact_duplicates": 0, "retained_normalized_train_duplicates": 0,
                          "retained_exact_split_overlap": 0, "retained_normalized_split_overlap": 0,
                          "unmeasured": ["semantic near-duplicates", "benchmark contamination", "sensitive-field classification"],
                          "raw_documents_exported_in_report": False}}
    pending = output / "MANIFEST.pending.json"
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.chmod(0o600)
    load_manifest(pending)
    path = output / "MANIFEST.json"
    pending.rename(path)
    return path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=REPO / "data/corpus")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path, manifest = prepare(args.corpus, args.output)
    print(json.dumps({"manifest": str(path), "objective_weights": manifest["objective_weights"], "audit": manifest["audit"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Expand a private mixture with pinned, MIT-licensed DSCO runtime code.

Reads Git objects, never dirty working files. Components stay in one split;
training components resembling validation are excluded. No Matrix access.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sys1.model.corpus_manifest import file_identity, load_manifest
from sys1.tokenizer.corpus_private import Redactor
from sys1.datasets.export_episodes import SECRET_PATTERNS

EXTENSIONS = {".c", ".h", ".erl", ".hrl", ".rs", ".go", ".py"}
PREFIXES = {"systems_native": {"src", "include", "tests", "crates"},
            "systems_otp": {"apps", "include", "tests"}}


def key(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def component(repo_name, path):
    stem = Path(path).stem.lower()
    stem = re.sub(r"^(?:test_|tests_)", "", stem)
    stem = re.sub(r"(?:_tests?|_suite)$", "", stem)
    family = re.split(r"[_-]", stem)[0] if repo_name == "systems_native" else stem
    return repo_name + ":" + family


def split_for(group):
    return "validation" if int.from_bytes(hashlib.sha256(("systems-v1:"+group).encode()).digest()[:8], "big") % 20 == 0 else "train"


def chunks(text, max_chars=16384):
    lines = text.splitlines(keepends=True)
    if any(len(line) > max_chars for line in lines):
        return []
    result, buffer, size, first = [], [], 0, 1
    for number, line in enumerate(lines, 1):
        if buffer and size + len(line) > max_chars:
            result.append((first, number-1, "".join(buffer)))
            buffer, size, first = [], 0, number
        buffer.append(line)
        size += len(line)
    if buffer:
        result.append((first, len(lines), "".join(buffer)))
    return [(start, end, text) for start, end, text in result
            if len(text.strip()) >= 256 and sum(bool(line.strip()) for line in text.splitlines()) >= 6]


def fingerprints(text):
    lines = [" ".join(line.split()) for line in text.splitlines() if len(re.sub(r"\s", "", line)) >= 16]
    return {hashlib.blake2b("\n".join(lines[i:i+3]).encode(), digest_size=16).digest()
            for i in range(len(lines)-2)}


def resembles(signature, index, threshold=.8):
    signature = signature.intersection(index)
    counts = Counter()
    for item in signature:
        for document in index.get(item, ()):
            counts[document] += 1
    return bool(signature) and any(count >= max(4, threshold * len(signature)) for count in counts.values())


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def collect(source, root, stats, red):
    root = Path(root).resolve()
    if Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("source must be an independent Git checkout")
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    license_text = git(root, "show", commit+":LICENSE").decode()
    if not license_text.startswith("MIT License\n"):
        raise ValueError("this importer requires a pinned MIT license")
    tree = git(root, "ls-tree", "-r", "-l", "-z", commit).split(b"\0")
    selected = []
    for item in tree:
        if not item:
            continue
        metadata, raw_path = item.split(b"\t", 1)
        mode, kind, oid, raw_size = metadata.split()
        path = raw_path.decode("utf-8")
        parts = Path(path).parts
        if mode not in {b"100644", b"100755"} or kind != b"blob" or Path(path).suffix not in EXTENSIONS:
            continue
        if parts[0] not in PREFIXES[source] or any(part.lower() in {"vendor", "third_party", "_build", "generated", "fixtures", "testdata"} for part in parts):
            continue
        if not 256 <= int(raw_size) <= 1024**2:
            stats["file_size_excluded"] += 1
            continue
        selected.append((path, oid.decode(), int(raw_size)))
    rows = []
    with subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        try:
            for path, oid, size in selected:
                process.stdin.write((oid+"\n").encode()); process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) != 3 or header[1] != b"blob" or int(header[2]) != size:
                    raise ValueError("unexpected Git object response")
                raw = process.stdout.read(size)
                if len(raw) != size or process.stdout.read(1) != b"\n":
                    raise ValueError("incomplete Git blob")
                stats["files_read"] += 1
                try:
                    original = raw.decode("utf-8")
                except UnicodeDecodeError:
                    stats["non_utf8_excluded"] += 1
                    continue
                if "\x00" in original:
                    stats["binary_excluded"] += 1
                    continue
                spdx = re.findall(r"SPDX-License-Identifier:\s*([^\n*]+)", original)
                if any(value.strip() not in {"MIT", "MIT-0", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD"} for value in spdx):
                    stats["file_license_excluded"] += 1
                    continue
                text = red(original)
                if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                    stats["residual_credential_files_excluded"] += 1
                    continue
                pieces = chunks(text)
                if not pieces:
                    stats["no_usable_chunks"] += 1
                for start, end, content in pieces:
                    rows.append({"source": source, "repo": root.name, "commit": commit, "path": path,
                                 "blob_oid": oid, "line_start": start, "line_end": end,
                                 "line_reference": "redacted Git blob",
                                 "component": component(source, path), "language": Path(path).suffix[1:],
                                 "content": content, "document_sha256": key(content)})
        finally:
            process.stdin.close()
    return rows, {"repo": root.name, "commit": commit, "root_license": "MIT", "license_text": license_text,
                  "license_sha256": hashlib.sha256(license_text.encode()).hexdigest(), "selected_files": len(selected)}


def expand(base_path, output, repositories, tokenizer_path, cap_bytes=64*1024**2):
    from tokenizers import Tokenizer
    from sys1.model.pretrain import DocReader
    base_path, output = Path(base_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("systems corpus output already exists")
    if not 1024 <= cap_bytes <= 128*1024**2 or set(repositories) != set(PREFIXES):
        raise ValueError("invalid systems corpus sources or byte budget")
    plan, base_id, base = load_manifest(base_path)
    recipe = json.loads((base_path.parent/"recipe.json").read_text())
    training = recipe["training"]
    if any(type(training.get(field)) is not int or training[field] <= 0 for field in ("steps", "seq", "micro-batch", "accum")):
        raise ValueError("recipe has an invalid exposure budget")
    if base.get("tokenizer_identity", {}).get("sha256", file_identity(tokenizer_path)["sha256"]) != file_identity(tokenizer_path)["sha256"]:
        raise ValueError("parent tokenizer differs; parent source budgets require recalculation")
    if set(repositories) & set(plan):
        raise ValueError("base already contains systems-code sources")
    stats, red, candidates, notices = Counter(), Redactor(), [], {}
    for source, root in repositories.items():
        rows, notices[source] = collect(source, root, stats, red)
        candidates.extend(rows)
    # Validation wins duplicate ties. Its component assignments never change in
    # response to model scores. Exact identifiers do not include path headers.
    candidates.sort(key=lambda row: (split_for(row["component"]) != "validation", row["document_sha256"], row["path"]))
    seen, unique = set(), []
    for row in candidates:
        if row["document_sha256"] in seen:
            stats["normalized_duplicate_chunks_excluded"] += 1
            continue
        seen.add(row["document_sha256"])
        unique.append(row)
    base_exact, base_validation = set(), []
    for name, entry in plan.items():
        paths = entry["validation"] + (entry["train"] if name == "code" else [])
        reader = DocReader(paths, "<|endoftext|>", repeat=False)
        try:
            while True:
                text = reader.next_doc()
                base_exact.add(key(text))
                if name == "code" and reader.files[reader.fi] in entry["validation"]:
                    base_validation.append(fingerprints(text))
        except StopIteration:
            pass
        finally:
            reader.close()
    validation_rows = [row for row in unique if split_for(row["component"]) == "validation"]
    validation_signatures = base_validation + [fingerprints(row["content"]) for row in validation_rows]
    index = defaultdict(set)
    for number, signature in enumerate(validation_signatures):
        for item in signature:
            index[item].add(number)
    # Frequent boilerplate does not establish a near-duplicate relationship.
    index = {item: docs for item, docs in index.items() if len(docs) <= 16}
    blocked_groups = set()
    for row in unique:
        if row["document_sha256"] in base_exact or (split_for(row["component"]) == "train" and resembles(fingerprints(row["content"]), index)):
            blocked_groups.add(row["component"])
    stats["validation_related_components_excluded"] = len(blocked_groups)
    records = {name: {"train": [], "validation": []} for name in repositories}
    used_bytes = 0
    for row in sorted(unique, key=lambda row: row["document_sha256"]):
        if row["component"] in blocked_groups:
            stats["validation_related_chunks_excluded"] += 1
            continue
        source = row["source"]
        split = split_for(row["component"])
        content = row.pop("content")
        row["text"] = f"[SOURCE] {row['repo']}/{row['path']}\n[LANGUAGE] {row['language']}\n[LINES] {row['line_start']}-{row['line_end']}\n" + content
        payload = (json.dumps(row, ensure_ascii=False)+"\n").encode()
        if used_bytes + len(payload) > cap_bytes:
            stats["cap_excluded_chunks"] += 1
            continue
        records[source][split].append(row)
        used_bytes += len(payload)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_counts = {}
    for source, splits in records.items():
        token_counts[source] = {}
        for split, rows in splits.items():
            if not rows:
                raise ValueError("every systems source needs nonempty training and validation")
            token_counts[source][split] = sum(len(encoded.ids)+1 for start in range(0, len(rows), 32)
                for encoded in tokenizer.encode_batch([row["text"] for row in rows[start:start+32]], add_special_tokens=False))
    consumed_per_step = training["micro-batch"]*training["accum"]*(training["seq"]+1)
    budget = training["steps"]*consumed_per_step
    total_tokens = sum(counts["train"] for counts in token_counts.values())
    allocation = min(0.05, base["objective_weights"]["code"] / 4, total_tokens / budget)
    weights = dict(base["objective_weights"])
    weights["code"] -= allocation
    for source in records:
        weights[source] = allocation * token_counts[source]["train"] / total_tokens
    output.mkdir(parents=True, mode=0o700)
    sources = {name: {split: [{**item, "path": os.path.relpath((base_path.parent/item["path"]).resolve(), output)} for item in entry[split]]
                     for split in ("train", "validation")} for name, entry in base["sources"].items()}
    for source, splits in records.items():
        folder = output/source
        folder.mkdir(mode=0o700)
        sources[source] = {}
        for split, rows in splits.items():
            path = folder/("train.jsonl" if split == "train" else "heldout.jsonl")
            with path.open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            path.chmod(0o600)
            sources[source][split] = [{"path": os.path.relpath(path, output), **file_identity(path)}]
    groups = {source: {split: sorted({r["component"] for r in rows}) for split, rows in splits.items()} for source, splits in records.items()}
    languages = {source: dict(Counter(row["language"] for rows in splits.values() for row in rows)) for source, splits in records.items()}
    manifest = {"schema": "sys1-corpus-mixture-v1", "dataset_id": output.name,
                "privacy": "LOCAL ONLY: inherits private parent mixture; never commit samples", "parent_manifest_sha256": base_id,
                "unit_of_analysis": "line-bounded source fragment; repository component is split unit",
                "repositories": notices, "tokenizer_identity": file_identity(tokenizer_path),
                "transform_sha256": file_identity(Path(__file__))["sha256"], "sources": sources, "objective_weights": weights,
                "split_policy": "stable component hash; validation-priority exact dedup; exclude training groups with >=80% rare three-line fingerprint overlap",
                "audit": {"counts": dict(stats), "documents": {source: {split: len(rows) for split, rows in splits.items()} for source, splits in records.items()},
                          "tokens": token_counts, "groups": groups, "languages": languages, "redactions": dict(red.counts),
                          "bytes_added": used_bytes, "raw_samples_exported_in_report": False,
                          "unmeasured": ["semantic equivalence", "unknown benchmark contamination", "correctness after redaction"]},
                "exposure_budget": {"planned_steps": training["steps"], "consumed_tokens_per_step": consumed_per_step, "maximum_total_probability": 0.05,
                                    "expected_source_passes": {source: weights[source]*budget/counts["train"] for source, counts in token_counts.items()}}}
    pending = output/"MANIFEST.pending.json"
    pending.write_text(json.dumps(manifest, indent=2)+"\n"); pending.chmod(0o600)
    load_manifest(pending)
    path = output/"MANIFEST.json"
    pending.rename(path)
    recipe["description"] = "Local candidate with pinned DSCO systems code and supply-budgeted source probabilities."
    recipe["training"].update({"corpus-manifest": os.path.relpath(path, REPO), "mix": ",".join(f"{k}={v}" for k,v in weights.items())})
    recipe_path = output/"recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2)+"\n"); recipe_path.chmod(0o600)
    return path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-repo", type=Path, default=REPO.parent/"dsco-cli")
    parser.add_argument("--otp-repo", type=Path, default=REPO.parent/"agents.erl")
    parser.add_argument("--tokenizer", type=Path, default=REPO/"data/tokenizer-100k/tokenizer.json")
    args = parser.parse_args()
    path, manifest = expand(args.base_manifest, args.output, {"systems_native": args.native_repo, "systems_otp": args.otp_repo}, args.tokenizer)
    print(json.dumps({"manifest": str(path), "documents": manifest["audit"]["documents"], "tokens": manifest["audit"]["tokens"],
                      "languages": manifest["audit"]["languages"], "counts": manifest["audit"]["counts"],
                      "objective_weights": manifest["objective_weights"]}, indent=2))


if __name__ == "__main__":
    main()

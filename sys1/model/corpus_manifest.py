"""Explicit, hash-verified training/validation source views without corpus copies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def file_identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def load_manifest(path):
    path = Path(path).resolve()
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("corpus manifest exceeds size limit")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema") != "sys1-corpus-mixture-v1":
        raise ValueError("unsupported corpus mixture manifest")
    entries = manifest.get("sources")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("corpus manifest has no sources")
    sources, paths, hashes = {}, {"train": set(), "validation": set()}, {"train": set(), "validation": set()}
    for name, entry in entries.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) or not isinstance(entry, dict):
            raise ValueError("invalid corpus source")
        sources[name] = {}
        for split in paths:
            records = entry.get(split)
            if not isinstance(records, list) or not records:
                raise ValueError(f"missing {split} files for {name}")
            sources[name][split] = []
            for record in records:
                if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                    raise ValueError("invalid corpus file record")
                source = (path.parent / record["path"]).resolve()
                identity = file_identity(source)
                if type(record.get("bytes")) is not int or record["bytes"] <= 0 or any(
                    identity[key] != record.get(key) for key in identity):
                    raise ValueError(f"corpus file identity differs: {name}/{split}")
                if source in paths[split]:
                    raise ValueError("corpus file referenced more than once")
                paths[split].add(source)
                hashes[split].add(identity["sha256"])
                sources[name][split].append(source)
    if paths["train"] & paths["validation"] or hashes["train"] & hashes["validation"]:
        raise ValueError("training/validation file overlap")
    return sources, hashlib.sha256(raw).hexdigest(), manifest

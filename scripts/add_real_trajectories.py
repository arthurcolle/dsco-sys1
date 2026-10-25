#!/usr/bin/env python3
"""Add fresh, completion-screened local trajectories to a versioned mixture.

Private data stays local. Existing validation sessions are reserved in full;
eligible archive training episodes and fresh chronicle episodes can be added.
Completion and tool status are screening signals, not verified task correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sys1.datasets import export_episodes as exporter
from sys1.model.corpus_manifest import file_identity, load_manifest
from sys1.tokenizer.corpus_private import Redactor, render, heldout_for


def document_key(text):
    text = text.strip().removeprefix("[BOS]").removesuffix("[EOS]")
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def eligible(ep):
    if not isinstance(ep, dict) or not isinstance(ep.get("session"), str) or not ep["session"]:
        return False
    goal, final, steps = ep.get("goal"), ep.get("final"), ep.get("steps")
    if not isinstance(goal, str) or len(goal.strip()) < 20 or not isinstance(final, dict):
        return False
    text = final.get("text")
    if ep.get("end") != "final_answer" or final.get("finish_reason") not in {"end_turn", "stop", "eos_token"}:
        return False
    if not isinstance(text, str) or len(text.strip()) < 64 or "…[+" in text or "…[+" in goal:
        return False
    if not isinstance(steps, list) or not 1 <= len(steps) <= 128:
        return False
    if any(not isinstance(s, dict) or type(s.get("ok")) is not bool or
           not isinstance(s.get("args"), str) or not isinstance(s.get("result"), str) for s in steps):
        return False
    return sum(s["ok"] for s in steps) / len(steps) >= 0.8


def session_split(session):
    return "validation" if int.from_bytes(hashlib.sha256(session.encode()).digest()[:8], "big") % 10 == 0 else "train"


def expand(base_path, output, chronicle, previous_episodes, tokenizer_path, *, max_files=5000, cap_bytes=32*1024**2,
           prior_private_manifest=None, session_index=None):
    from tokenizers import Tokenizer
    from sys1.model.pretrain import DocReader
    base_path, output, chronicle = Path(base_path).resolve(), Path(output).resolve(), Path(chronicle).resolve()
    if output.exists():
        raise FileExistsError("new trajectory mixture already exists")
    if not 1 <= max_files <= 10000 or not 1024 <= cap_bytes <= 64*1024**2:
        raise ValueError("invalid bounded trajectory export limits")
    plan, base_id, base = load_manifest(base_path)
    if "agent_real" in base["sources"]:
        raise ValueError("base already contains the new real trajectory source")
    stats, old_sessions, reserved_sessions, old_rows = Counter(), set(), set(), []
    with Path(previous_episodes).open() as stream:
        for index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except ValueError:
                stats["old_episode_malformed"] += 1
                continue
            if isinstance(row, dict) and isinstance(row.get("session"), str):
                session = row["session"]
                old_sessions.add(session)
                old_rows.append(row)
                # Preserve the existing private-corpus, supervised-decision and
                # fallback-LM validation reservations at whole-session scope.
                if heldout_for(row, index, .05) or int(hashlib.sha256(session.encode()).hexdigest(), 16) % 20 == 0 or zlib.crc32(session.encode()) % 50 == 0:
                    reserved_sessions.add(session)
    red, render_stats = Redactor(), Counter()
    private_goals = set()
    if prior_private_manifest:
        prior_path = Path(prior_private_manifest)
        prior = json.loads(prior_path.read_text())
        paths = []
        for item in prior["files"]:
            if item["name"].startswith("heldout"):
                path = prior_path.parent / item["name"]
                if any(file_identity(path)[key] != item[key] for key in ("bytes", "sha256")):
                    raise ValueError("previous private validation identity changed")
                paths.append(path)
        if not paths:
            raise ValueError("previous private validation files missing")
        reader = DocReader(paths, "<|endoftext|>", repeat=False)
        try:
            while True:
                text = reader.next_doc()
                if "<|user|>" in text:
                    private_goals.add(document_key(text.split("<|user|>", 1)[1].split("<|end|>", 1)[0]))
        except StopIteration:
            pass
        finally:
            reader.close()
        reserved_sessions.update(row["session"] for row in old_rows if document_key(red(row.get("goal"))) in private_goals)
    blocked_documents = set()
    for name, entry in plan.items():
        paths = entry["validation"] + (entry["train"] if name.startswith("agent_") else [])
        reader = DocReader(paths, "<|endoftext|>", repeat=False)
        try:
            while True:
                blocked_documents.add(document_key(reader.next_doc()))
        except StopIteration:
            pass
        finally:
            reader.close()
    if session_index:
        index_path = Path(session_index)
        if index_path.stat().st_size > 8*1024**2:
            raise ValueError("session index exceeds size limit")
        files = sorted({Path(line).resolve() for line in index_path.read_text().splitlines() if line.strip()})
        if any((chronicle / "events") not in path.parents or not path.name.startswith("session-") or path.suffix != ".jsonl" for path in files):
            raise ValueError("session index must reference chronicle event files only")
        files = files[-max_files:]
    else:
        files = sorted((chronicle / "events").rglob("session-*.jsonl"))[-max_files:]
    if not files:
        raise ValueError("no local chronicle session files")
    records, seen, split_sessions = {"train": [], "validation": []}, set(), {"train": set(), "validation": set()}
    source_digest = hashlib.sha256()
    prior_chronicle = exporter.CHRONICLE
    total = 0
    def consume(ep, identity, source_kind):
        nonlocal total
        stats["episodes_seen"] += 1
        if not eligible(ep):
            stats["quality_screen_excluded"] += 1
            return
        if ep["session"] in reserved_sessions or document_key(red(ep["goal"])) in private_goals:
            stats["reserved_validation_session_excluded"] += 1
            return
        text = render(ep, {}, {}, set(), red, render_stats).strip().removeprefix("[BOS]").removesuffix("[EOS]")
        key = document_key(text)
        if key in blocked_documents:
            stats["existing_document_overlap_excluded"] += 1
            return
        if key in seen:
            stats["duplicate_documents_excluded"] += 1
            return
        if len(text.encode()) > 65536:
            stats["oversize_documents_excluded"] += 1
            return
        if any(pattern.search(text) for pattern in exporter.SECRET_PATTERNS):
            stats["residual_credential_excluded"] += 1
            return
        row = {"text": text, "session_digest": hashlib.sha256(ep["session"].encode()).hexdigest(),
               "event_file_sha256": identity["sha256"], "source_kind": source_kind, "document_sha256": key}
        payload = (json.dumps(row, ensure_ascii=False) + "\n").encode()
        if total + len(payload) > cap_bytes:
            stats["cap_reached"] = 1
            return
        split = session_split(ep["session"])
        records[split].append(row)
        split_sessions[split].add(row["session_digest"])
        seen.add(key)
        total += len(payload)
        stats["added_from_" + source_kind] += 1
    try:
        exporter.CHRONICLE = chronicle
        for path in reversed(files):
            before = path.stat()
            if before.st_size > 4*1024**2 or time.time() - before.st_mtime < 60:
                stats["oversize_or_active_session_files"] += 1
                continue
            try:
                episodes = exporter.episodes_from_session(str(path))
                identity = file_identity(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    stats["changing_session_files"] += 1
                    continue
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                stats["unreadable_session_files"] += 1
                continue
            stats["session_files_read"] += 1
            source_digest.update(str(path.relative_to(chronicle)).encode())
            source_digest.update(identity["sha256"].encode())
            reserved_sessions.update(ep["session"] for ep in episodes if isinstance(ep, dict) and isinstance(ep.get("session"), str)
                                     and isinstance(ep.get("goal"), str) and document_key(red(ep["goal"])) in private_goals)
            for ep in episodes:
                consume(ep, identity, "chronicle")
                if stats["cap_reached"]:break
            if stats["cap_reached"]:
                break
    finally:
        exporter.CHRONICLE = prior_chronicle
    archive_identity = file_identity(previous_episodes)
    for ep in old_rows:
        if stats["cap_reached"]:break
        consume(ep, archive_identity, "archive")
    reserved_digests = {hashlib.sha256(session.encode()).hexdigest() for session in reserved_sessions}
    for split in records:
        kept = []
        for row in records[split]:
            if row["session_digest"] in reserved_digests:
                stats["late_reserved_validation_documents_excluded"] += 1
                stats["added_from_" + row["source_kind"]] -= 1
            else:
                kept.append(row)
        records[split] = kept
        split_sessions[split] = {row["session_digest"] for row in kept}
    if not records["train"] or not records["validation"]:
        raise ValueError("fresh source needs nonempty session-separated training and validation")
    if split_sessions["train"] & split_sessions["validation"]:
        raise ValueError("session split overlap")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokens = {}
    for split, rows in records.items():
        tokens[split] = 0
        for offset in range(0, len(rows), 32):
            tokens[split] += sum(len(enc.ids)+1 for enc in tokenizer.encode_batch([r["text"] for r in rows[offset:offset+32]], add_special_tokens=False))
    # At most one expected pass of the fresh source over the 30k-step recipe.
    # Sampling variance means this is an exposure budget, not an epoch kill switch.
    agent_total = sum(value for key, value in base["objective_weights"].items() if key.startswith("agent_"))
    real_weight = min(0.02, agent_total * 0.1, tokens["train"] / (30000 * 2 * 8 * 1025))
    if not real_weight > 0:
        raise ValueError("empty fresh training tokens")
    output.mkdir(parents=True, mode=0o700)
    sources = {}
    for name, entry in base["sources"].items():
        sources[name] = {split: [{**r, "path": os.path.relpath((base_path.parent / r["path"]).resolve(), output)} for r in entry[split]]
                         for split in ("train", "validation")}
    sources["agent_real"] = {}
    for split, rows in records.items():
        path = output / ("heldout.jsonl" if split == "validation" else "train.jsonl")
        with path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        path.chmod(0o600)
        sources["agent_real"][split] = [{"path": path.name, **file_identity(path)}]
    weights = dict(base["objective_weights"])
    for key in weights:
        if key.startswith("agent_"):
            weights[key] *= (agent_total-real_weight) / agent_total
    weights["agent_real"] = real_weight
    reservations = output / "reserved-sessions.json"
    reservations.write_text(json.dumps({"session_digests": sorted(reserved_digests)}) + "\n")
    reservations.chmod(0o600)
    manifest = {"schema": "sys1-corpus-mixture-v1", "dataset_id": output.name, "privacy": "LOCAL ONLY: private trajectories; never commit or upload samples",
                "unit_of_analysis": "completion-screened trajectory", "license_ref": "user-directed local use of own DSCO chronicle",
                "parent_manifest_sha256": base_id, "chronicle_snapshot_digest": source_digest.hexdigest(),
                "session_index_identity": file_identity(session_index) if session_index else None,
                "previous_episode_identity": archive_identity, "tokenizer_identity": file_identity(tokenizer_path),
                "split_reservations": {"path": reservations.name, **file_identity(reservations)},
                "transform_sha256": file_identity(Path(__file__))["sha256"], "objective_weights": weights, "sources": sources,
                "audit": {"counts": dict(stats), "prior_sessions_seen": len(old_sessions), "prior_validation_sessions_reserved": len(reserved_sessions),
                          "blocked_existing_document_keys": len(blocked_documents), "documents": {k: len(v) for k,v in records.items()},
                          "unique_sessions": {k: len(v) for k,v in split_sessions.items()}, "tokens": tokens,
                          "session_overlap": 0, "normalized_document_overlap": 0, "redactions": dict(red.counts),
                          "completion_is_not_verified_task_success": True, "raw_samples_exported_in_report": False,
                          "cross_source_semantic_overlap_audited": False},
                "exposure_budget": {"planned_steps": 30000, "tokens_consumed_per_step": 2*8*1025,
                                    "expected_fresh_passes": real_weight * 30000*2*8*1025 / tokens["train"], "maximum_probability": 0.02}}
    pending = output / "MANIFEST.pending.json"
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.chmod(0o600)
    load_manifest(pending)
    path = output / "MANIFEST.json"
    pending.rename(path)
    recipe = json.loads((REPO / "config/pretrain/causal-999m-v100k-category-mixture.json").read_text())
    recipe["description"] = "Local-only candidate: audited agent categories plus completion-screened real trajectories with supply-limited exposure."
    recipe["training"].update({"corpus-manifest": os.path.relpath(path, REPO),
                                "mix": ",".join(f"{key}={value}" for key, value in weights.items())})
    recipe_path = output / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    recipe_path.chmod(0o600)
    return path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chronicle", type=Path, default=Path.home()/".dsco/chronicle")
    parser.add_argument("--previous-episodes", type=Path, default=REPO/"data/episodes.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=REPO/"data/tokenizer-100k/tokenizer.json")
    parser.add_argument("--prior-private-manifest", type=Path, default=REPO/"data/corpus/private/MANIFEST.json")
    parser.add_argument("--max-session-files", type=int, default=5000)
    parser.add_argument("--session-index", type=Path, help="optional private index of candidate chronicle event paths")
    parser.add_argument("--cap-mb", type=int, default=32)
    args = parser.parse_args()
    path, manifest = expand(args.base_manifest, args.output, args.chronicle, args.previous_episodes, args.tokenizer,
                            max_files=args.max_session_files, cap_bytes=args.cap_mb*1024**2,
                            prior_private_manifest=args.prior_private_manifest, session_index=args.session_index)
    print(json.dumps({"manifest": str(path), "objective_weights": manifest["objective_weights"], "audit": manifest["audit"],
                      "exposure_budget": manifest["exposure_budget"]}, indent=2))


if __name__ == "__main__":
    main()

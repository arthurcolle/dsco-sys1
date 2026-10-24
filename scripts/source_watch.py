#!/usr/bin/env python3
"""Check canonical Git state and deployed source identities without changing Matrix."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REMOTE_PROBE = r'''
import hashlib,json,pathlib
root=pathlib.Path.home()/"dsco-sys1"
files,skipped={},[]
paths=[root/"pyproject.toml"]
for folder in ["sys1","docs","tests"]:
 p=root/folder
 if p.exists():
  paths.extend(f for f in p.rglob("*") if f.suffix in [".py",".md"] or (folder=="sys1" and f.parent.name=="vocab" and f.suffix==".json"))
total=0
for p in sorted(set(paths)):
 if not p.is_file() or p.is_symlink():continue
 rel=str(p.relative_to(root));size=p.stat().st_size
 if size>2*1024**2 or total+size>64*1024**2 or len(files)>=1024:
  skipped.append(rel);continue
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(65536),b""):h.update(b)
 files[rel]={"sha256":h.hexdigest(),"bytes":size};total+=size
print(json.dumps({"files":files,"skipped":skipped,"bytes_hashed":total}))
'''


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def git(repo: Path, *args: str) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(["git", *args], cwd=repo, env=env, text=True,
                            capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("git_" + args[0] + "_failed")
    return result.stdout.strip()


def git_state(repo: Path, update: bool) -> dict:
    git(repo, "fetch", "--quiet", "origin", "main")
    branch = git(repo, "branch", "--show-current")
    dirty = bool(git(repo, "status", "--porcelain"))
    behind, ahead = map(int, git(repo, "rev-list", "--left-right", "--count", "origin/main...HEAD").split())
    updated = False
    if update and branch == "main" and not dirty and behind and not ahead:
        # Git itself also refuses a non-fast-forward or files changed since the probe.
        git(repo, "merge", "--ff-only", "origin/main")
        behind, ahead, updated = 0, 0, True
    return {"branch": branch, "head": git(repo, "rev-parse", "HEAD"),
            "remote_head": git(repo, "rev-parse", "origin/main"),
            "behind": behind, "ahead": ahead, "local_changes": dirty, "fast_forwarded": updated}


def matrix_inventory(repo: Path) -> dict:
    sys.path.insert(0, str(repo.parent / "fleet-mesh"))
    import fleet_exec
    result = fleet_exec.run_fanout(
        ["/usr/bin/python3", "-c", REMOTE_PROBE], nodes=fleet_exec.select_fleet_nodes(["matrix"]),
        timeout=12, idempotent=True, max_attempts=2,
    )
    row = result["results"][0]
    if not row.get("ok"):
        raise RuntimeError("matrix_" + row.get("status", "unreachable"))
    return json.loads(row["stdout"])


def compare(previous: dict, current: dict) -> dict:
    old, new = previous["files"], current["files"]
    return {"added": sorted(new.keys() - old.keys()), "removed": sorted(old.keys() - new.keys()),
            "changed": sorted(name for name in old.keys() & new.keys()
                              if old[name]["sha256"] != new[name]["sha256"]),
            "skipped": current.get("skipped", [])}


def check(repo: Path, state: Path, *, update: bool = False, accept_baseline: bool = False,
          probe=None) -> dict:
    baseline_path = state / "deployment-baseline.json"
    record = {"observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "matrix_writes": False, "automatic_commit_or_push": False}
    try:
        record["git"] = git_state(repo, update)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        record["git_error"] = str(error) if isinstance(error, RuntimeError) else "git_timeout"
    try:
        inventory = (probe or matrix_inventory)(repo)
        if not isinstance(inventory.get("files"), dict) or not inventory["files"]:
            raise RuntimeError("empty_matrix_source_inventory")
        baseline = read_json(baseline_path)
        if accept_baseline:
            if baseline:
                raise RuntimeError("baseline_already_exists_review_before_replacing")
            atomic_json(baseline_path, inventory)
            baseline = inventory
        if not baseline:
            record["matrix_error"] = "deployment_baseline_not_initialized"
        else:
            record["matrix"] = compare(baseline, inventory)
            record["matrix"]["source_files"] = len(inventory["files"])
    except (RuntimeError, ValueError, OSError) as error:
        record["matrix_error"] = str(error) if isinstance(error, RuntimeError) else type(error).__name__
    changes = record.get("matrix", {})
    source_drift = any(changes.get(key) for key in ("added", "removed", "changed", "skipped"))
    git_drift = bool(record.get("git", {}).get("behind") or record.get("git", {}).get("ahead"))
    record["needs_review"] = source_drift or git_drift or "git_error" in record or "matrix_error" in record
    record["status"] = "needs_review" if record["needs_review"] else "current"
    atomic_json(state / "latest.json", record)
    return record


def notify_once(state: Path, record: dict) -> None:
    signal = {key: record.get(key) for key in ("status", "matrix", "matrix_error", "git_error")}
    signal["git_drift"] = {key: record.get("git", {}).get(key) for key in ("ahead", "behind")}
    signature = hashlib.sha256(json.dumps(signal, sort_keys=True).encode()).hexdigest()
    previous = read_json(state / "notification.json")
    if record["needs_review"] and previous.get("signature") != signature:
        subprocess.run(["/usr/bin/osascript", "-e", 'display notification "Source or Git drift needs review. See dsco-sys1/runs/source-sync/latest.json." with title "DSCO System One"'],
                       capture_output=True, timeout=5, check=False)
    atomic_json(state / "notification.json", {"signature": signature})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="fast-forward a clean main checkout only")
    parser.add_argument("--accept-baseline", action="store_true", help="initialize a deployment baseline once")
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    state = REPO / "runs/source-sync"
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        record = check(REPO, state, update=args.update, accept_baseline=args.accept_baseline)
        if args.notify:
            notify_once(state, record)
        print(json.dumps(record))
        return 1 if record["needs_review"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

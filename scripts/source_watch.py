#!/usr/bin/env python3
"""Check canonical Git state and deployed source identities without changing Matrix."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sys1.model.diagnostics import diagnose_training
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
payload={"files":files,"skipped":skipped,"bytes_hashed":total}
'''


def remote_probe(run_name: str, trainer_pid: int | None = None) -> str:
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_name):
        raise ValueError("invalid training run name")
    if trainer_pid is not None and (type(trainer_pid) is not int or not 0 < trainer_pid < 2**31):
        raise ValueError("invalid trainer PID")
    trailer = '''
import time,subprocess
run=root/"runs/pretrain"/RUN_NAME
log=run/"log.jsonl"
try:
 if not log.is_file() or log.is_symlink():raise ValueError("training_log_missing")
 if log.stat().st_size>8*1024**2:raise ValueError("training_log_too_large")
 raw=log.read_bytes();lines=raw.splitlines();rows=[];malformed=0
 for index,line in enumerate(lines):
  try:rows.append(json.loads(line))
  except ValueError:
   if index<len(lines)-1 or raw.endswith(b"\\n"):malformed+=1
 retained=[]
 for p in (run/"checkpoints").glob("step_*.pt"):
  try:retained.append(int(p.stem.split("_")[1]))
  except ValueError:pass
 payload["training"]=diagnose_training(rows,retained,time.time())
 if malformed:payload["training"]["alerts"].append("malformed_training_log")
 legacy="b97bdd945cdd18e1f20bf7358b868542be097ea88df2872ba4197593351a7b12"
 if files.get("sys1/model/pretrain.py",{}).get("sha256")==legacy:
  payload["training"]["sampling"]="legacy_prefix"
  payload["training"]["alerts"].append("legacy_prefix_validation_requires_coverage_review")
 if TRAINER_PID is not None:
  process=subprocess.run(["ps","-p",str(TRAINER_PID),"-o","pid=,lstart=,state=,time="],capture_output=True,text=True,timeout=5)
  payload["training"]["registered_pid"]=TRAINER_PID
  payload["training"]["registered_pid_present"]=process.returncode==0
  payload["training"]["process_metadata"]=process.stdout.strip()
  if process.returncode and payload["training"]["status"] not in ["completed","early_stopped"]:
   payload["training"]["alerts"].append("registered_trainer_process_missing")
except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
 payload["training"]={"status":"unavailable","alerts":["training_diagnostics_unavailable"],"automatic_intervention":False}
print(json.dumps(payload))
'''
    return REMOTE_PROBE + "\n" + inspect.getsource(diagnose_training) + "\nRUN_NAME=" + repr(run_name) + "\nTRAINER_PID=" + repr(trainer_pid) + "\n" + trailer


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
    config = read_json(repo / "runs/source-sync/watch-config.json")
    run_name = config.get("training_run", "sys1-999m-v100k-run1")
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_name):
        raise ValueError("invalid training run name")
    manifest = read_json(repo / "runs/pretrain" / run_name / "run-manifest.json")
    pid = config.get("trainer_pid", manifest.get("live_observation", {}).get("pid"))
    source = remote_probe(run_name, pid)
    result = fleet_exec.run_fanout(
        ["/usr/bin/python3", "-c", source], nodes=fleet_exec.select_fleet_nodes(["matrix"]),
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
        if "training" in inventory:
            record["training"] = inventory["training"]
    except (RuntimeError, ValueError, OSError) as error:
        record["matrix_error"] = str(error) if isinstance(error, RuntimeError) else type(error).__name__
    changes = record.get("matrix", {})
    source_drift = any(changes.get(key) for key in ("added", "removed", "changed", "skipped"))
    git_drift = bool(record.get("git", {}).get("behind") or record.get("git", {}).get("ahead"))
    source_review = source_drift or git_drift or "git_error" in record or "matrix_error" in record
    record["source_status"] = "needs_review" if source_review else "current"
    record["needs_review"] = source_review or bool(record.get("training", {}).get("alerts"))
    record["status"] = "needs_review" if record["needs_review"] else "current"
    atomic_json(state / "latest.json", record)
    return record


def notify_once(state: Path, record: dict) -> None:
    signal = {key: record.get(key) for key in ("status", "matrix", "matrix_error", "git_error")}
    signal["git_drift"] = {key: record.get("git", {}).get(key) for key in ("ahead", "behind")}
    signal["training_alerts"] = sorted(record.get("training", {}).get("alerts", []))
    signal["training_status"] = record.get("training", {}).get("status")
    signature = hashlib.sha256(json.dumps(signal, sort_keys=True).encode()).hexdigest()
    previous = read_json(state / "notification.json")
    if record["needs_review"] and previous.get("signature") != signature:
        subprocess.run(["/usr/bin/osascript", "-e", 'display notification "Source or training quality needs review. See dsco-sys1/runs/source-sync/latest.json." with title "DSCO System One"'],
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

"""Immutable experiment identity and atomic, exclusively locked case records."""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
            size += len(block)
    return {"bytes": size, "sha256": h.hexdigest()}


def atomic_json(path, value):
    path = Path(path)
    payload = canonical(value) + "\n"
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Registry:
    def __init__(self, root, plan, resume=False):
        self.root, self.plan = Path(root), plan
        self.fingerprint = digest(plan)
        if self.root.exists() and not resume:
            raise FileExistsError("run exists; explicit --resume and identical inputs required")
        if not self.root.exists() and resume:
            raise FileNotFoundError("cannot resume a missing run")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(exist_ok=resume, mode=0o700)
        self._lock = None

    @contextmanager
    def locked(self):
        with (self.root / ".lock").open("a") as stream:
            os.chmod(stream.name, 0o600)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock = stream
            try:
                path = self.root / "plan.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["fingerprint"] != self.fingerprint or record["plan"] != self.plan:
                        raise ValueError("run identity differs; create a new run, never relabel old results")
                else:
                    if any(p.name != ".lock" for p in self.root.iterdir()):
                        raise ValueError("run has files but no immutable plan")
                    atomic_json(path, {"schema": "sys1-neuron-run-v1", "fingerprint": self.fingerprint,
                                      "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "plan": self.plan})
                yield self
            finally:
                self._lock = None
                fcntl.flock(stream, fcntl.LOCK_UN)

    def case(self, key):
        path = self.root / ("case-" + digest(key) + ".json")
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record["fingerprint"] != self.fingerprint or record["key"] != key or digest(record["result"]) != record["result_digest"]:
            raise ValueError("case identity or result digest differs")
        return record["result"]

    def save(self, key, result):
        if self._lock is None:
            raise RuntimeError("case writes require the exclusive run lock")
        if self.case(key) is not None:
            raise FileExistsError("case is immutable")
        atomic_json(self.root / ("case-" + digest(key) + ".json"), {
            "fingerprint": self.fingerprint, "key": key, "result": result, "result_digest": digest(result)})

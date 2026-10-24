from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def watch():
    path = Path(__file__).resolve().parents[1] / "scripts/source_watch.py"
    spec = importlib.util.spec_from_file_location("source_watch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(path, *args):
    result = subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def repositories(tmp_path):
    remote, writer, checkout = [tmp_path / name for name in ("remote.git", "writer", "checkout")]
    command(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    writer.mkdir()
    command(writer, "init", "--initial-branch=main")
    command(writer, "config", "user.name", "Source Watch Test")
    command(writer, "config", "user.email", "source-watch@example.invalid")
    (writer / "code.py").write_text("first\n")
    command(writer, "add", "code.py")
    command(writer, "commit", "-m", "initial")
    command(writer, "remote", "add", "origin", str(remote))
    command(writer, "push", "origin", "main")
    command(tmp_path, "clone", "--branch", "main", str(remote), str(checkout))
    return writer, checkout


def advance(writer):
    (writer / "code.py").write_text("second\n")
    command(writer, "commit", "-am", "remote update")
    command(writer, "push", "origin", "main")


def test_clean_checkout_fast_forwards(watch, repositories):
    writer, checkout = repositories
    advance(writer)
    result = watch.git_state(checkout, update=True)
    assert result["fast_forwarded"] is True
    assert result["head"] == result["remote_head"]
    assert (checkout / "code.py").read_text() == "second\n"


def test_dirty_checkout_is_never_overwritten(watch, repositories):
    writer, checkout = repositories
    (checkout / "code.py").write_text("local unsaved work\n")
    advance(writer)
    result = watch.git_state(checkout, update=True)
    assert result["behind"] == 1
    assert result["fast_forwarded"] is False
    assert (checkout / "code.py").read_text() == "local unsaved work\n"


def test_feature_branch_is_never_automatically_moved(watch, repositories):
    writer, checkout = repositories
    command(checkout, "switch", "-c", "work")
    before = command(checkout, "rev-parse", "HEAD")
    advance(writer)
    result = watch.git_state(checkout, update=True)
    assert result["behind"] == 1
    assert result["head"] == before


def test_deployment_drift_is_reported_without_resetting_baseline(watch, repositories, tmp_path):
    _, checkout = repositories
    state = tmp_path / "state"
    first = {"files": {"sys1/model/pretrain.py": {"sha256": "a", "bytes": 10}}, "skipped": []}
    result = watch.check(checkout, state, accept_baseline=True, probe=lambda _: first)
    assert result["status"] == "current"
    original = (state / "deployment-baseline.json").read_bytes()
    second = {"files": {"sys1/model/pretrain.py": {"sha256": "b", "bytes": 11}}, "skipped": []}
    result = watch.check(checkout, state, probe=lambda _: second)
    assert result["matrix"]["changed"] == ["sys1/model/pretrain.py"]
    assert result["matrix_writes"] is False
    assert result["needs_review"] is True
    assert (state / "deployment-baseline.json").read_bytes() == original
    result = watch.check(checkout, state, accept_baseline=True, probe=lambda _: second)
    assert result["matrix_error"] == "baseline_already_exists_review_before_replacing"
    assert (state / "deployment-baseline.json").read_bytes() == original


def test_unreachable_matrix_is_not_reported_current(watch, repositories, tmp_path):
    _, checkout = repositories
    def unavailable(_):
        raise RuntimeError("matrix_unreachable")
    result = watch.check(checkout, tmp_path / "state", probe=unavailable)
    assert result["needs_review"] is True
    assert result["matrix_error"] == "matrix_unreachable"


def test_training_alert_does_not_masquerade_as_source_drift(watch, repositories, tmp_path):
    _, checkout = repositories
    state = tmp_path / "state"
    inventory = {"files": {"sys1/model/pretrain.py": {"sha256": "a", "bytes": 10}}, "skipped": [],
                 "training": {"status": "possible_overfitting", "alerts": ["possible_overfitting"]}}
    report = watch.check(checkout, state, accept_baseline=True, probe=lambda _: inventory)
    assert report["source_status"] == "current"
    assert report["needs_review"] is True
    assert report["matrix"]["changed"] == []
    assert report["training"]["status"] == "possible_overfitting"


def test_remote_probe_exports_only_aggregates_and_does_not_write_training_files(watch, tmp_path, monkeypatch, capsys):
    from test_training_diagnostics import history
    root = tmp_path / "dsco-sys1"
    (root / "sys1/model").mkdir(parents=True)
    code = root / "sys1/model/pretrain.py"
    code.write_text("# source fixture\n")
    run = root / "runs/pretrain/demo"
    (run / "checkpoints").mkdir(parents=True)
    log = run / "log.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in history()))
    checkpoint = run / "checkpoints/step_00000010.pt"
    checkpoint.write_bytes(b"unchanged-checkpoint")
    before = (code.read_bytes(), log.read_bytes(), checkpoint.read_bytes())
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    exec(compile(watch.remote_probe("demo"), "<read-only-probe>", "exec"), {})
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["training"]["best_validation"]["step"] == 10
    assert "PRIVATE_EXAMPLE" not in output
    assert before == (code.read_bytes(), log.read_bytes(), checkpoint.read_bytes())
    with pytest.raises(ValueError):
        watch.remote_probe("../other")


def test_notification_does_not_repeat_for_every_new_training_step(watch, tmp_path, monkeypatch):
    import copy
    calls = []
    monkeypatch.setattr(watch.subprocess, "run", lambda *a, **k: calls.append(a))
    record = {"status": "needs_review", "needs_review": True, "git": {"ahead": 0, "behind": 0},
              "matrix": {}, "training": {"status": "improving", "alerts": ["lost_best"], "step": 10}}
    watch.notify_once(tmp_path, record)
    later = copy.deepcopy(record)
    later["training"]["step"] = 20
    watch.notify_once(tmp_path, later)
    assert len(calls) == 1

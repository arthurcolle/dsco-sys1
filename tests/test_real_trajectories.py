import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import pytest

from sys1.model.corpus_manifest import file_identity, load_manifest
from test_pretrain import tokenizer


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/add_real_trajectories.py"
    spec = importlib.util.spec_from_file_location("add_real_trajectories", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode(session, goal=None):
    return {"session": session, "ts": "2026-10-01", "end": "final_answer",
            "goal": goal or f"Complete this useful systems task for {session}",
            "steps": [{"tool": "read_file", "args": "{}", "result": "completed successfully", "ok": True}],
            "final": {"finish_reason": "end_turn", "text": "The requested systems operation completed with the expected result and verification."}}


def fixture_data(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    sources = {}
    for name in ("en", "agent_opcodes"):
        sources[name] = {}
        for split in ("train", "validation"):
            path = base / f"{name}-{split}.jsonl"
            path.write_text(json.dumps({"text": f"baseline {name} {split} text"}) + "\n")
            sources[name][split] = [{"path": path.name, **file_identity(path)}]
    manifest = base / "MANIFEST.json"
    manifest.write_text(json.dumps({"schema": "sys1-corpus-mixture-v1", "sources": sources,
                                    "objective_weights": {"en": 0.8, "agent_opcodes": 0.2}}))
    chronicle = tmp_path / "chronicle"
    (chronicle / "events").mkdir(parents=True)
    for name in ("session-a.jsonl", "session-z.jsonl"):
        path = chronicle / "events" / name
        path.write_text('{}\n')
        os.utime(path, (time.time()-120, time.time()-120))
    previous = tmp_path / "previous.jsonl"
    previous.write_text("")
    tok = tmp_path / "tokenizer.json"
    tokenizer().save(str(tok))
    return manifest, chronicle, previous, tok


def sessions(builder):
    found = {"train": [], "validation": []}
    for i in range(100):
        name = f"fixture-session-{i}"
        found[builder.session_split(name)].append(name)
    return found


@pytest.mark.parametrize("mutation", ["unfinished", "truncated", "unresolved_tool", "failed_tool", "empty_goal"])
def test_incomplete_or_unusable_episodes_are_rejected(builder, mutation):
    row = episode("session")
    assert builder.eligible(row)
    if mutation == "unfinished":row["end"] = "session_end"
    if mutation == "truncated":row["final"]["finish_reason"] = "max_tokens"
    if mutation == "unresolved_tool":row["steps"][0]["ok"] = None
    if mutation == "failed_tool":row["steps"][0]["ok"] = False
    if mutation == "empty_goal":row["goal"] = ""
    assert not builder.eligible(row)


def test_expansion_reserves_sessions_redacts_and_binds_exposure(tmp_path, builder, monkeypatch):
    base, chronicle, previous, tok = fixture_data(tmp_path)
    names = sessions(builder)
    rows = [episode(names["train"][0]), episode(names["validation"][0])]
    secret = "sk-" + "A" * 30
    rows[0]["steps"][0]["args"] = secret
    monkeypatch.setattr(builder.exporter, "episodes_from_session", lambda _: copy.deepcopy(rows))
    original = builder.exporter.CHRONICLE
    path, manifest = builder.expand(base, tmp_path / "new", chronicle, previous, tok)
    assert builder.exporter.CHRONICLE == original
    sources, _, _ = load_manifest(path)
    assert len(sources) == 3
    records = {split: [json.loads(line) for line in sources["agent_real"][split][0].read_text().splitlines()]
               for split in ("train", "validation")}
    assert len(records["train"]) == len(records["validation"]) == 1
    assert secret not in sources["agent_real"]["train"][0].read_text()
    assert manifest["audit"]["redactions"]["credential"] > 0
    assert manifest["exposure_budget"]["expected_fresh_passes"] <= 1 + 1e-12
    assert sum(manifest["objective_weights"].values()) == pytest.approx(1)
    assert manifest["objective_weights"]["en"] == 0.8
    assert (path.parent / "recipe.json").is_file()
    with pytest.raises(FileExistsError):
        builder.expand(base, path.parent, chronicle, previous, tok)


def test_later_validation_goal_reservation_removes_earlier_session_records(tmp_path, builder, monkeypatch):
    base, chronicle, previous, tok = fixture_data(tmp_path)
    names = sessions(builder)
    reserved = names["train"][0]
    held_goal = "This entire private validation task must remain reserved"
    prior_path = tmp_path / "heldout-private.txt"
    prior_path.write_text(f"[BOS]<|user|>{held_goal}<|end|>\n<|assistant|>answer<|end|>[EOS]\n")
    prior = tmp_path / "private-manifest.json"
    prior.write_text(json.dumps({"files": [{"name": prior_path.name, **file_identity(prior_path)}]}))
    data = {"session-z.jsonl": [episode(reserved), episode(names["train"][1]), episode(names["validation"][0])],
            "session-a.jsonl": [episode(reserved, held_goal)]}
    monkeypatch.setattr(builder.exporter, "episodes_from_session", lambda path: copy.deepcopy(data[Path(path).name]))
    path, manifest = builder.expand(base, tmp_path / "new", chronicle, previous, tok, prior_private_manifest=prior)
    digest = hashlib.sha256(reserved.encode()).hexdigest()
    reserved_set = set(json.loads((path.parent / "reserved-sessions.json").read_text())["session_digests"])
    assert digest in reserved_set
    assert manifest["audit"]["counts"]["late_reserved_validation_documents_excluded"] == 1
    for split in ("train.jsonl", "heldout.jsonl"):
        assert all(json.loads(line)["session_digest"] != digest for line in (path.parent / split).read_text().splitlines())


def test_session_index_cannot_reference_files_outside_chronicle(tmp_path, builder):
    base, chronicle, previous, tok = fixture_data(tmp_path)
    index = tmp_path / "index.txt"
    index.write_text(str(previous) + "\n")
    with pytest.raises(ValueError, match="chronicle event files only"):
        builder.expand(base, tmp_path / "new", chronicle, previous, tok, session_index=index)
    assert not (tmp_path / "new").exists()

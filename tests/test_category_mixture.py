import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from sys1.model.corpus_manifest import file_identity, load_manifest
from sys1.model import pretrain as pt
from sys1.model.validation import ValidationGuard
from test_pretrain import data_args, tokenizer


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_agent_mixture.py"
    spec = importlib.util.spec_from_file_location("prepare_agent_mixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_corpus(root, builder):
    root.mkdir()
    agent = root / "agent"
    agent.mkdir()
    parent = {"version": 1, "format": "one sample per line; newlines escaped as \\n", "files": {},
              "heldout_by_category": {}, "provenance": "synthetic fixture"}
    validation = []
    for category in builder.CATEGORY_SHARES:
        lines = [f"{category} held only {i}\n" for i in range(16)]
        train = [f"{category} alpha beta {i}\n" for i in range(32)]
        if category == "opcodes":
            train += [f"{category}  held  only  0 \n", train[0], f"{category}  alpha  beta  1 \n"]
        path = agent / (category + ".txt")
        path.write_text("".join(train))
        parent["files"][path.name] = {**file_identity(path), "lines": len(train)}
        parent["heldout_by_category"][category] = {"lines": len(lines), "bytes": len("".join(lines).encode())}
        validation += lines
    path = agent / "heldout.txt"
    path.write_text("".join(validation))
    parent["files"][path.name] = {**file_identity(path), "lines": len(validation)}
    (agent / "MANIFEST.json").write_text(json.dumps(parent))
    for source in ("en", "code"):
        folder = root / source
        folder.mkdir()
        (folder / "train.txt").write_text(f"{source} alpha beta\n" * 64)
        (folder / "heldout.txt").write_text(f"{source} held only\n" * 64)
        (folder / "MANIFEST.json").write_text('{"license_ref":"synthetic fixture"}')
    return root


def build_view(tmp_path, builder):
    parent = parent_corpus(tmp_path / "parent", builder)
    path, manifest = builder.prepare(parent, tmp_path / "derived")
    return parent, path, manifest


def view_args(parent, path, builder):
    args = data_args(parent)
    args.corpus_manifest = str(path)
    args.mix = ",".join(f"{key}={value}" for key, value in builder.WEIGHTS.items())
    args.validation_sampling = "reservoir"
    return args


def test_derived_view_repairs_overlap_and_duplicates_without_changing_parent(tmp_path, builder):
    parent = parent_corpus(tmp_path / "parent", builder)
    before = {str(p.relative_to(parent)): file_identity(p) for p in parent.rglob("*") if p.is_file()}
    path, manifest = builder.prepare(parent, tmp_path / "derived")
    assert before == {str(p.relative_to(parent)): file_identity(p) for p in parent.rglob("*") if p.is_file()}
    assert manifest["audit"]["normalized_split_overlap"] == 1
    assert manifest["audit"]["exact_train_duplicates"] == 1
    assert manifest["audit"]["normalized_train_duplicates"] == 2
    assert manifest["audit"]["excluded_train_documents"] == 3
    sources, _, _ = load_manifest(path)
    normalize = lambda line: " ".join(line.split())
    validation = {normalize(line) for name in sources if name.startswith("agent_") for line in sources[name]["validation"][0].read_text().splitlines()}
    training = [normalize(line) for name in sources if name.startswith("agent_") for line in sources[name]["train"][0].read_text().splitlines()]
    assert not set(training) & validation
    assert len(training) == len(set(training))
    assert sources["agent_decisions"]["train"][0] == parent / "agent/decisions.txt"
    assert sources["agent_opcodes"]["train"][0] == path.parent / "agent_opcodes/train.txt"
    with pytest.raises(FileExistsError):
        builder.prepare(parent, path.parent)


def test_category_sampler_covers_every_category_and_resumes_exactly(tmp_path, builder):
    parent, path, _ = build_view(tmp_path, builder)
    args, tok = view_args(parent, path, builder), tokenizer()
    mix, heldout, _ = pt.build_data(args, tok, 1)
    assert set(mix.names) == set(heldout) == set(builder.WEIGHTS)
    mix.batch(2048)
    assert all(mix.drawn[name] >= 20 for name in mix.names)
    state = copy.deepcopy(mix.state())
    expected = mix.batch(64)
    restored, _, _ = pt.build_data(args, tok, 1)
    restored.load_state(state)
    assert torch.equal(expected, restored.batch(64))
    path.write_text(path.read_text() + "\n")
    changed, _, _ = pt.build_data(args, tok, 1)
    with pytest.raises(ValueError, match="manifest identity"):
        changed.load_state(state)


def test_category_guard_catches_opcode_regression_hidden_by_aggregate_gain(tmp_path, builder):
    parent, path, _ = build_view(tmp_path, builder)
    mix, _, _ = pt.build_data(view_args(parent, path, builder), tokenizer(), 1)
    guard = ValidationGuard(mix.reference_weights, max_domain_regression=0.15, domain_patience=2)
    base = {key: 4.0 for key in mix.names}
    first = guard.observe(1, base)
    later = {key: 2.0 for key in mix.names}
    later["agent_opcodes"] = 5.0
    result = guard.observe(2, later)
    assert result["weighted_loss"] < first["weighted_loss"]
    assert result["best_step"] == 1
    assert guard.observe(3, later)["stop_reason"] == "domain_regression:agent_opcodes"


def test_modified_referenced_training_file_is_rejected(tmp_path, builder):
    parent, path, _ = build_view(tmp_path, builder)
    with (parent / "agent/decisions.txt").open("a") as stream:
        stream.write("unexpected change\n")
    with pytest.raises(ValueError, match="identity differs"):
        load_manifest(path)


def test_overlap_alias_cannot_be_declared_as_validation(tmp_path, builder):
    _, path, manifest = build_view(tmp_path, builder)
    manifest["sources"]["agent_schemas"]["validation"] = manifest["sources"]["agent_schemas"]["train"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        load_manifest(path)


def test_missing_category_weight_cannot_be_silently_renormalized(tmp_path, builder):
    parent, path, _ = build_view(tmp_path, builder)
    args = view_args(parent, path, builder)
    args.mix = "en=0.5,code=0.3"
    with pytest.raises(ValueError, match="exactly the manifest sources"):
        pt.build_data(args, tokenizer(), 1)


def test_parent_hash_mismatch_is_rejected_before_creating_output(tmp_path, builder):
    parent = parent_corpus(tmp_path / "parent", builder)
    (parent / "agent/opcodes.txt").write_text("changed\n")
    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="differs from its manifest"):
        builder.prepare(parent, output)
    assert not output.exists()

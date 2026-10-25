import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from sys1.model.corpus_manifest import file_identity, load_manifest
from test_pretrain import tokenizer


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/add_systems_corpus.py"
    spec = importlib.util.spec_from_file_location("add_systems_corpus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def code(name):
    return "\n".join(f"int {name}_operation_{i}(int value) {{ return value + {i}; }}" for i in range(16)) + "\n"


def repository(root, source, builder):
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "dsco")
    git(root, "config", "user.email", "agent@distributed.systems")
    (root / "LICENSE").write_text("MIT License\n\nCopyright DSCO test fixture\n")
    counts = {"train": 0, "validation": 0}
    selected = []
    for i in range(1000):
        name = f"module{i}"
        path = (Path("src") / (name+".c")) if source == "systems_native" else Path("apps/demo/src") / (name+".erl")
        split = builder.split_for(builder.component(source, str(path)))
        if counts[split] >= 2:
            continue
        counts[split] += 1
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code(name+source))
        selected.append(path)
        if min(counts.values()) == 2:
            break
    (root / "vendor").mkdir()
    (root / "vendor/ignored.c").write_text(code("vendor"))
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return selected


def base_manifest(root, tok):
    root.mkdir()
    sources = {}
    for name in ("en", "code"):
        sources[name] = {}
        for split in ("train", "validation"):
            path = root / (name+"-"+split+".jsonl")
            path.write_text(json.dumps({"text": f"baseline {name} {split} only"})+"\n")
            sources[name][split] = [{"path": path.name, **file_identity(path)}]
    path = root / "MANIFEST.json"
    path.write_text(json.dumps({"schema": "sys1-corpus-mixture-v1", "sources": sources,
                                "objective_weights": {"en": .7, "code": .3}, "tokenizer_identity": file_identity(tok)}))
    (root / "recipe.json").write_text(json.dumps({"schema": "sys1-pretrain-recipe-v1", "objective": "causal-next-token-cross-entropy",
        "training": {"steps": 30000, "seq": 1024, "micro-batch": 2, "accum": 8}}))
    return path


def test_related_files_share_a_component_and_split(builder):
    paths = ["src/provider_openai.c", "include/provider.h", "tests/test_provider_openai.c"]
    assert len({builder.component("systems_native", path) for path in paths}) == 1
    assert builder.component("systems_otp", "apps/agents/src/agent_registry.erl") == builder.component("systems_otp", "tests/agent_registry_tests.erl")


def test_chunks_keep_line_provenance_and_do_not_truncate_long_lines(builder):
    text = code("example") * 4
    result = builder.chunks(text, max_chars=400)
    lines = text.splitlines(keepends=True)
    assert len(result) > 1
    assert all(content == "".join(lines[start-1:end]) for start, end, content in result)
    assert builder.chunks("a"*1000, max_chars=400) == []


def test_near_duplicate_uses_informative_fingerprints(builder):
    signature = builder.fingerprints(code("example"))
    index = {item: {0} for item in signature}
    assert builder.resembles(signature, index)
    assert not builder.resembles(builder.fingerprints(code("different")), index)


def test_expansion_reads_pinned_git_not_dirty_files_and_preserves_splits(tmp_path, builder):
    tok = tmp_path / "tokenizer.json"
    tokenizer().save(str(tok))
    repositories = {name: tmp_path/name for name in builder.PREFIXES}
    selected = {name: repository(path, name, builder) for name, path in repositories.items()}
    native = repositories["systems_native"]
    train_paths = [p for p in selected["systems_native"] if builder.split_for(builder.component("systems_native", str(p))) == "train"]
    held_path = next(p for p in selected["systems_native"] if builder.split_for(builder.component("systems_native", str(p))) == "validation")
    (native / train_paths[0]).write_text((native / held_path).read_text().replace("return value + 0;", "return value + 999;", 1))
    git(native, "add", ".")
    git(native, "commit", "-qm", "add near-duplicate training component")
    dirty_path = native / train_paths[1]
    dirty_path.write_text("UNCOMMITTED_PRIVATE_EDIT")
    base = base_manifest(tmp_path / "base", tok)
    path, manifest = builder.expand(base, tmp_path / "expanded", repositories, tok)
    assert dirty_path.read_text() == "UNCOMMITTED_PRIVATE_EDIT"
    resolved, _, _ = load_manifest(path)
    assert len(resolved) == 4
    for name in repositories:
        records = {split: [json.loads(line) for p in resolved[name][split] for line in p.read_text().splitlines()]
                   for split in ("train", "validation")}
        assert not {row["component"] for row in records["train"]} & {row["component"] for row in records["validation"]}
        assert not {row["document_sha256"] for row in records["train"]} & {row["document_sha256"] for row in records["validation"]}
        assert all("UNCOMMITTED_PRIVATE_EDIT" not in row["text"] and "vendor/" not in row["path"] for rows in records.values() for row in rows)
        if name == "systems_native":
            assert not any(row["component"] == builder.component(name, str(train_paths[0])) for rows in records.values() for row in rows)
            assert any(row["path"] == str(train_paths[1]) for rows in records.values() for row in rows)
        assert manifest["repositories"][name]["commit"] == git(repositories[name], "rev-parse", "HEAD")
        assert manifest["exposure_budget"]["expected_source_passes"][name] <= 1 + 1e-12
    assert sum(manifest["objective_weights"].values()) == pytest.approx(1)
    assert manifest["objective_weights"]["en"] == .7
    with pytest.raises(FileExistsError):
        builder.expand(base, path.parent, repositories, tok)

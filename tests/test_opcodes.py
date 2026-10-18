"""Op-code table invariants: append-only codes, full definitions, closed references."""
from __future__ import annotations

import json

from sys1.paths import OPCODES_JSON
from sys1.vocab import definitions, opcodes

PINNED = 1500


def test_count_matches_header(opcodes_doc, ops):
    assert opcodes_doc["count"] == len(ops)


def test_mnemonics_unique(ops):
    mnemonics = [o["mnemonic"] for o in ops]
    assert len(mnemonics) == len(set(mnemonics))


def test_tokens_unique_and_derived(ops):
    tokens = [o["token"] for o in ops]
    assert len(tokens) == len(set(tokens))
    for o in ops:
        assert o["token"] == "<|op:" + o["mnemonic"].lower() + "|>"


def test_codes_contiguous(ops):
    assert [o["code"] for o in ops] == list(range(len(ops)))


def test_mnemonic_shape(ops):
    for o in ops:
        assert opcodes.MNEMONIC.match(o["mnemonic"]), o["mnemonic"]
        assert o["mnemonic"].split(".")[0] == o["family"]
        assert o["effect"] in {"pure", "observe", "local", "egress", "control"}


def test_first_1500_pinned(ops, pinned_mnemonics):
    assert len(pinned_mnemonics) == PINNED
    assert len(ops) >= PINNED
    current = [o["mnemonic"] for o in ops[:PINNED]]
    drift = [(i, a, b) for i, (a, b) in enumerate(zip(pinned_mnemonics, current)) if a != b]
    assert not drift, f"op-codes are append-only; first changed codes: {drift[:5]}"


def test_no_default_definitions(ops):
    defaults = [o["mnemonic"] for o in ops if o.get("definition") == "default"]
    assert not defaults, defaults[:10]
    assert {o["definition"] for o in ops} <= {"verb", "single"}


def test_inverse_and_failure_refs_exist(ops, by_mnemonic):
    bad_inverse = [(o["mnemonic"], o["inverse"]) for o in ops if o.get("inverse") and o["inverse"] not in by_mnemonic]
    bad_fail = [(o["mnemonic"], f) for o in ops for f in o.get("failure_ops", []) if f not in by_mnemonic]
    assert not bad_inverse, bad_inverse[:10]
    assert not bad_fail, bad_fail[:10]


def test_dsco_grounding_targets_exist(by_mnemonic):
    missing = sorted(set(opcodes.DSCO_GROUNDING) - set(by_mnemonic))
    assert not missing, missing


def test_opcodes_json_matches_regeneration(tmp_path, monkeypatch, opcodes_doc):
    """definitions.main() writes next to its __file__; point it at a temp dir so the
    checked-in opcodes.json is never overwritten."""
    monkeypatch.setattr(definitions, "__file__", str(tmp_path / "definitions.py"))
    assert definitions.main() == 0
    regenerated = json.loads((tmp_path / "opcodes.json").read_text())
    assert regenerated["count"] == opcodes_doc["count"]
    for i, (a, b) in enumerate(zip(regenerated["ops"], opcodes_doc["ops"])):
        assert a == b, f"opcodes.json is stale at code {i} ({b['mnemonic']}); rerun python -m sys1.vocab.definitions"
    assert regenerated == opcodes_doc
    assert json.loads(OPCODES_JSON.read_text()) == opcodes_doc  # untouched

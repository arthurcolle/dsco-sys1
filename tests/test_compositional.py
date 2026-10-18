"""Compositional fam/verb/obj tokens cover every fused FAMILY.VERB.OBJECT op."""
from __future__ import annotations

import json

import pytest

from sys1.paths import COMPOSITIONAL_JSON
from sys1.vocab import compositional


@pytest.fixture(scope="module")
def comp() -> dict:
    return json.loads(COMPOSITIONAL_JSON.read_text())


def test_every_fused_pair_decomposes(ops, comp):
    tokens = set(comp["tokens"])
    decomposition = comp["decomposition"]
    fused = [o for o in ops if o["mnemonic"].count(".") == 2]
    assert fused
    for o in fused:
        fam, verb, obj = o["mnemonic"].lower().split(".")
        expected = [f"<|fam:{fam}|>", f"<|verb:{verb}|>", f"<|obj:{obj}|>"]
        assert decomposition.get(o["token"]) == expected, o["mnemonic"]
        assert set(expected) <= tokens, o["mnemonic"]
    assert len(decomposition) == len(fused)


def test_singles_not_decomposed(ops, comp):
    for o in ops:
        if o["mnemonic"].count(".") == 1:
            assert o["token"] not in comp["decomposition"]


def test_tokens_unique_and_disjoint_from_fused(ops, comp):
    tokens = comp["tokens"]
    assert len(tokens) == len(set(tokens))
    assert not set(tokens) & {o["token"] for o in ops}
    assert len(tokens) == comp["families"] + comp["verbs"] + comp["objects"]


def test_compositional_json_matches_build(comp):
    assert compositional.build() == comp

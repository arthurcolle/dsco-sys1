"""jevmr prompt families: options exist, oracles are closed, generation is seeded."""
from __future__ import annotations

import itertools
import random

import pytest

from sys1.jevmr.families import FAMILIES, OPS
from sys1.jevmr import fractal
from sys1.jevmr.jevmr import item_id

BASE = {f.name: f for f in fractal.BASE}
MODEL = "test-model"


def combos(fam):
    keys = list(fam.slots)
    return [dict(zip(keys, v)) for v in itertools.product(*(fam.slots[k] for k in keys))]


def test_variants_registered():
    assert len(BASE) >= 8
    assert set(fractal.VARIANTS) <= set(FAMILIES)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_options_exist(name, by_mnemonic):
    fam = FAMILIES[name]
    assert fam.options
    assert len(fam.options) == len(set(fam.options))
    missing = [o for o in fam.options if o not in by_mnemonic or o not in OPS]
    assert not missing, missing


@pytest.mark.parametrize("name", sorted(BASE))
def test_base_valid_nonempty_and_oracle_closed(name):
    fam = BASE[name]
    valid = [c for c in combos(fam) if fam.valid(c)]
    assert valid, f"{name}: valid() rejects every slot combination"
    golds = {fam.oracle(c) for c in valid}
    assert golds <= set(fam.options), golds - set(fam.options)


@pytest.mark.parametrize("name", sorted(BASE))
def test_base_items_render(name):
    fam = BASE[name]
    items = fam.items(random.Random(7))
    assert items
    assert len(items) == len([c for c in combos(fam) if fam.valid(c)]) * len(fam.phrasings)
    for it in items:
        assert it["gold"] in fam.options
        assert "{" not in it["state"], it["state"]
        for text, answer in it["probes"].values():
            assert isinstance(text, str) and isinstance(answer, bool)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_all_families_nonempty_and_gold_in_options(name):
    fam = FAMILIES[name]
    items = fam.items(random.Random(7), 4)
    assert items, f"{name}: generated no items"
    for it in items:
        assert it["family"] == name
        assert it["gold"] in fam.options, (name, it["gold"])


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_item_ids_stable_for_same_seed(name):
    fam = FAMILIES[name]

    def ids():
        return [item_id(it, fam.request(it, MODEL)) for it in fam.items(random.Random(7), 4)]

    first, second = ids(), ids()
    assert first == second
    assert len(first) == len(set(first)), f"{name}: duplicate item ids"

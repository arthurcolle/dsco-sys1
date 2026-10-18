"""Reserved-token layout of the constructed vocabulary."""
from __future__ import annotations

from sys1.tokenizer import spec


def test_vocab_size_multiple_of_128():
    assert spec.VOCAB_SIZE % 128 == 0


def test_reserved_deterministic_and_unique():
    a = spec.reserved_tokens(include_tools=False)
    b = spec.reserved_tokens(include_tools=False)
    assert a == b
    assert len(a) == len(set(a))


def test_reserved_order(ops):
    r = spec.reserved_tokens(include_tools=False)
    nc, nr = len(spec.CONTROL), len(spec.ROLES)
    assert r[:nc] == spec.CONTROL
    assert r[nc:nc + nr] == spec.ROLES
    by_code = [o["token"] for o in sorted(ops, key=lambda o: o["code"])]
    assert r[nc + nr:nc + nr + len(by_code)] == by_code


def test_reserved_ends_with_future_slots():
    r = spec.reserved_tokens(include_tools=False)
    tail = r[-spec.FUTURE_SLOTS:]
    assert tail == [f"<|reserved_{i}|>" for i in range(spec.FUTURE_SLOTS)]


def test_bpe_budget_positive():
    budget = spec.bpe_budget(include_tools=False)
    assert budget > 0
    assert budget + len(spec.reserved_tokens(include_tools=False)) == spec.VOCAB_SIZE

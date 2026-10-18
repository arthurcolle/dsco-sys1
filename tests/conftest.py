"""Pytest configuration for the sys1 test suite.

Equivalent of a [tool.pytest.ini_options] section, kept here so pyproject.toml
stays untouched. The suite is CPU-only, needs no network or private data, and
should finish in well under 60 s.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch a GPU


def pytest_configure(config):
    # [tool.pytest.ini_options] equivalents
    config.option.strict_markers = True
    config.addinivalue_line("markers", "slow: tests that take more than a few seconds")
    config.addinivalue_line("filterwarnings", "error::pytest.PytestUnraisableExceptionWarning")


@pytest.fixture(scope="session")
def opcodes_doc() -> dict:
    from sys1.paths import OPCODES_JSON
    return json.loads(OPCODES_JSON.read_text())


@pytest.fixture(scope="session")
def ops(opcodes_doc) -> list[dict]:
    return opcodes_doc["ops"]


@pytest.fixture(scope="session")
def by_mnemonic(ops) -> dict[str, dict]:
    return {o["mnemonic"]: o for o in ops}


@pytest.fixture(scope="session")
def pinned_mnemonics() -> list[str]:
    return json.loads((FIXTURES / "opcodes_v1_mnemonics.json").read_text())["mnemonics"]

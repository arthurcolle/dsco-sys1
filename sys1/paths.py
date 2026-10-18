"""Canonical file locations, independent of the current working directory."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
VOCAB = PACKAGE / "vocab"
OPCODES_JSON = VOCAB / "opcodes.json"
COMPOSITIONAL_JSON = VOCAB / "compositional.json"
DATA = REPO / "data"   # local, gitignored: exported episodes, decisions, labels
RUNS = REPO / "runs"   # local, gitignored: checkpoints, Jev results, reports

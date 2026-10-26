#!/usr/bin/env python3
"""Run offline SystemOne neuron diagnostics with explicit checkpoint/probe identities."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sys1.neurons.cli import main

if __name__ == "__main__":
    main()

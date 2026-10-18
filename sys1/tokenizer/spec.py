"""Layout of the constructed ~100K sys1 vocabulary.

A byte-level BPE base is learned from the corpus mix; designed regions are
reserved on top of it so every agent-mechanics concept is one token:

  control      special markers for decision formatting and chat structure
  opcodes      every fused op-code token from sys1/vocab/opcodes.json (append-only)
  compositional family / verb / object tokens from sys1/vocab/compositional.json
  tools        one token per known dsco tool name, <|tool:name|>
  roles/format role, channel and structured-output markers
  future       unused reserved slots so later op-codes never shift BPE ids

VOCAB_SIZE is a multiple of 128 for GPU-friendly embedding shapes. The BPE
trainer learns exactly VOCAB_SIZE - len(reserved_tokens()) merges+bytes.
"""
from __future__ import annotations

import json

from sys1.paths import COMPOSITIONAL_JSON, DATA, OPCODES_JSON

VOCAB_SIZE = 100_352  # 784 x 128; comparable to cl100k_base (100,256)

CONTROL = ["[PAD]", "[EOS]", "[BOS]", "[UNK]", "[GOAL]", "[STEP]", "[OPTS]", "[OPT]", "[DECIDE]",
           "[STATE]", "[OBS]", "[ACT]", "[RESULT]", "[POLICY]", "[ANSWER]", "[SEP]", "[MASK]"]
ROLES = ["<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|developer|>", "<|end|>",
         "<|json|>", "<|/json|>", "<|code|>", "<|/code|>", "<|think|>", "<|/think|>",
         "<|choice|>", "<|score|>", "<|noul|>", "<|box|>", "<|confidence|>"]
FUTURE_SLOTS = 2048  # headroom for op-code growth without retraining the BPE base


def _ops() -> list[str]:
    return [op["token"] for op in json.loads(OPCODES_JSON.read_text())["ops"]]


def _compositional() -> list[str]:
    return json.loads(COMPOSITIONAL_JSON.read_text())["tokens"]


def _tools(limit: int = 512) -> list[str]:
    """Most frequent dsco tool names from local episodes, if present."""
    path = DATA / "episodes.jsonl"
    if not path.exists():
        return []
    from collections import Counter
    counts = Counter()
    with open(path) as f:
        for line in f:
            for step in json.loads(line).get("steps", []):
                name = step.get("tool")
                if name and name.replace("_", "").replace("-", "").replace(".", "").isalnum() and len(name) <= 48:
                    counts[name] += 1
    return [f"<|tool:{name}|>" for name, _ in counts.most_common(limit)]


def reserved_tokens(include_tools: bool = True) -> list[str]:
    """Ordered reserved region. Order is part of the contract: append only."""
    tokens = CONTROL + ROLES + _ops() + _compositional()
    if include_tools:
        tokens += _tools()
    tokens += [f"<|reserved_{i}|>" for i in range(FUTURE_SLOTS)]
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def bpe_budget(include_tools: bool = True) -> int:
    return VOCAB_SIZE - len(reserved_tokens(include_tools))


if __name__ == "__main__":
    r = reserved_tokens()
    print(json.dumps({"vocab_size": VOCAB_SIZE, "reserved": len(r), "bpe_budget": VOCAB_SIZE - len(r),
                      "regions": {"control": len(CONTROL), "roles": len(ROLES), "opcodes": len(_ops()),
                                  "compositional": len(_compositional()), "tools": len(_tools()),
                                  "future": FUTURE_SLOTS}}))

#!/usr/bin/env python3
"""Compositional op-code tokens alongside the 1,500 fused ones.

    python -m sys1.vocab.compositional      # writes compositional.json

Fused tokens (<|op:ctx.mask.observation_block|>) stay canonical. These extra
tokens let the model express verb+object pairs it never saw fused:
  <|fam:ctx|>   one per family
  <|verb:mask|> one per verb used by any op-code
  <|obj:observation_block|> one per object used by any op-code
A fused op FAMILY.VERB.OBJECT decomposes into [fam, verb, obj]; singles
(FAMILY.NAME) stay fused only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sys1.vocab import opcodes


def build() -> dict:
    ops = opcodes.build()
    fams, verbs, objs, decomposition = set(), set(), set(), {}
    for op in ops:
        parts = op["mnemonic"].split(".")
        fams.add(parts[0])
        if len(parts) == 3:
            verbs.add(parts[1].lower())
            objs.add(parts[2].lower())
            decomposition[op["token"]] = [f"<|fam:{parts[0].lower()}|>", f"<|verb:{parts[1].lower()}|>",
                                          f"<|obj:{parts[2].lower()}|>"]
    tokens = ([f"<|fam:{f.lower()}|>" for f in sorted(fams)] + [f"<|verb:{v}|>" for v in sorted(verbs)] +
              [f"<|obj:{o}|>" for o in sorted(objs)])
    fused = {op["token"] for op in ops}
    assert not fused & set(tokens), "compositional tokens collide with fused tokens"
    assert len(tokens) == len(set(tokens))
    return {"version": 1, "families": len(fams), "verbs": len(verbs), "objects": len(objs),
            "tokens": tokens, "decomposition": decomposition}


def main() -> int:
    data = build()
    out = Path(__file__).with_name("compositional.json")
    out.write_text(json.dumps(data, indent=1))
    print(json.dumps({k: data[k] for k in ("families", "verbs", "objects")} | {"tokens": len(data["tokens"]),
                                                                              "decomposed_ops": len(data["decomposition"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

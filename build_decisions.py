#!/usr/bin/env python3
"""Turn exported episodes into typed decision examples and train a tokenizer.

    .venv/bin/python build_decisions.py --episodes data/episodes.jsonl --out data/decisions

Each tool step (and each episode's final answer) becomes one decision:
  state   goal + the most recent steps (tool, ok, truncated args/result)
  options shuffled candidates: the action actually taken, sampled
          alternatives, and FINAL (answer and stop)
  labels  choice  index of the action taken
          noul    did that tool call succeed (masked for FINAL)
          value   did the episode end in a final answer
Splits are by session so no session appears in both train and validation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["[PAD]", "[EOS]", "[GOAL]", "[STEP]", "[OPTS]", "[OPT]", "[DECIDE]"]
# Op-codes for agent mechanics (opcodes.py) are reserved as single tokens.
_OPCODES = Path(__file__).with_name("opcodes.json")
if _OPCODES.exists():
    SPECIAL += [op["token"] for op in json.loads(_OPCODES.read_text())["ops"]]
FINAL = "FINAL"
HISTORY, OPTIONS = 6, 16
GOAL_CHARS, ARG_CHARS, RESULT_CHARS = 800, 160, 160


def clip(text, n):
    text = (text or "").replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def state_text(goal, steps):
    parts = [f"[GOAL] {clip(goal, GOAL_CHARS)}"]
    for s in steps[-HISTORY:]:
        ok = "?" if s.get("ok") is None else ("1" if s["ok"] else "0")
        parts.append(f"[STEP] {s.get('tool')} ok={ok} args={clip(s.get('args'), ARG_CHARS)} "
                     f"result={clip(s.get('result'), RESULT_CHARS)}")
    return " ".join(parts)


def split_of(session: str) -> str:
    return "val" if int(hashlib.sha256((session or "").encode()).hexdigest(), 16) % 20 == 0 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--max-steps-per-episode", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    episodes = [json.loads(line) for line in open(args.episodes)]
    freq = collections.Counter(s["tool"] for e in episodes for s in e["steps"] if s.get("tool"))
    pool = [t for t, _ in freq.most_common(96)]
    weights = [freq[t] ** 0.5 for t in pool]  # flattened so rare tools appear as negatives
    args.out.mkdir(parents=True, exist_ok=True)
    files = {k: open(args.out / f"{k}.jsonl", "w") for k in ("train", "val")}
    counts = collections.Counter()
    texts_for_tokenizer = []
    for ep in episodes:
        split = split_of(ep.get("session"))
        reached = ep.get("end") == "final_answer"
        steps = ep["steps"][: args.max_steps_per_episode]
        decisions = [(i, s["tool"]) for i, s in enumerate(steps) if s.get("tool")]
        if reached:
            decisions.append((len(steps), FINAL))
        for i, gold in decisions:
            cands = {gold, FINAL}
            while len(cands) < OPTIONS:
                cands.add(rng.choices(pool, weights)[0])
            options = list(cands)
            rng.shuffle(options)
            ok = steps[i].get("ok") if gold != FINAL else None
            example = {"state": state_text(ep["goal"], steps[:i]), "options": options,
                       "choice": options.index(gold),
                       "noul": None if ok is None else int(bool(ok)), "value": int(reached),
                       "prev_tool": steps[i - 1]["tool"] if i > 0 else None, "gold": gold}
            files[split].write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[split] += 1
            if split == "train" and rng.random() < 0.25:
                texts_for_tokenizer.append(example["state"] + " " + " ".join(options))
    for f in files.values():
        f.close()
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=SPECIAL, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts_for_tokenizer, trainer)
    tok.save(str(args.out / "tokenizer.json"))
    meta = {"train": counts["train"], "val": counts["val"], "options": OPTIONS, "history": HISTORY,
            "vocab": tok.get_vocab_size(), "tool_pool": len(pool), "episodes": len(episodes)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Mine higher-order abstractions for the constructed 100K tokenizer.

    python -m sys1.tokenizer.abstractions --per-source-mb 16 --top 2048
    python -m sys1.tokenizer.abstractions --apply runs/abstractions/candidates.json --out data/tokenizer-100k-abs

BPE learns frequent byte pairs. It does not learn recurring multi-token units
that agent text is built from: JSON argument shapes, role/op/tool call frames,
shell command prefixes, code idioms, stock phrases. This module finds them:

  1. tokenize a sample of each corpus source with the current tokenizer
  2. count token n-grams of length 2..MAX_N, growing only from frequent
     shorter n-grams (Apriori), never across document boundaries
  3. score gain = count x (n - 1) tokens saved
  4. suppress sub-patterns largely explained by a selected longer pattern
  5. measure real compression on held-out text with greedy longest match

--apply writes a new tokenizer whose reserved future slots carry the selected
abstractions as added tokens. Every existing token id stays the same, so
models trained on the base tokenizer keep working; only the reserved slots
gain meaning.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

from sys1.paths import DATA, RUNS
from sys1.tokenizer import spec

MAX_N = 10  # longer recurring spans are templates, not abstractions
SOURCES = ("en", "code", "agent", "private")
# Synthetic agent text is templated and would dominate raw counts; real
# trajectories carry the most signal for an agent tokenizer.
SOURCE_WEIGHT = {"en": 0.75, "code": 0.75, "agent": 0.0, "private": 1.0}
MIN_DOCS = 100  # a pattern must recur across this many distinct documents
ESCAPED = {"en", "code", "agent"}  # one doc per line, newlines escaped


def _byte_level(text: str) -> str:
    """GPT-2 byte-to-unicode form used by ByteLevel BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    table = dict(zip(bs, map(chr, cs)))
    return "".join(table[b] for b in text.encode("utf-8"))


def unescape(line: str) -> str:
    return line.replace("\\n", "\n").replace("\\r", "\r").replace("\\\\", "\\")


def read_docs(source: str, budget: int, heldout: bool) -> list[str]:
    root = DATA / "corpus" / source
    files = sorted(root.glob("heldout*.txt") if heldout else
                   [p for p in root.glob("*.txt") if not p.name.startswith("heldout")])
    docs, used = [], 0
    for path in files:
        if source in ESCAPED:
            with open(path, errors="replace") as f:
                for line in f:
                    text = unescape(line.rstrip("\n"))
                    if source == "code":
                        text = re.sub(r"^<lang=[^>]+> ", "", text)
                    docs.append(text)
                    used += len(text)
                    if used >= budget:
                        return docs
        else:  # private: [BOS] ... [EOS] blocks
            buf = []
            with open(path, errors="replace") as f:
                for line in f:
                    buf.append(line)
                    if line.rstrip().endswith("[EOS]"):
                        text = "".join(buf)
                        buf = []
                        docs.append(text)
                        used += len(text)
                        if used >= budget:
                            return docs
    return docs


def mine(docs: list[tuple[str, list[int]]], min_count: int, max_n: int) -> dict[tuple, list]:
    """Apriori n-gram counting with document frequency and source-weighted
    counts. Returns gram -> [raw_count, weighted_count, doc_freq]. An n-gram is
    counted only if its (n-1)-prefix and (n-1)-suffix were frequent."""
    def count(n: int, prev: set | None) -> dict[tuple, list]:
        stats: dict[tuple, list] = {}
        for d, (src, s) in enumerate(docs):
            w = SOURCE_WEIGHT.get(src, 1.0)
            for i in range(len(s) - n + 1):
                gram = tuple(s[i:i + n])
                if prev is not None and (gram[:-1] not in prev or gram[1:] not in prev):
                    continue
                st = stats.get(gram)
                if st is None:
                    stats[gram] = [1, w, 1, d]
                else:
                    st[0] += 1
                    st[1] += w
                    if st[3] != d:
                        st[2] += 1
                        st[3] = d
        return {g: st[:3] for g, st in stats.items() if st[0] >= min_count and st[2] >= MIN_DOCS}

    frequent = count(2, None)
    result = dict(frequent)
    n = 2
    while frequent and n < max_n:
        n += 1
        frequent = count(n, set(frequent))
        result.update(frequent)
    return result


def categorize(text: str) -> str:
    if "<|" in text or "[" in text and "]" in text and any(t in text for t in ("[GOAL]", "[STEP]", "[OPT]", "[DECIDE]")):
        return "frame"        # role / op / tool / control token sequences
    if re.search(r'["{}\[\]:,]', text) and re.search(r'"\w+"\s*:', text):
        return "json"
    if re.match(r"\s*(git|cd|ls|cat|grep|rg|sed|python3?|make|uv|pip|npm|curl|echo)\b", text):
        return "shell"
    if re.search(r"(def |return |import |self\.|=>|;\s*$|\)\s*\{|#include|const |let |fn )", text):
        return "code"
    if re.fullmatch(r"[\s\W]+", text):
        return "whitespace_punct"
    return "phrase"


def _overlaps(a: tuple, b: tuple) -> bool:
    """True when a shifted window of one pattern covers most of the other."""
    k = min(len(a), len(b))
    for m in range(max(2, k // 2 + 1), k + 1):
        if a[-m:] == b[:m] or b[-m:] == a[:m]:
            return True
    return False


def select(grams: dict[tuple, list], tok: Tokenizer, top: int, reserved_ids: set[int]) -> list[dict]:
    scored = sorted(((st[1] * (len(g) - 1), g, st) for g, st in grams.items()), reverse=True)
    chosen, chosen_meta = [], []
    for gain, g, (raw, weighted, df) in scored:
        if len(chosen) >= top:
            break
        text = tok.decode(list(g), skip_special_tokens=False)
        if not text.strip() or len(text) > 80:
            continue
        # A string the tokenizer already has as one token would duplicate a vocab entry.
        if len(tok.encode(text).ids) <= 1 or text in tok.get_vocab():
            continue
        redundant = False
        for cg, ctext, craw in chosen_meta:
            similar = raw <= 1.5 * craw
            if similar and (text in ctext or _overlaps(g, cg)):
                redundant = True
                break
        if redundant:
            continue
        chosen.append({"ids": list(g), "text": text, "n": len(g), "count": raw, "doc_freq": df,
                       "weighted_gain": round(gain, 1), "category": categorize(text),
                       "has_reserved": any(i in reserved_ids for i in g)})
        chosen_meta.append((g, text, raw))
    return chosen


def compress(seqs: list[list[int]], patterns: list[list[int]]) -> int:
    """Greedy longest-match replacement; returns resulting token count."""
    by_first = collections.defaultdict(list)
    for p in patterns:
        by_first[p[0]].append(p)
    for v in by_first.values():
        v.sort(key=len, reverse=True)
    total = 0
    for s in seqs:
        i = 0
        while i < len(s):
            for p in by_first.get(s[i], ()):
                if s[i:i + len(p)] == p:
                    i += len(p)
                    break
            else:
                i += 1
            total += 1
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", type=Path, default=DATA / "tokenizer-100k" / "tokenizer.json")
    ap.add_argument("--per-source-mb", type=float, default=16)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--top", type=int, default=spec.FUTURE_SLOTS)
    ap.add_argument("--max-n", type=int, default=MAX_N)
    ap.add_argument("--out-dir", type=Path, default=RUNS / "abstractions")
    ap.add_argument("--apply", type=Path, help="candidates.json to write into reserved slots")
    ap.add_argument("--out", type=Path, default=DATA / "tokenizer-100k-abs")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(a.tokenizer))
    reserved = spec.reserved_tokens()
    reserved_ids = {tok.token_to_id(t) for t in reserved if tok.token_to_id(t) is not None}

    if a.apply:
        cands = json.loads(a.apply.read_text())["candidates"]
        future = [t for t in reserved if t.startswith("<|reserved_")]
        assert len(cands) <= len(future), "more abstractions than reserved slots"
        data = json.loads(a.tokenizer.read_text())
        by_content = {t["content"]: t for t in data["added_tokens"]}
        model_vocab = data["model"]["vocab"]
        mapping = []
        for slot, cand in zip(future, cands):
            entry = by_content[slot]
            # Reserved tokens live in both the added-token table and the model
            # vocabulary; rename both so the slot is not counted twice.
            if slot in model_vocab:
                model_vocab[cand["text"]] = model_vocab.pop(slot)  # added tokens are looked up by raw content
            entry["content"] = cand["text"]
            entry["special"] = False  # matched in ordinary text, decoded verbatim
            mapping.append({"id": entry["id"], "slot": slot, "text": cand["text"]})
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False))
        (a.out / "abstractions.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1))
        new = Tokenizer.from_file(str(a.out / "tokenizer.json"))
        assert new.get_vocab_size() == tok.get_vocab_size(), "vocab size changed"
        print(json.dumps({"applied": len(mapping), "vocab": new.get_vocab_size(), "out": str(a.out)}))
        return 0

    budget = int(a.per_source_mb * 1e6)
    train, held = {}, {}
    for src in SOURCES:
        docs = read_docs(src, budget, heldout=False)
        hdocs = read_docs(src, budget // 4, heldout=True)
        if not docs:
            continue
        train[src] = [(src, e.ids) for e in tok.encode_batch(docs)]
        held[src] = [e.ids for e in tok.encode_batch(hdocs)]
        print(json.dumps({"source": src, "docs": len(docs), "tokens": sum(len(x) for _, x in train[src])}), flush=True)

    # Weight-0 sources (the templated synthetic agent corpus) are measured, not mined.
    all_docs = [d for src, v in train.items() if SOURCE_WEIGHT.get(src, 1.0) > 0 for d in v]
    grams = mine(all_docs, a.min_count, a.max_n)
    cands = select(grams, tok, a.top, reserved_ids)
    # which sources each candidate comes from (presence in a sample of each source)
    patterns = [c["ids"] for c in cands]
    report = {"tokenizer": str(a.tokenizer), "frequent_ngrams": len(grams), "selected": len(cands),
              "by_category": dict(collections.Counter(c["category"] for c in cands)),
              "held_out": {}}
    for src, seqs in held.items():
        base = sum(map(len, seqs))
        after = compress(seqs, patterns)
        report["held_out"][src] = {"tokens_before": base, "tokens_after": after,
                                   "reduction": round(1 - after / base, 4) if base else None}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "candidates.json").write_text(json.dumps({"candidates": cands}, ensure_ascii=False, indent=1))
    (a.out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    for c in cands[:25]:
        print(f"{c['weighted_gain']:>9} x{c['count']:<7} df={c['doc_freq']:<6} n={c['n']:<2} {c['category']:<16} {c['text'][:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare tokenizers on the held-out corpus slices.

Metrics per tokenizer and slice (data/corpus/{en,code,agent,private}/heldout.txt):
  bytes_per_token     UTF-8 bytes / tokens (higher = better compression)
  tokens_per_1k_chars tokens per 1,000 characters (lower = better)
  roundtrip_pct       % of documents where decode(encode(doc)) == doc
plus, per tokenizer, the fraction of reserved op-code / compositional / tool / control+role
tokens that encode as exactly one id, and a per-language breakdown for the code slice.

Candidates: data/tokenizer-100k/tokenizer.json (if present), data/decisions/tokenizer.json (32K),
tiktoken cl100k_base and o200k_base. Writes runs/tokenizer-eval/report.json.

    python -m sys1.tokenizer.evaluate [--corpus DIR] [--max-docs N] [--max-chars N]

Held-out formats follow the corpus builders (see load_docs); private uses heldout-*.txt shards.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from sys1.paths import DATA, RUNS
from sys1.tokenizer import spec

SLICES = ["en", "code", "agent", "private"]


# ---------------------------------------------------------------- tokenizer adapters

class HFTok:
    def __init__(self, name: str, path: Path):
        from tokenizers import Tokenizer
        self.name, self.path = name, str(path)
        self.tok = Tokenizer.from_file(str(path))
        self.vocab_size = self.tok.get_vocab_size(with_added_tokens=True)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def decode_batch(self, ids: list[list[int]]) -> list[str]:
        return self.tok.decode_batch(ids, skip_special_tokens=False)

    def encode_one(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids


class TiktokenTok:
    def __init__(self, name: str):
        import tiktoken
        self.name, self.path = name, f"tiktoken:{name}"
        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        # disallowed_special=() treats "<|endoftext|>" etc. in data as plain text
        return self.enc.encode_ordinary_batch(texts)

    def decode_batch(self, ids: list[list[int]]) -> list[str]:
        return self.enc.decode_batch(ids)

    def encode_one(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)


def load_tokenizers(include_missing: bool = False) -> tuple[list, dict]:
    toks, skipped = [], {}
    for name, path in [("sys1-100k", DATA / "tokenizer-100k" / "tokenizer.json"),
                       ("sys1-32k", DATA / "decisions" / "tokenizer.json")]:
        if path.exists():
            try:
                toks.append(HFTok(name, path))
            except Exception as e:  # half-written file from a concurrent trainer
                skipped[name] = f"load failed: {e}"
        else:
            skipped[name] = f"missing: {path}"
    for name in ["cl100k_base", "o200k_base"]:
        try:
            toks.append(TiktokenTok(name))
        except Exception as e:
            skipped[name] = f"load failed: {e}"
    return toks, skipped


# ---------------------------------------------------------------- corpus loading

_LANG_HINTS = [  # (lang, regex) checked in order; crude fallback when no label is present
    ("python", r"^\s*(def |class |import |from \S+ import )|\bself\.|elif "),
    ("rust", r"\bfn \w+\(|\blet mut\b|\bimpl\b|::new\(|pub fn"),
    ("go", r"^package \w+|\bfunc \w+\(|:= "),
    ("c/c++", r"#include\s*[<\"]|\bstd::|\bprintf\(|->\w+|\bsize_t\b"),
    ("javascript/ts", r"\bconst \w+ = |=> \{|\bfunction \w+\(|console\.log|\binterface \w+ \{"),
    ("java", r"\bpublic (static )?(class|void)\b|System\.out"),
    ("shell", r"^#!/bin/(ba)?sh|^\s*(echo|export|fi|done)\b"),
    ("erlang/elixir", r"-module\(|\bdefmodule\b|\bdefp? \w+"),
    ("json/yaml", r"^\s*[\{\[]\s*\"|^\w[\w-]*:\s"),
]


def guess_lang(text: str) -> str:
    head = text[:4000]
    for lang, rx in _LANG_HINTS:
        if re.search(rx, head, re.M):
            return lang
    return "unknown"


def unescape(line: str) -> str:
    """Inverse of the corpus builders' escape: \\\\ -> backslash, \\n -> LF, \\r -> CR."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n:
            nxt = line[i + 1]
            out.append({"n": "\n", "r": "\r", "\\": "\\"}.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


_LANG_PREFIX = re.compile(r"^<lang=([^>]{1,40})> ")


def heldout_files(corpus: Path, sl: str) -> list[Path]:
    d = corpus / sl
    single = d / "heldout.txt"
    return [single] if single.exists() else sorted(d.glob("heldout*.txt"))


def load_docs(paths: list[Path], sl: str, max_docs: int, max_chars: int) -> list[dict]:
    """Return [{"text":..., "lang":...|None}], capped by doc count and total chars.

    Formats (matching sys1/tokenizer/corpus_*.py): JSONL with "text"; en/code one escaped doc
    per line (code prefixed "<lang=TAG> "); agent one sample per line with only newlines escaped;
    private multi-line episodes terminated by "[EOS]".
    """
    docs, total = [], 0

    def add(text: str, lang=None) -> bool:
        nonlocal total
        if text:
            docs.append({"text": text, "lang": lang})
            total += len(text)
        return len(docs) < max_docs and total < max_chars

    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            first = f.readline()
            f.seek(0)
            try:
                o = json.loads(first)
                is_jsonl = isinstance(o, dict) and "text" in o
            except Exception:
                is_jsonl = False
            if is_jsonl:
                for line in f:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    lang = o.get("lang") or o.get("language") or (o.get("meta") or {}).get("lang")
                    if not add(o.get("text") or "", lang):
                        return docs
            elif sl == "private":
                buf = []
                for line in f:
                    buf.append(line)
                    if "[EOS]" in line:
                        if not add("".join(buf).strip("\n")):
                            return docs
                        buf = []
                if buf and not add("".join(buf).strip("\n")):
                    return docs
            else:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    lang = None
                    if sl == "code":
                        m = _LANG_PREFIX.match(line)
                        if m:
                            lang, line = m.group(1), line[m.end():]
                    text = line.replace("\\n", "\n") if sl == "agent" else unescape(line)
                    if not add(text, lang):
                        return docs
    return docs


# ---------------------------------------------------------------- metrics

def measure(tok, docs: list[dict], batch: int = 256) -> dict:
    n_tok = n_bytes = n_chars = ok = 0
    by_lang = defaultdict(lambda: [0, 0, 0, 0, 0])  # docs, tokens, bytes, chars, ok
    t0 = time.time()
    for i in range(0, len(docs), batch):
        chunk = docs[i:i + batch]
        texts = [d["text"] for d in chunk]
        ids = tok.encode_batch(texts)
        dec = tok.decode_batch(ids)
        for d, t, x, y in zip(chunk, texts, ids, dec):
            b, c, good = len(t.encode("utf-8")), len(t), int(x is not None and y == t)
            n_tok += len(x); n_bytes += b; n_chars += c; ok += good
            if d.get("lang") is not None:
                s = by_lang[d["lang"]]
                s[0] += 1; s[1] += len(x); s[2] += b; s[3] += c; s[4] += good
    out = _summ(len(docs), n_tok, n_bytes, n_chars, ok)
    out["seconds"] = round(time.time() - t0, 2)
    if by_lang:
        out["by_lang"] = {k: _summ(*v) for k, v in sorted(by_lang.items(), key=lambda kv: -kv[1][2])}
    return out


def _summ(docs, toks, nbytes, chars, ok) -> dict:
    return {"docs": docs, "tokens": toks, "bytes": nbytes, "chars": chars,
            "bytes_per_token": round(nbytes / toks, 4) if toks else None,
            "tokens_per_1k_chars": round(1000 * toks / chars, 2) if chars else None,
            "roundtrip_pct": round(100 * ok / docs, 3) if docs else None}


def reserved_groups() -> dict[str, list[str]]:
    groups = {"control_roles": spec.CONTROL + spec.ROLES,
              "opcodes": spec._ops(),
              "compositional": spec._compositional(),
              "tools": spec._tools()}
    return {k: list(dict.fromkeys(v)) for k, v in groups.items()}


def single_id(tok, groups: dict[str, list[str]]) -> dict:
    out = {}
    for g, toks in groups.items():
        if not toks:
            out[g] = {"n": 0, "single_id_frac": None}
            continue
        lens = [len(tok.encode_one(t)) for t in toks]
        one = sum(1 for n in lens if n == 1)
        out[g] = {"n": len(toks), "single_id": one, "single_id_frac": round(one / len(toks), 4),
                  "mean_ids": round(sum(lens) / len(lens), 3),
                  "examples_multi": [t for t, n in zip(toks, lens) if n != 1][:3]}
    return out


# ---------------------------------------------------------------- report

def fmt(v, w=8, p=2):
    return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def print_table(report: dict) -> None:
    names = [t["name"] for t in report["tokenizers"]]
    print(f"\n== corpus slices (B/tok higher=better, tok/1kc lower=better, RT% roundtrip) ==")
    print(f"{'slice':<9}{'tokenizer':<13}{'docs':>7}{'B/tok':>8}{'tok/1kc':>9}{'RT%':>8}")
    for sl, res in report["slices"].items():
        for n in names:
            r = res.get(n)
            if r:
                print(f"{sl:<9}{n:<13}{r['docs']:>7}{fmt(r['bytes_per_token'])}"
                      f"{fmt(r['tokens_per_1k_chars'], 9, 1)}{fmt(r['roundtrip_pct'], 8, 1)}")
    code = report["slices"].get("code") or {}
    langs = {}
    for n in names:
        for lang, r in (code.get(n, {}).get("by_lang") or {}).items():
            langs.setdefault(lang, {})[n] = r
    if langs:
        print("\n== code by language: bytes/token ==")
        print(f"{'lang':<16}{'docs':>6}" + "".join(f"{n:>13}" for n in names))
        for lang, per in list(langs.items())[:20]:
            d = next(iter(per.values()))["docs"]
            print(f"{lang:<16}{d:>6}" + "".join(fmt(per.get(n, {}).get("bytes_per_token"), 13) for n in names))
    print("\n== reserved tokens encoding as exactly one id ==")
    groups = list(report["reserved_groups"].keys())
    print(f"{'tokenizer':<13}" + "".join(f"{g + '(' + str(report['reserved_groups'][g]) + ')':>22}" for g in groups))
    for t in report["tokenizers"]:
        s = report["single_id"][t["name"]]
        print(f"{t['name']:<13}" + "".join(fmt(None if s[g]['single_id_frac'] is None else 100 * s[g]['single_id_frac'], 21, 1) + "%"
                                          for g in groups))
    if report["skipped"]:
        print("\nskipped:", json.dumps(report["skipped"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=DATA / "corpus")
    ap.add_argument("--out", type=Path, default=RUNS / "tokenizer-eval" / "report.json")
    ap.add_argument("--max-docs", type=int, default=20000, help="per slice")
    ap.add_argument("--max-chars", type=int, default=10_000_000, help="per slice")
    ap.add_argument("--require-100k", action="store_true", help="exit 2 if the 100K tokenizer is absent")
    a = ap.parse_args(argv)

    toks, skipped = load_tokenizers()
    if a.require_100k and not any(t.name == "sys1-100k" for t in toks):
        print("sys1-100k tokenizer not found", file=sys.stderr)
        return 2
    groups = reserved_groups()
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "corpus": str(a.corpus),
              "limits": {"max_docs": a.max_docs, "max_chars": a.max_chars},
              "tokenizers": [{"name": t.name, "source": t.path, "vocab_size": t.vocab_size} for t in toks],
              "skipped": skipped, "reserved_groups": {g: len(v) for g, v in groups.items()},
              "slices": {}, "single_id": {}}
    for sl in SLICES:
        paths = heldout_files(a.corpus, sl)
        if not paths:
            report["slices"][sl] = {"missing": str(a.corpus / sl / "heldout*.txt")}
            continue
        docs = load_docs(paths, sl, a.max_docs, a.max_chars)
        if sl == "code":
            for d in docs:
                d["lang"] = d["lang"] or guess_lang(d["text"])
        else:
            for d in docs:
                d["lang"] = None
        report["slices"][sl] = {t.name: measure(t, docs) for t in toks}
        report["slices"][sl]["_source"] = {"paths": [str(p) for p in paths], "docs": len(docs),
                                           "chars": sum(len(d["text"]) for d in docs)}
        del docs
    for t in toks:
        report["single_id"][t.name] = single_id(t, groups)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print_table(report)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train the constructed sys1 tokenizer (byte-level BPE + reserved region).

Layout: ids [0, len(reserved)) are sys1.tokenizer.spec.reserved_tokens() in spec
order, then the 256-byte ByteLevel alphabet, then learned merges, for exactly
spec.VOCAB_SIZE ids in total.

Corpus: a weighted mixture streamed from data/corpus/{en,code,agent,private}/*.txt
(heldout*.txt excluded), bounded by a total byte cap. Each source's share of the
cap is its renormalised weight. When a source holds more bytes than its share,
newline-aligned blocks are kept with probability share/total (seeded), so the
sample spans the whole source instead of its prefix.

  python -m sys1.tokenizer.train_bpe                       # full run -> data/tokenizer-100k
  python -m sys1.tokenizer.train_bpe --vocab-size 20000 --files a.txt b.txt --out runs/tok-test
"""
from __future__ import annotations

import os

os.environ.setdefault("RAYON_RS_NUM_CPUS", "4")  # must precede the tokenizers import

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from sys1.paths import DATA
from sys1.tokenizer import spec

# cl100k/o200k-style split: contractions, letter runs (with one optional leading
# non-letter/non-digit), digits in groups of 1-3, punctuation runs with trailing
# newlines, newline runs, and whitespace runs that leave the last space for the
# following word.
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

SOURCES = ("en", "code", "agent", "private")
DEFAULT_WEIGHTS = {"en": 0.40, "code": 0.30, "agent": 0.15, "private": 0.15}
BLOCK_BYTES = 1 << 16  # read granularity; blocks are extended to the next newline
MAX_EXTEND = 1 << 18  # longest newline search past a block boundary
BATCH_BYTES = 1 << 20  # raw bytes of documents handed to the trainer per iterator item


def available_gb() -> float:
    """free + inactive + speculative pages (macOS vm_stat); inf if unavailable."""
    try:
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return float("inf")
    page, pages = 16384, 0
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        for key in ("Pages free", "Pages inactive", "Pages speculative"):
            if line.startswith(key + ":"):
                pages += int(line.split(":")[1].strip().rstrip("."))
    return pages * page / 1e9


def source_files(corpus: Path, name: str) -> list[Path]:
    d = corpus / name
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.txt") if not p.name.startswith("heldout") and p.stat().st_size > 0)


def iter_blocks(path: Path):
    """Newline-aligned byte blocks of ~BLOCK_BYTES."""
    with open(path, "rb") as f:
        while True:
            block = f.read(BLOCK_BYTES)
            if not block:
                return
            if not block.endswith(b"\n"):
                block += f.readline(MAX_EXTEND)  # bounded so huge lines cannot balloon a block
            yield block


_ESC = re.compile(r"\\(.)", re.S)
_ESC_MAP = {"n": "\n", "r": "\r", "\\": "\\"}
_LANG_PREFIX = re.compile(r"^<lang=[^>\s]*> ")
# Reserved-token strings already present in agent/private text are stripped so
# the BPE does not spend merges on fragments like "<|" and "|>"; at encode time
# these strings are matched whole as special tokens anyway.
_RESERVED_RE = re.compile(r"<\|[A-Za-z0-9_:./\-]{1,64}\|>|\[(?:" + "|".join(
    re.escape(t[1:-1]) for t in spec.CONTROL) + r")\]")


def unescape(line: str) -> str:
    """Inverse of the corpus line escaping: \\ -> backslash, \\n -> LF, \\r -> CR."""
    if "\\" not in line:
        return line
    return _ESC.sub(lambda m: _ESC_MAP.get(m.group(1), m.group(0)), line)


def line_escaped(corpus_dir: Path) -> bool:
    """True when the source MANIFEST declares one escaped document per line."""
    try:
        fmt = json.loads((corpus_dir / "MANIFEST.json").read_text()).get("format", "")
    except (OSError, ValueError):
        return False
    return "\\n" in fmt or "escape" in fmt


class Mixture:
    """Streams a weighted, byte-capped sample of documents; records bytes emitted."""

    def __init__(self, groups: dict[str, list[Path]], weights: dict[str, float], byte_cap: int, seed: int,
                 escaped: dict[str, bool] | None = None):
        present = {k: v for k, v in groups.items() if v and weights.get(k, 0) > 0}
        total_w = sum(weights[k] for k in present)
        self.groups = present
        self.share = {k: int(byte_cap * weights[k] / total_w) for k in present}
        self.available = {k: sum(p.stat().st_size for p in v) for k, v in present.items()}
        self.emitted = {k: 0 for k in present}
        self.escaped = {k: bool((escaped or {}).get(k)) for k in present}
        self.seed = seed

    def _docs(self, name: str, block: bytes) -> list[str]:
        text = block.decode("utf-8", errors="replace")
        if self.escaped[name]:
            docs = [unescape(_LANG_PREFIX.sub("", ln)) for ln in text.split("\n") if ln]
        else:
            docs = [text]
        return [_RESERVED_RE.sub("", d) for d in docs]

    def __iter__(self):
        buf, size = [], 0
        for name, files in self.groups.items():
            share, avail = self.share[name], self.available[name]
            keep = min(1.0, 1.1 * share / avail) if avail else 0.0  # slight oversample; the cap trims
            rng = random.Random(f"{self.seed}:{name}")
            for path in files:
                for block in iter_blocks(path):
                    if self.emitted[name] >= share:
                        break
                    if keep < 1.0 and rng.random() >= keep:
                        continue
                    self.emitted[name] += len(block)
                    buf.extend(self._docs(name, block))
                    size += len(block)
                    if size >= BATCH_BYTES:
                        yield buf
                        buf, size = [], 0
                if self.emitted[name] >= share:
                    break
            print(f"[mix] {name}: {self.emitted[name]:,} / share {share:,} bytes "
                  f"(available {avail:,}, keep p={keep:.3f}, escaped={self.escaped[name]})",
                  file=sys.stderr, flush=True)
        if buf:
            yield buf


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train(args) -> dict:
    reserved = spec.reserved_tokens()
    vocab_size = args.vocab_size or spec.VOCAB_SIZE
    budget = vocab_size - len(reserved)
    if budget < 256:
        raise SystemExit(f"vocab {vocab_size} leaves {budget} BPE slots (< 256-byte alphabet)")

    weights = dict(DEFAULT_WEIGHTS)
    for item in args.weights or []:
        k, v = item.split("=")
        weights[k] = float(v)
    if args.files:
        groups = {"files": [Path(p) for p in args.files]}
        weights = {"files": 1.0}
    else:
        groups = {name: source_files(Path(args.corpus), name) for name in SOURCES}
    escaped = {k: line_escaped(Path(args.corpus) / k) for k in groups} if not args.files else {}
    missing = [k for k, v in groups.items() if not v]
    mix = Mixture(groups, weights, int(args.byte_cap), args.seed, escaped)
    if not mix.groups:
        raise SystemExit("no corpus text found")

    gb, deadline = available_gb(), time.time() + 60 * args.wait_mem_minutes
    while gb < args.min_free_gb and time.time() < deadline:
        print(f"[mem] {gb:.2f} GB available < {args.min_free_gb}; waiting", file=sys.stderr, flush=True)
        time.sleep(60)
        gb = available_gb()
    print(f"[mem] available {gb:.2f} GB; sources {list(mix.groups)}; missing {missing}; "
          f"shares {mix.share}", file=sys.stderr, flush=True)
    if gb < args.min_free_gb:
        raise SystemExit(f"only {gb:.2f} GB available (< {args.min_free_gb}); refusing to train")

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=reserved,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        max_token_length=args.max_token_length,
    )
    t0 = time.time()
    tok.train_from_iterator(mix, trainer=trainer)
    elapsed = time.time() - t0

    size = tok.get_vocab_size(with_added_tokens=True)
    assert size == vocab_size, f"vocab size {size} != target {vocab_size} (corpus too small for the merge budget?)"
    for i, t in enumerate(reserved):
        assert tok.token_to_id(t) == i, f"reserved token {t!r} at id {tok.token_to_id(t)}, expected {i}"
    probe = "def f(x):\n    return x + 12345  # it's <|op:ctrl.start|> héllo"
    assert tok.decode(tok.encode(probe).ids, skip_special_tokens=False) == probe, "round-trip failed"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "tokenizer.json"
    tok.save(str(tok_path))
    sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    config = {
        "vocab_size": size,
        "reserved_count": len(reserved),
        "bpe_budget": budget,
        "spec_vocab_size": spec.VOCAB_SIZE,
        "weights": {k: weights[k] for k in mix.groups},
        "byte_cap": int(args.byte_cap),
        "bytes_per_source": mix.emitted,
        "bytes_available_per_source": mix.available,
        "missing_sources": missing,
        "line_escaped_sources": {k: v for k, v in mix.escaped.items() if v},
        "files": {k: [str(p) for p in v] for k, v in mix.groups.items()},
        "split_pattern": SPLIT_PATTERN,
        "min_frequency": args.min_frequency,
        "max_token_length": args.max_token_length,
        "seed": args.seed,
        "train_seconds": round(elapsed, 1),
        "tokenizer_sha256": sha,
        "tokenizers_version": __import__("tokenizers").__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(DATA / "corpus"))
    ap.add_argument("--out", default=str(DATA / "tokenizer-100k"))
    ap.add_argument("--weights", nargs="*", help="override as source=weight, e.g. en=0.5 code=0.3")
    ap.add_argument("--byte-cap", type=float, default=1.2e9)
    ap.add_argument("--vocab-size", type=int, default=0, help="test override; default spec.VOCAB_SIZE")
    ap.add_argument("--files", nargs="*", help="train on these files instead of the corpus mixture")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--max-token-length", type=int, default=32)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--wait-mem-minutes", type=float, default=0, help="poll this long for free memory")
    ap.add_argument("--seed", type=int, default=0)
    cfg = train(ap.parse_args(argv))
    print(json.dumps({k: cfg[k] for k in ("vocab_size", "reserved_count", "bpe_budget", "bytes_per_source",
                                         "missing_sources", "train_seconds", "tokenizer_sha256")}))


if __name__ == "__main__":
    main()

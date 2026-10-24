"""Stream a permissively licensed English corpus into plain-text shards.

Source: HuggingFaceFW/fineweb-edu, config ``sample-10BT`` (license ODC-By 1.0,
per the dataset card front matter). Documents are streamed, never downloaded
whole, and written to ``data/corpus/en/``:

  en-00000.txt ...  one document per line
  heldout.txt       the last 5% of accepted documents (same format)
  MANIFEST.json     dataset id, config, license, counts, sha256 per file

Line format: each document is escaped so it fits on one line. A backslash
becomes ``\\\\``, a newline becomes ``\\n`` and a carriage return ``\\r``.
``unescape`` reverses this exactly.

Filtering: exact dedup on a hash of whitespace-normalized lowercased text, a
minimum length, and the dataset's own ``language_score`` threshold.
The byte cap counts every output text file, held-out docs included.

Run:  python -m sys1.tokenizer.corpus_en [--max-bytes N] [--shard-bytes N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from sys1.paths import DATA

DATASET_ID = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
LICENSE = "odc-by"
LICENSE_URL = "https://opendatacommons.org/licenses/by/1-0/"
OUT_DIR = DATA / "corpus" / "en"
MAX_BYTES = 600 * 1000 * 1000
SHARD_BYTES = 64 * 1000 * 1000
HELDOUT_FRACTION = 0.05
MIN_CHARS = 200
MIN_LANG_SCORE = 0.9

_WS = re.compile(r"\s+")


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def unescape(line: str) -> str:
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


def _dedup_key(text: str) -> bytes:
    return hashlib.blake2b(_WS.sub(" ", text).strip().lower().encode(), digest_size=16).digest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stream():
    from datasets import load_dataset
    return load_dataset(DATASET_ID, name=CONFIG, split="train", streaming=True)


def _split_heldout(shards: list[Path], n_heldout: int, heldout: Path) -> None:
    """Move the last n_heldout lines across the shard tail into heldout.txt."""
    tail: list[bytes] = []
    while n_heldout > len(tail) and shards:
        last = shards[-1]
        lines = last.read_bytes().splitlines(keepends=True)
        need = n_heldout - len(tail)
        if need >= len(lines):
            tail = lines + tail
            last.unlink()
            shards.pop()
        else:
            tail = lines[-need:] + tail
            last.write_bytes(b"".join(lines[:-need]))
    heldout.write_bytes(b"".join(tail))


def build(max_bytes: int = MAX_BYTES, shard_bytes: int = SHARD_BYTES, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("en-*.txt"):
        old.unlink()
    seen: set[bytes] = set()
    stats = {"streamed": 0, "dup": 0, "short": 0, "lang": 0}
    shards: list[Path] = []
    total = docs = cur_bytes = 0
    fh = None
    t0 = time.time()
    for row in _stream():
        stats["streamed"] += 1
        text = (row.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            stats["short"] += 1
            continue
        if (row.get("language_score") or 0.0) < MIN_LANG_SCORE:
            stats["lang"] += 1
            continue
        key = _dedup_key(text)
        if key in seen:
            stats["dup"] += 1
            continue
        line = (escape(text) + "\n").encode("utf-8")
        if total + len(line) > max_bytes:
            break
        seen.add(key)
        if fh is None or cur_bytes + len(line) > shard_bytes:
            if fh:
                fh.close()
            shards.append(out_dir / f"en-{len(shards):05d}.txt")
            fh = shards[-1].open("wb")
            cur_bytes = 0
        fh.write(line)
        cur_bytes += len(line)
        total += len(line)
        docs += 1
        if docs % 20000 == 0:
            rate = total / max(time.time() - t0, 1e-9) / 1e6
            print(f"[corpus_en] docs={docs} bytes={total} ({rate:.1f} MB/s) {stats}", file=sys.stderr, flush=True)
    if fh:
        fh.close()

    n_heldout = max(1, int(round(docs * HELDOUT_FRACTION)))
    heldout = out_dir / "heldout.txt"
    _split_heldout(shards, n_heldout, heldout)

    def entry(p: Path) -> dict:
        data_lines = sum(1 for _ in p.open("rb"))
        return {"file": p.name, "docs": data_lines, "bytes": p.stat().st_size, "sha256": _sha256(p)}

    shard_entries = [entry(p) for p in shards]
    held_entry = entry(heldout)
    manifest = {
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "split": "train",
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "license_source": f"https://huggingface.co/datasets/{DATASET_ID} (card front matter: license: odc-by)",
        "format": "one document per line; escapes: \\\\ -> backslash, \\n -> LF, \\r -> CR (see unescape())",
        "filters": {"min_chars": MIN_CHARS, "min_language_score": MIN_LANG_SCORE,
                    "dedup": "exact, blake2b-128 of whitespace-normalized lowercased text"},
        "max_bytes": max_bytes,
        "stream_stats": stats,
        "docs": docs,
        "bytes": total,
        "train": {"docs": sum(e["docs"] for e in shard_entries),
                  "bytes": sum(e["bytes"] for e in shard_entries), "shards": shard_entries},
        "heldout": {**held_entry, "fraction": HELDOUT_FRACTION, "selection": "last 5% of accepted docs in stream order"},
        "created_unix": int(time.time()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    m = build(a.max_bytes, a.shard_bytes, a.out)
    print(json.dumps({k: m[k] for k in ("docs", "bytes", "stream_stats")}
                     | {"train_docs": m["train"]["docs"], "heldout_docs": m["heldout"]["docs"],
                        "shards": len(m["train"]["shards"])}), flush=True)
    sys.stderr.flush()
    # Breaking out of a streaming IterableDataset leaves HF fetch threads alive that
    # block interpreter shutdown; all files are closed and the manifest is written.
    os._exit(0)


if __name__ == "__main__":
    main()

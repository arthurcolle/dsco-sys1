"""Backend-neutral corpus readers shared by the Torch and native MLX trainers."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import zlib

SNIFF_BYTES = 1 << 16       # bytes read to decide a .txt shard's layout

ENCODE_BATCH = 64          # documents tokenized per refill (tokenizers batches in Rust)

MAX_DOC_CHARS = 1 << 20    # cap a single runaway .txt document at 1 MiB

EPISODE_HELDOUT_MOD = 50   # 1 in 50 episode sessions goes to the fallback heldout set

def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" \
        else open(path, encoding="utf-8", errors="replace", newline="")

def _shard_kind(path: Path, txt_sep: str) -> str:
    name = path.name[:-3] if path.name.endswith(".gz") else path.name
    if name.endswith(".jsonl"):
        return "jsonl"
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        head = f.read(SNIFF_BYTES).decode("utf-8", "replace")
    first = head.split("\n", 1)[0]
    if "[EOS]" in first:   # whole [BOS]...[EOS] sample on one line: escaped layout
        return "lines"
    blocks = head.startswith("[BOS]") or "[EOS]\n\n[BOS]" in head \
        or f"\n{txt_sep}\n" in head or head.startswith(txt_sep + "\n")
    return "blocks" if blocks else "lines"

def _is_heldout(path: Path) -> bool:
    return path.name.startswith("heldout")

def _json_text(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, str):
        return obj
    text = obj.get("text") or obj.get("content")
    return text if isinstance(text, str) and text else None

def render_episode(ep: dict) -> str:
    """Plain-text view of one exported dsco episode for LM pretraining."""
    parts = [f"[GOAL] {ep.get('goal', '')}"]
    for s in ep.get("steps", []):
        status = "ok" if s.get("ok") else f"fail {s.get('failure') or ''}".strip()
        parts.append(f"[STEP] {s.get('tool')} {s.get('args', '')}\n[RESULT] {status}\n{(s.get('result') or '')[:4000]}")
    final = ep.get("final") or {}
    if final.get("text"):
        parts.append(f"[ANSWER] {final['text']}")
    return "\n".join(parts)

class DocReader:
    """Resumable document stream over a list of shard files, looping by epoch.

    Cursor = (epoch, file index, position). Position is a byte offset for plain
    files and a line count for gzip files (not cheaply seekable)."""

    def __init__(self, files: list[Path], txt_sep: str, episode_filter=None, repeat: bool = True):
        self.files, self.txt_sep, self.episode_filter = files, txt_sep, episode_filter
        self.repeat = repeat
        self.epoch, self.fi, self.pos = 0, 0, 0
        self._fh, self._gz = None, False
        self.kinds = [None if episode_filter else _shard_kind(f, txt_sep) for f in files]

    def state(self) -> dict:
        f = self.files[self.fi]
        return {"epoch": self.epoch, "fi": self.fi, "pos": self.pos, "file": str(f), "size": f.stat().st_size}

    def load_state(self, st: dict):
        """Restore by file path, so shards added to the corpus later do not shift the cursor."""
        self.close()
        self.epoch, self.fi, self.pos = st["epoch"], st["fi"], st["pos"]
        names = [str(f) for f in self.files]
        if st.get("file") in names:
            self.fi = names.index(st["file"])
            if self.files[self.fi].stat().st_size < st.get("size", 0):
                print(f"warning: shard {st['file']} shrank since the checkpoint; restarting it")
                self.pos = 0
        elif st.get("file"):
            print(f"warning: checkpoint shard {st['file']} missing; restarting this source at file {self.fi}")
            self.fi, self.pos = min(self.fi, len(self.files) - 1), 0

    def close(self):
        if self._fh:
            self._fh.close()
        self._fh = None

    def _ensure_open(self):
        if self._fh is not None:
            return
        path = self.files[self.fi]
        self._gz = path.suffix == ".gz"
        if self._gz:
            self._fh = gzip.open(path, "rb")
            for _ in range(self.pos):
                if not self._fh.readline():
                    break
        else:
            self._fh = open(path, "rb")
            self._fh.seek(self.pos)

    def _readline(self) -> bytes:
        line = self._fh.readline()
        if line:
            self.pos += 1 if self._gz else len(line)
        return line

    def _advance_file(self):
        self.close()
        self.fi, self.pos = self.fi + 1, 0
        if self.fi >= len(self.files):
            self.fi, self.epoch = 0, self.epoch + 1

    def next_doc(self) -> str:
        if not self.repeat and self.epoch:
            raise StopIteration
        empty_files = 0
        while True:
            self._ensure_open()
            path = self.files[self.fi]
            if self.episode_filter is not None:
                line = self._readline()
                if not line:
                    empty_files = self._eof(empty_files)
                    continue
                empty_files = 0
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self.episode_filter(ep):
                    return render_episode(ep)
            elif self.kinds[self.fi] == "jsonl":
                line = self._readline()
                if not line:
                    empty_files = self._eof(empty_files)
                    continue
                empty_files = 0
                text = _json_text(line.decode("utf-8", "replace"))
                if text:
                    return text
            elif self.kinds[self.fi] == "lines":
                line = self._readline()
                if not line:
                    empty_files = self._eof(empty_files)
                    continue
                empty_files = 0
                text = line.decode("utf-8", "replace").rstrip("\r\n").replace("\\n", "\n")
                text = text.removeprefix("[BOS]").removesuffix("[EOS]")
                if text.strip():
                    return text
            else:
                buf, size = [], 0
                while True:
                    line = self._readline()
                    if not line:
                        break
                    s = line.decode("utf-8", "replace")
                    bare = s.rstrip("\r\n")
                    if bare == self.txt_sep:
                        break
                    if bare.endswith("[EOS]"):
                        buf.append(bare[: -len("[EOS]")])
                        break
                    buf.append(s)
                    size += len(s)
                    if size >= MAX_DOC_CHARS:
                        break
                text = "".join(buf).strip("\n")
                if text.startswith("[BOS]"):
                    text = text[len("[BOS]"):]
                if not line and not buf:
                    empty_files = self._eof(empty_files)
                    continue
                empty_files = 0
                if text:
                    return text

    def _eof(self, empty_files: int) -> int:
        self._advance_file()
        if not self.repeat and self.epoch:
            raise StopIteration
        if empty_files > 2 * len(self.files):
            raise RuntimeError(f"no documents in {[str(f) for f in self.files]}")
        return empty_files + 1

def _shards(d: Path) -> list[Path]:
    pats = ("*.jsonl", "*.jsonl.gz", "*.txt", "*.txt.gz")
    files = sorted({p for pat in pats for p in d.rglob(pat)})
    return [p for p in files if p.stat().st_size > 0]

def _episode_heldout(ep: dict) -> bool:
    return zlib.crc32(str(ep.get("session", "")).encode()) % EPISODE_HELDOUT_MOD == 0

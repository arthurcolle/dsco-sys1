#!/usr/bin/env python3
"""Causal-LM pretraining of Sys1Model over a weighted corpus mixture.

    python -m sys1.model.pretrain --name tiny-smoke --preset sys1-tiny \
        --tokenizer data/decisions/tokenizer.json --seq 512 --steps 60

Data: data/corpus/<source>/ for each source in --mix (default en,code,agent,private).
Training shards are *.jsonl / *.jsonl.gz (one JSON object per line, text under
"text" or "content") and *.txt / *.txt.gz in one of two layouts, sniffed per file:
  lines   one document per line, real newlines escaped as a literal backslash-n
  blocks  multi-line documents, each ending at a line that ends with [EOS] or is
          exactly --txt-sep (leading [BOS] / trailing [EOS] are stripped; the
          packer inserts its own [EOS])
Files named heldout*.txt / heldout*.jsonl in a source dir are eval-only.
If no corpus source has data, the "agent" source falls back to text rendered from
data/episodes.jsonl (1 in 50 sessions held out).

Each source is tokenized on the fly, documents are joined with [EOS], and the
stream is cut into fixed seq+1 windows. Every sequence comes from one source,
picked by a seeded weighted draw, so the mixture is exact in expectation and the
data cursor is small: per-source (file, offset, leftover tokens) plus the draw RNG.

Checkpoints hold model, optimizer, data cursor and all RNG states, so a resumed
run continues the same batches and the same loss curve. Logs go to
runs/pretrain/<name>/log.jsonl. Device-agnostic: cuda (bf16 autocast, TF32,
fused AdamW, optional torch.compile), mps (bf16 if a probe step agrees with fp32,
else fp32), cpu (fp32).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import signal
import time
import zlib
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from sys1.model.model import PRESETS, Sys1Model, param_count
from sys1.paths import DATA, RUNS

SOURCES = ("en", "code", "agent", "private")
SNIFF_BYTES = 1 << 16       # bytes read to decide a .txt shard's layout
ENCODE_BATCH = 64          # documents tokenized per refill (tokenizers batches in Rust)
MAX_DOC_CHARS = 1 << 20    # cap a single runaway .txt document at 1 MiB
EPISODE_HELDOUT_MOD = 50   # 1 in 50 episode sessions goes to the fallback heldout set

# Dense bf16 peak TFLOPs per device, for the MFU estimate (override with --peak-tflops).
PEAK_TFLOPS = {"H200": 989.0, "H100": 989.0, "GH200": 989.0, "B200": 2250.0, "A100": 312.0,
               "L40S": 362.0, "L40": 181.0, "A10G": 125.0, "A10": 125.0, "L4": 121.0,
               "RTX 4090": 165.0, "RTX 3090": 71.0, "A6000": 155.0, "V100": 125.0, "T4": 65.0}


# --------------------------------------------------------------------------- data

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

    def __init__(self, files: list[Path], txt_sep: str, episode_filter=None):
        self.files, self.txt_sep, self.episode_filter = files, txt_sep, episode_filter
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
        if empty_files > 2 * len(self.files):
            raise RuntimeError(f"no documents in {[str(f) for f in self.files]}")
        return empty_files + 1


class PackedSource:
    """Tokenized, [EOS]-joined, fixed-window stream for one mixture source."""

    def __init__(self, name: str, reader: DocReader, tok: Tokenizer, eos: int, window: int):
        self.name, self.reader, self.tok, self.eos, self.window = name, reader, tok, eos, window
        self.buf: list[int] = []

    def next_window(self) -> list[int]:
        while len(self.buf) < self.window:
            docs = [self.reader.next_doc() for _ in range(ENCODE_BATCH)]
            for enc in self.tok.encode_batch(docs, add_special_tokens=False):
                self.buf.extend(enc.ids)
                self.buf.append(self.eos)
        out, self.buf = self.buf[: self.window], self.buf[self.window:]
        return out

    def state(self) -> dict:
        return {"reader": self.reader.state(), "buf": np.asarray(self.buf, dtype=np.int32)}

    def load_state(self, st: dict):
        self.reader.load_state(st["reader"])
        self.buf = [int(x) for x in st["buf"]]


class Mixture:
    def __init__(self, sources: dict[str, PackedSource], weights: dict[str, float], seed: int):
        self.sources = sources
        self.names = list(sources)
        total = sum(weights[n] for n in self.names)
        self.weights = [weights[n] / total for n in self.names]
        self.rng = random.Random(seed)
        self.drawn = {n: 0 for n in self.names}

    def batch(self, bsz: int) -> torch.Tensor:
        rows = []
        for _ in range(bsz):
            name = self.rng.choices(self.names, self.weights)[0]
            self.drawn[name] += 1
            rows.append(self.sources[name].next_window())
        return torch.tensor(rows, dtype=torch.long)

    def state(self) -> dict:
        return {"rng": self.rng.getstate(), "drawn": dict(self.drawn),
                "sources": {n: s.state() for n, s in self.sources.items()}}

    def load_state(self, st: dict):
        self.rng.setstate(st["rng"])
        if set(st["sources"]) != set(self.names):
            print(f"warning: mixture sources changed {sorted(st['sources'])} -> {self.names}; "
                  "draws will not match the original run")
        self.drawn = {n: st["drawn"].get(n, 0) for n in self.names}
        for n, s in self.sources.items():
            if n in st["sources"]:
                s.load_state(st["sources"][n])


def _shards(d: Path) -> list[Path]:
    pats = ("*.jsonl", "*.jsonl.gz", "*.txt", "*.txt.gz")
    files = sorted({p for pat in pats for p in d.rglob(pat)})
    return [p for p in files if p.stat().st_size > 0]


def _episode_heldout(ep: dict) -> bool:
    return zlib.crc32(str(ep.get("session", "")).encode()) % EPISODE_HELDOUT_MOD == 0


def heldout_windows(paths: list[Path] | None, episodes: Path | None, tok, eos, window, n: int, txt_sep: str):
    """First n packed windows of a source's heldout set (deterministic)."""
    if paths:
        reader = DocReader(paths, txt_sep)
    else:
        reader = DocReader([episodes], txt_sep, episode_filter=_episode_heldout)
    src = PackedSource("heldout", reader, tok, eos, window)
    out = []
    while len(out) < n and reader.epoch == 0:
        out.append(src.next_window())
    reader.close()
    return torch.tensor(out, dtype=torch.long) if out else None


def build_data(args, tok, eos):
    window = args.seq + 1
    corpus = Path(args.corpus)
    weights = parse_mix(args.mix)
    sources, heldout, origin = {}, {}, {}
    for name, w in weights.items():
        if w <= 0:
            continue
        d = corpus / name
        found = _shards(d) if d.is_dir() else []
        files = [p for p in found if not _is_heldout(p)]
        held = [p for p in found if _is_heldout(p)]
        if not files:
            continue
        reader = DocReader(files, args.txt_sep)
        sources[name] = PackedSource(name, reader, tok, eos, window)
        heldout[name] = heldout_windows(held, None, tok, eos, window, args.eval_seqs, args.txt_sep) if held else None
        origin[name] = {"shards": len(files), "bytes": sum(p.stat().st_size for p in files),
                        "layouts": sorted(set(reader.kinds)), "heldout": [str(p) for p in held] or None}
    if not sources:
        episodes = Path(args.episodes)
        if not episodes.exists():
            raise SystemExit(f"no corpus under {corpus} and no fallback {episodes}")
        name = "agent"
        weights = {name: 1.0}
        reader = DocReader([episodes], args.txt_sep, episode_filter=lambda ep: not _episode_heldout(ep))
        sources[name] = PackedSource(name, reader, tok, eos, window)
        heldout[name] = heldout_windows(None, episodes, tok, eos, window, args.eval_seqs, args.txt_sep)
        origin[name] = {"fallback": str(episodes), "heldout": f"crc32(session) % {EPISODE_HELDOUT_MOD} == 0"}
    mix = Mixture(sources, weights, args.seed)
    return mix, heldout, origin


def parse_mix(spec: str) -> dict[str, float]:
    """'en=0.5,code=0.25,agent=0.15,private=0.1' or 'en,code' (equal weights)."""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition("=")
        out[name.strip()] = float(w) if w else 1.0
    return out


# --------------------------------------------------------------------------- device / precision

def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, dtype):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def probe_bf16(model, x, device) -> tuple[bool, dict]:
    """One forward+backward in fp32 and in bf16 autocast on the same batch.
    bf16 is accepted when its loss is finite, within 2% of fp32, and grads are finite."""
    result = {}
    try:
        losses = {}
        for tag, dt in (("fp32", None), ("bf16", torch.bfloat16)):
            model.zero_grad(set_to_none=True)
            with autocast_ctx(device, dt):
                loss = lm_loss(model, x)
            loss.backward()
            gn = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None))
            losses[tag] = (loss.item(), gn.item())
        model.zero_grad(set_to_none=True)
        (l32, g32), (l16, g16) = losses["fp32"], losses["bf16"]
        rel = abs(l16 - l32) / max(abs(l32), 1e-6)
        result = {"loss_fp32": l32, "loss_bf16": l16, "rel_diff": rel, "gnorm_fp32": g32, "gnorm_bf16": g16}
        ok = math.isfinite(l16) and math.isfinite(g16) and rel < 0.02
        return ok, result
    except Exception as e:  # unsupported op under MPS autocast, etc.
        model.zero_grad(set_to_none=True)
        return False, {"error": repr(e)}


def peak_flops(device: torch.device, override: float) -> float:
    if override > 0:
        return override * 1e12
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        for key, tf in PEAK_TFLOPS.items():
            if key in name:
                return tf * 1e12
    return 0.0  # unknown (MPS / CPU): MFU logged as null unless --peak-tflops given


def flops_per_token(cfg, seq: int) -> float:
    """6N (fwd+bwd matmuls, incl. tied LM head) + attention score/value matmuls."""
    pc = param_count(cfg)
    n_matmul = pc["layers"] + cfg.vocab_size * cfg.d_model
    attn = 12 * cfg.n_layers * cfg.d_model * seq / 2  # causal: ~half the score matrix
    return 6 * n_matmul + attn


# --------------------------------------------------------------------------- training

def lm_loss(model, x):
    inp, tgt = x[:, :-1], x[:, 1:]
    logits = model.lm_logits(model.hidden(inp))
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1))


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    floor = args.lr * args.min_lr_ratio
    return floor + 0.5 * (args.lr - floor) * (1 + math.cos(math.pi * t))


def param_groups(model, wd: float, decay_embeddings: bool):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = name.startswith("embed.")
        if p.ndim < 2 or (is_embed and not decay_embeddings):
            no_decay.append(p)   # norms, biases, (optionally) embeddings
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def rng_state(device) -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(st: dict, device):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if device.type == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])
    elif device.type == "mps" and "mps" in st:
        torch.mps.set_rng_state(st["mps"])


def save_checkpoint(out: Path, step: int, model, opt, mix, args, precision, keep: int):
    ckdir = out / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    raw = getattr(model, "_orig_mod", model)
    blob = {"step": step, "model": raw.state_dict(), "optimizer": opt.state_dict(),
            "data": mix.state(), "rng": rng_state(next(raw.parameters()).device),
            "config": asdict(raw.cfg), "args": vars(args), "precision": precision}
    path = ckdir / f"step_{step:08d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)
    (ckdir / "latest.tmp").write_text(path.name)
    os.replace(ckdir / "latest.tmp", ckdir / "latest")
    for old in sorted(ckdir.glob("step_*.pt"))[:-keep] if keep > 0 else []:
        old.unlink()
    return path


def find_resume(out: Path, resume: str) -> Path | None:
    if not resume:
        return None
    if resume != "auto":
        return Path(resume)
    latest = out / "checkpoints" / "latest"
    if latest.exists():
        p = out / "checkpoints" / latest.read_text().strip()
        if p.exists():
            return p
    return None


@torch.no_grad()
def evaluate(model, heldout, args, device, amp):
    model.eval()
    res = {}
    for name, xs in heldout.items():
        if xs is None:
            continue
        tot, n = 0.0, 0
        for i in range(0, len(xs), args.micro_batch):
            x = xs[i: i + args.micro_batch].to(device)
            with autocast_ctx(device, amp):
                tot += lm_loss(model, x).item() * len(x)
            n += len(x)
        res[name] = tot / n
    model.train()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--preset", default="sys1-tiny", choices=sorted(PRESETS))
    ap.add_argument("--tokenizer", default=str(DATA / "decisions" / "tokenizer.json"))
    ap.add_argument("--corpus", default=str(DATA / "corpus"))
    ap.add_argument("--episodes", default=str(DATA / "episodes.jsonl"), help="fallback text source")
    ap.add_argument("--mix", default="en=0.45,code=0.25,agent=0.2,private=0.1",
                    help="source=weight,...; missing/empty sources are dropped and weights renormalized")
    ap.add_argument("--txt-sep", default="<|endoftext|>", help="document separator line in .txt shards")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--steps", type=int, default=10000, help="optimizer steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--decay-embeddings", action="store_true", help="also weight-decay the (tied) embedding")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true", help="torch.compile (CUDA)")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep", type=int, default=3, help="checkpoints to keep (0 = all)")
    ap.add_argument("--resume", default="auto", help="'auto' (latest in run dir), a path, or '' for fresh")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-seqs", type=int, default=32, help="heldout windows per source")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--peak-tflops", type=float, default=0.0, help="device peak for MFU; auto on known CUDA GPUs")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stop-after", type=int, default=0, help="exit after this many steps this invocation (testing)")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    out = RUNS / "pretrain" / args.name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("[EOS]")
    if eos is None:
        raise SystemExit("tokenizer has no [EOS] token")
    cfg = PRESETS[args.preset]
    cfg = type(cfg)(**{**asdict(cfg), "vocab_size": tok.get_vocab_size(), "max_seq": max(cfg.max_seq, args.seq)})
    model = Sys1Model(cfg).to(device)
    mix, heldout, origin = build_data(args, tok, eos)

    resume_path = find_resume(out, args.resume)
    blob = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path else None

    # Precision: CUDA bf16 autocast; MPS bf16 only if the probe agrees with fp32; CPU fp32.
    probe = None
    if blob is not None:
        precision = blob["precision"]
    elif args.precision != "auto":
        precision = args.precision
    elif device.type == "cuda":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    elif device.type == "mps":
        xs = next((h for h in heldout.values() if h is not None), None)
        x = (xs[: min(2, len(xs))] if xs is not None else torch.randint(0, cfg.vocab_size, (2, args.seq + 1))).to(device)
        ok, probe = probe_bf16(model, x, device)
        precision = "bf16" if ok else "fp32"
    else:
        precision = "fp32"
    amp = torch.bfloat16 if precision == "bf16" else None

    fused = device.type == "cuda"
    opt = torch.optim.AdamW(param_groups(model, args.wd, args.decay_embeddings), lr=args.lr,
                            betas=(args.beta1, args.beta2), eps=1e-8, fused=fused)
    step = 0
    if blob is not None:
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        mix.load_state(blob["data"])
        set_rng_state(blob["rng"], device)
        step = blob["step"]
        del blob
    train_model = torch.compile(model) if args.compile else model

    log_f = open(out / "log.jsonl", "a")

    def log(rec):
        rec = {"t": round(time.time(), 3), **rec}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps(rec), flush=True)

    pc = param_count(cfg)
    fpt = flops_per_token(cfg, args.seq)
    peak = peak_flops(device, args.peak_tflops)
    tokens_per_step = args.micro_batch * args.accum * args.seq
    log({"event": "resume" if resume_path else "start", "step": step, "resume_from": str(resume_path) if resume_path else None,
         "device": str(device), "precision": precision, "bf16_probe": probe, "preset": args.preset,
         "config": asdict(cfg), "params": pc["total"], "non_embedding": pc["non_embedding"],
         "sources": origin, "weights": dict(zip(mix.names, mix.weights)), "tokens_per_step": tokens_per_step,
         "flops_per_token": fpt, "peak_flops": peak or None, "torch": torch.__version__})

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    train_model.train()
    t_last, tok_last, ran = time.time(), 0, 0
    while step < args.steps and not stop["flag"]:
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for _ in range(args.accum):
            x = mix.batch(args.micro_batch).to(device, non_blocking=True)
            with autocast_ctx(device, amp):
                loss = lm_loss(train_model, x)
            (loss / args.accum).backward()
            loss_sum += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) if args.clip > 0 else torch.tensor(0.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        ran += 1
        tok_last += tokens_per_step

        if step % args.log_every == 0 or step == args.steps:
            loss_v = (loss_sum / args.accum).item()
            sync()
            dt = time.time() - t_last
            tps = tok_last / dt
            log({"event": "train", "step": step, "loss": round(loss_v, 5), "lr": lr,
                 "grad_norm": round(float(gnorm), 4), "tok_per_s": round(tps, 1),
                 "mfu": round(tps * fpt / peak, 4) if peak else None,
                 "tokens": step * tokens_per_step, "drawn": dict(mix.drawn)})
            if not math.isfinite(loss_v):
                log({"event": "abort", "step": step, "reason": "non-finite loss"})
                break
            t_last, tok_last = time.time(), 0
        if args.eval_every and step % args.eval_every == 0:
            log({"event": "eval", "step": step, "heldout_loss": evaluate(model, heldout, args, device, amp)})
            t_last, tok_last = time.time(), 0
        if args.ckpt_every and step % args.ckpt_every == 0:
            p = save_checkpoint(out, step, model, opt, mix, args, precision, args.keep)
            log({"event": "checkpoint", "step": step, "path": str(p)})
            t_last, tok_last = time.time(), 0
        if args.stop_after and ran >= args.stop_after:
            break

    if stop["flag"] or (args.ckpt_every and step % args.ckpt_every):
        p = save_checkpoint(out, step, model, opt, mix, args, precision, args.keep)
        log({"event": "checkpoint", "step": step, "path": str(p)})
    if step >= args.steps:
        if not (args.eval_every and step % args.eval_every == 0):
            log({"event": "eval", "step": step, "heldout_loss": evaluate(model, heldout, args, device, amp)})
        log({"event": "done", "step": step})
    log_f.close()


if __name__ == "__main__":
    main()

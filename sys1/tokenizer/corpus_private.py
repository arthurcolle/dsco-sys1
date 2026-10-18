#!/usr/bin/env python3
"""Render private dsco agent episodes into role-marked tokenizer corpus text.

    python -m sys1.tokenizer.corpus_private [--cap-mb 400] [--heldout 0.05]

Inputs (local, gitignored):
  data/episodes.jsonl                 goal, steps (tool/args/result/ok), final
  data/labels/labeled_episodes.jsonl  op-code labels per step, line-aligned

Output: data/corpus/private/{train,heldout}-NNNNN.txt + MANIFEST.json.
LOCAL ONLY. This corpus holds private prompts, file contents and tool output.
Nothing is written outside data/corpus/private/. Credentials are redacted again
(patterns from sys1/datasets/export_episodes.py), plus emails, long hex/base64
blobs and absolute home paths.

Episode layout:
  [BOS]<|user|>goal<|end|>
  <|assistant|><call ops><|tool:name|><|json|>args<|/json|><|end|>
  <|tool|><result ops>result<|end|>          (per step)
  <|assistant|><final ops>final text<|end|>[EOS]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from sys1.datasets.export_episodes import SECRET_PATTERNS
from sys1.paths import DATA, OPCODES_JSON
from sys1.tokenizer import spec

EPISODES = DATA / "episodes.jsonl"
LABELS = DATA / "labels" / "labeled_episodes.jsonl"
OUT_DIR = DATA / "corpus" / "private"
SHARD_BYTES = 64 * 1024 * 1024

EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
HOME_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+(?=/|\b)")
HEX_RE = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{41,}(?![0-9A-Za-z])")
B64_RE = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{41,}={0,2}(?![A-Za-z0-9+/=_-])")


def _looks_b64(s: str) -> bool:
    """Random-looking blob: mixed case + digits, and not a path or identifier."""
    body = s.rstrip("=")
    if body.count("/") > len(body) // 12 or body.count("_") + body.count("-") > len(body) // 10:
        return False
    return any(c.isdigit() for c in body) and any(c.isupper() for c in body) and any(c.islower() for c in body)


class Redactor:
    def __init__(self) -> None:
        self.counts: Counter = Counter({k: 0 for k in ("credential", "private_key", "email", "home_path", "hex_blob", "base64_blob")})

    def __call__(self, text: str | None) -> str:
        if not text:
            return ""
        for i, pattern in enumerate(SECRET_PATTERNS):
            kind = "private_key" if i == len(SECRET_PATTERNS) - 1 else "credential"

            def sub(m: re.Match, kind: str = kind) -> str:
                self.counts[kind] += 1
                return (m.group(1) + m.group(2) + "[REDACTED]") if m.re.groups >= 2 else "[REDACTED]"
            text = pattern.sub(sub, text)
        text = self._sub(EMAIL_RE, "email", "[EMAIL]", text)
        text = self._sub(HOME_RE, "home_path", "~", text)
        text = self._sub(HEX_RE, "hex_blob", "[HEX]", text)

        def b64(m: re.Match) -> str:
            if not _looks_b64(m.group(0)):
                return m.group(0)
            self.counts["base64_blob"] += 1
            return "[BLOB]"
        return B64_RE.sub(b64, text)

    def _sub(self, pattern: re.Pattern, kind: str, repl: str, text: str) -> str:
        text, n = pattern.subn(repl, text)
        self.counts[kind] += n
        return text


def _op_tokens() -> dict[str, str]:
    return {op["mnemonic"]: op["token"] for op in json.loads(OPCODES_JSON.read_text())["ops"]}


def render(ep: dict, lab: dict, ops: dict[str, str], tools: set[str], red: Redactor, stats: Counter) -> str:
    def toks(names: list[str]) -> str:
        out = []
        for n in names:
            if n in ops:
                out.append(ops[n])
            else:
                stats["unknown_ops"] += 1
        return "".join(out)

    parts = ["[BOS]<|user|>", red(ep.get("goal")), "<|end|>\n"]
    lsteps = lab.get("steps") or []
    used = 0
    for i, step in enumerate(ep.get("steps") or []):
        sops = lsteps[i]["ops"] if i < len(lsteps) else []
        used += len(sops)
        call = [o for o in sops if not o.startswith("ACT.RESULT")]
        result = [o for o in sops if o.startswith("ACT.RESULT")]
        name = step.get("tool") or "unknown"
        tool_tok = f"<|tool:{name}|>"
        if tool_tok not in tools:
            stats["tools_untokenized"] += 1
            tool_tok = f"[ACT]{red(name)}"
        args = step.get("args")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        parts += ["<|assistant|>", toks(call), tool_tok, "<|json|>", red(args), "<|/json|><|end|>\n",
                  "<|tool|>", toks(result), red(step.get("result")), "<|end|>\n"]
        stats["steps"] += 1
    final = ep.get("final") or {}
    parts += ["<|assistant|>", toks((lab.get("ops") or [])[used:]), red(final.get("text")), "<|end|>[EOS]\n\n"]
    return "".join(parts)


def heldout_for(ep: dict, lineno: int, frac: float) -> bool:
    key = f"{ep.get('session')}:{ep.get('ts')}:{lineno}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64 < frac


class ShardWriter:
    def __init__(self, split: str) -> None:
        self.split, self.idx, self.fh, self.size, self.files = split, -1, None, 0, []

    def write(self, text: str) -> int:
        data = text.encode("utf-8")
        if self.fh is None or self.size + len(data) > SHARD_BYTES:
            self._roll()
        self.fh.write(data)
        self.size += len(data)
        return len(data)

    def _roll(self) -> None:
        if self.fh:
            self.fh.close()
        self.idx += 1
        path = OUT_DIR / f"{self.split}-{self.idx:05d}.txt"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self.fh = os.fdopen(fd, "wb")
        self.size = 0
        self.files.append(path)

    def close(self) -> None:
        if self.fh:
            self.fh.close()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cap-mb", type=int, default=400)
    ap.add_argument("--heldout", type=float, default=0.05)
    args = ap.parse_args()
    cap = args.cap_mb * 1024 * 1024

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT_DIR, 0o700)
    for old in OUT_DIR.glob("*.txt"):
        old.unlink()

    ops = _op_tokens()
    tools = set(spec._tools())
    red, stats = Redactor(), Counter()
    writers = {"train": ShardWriter("train"), "heldout": ShardWriter("heldout")}
    split_eps, split_bytes = Counter(), Counter()
    total, truncated = 0, False
    with open(EPISODES) as fe, open(LABELS) as fl:
        for lineno, (le, ll) in enumerate(zip(fe, fl)):
            ep, lab = json.loads(le), json.loads(ll)
            if ep.get("session") != lab.get("session") or len(ep.get("steps") or []) != len(lab.get("steps") or []):
                stats["label_misaligned"] += 1
                lab = {"steps": [], "ops": []}
            text = render(ep, lab, ops, tools, red, stats)
            n = len(text.encode("utf-8"))
            if total + n > cap:
                truncated = True
                break
            split = "heldout" if heldout_for(ep, lineno, args.heldout) else "train"
            writers[split].write(text)
            total += n
            split_eps[split] += 1
            split_bytes[split] += n
    for w in writers.values():
        w.close()

    files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)}
             for w in writers.values() for p in w.files]
    manifest = {
        "privacy": "LOCAL ONLY - private dsco trajectories; never upload, publish or copy off this machine",
        "sources": [str(EPISODES.name), "labels/" + LABELS.name],
        "episodes": dict(split_eps), "bytes": dict(split_bytes), "total_bytes": total,
        "cap_bytes": cap, "truncated_at_cap": truncated, "heldout_frac": args.heldout,
        "split_key": "sha256(session:ts:lineno) per episode",
        "steps": stats["steps"], "tools_untokenized": stats["tools_untokenized"],
        "unknown_ops": stats["unknown_ops"], "label_misaligned": stats["label_misaligned"],
        "redactions": dict(red.counts), "files": files,
    }
    mpath = OUT_DIR / "MANIFEST.json"
    fd = os.open(mpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()

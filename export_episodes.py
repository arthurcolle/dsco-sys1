#!/usr/bin/env python3
"""Reconstruct dsco agent episodes from the local chronicle into training data.

    python3 export_episodes.py --out data/episodes.jsonl [--since 2026-09-01] [--limit N]

Reads ~/.dsco/chronicle/events/**/session-*.jsonl and the content-addressed
blobs they reference. Writes one JSON line per episode (a user request and the
agent's tool steps until the turn ends). Read-only on the chronicle.

The output can contain private content (prompts, file contents, tool output).
It stays local under data/ (gitignored). Obvious credentials are redacted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CHRONICLE = Path.home() / ".dsco" / "chronicle"
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ARGS_MAX, RESULT_MAX, TEXT_MAX = 768, 768, 2048

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|authorization)([\"'\s:=]+)(Bearer\s+)?[A-Za-z0-9_\-\.]{12,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
]


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) + m.group(2) + "[REDACTED]") if m.re.groups >= 2 else "[REDACTED]", text)
    return text


def blob(sha: str | None, limit: int) -> str | None:
    if not sha:
        return None
    if sha == EMPTY_SHA:
        return ""
    path = CHRONICLE / "blobs" / "sha256" / sha[:2] / sha[2:4] / f"{sha}.blob"
    try:
        with open(path, "rb") as f:
            raw = f.read(limit * 4)
    except OSError:
        return None
    text = raw.decode("utf-8", errors="replace")
    if len(text) > limit:
        text = text[:limit] + f"…[+{path.stat().st_size - limit} bytes]"
    return redact(text)


def _user_texts(node, out: list[str]) -> None:
    """Collect user-authored text from chat or Responses-style request JSON."""
    if isinstance(node, dict):
        if node.get("role") == "user":
            content = node.get("content")
            if isinstance(content, str):
                out.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "input_text") and isinstance(part.get("text"), str):
                        out.append(part["text"])
            return
        for value in node.values():
            _user_texts(value, out)
    elif isinstance(node, list):
        for value in node:
            _user_texts(value, out)


def goal_from_request(sha: str | None) -> str | None:
    """The newest genuine user instruction in a logged request."""
    if not sha:
        return None
    path = CHRONICLE / "blobs" / "sha256" / sha[:2] / sha[2:4] / f"{sha}.blob"
    try:
        request = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    texts: list[str] = []
    _user_texts(request.get("messages") or request.get("input") or [], texts)
    for text in reversed(texts):
        stripped = text.strip()
        # Harness-injected observations are not user goals.
        if stripped and not stripped.startswith(("Tool observation", "[Earlier tool output cleared", "<system")):
            return redact(stripped[:TEXT_MAX])
    return None


def episodes_from_session(path: str) -> list[dict]:
    events = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event_type") in {"user.message", "llm.request.created", "llm.response.completed",
                                       "tool.call.created", "tool.call.completed"}:
                events.append(d)
    events.sort(key=lambda d: d.get("seq") or 0)
    out, current, pending = [], None, {}

    def close(reason: str):
        nonlocal current
        if current and current["steps"] is not None and (current["steps"] or current.get("final")):
            current["end"] = reason
            out.append(current)
        current = None

    for d in events:
        t, p = d["event_type"], d.get("payload") or {}
        if t == "user.message":
            close("next_user_message")
            goal = blob(p.get("blob_sha256"), TEXT_MAX)
            if goal is None:
                continue
            current = {"session": d.get("session_id"), "ts": d.get("wall_time"), "goal": goal,
                       "model": None, "steps": [], "llm_turns": 0, "final": None}
        elif t == "llm.request.created":
            if current is None:
                goal = goal_from_request(p.get("request_blob_sha256"))
                if goal:
                    current = {"session": d.get("session_id"), "ts": d.get("wall_time"), "goal": goal,
                               "model": p.get("model"), "steps": [], "llm_turns": 0, "final": None}
        elif current is None:
            continue
        elif t == "llm.response.completed":
            current["llm_turns"] += 1
            current["model"] = current["model"] or p.get("model")
            if p.get("finish_reason") != "tool_use":
                current["final"] = {"finish_reason": p.get("finish_reason"),
                                    "status": p.get("response_status"),
                                    "text": blob(p.get("output_blob_sha256"), TEXT_MAX)}
                close("final_answer")
        elif t == "tool.call.created":
            step = {"tool": p.get("tool_name"), "args": blob(p.get("args_blob_sha256"), ARGS_MAX),
                    "ok": None, "failure": None, "latency_ms": None, "result": None,
                    "llm_turn": current["llm_turns"]}
            current["steps"].append(step)
            pending[d.get("turn_id") or p.get("tool_id")] = step
        elif t == "tool.call.completed":
            step = pending.pop(d.get("turn_id"), None)
            if step is None:
                for candidate in reversed(current["steps"]):
                    if candidate["ok"] is None and candidate["tool"] == p.get("tool_name"):
                        step = candidate
                        break
            if step is not None:
                step.update(ok=p.get("ok"), failure=p.get("failure_class"),
                            latency_ms=p.get("latency_ms"),
                            result=blob(p.get("result_blob_sha256"), RESULT_MAX))
    close("session_end")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--since", default="", help="YYYY-MM-DD lower bound on session directory date")
    parser.add_argument("--limit", type=int, default=0, help="max session files (0 = all)")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()
    files = sorted(str(p) for p in (CHRONICLE / "events").rglob("session-*.jsonl"))
    if args.since:
        y, m, d = args.since.split("-")
        files = [f for f in files if "/".join(Path(f).parts[-4:-1]) >= f"{y}/{m}/{d}"]
    if args.limit:
        files = files[-args.limit:]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ep = n_steps = n_final = 0
    with open(args.out, "w") as out, ProcessPoolExecutor(args.workers) as pool:
        for episodes in pool.map(episodes_from_session, files, chunksize=64):
            for ep in episodes:
                out.write(json.dumps(ep, ensure_ascii=False) + "\n")
                n_ep += 1
                n_steps += len(ep["steps"])
                n_final += ep["end"] == "final_answer"
    print(json.dumps({"session_files": len(files), "episodes": n_ep, "tool_steps": n_steps,
                      "ended_with_final_answer": n_final, "out": str(args.out)}))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Adjudicate Jev misses with independent dsco workers (Claude and Codex).

    python -m sys1.jevmr.adjudicate --run runs/jev/fractal-v1 --out runs/jev/fractal-v1/adjudication

For every item Jev got wrong in variants whose gold follows from stated rules
(explicit compositions and ablations), batches of situations go to dsco
workers on different providers. Each miss is classified:
  jev_error      both workers agree with the rule-derived gold
  oracle_suspect both workers agree with Jev (the situation or rule is likely ambiguous)
  ambiguous      workers disagree with each other or pick a third option
The pool spawns a worker only while enough memory is free, and caps concurrency.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sys1.jevmr import fractal  # noqa: F401  (registers variants)
from sys1.jevmr.families import FAMILIES, OPS
from sys1.jevmr.jevmr import item_id, read_results

DSCO = "/Users/arthurcolle/Dsco/dsco-cli/dsco"
WORKERS = {
    "claude": ["--provider", "anthropic", "--model", "claude-opus-5-5"],
    "codex": ["--provider", "openai-codex", "--model", "gpt-6-astra"],
}
BATCH = 10
MIN_FREE_GB = 4.0
MAX_CONCURRENT = 8
# Adjudication is a text-only judgment; tool choice "none" is the supported
# setting for that and keeps dsco's headless execution contract from demanding a tool.
WORKER_ENV = {**__import__("os").environ, "DSCO_TOOL_CHOICE": "none"}


def available_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1)) for k in ("Pages free", "Pages inactive", "Pages speculative"))
    return pages * page / 1e9


class MemoryGate:
    """Admit a worker only when concurrency and free memory allow it."""

    def __init__(self, max_concurrent: int, min_free_gb: float):
        self.sem = threading.Semaphore(max_concurrent)
        self.min_free = min_free_gb
        self.lock = threading.Lock()
        self.peak_running = 0
        self.running = 0
        self.waits = 0

    def __enter__(self):
        self.sem.acquire()
        while True:
            with self.lock:
                if available_gb() >= self.min_free:
                    self.running += 1
                    self.peak_running = max(self.peak_running, self.running)
                    return self
                self.waits += 1
            time.sleep(5)

    def __exit__(self, *exc):
        with self.lock:
            self.running -= 1
        self.sem.release()


def load_misses(run: Path) -> list[dict]:
    job = json.loads((run / "job.json").read_text())
    rows = {r["id"]: r for r in read_results(run / "results.jsonl") if not r.get("error")}
    misses = []
    rng = random.Random(job["seed"])  # replay the runner's single shared stream, in job order
    for fam_name in job["families"]:
        fam = FAMILIES[fam_name]
        items = fam.items(rng, job.get("limit") or None)
        if not (fam_name.startswith("compose:") or fam_name.startswith("ablate:")):
            continue
        for item in items:
            body = fam.request(item, job["model"])
            row = rows.get(item_id(item, body))
            if row and row["choice"] != row["gold"]:
                misses.append({"id": row["id"], "family": fam_name, "state": item["state"],
                               "options": list(body["questions"]["op"]["criteria"]),
                               "gold": row["gold"], "jev": row["choice"], "jev_confidence": row["confidence"]})
    return misses


def prompt(batch: list[dict]) -> str:
    lines = ["You are adjudicating agent-operation decisions. Do not use any tools; answer directly from the text.",
             "For each case, pick the single best option given the situation and any policy stated in it.",
             'Reply with one JSON object per line and nothing else: {"id": "<id>", "answer": "<MNEMONIC>", "confidence": <0-1>}', ""]
    for c in batch:
        lines.append(f"CASE {c['id']}")
        lines.append(f"Situation: {c['state']}")
        lines.append("Options:")
        for o in c["options"]:
            lines.append(f"- {o}: {OPS[o]['desc']}")
        lines.append("")
    return "\n".join(lines)


def run_worker(name: str, batch: list[dict], gate: MemoryGate, timeout: int = 900) -> dict:
    with gate:
        t0 = time.monotonic()
        try:
            proc = subprocess.run([DSCO, *WORKERS[name], "-C", "--output-format", "json", "-p", prompt(batch)],
                                  capture_output=True, text=True, timeout=timeout, env=WORKER_ENV)
        except subprocess.TimeoutExpired:
            return {"worker": name, "answers": {}, "seconds": timeout, "returncode": -1, "stderr_tail": "timeout"}
    text = proc.stdout
    try:  # --output-format json wraps the final result
        obj = json.loads(text)
        text = obj.get("result") or obj.get("text") or text
    except (json.JSONDecodeError, AttributeError):
        pass
    answers = {}
    for m in re.finditer(r'\{[^{}]*"id"\s*:\s*"([0-9a-f]{16})"[^{}]*\}', text):
        try:
            rec = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        answers[rec["id"]] = rec
    return {"worker": name, "answers": answers, "seconds": round(time.monotonic() - t0, 1),
            "returncode": proc.returncode, "stderr_tail": proc.stderr[-300:] if proc.returncode else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="max misses to adjudicate (0 = all)")
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    a = ap.parse_args()
    misses = load_misses(a.run)
    random.Random(3).shuffle(misses)
    if a.limit:
        misses = misses[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    batches = [misses[i:i + BATCH] for i in range(0, len(misses), BATCH)]
    gate = MemoryGate(a.max_concurrent, a.min_free_gb)
    jobs = [(w, b) for b in batches for w in WORKERS]
    print(json.dumps({"misses": len(misses), "batches": len(batches), "worker_runs": len(jobs),
                      "available_gb": round(available_gb(), 1)}), flush=True)
    votes = collections.defaultdict(dict)
    log = open(a.out / "worker_runs.jsonl", "a")
    def with_retry(job):
        res = run_worker(*job, gate)
        if len(res["answers"]) < 0.8 * len(job[1]):  # incomplete batch: one retry
            again = run_worker(*job, gate)
            again["answers"] = {**res["answers"], **again["answers"]}
            again["retried"] = True
            res = again
        return res

    with ThreadPoolExecutor(a.max_concurrent) as pool:
        for res in pool.map(with_retry, jobs):
            for i, rec in res["answers"].items():
                votes[i][res["worker"]] = rec.get("answer")
            log.write(json.dumps({k: v for k, v in res.items() if k != "answers"} | {"n_answers": len(res["answers"])}) + "\n")
            log.flush()
            print(json.dumps({k: res[k] for k in ("worker", "seconds", "returncode")} | {"answers": len(res["answers"])}), flush=True)
    verdicts = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    rows = []
    for c in misses:
        v = votes.get(c["id"], {})
        cl, cx = v.get("claude"), v.get("codex")
        if cl is None or cx is None:
            verdict = "missing_vote"
        elif cl == cx == c["gold"]:
            verdict = "jev_error"
        elif cl == cx == c["jev"]:
            verdict = "oracle_suspect"
        else:
            verdict = "ambiguous"
        verdicts[verdict] += 1
        kind = c["family"].split(":")[0] + (f"/{c['family'].count('+') + 1}-way" if "compose" in c["family"] else "")
        by_kind[kind][verdict] += 1
        rows.append({**{k: c[k] for k in ("id", "family", "gold", "jev", "jev_confidence")}, "claude": cl, "codex": cx, "verdict": verdict})
    with open(a.out / "verdicts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    agree = sum(1 for r in rows if r["claude"] and r["claude"] == r["codex"])
    report = {"misses": len(misses), "verdicts": dict(verdicts), "by_kind": {k: dict(v) for k, v in by_kind.items()},
              "claude_codex_agreement": round(agree / max(1, sum(1 for r in rows if r["claude"] and r["codex"])), 3),
              "memory_gate": {"peak_running": gate.peak_running, "waits_for_memory": gate.waits,
                              "min_free_gb": a.min_free_gb, "max_concurrent": a.max_concurrent}}
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

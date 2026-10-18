#!/usr/bin/env python3
"""Map-reduce Jev decisions over generated prompt families.

    python -m jevmr.jevmr run --families ctx_pressure,tool_failure --out runs/jev/pilot --max-requests 200
    python -m jevmr.jevmr reduce --out runs/jev/pilot
    python -m jevmr.jevmr loop jevmr/jobs        # run every job spec that is not finished

map:    one request per generated situation; the op-code Choice and the Noul
        probes share the request (TypeSafe's fan-out pattern).
reduce: accuracy against the rule-derived answer, calibration (ECE), accuracy
        at confidence thresholds with coverage, probe accuracy, confusions,
        latency, tokens and cost.
Results append to results.jsonl, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevmr.families import FAMILIES
from jevmr import fractal  # registers fractal variants into FAMILIES

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
USD_PER_MTOK = 0.042  # input tokens only; output is free (docs.typesafe.ai/models)


def item_id(item: dict, body: dict) -> str:
    """Scoped to the exact request (model, question, options, state): a re-run
    with another model or a changed family never reuses old answers."""
    return hashlib.sha256(json.dumps([item["family"], body], sort_keys=True).encode()).hexdigest()[:16]


def read_results(path: Path) -> list[dict]:
    """Tolerate a torn final line from an interrupted write."""
    rows = []
    if path.exists():
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def valid_answer(resp, options) -> str | None:
    try:
        op = resp["answers"]["op"]
        conf = op.get("confidence")
        if op.get("choice") not in options:
            return "malformed: choice not among options"
        if not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
            return "malformed: confidence missing or out of range"
    except (KeyError, TypeError, AttributeError):
        return "malformed: missing answers.op"
    return None


def call(body: dict, key: str, attempts: int = 6) -> tuple[dict | None, str | None, float, int]:
    """Returns (response, error, total seconds including retries, POST attempts)."""
    data = json.dumps(body).encode()
    delay = 0.5
    t_start = time.monotonic()
    for attempt in range(attempts):
        req = urllib.request.Request(ENDPOINT, data=data, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        elapsed = lambda: time.monotonic() - t_start
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                try:
                    return json.loads(resp.read()), None, elapsed(), attempt + 1
                except json.JSONDecodeError:
                    return None, "malformed: response is not JSON", elapsed(), attempt + 1
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt + 1 < attempts:
                retry_after = e.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else delay
                time.sleep(min(wait, 30.0))  # bounded: never stall a worker indefinitely
                delay = min(delay * 2, 16)
                continue
            return None, f"http_{e.code}: {e.read()[:200].decode(errors='replace')}", elapsed(), attempt + 1
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay = min(delay * 2, 16)
                continue
            return None, f"transport: {e}", elapsed(), attempt + 1
    raise AssertionError("unreachable")


def run(families: list[str], out: Path, limit: int, max_requests: int, concurrency: int, model: str, seed: int) -> dict:
    """max_requests caps total POST attempts for this output directory across all resumes."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is not set")
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}; known: {sorted(FAMILIES)}")
    out.mkdir(parents=True, exist_ok=True)
    lock_file = open(out / ".lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another runner holds {out}; refusing to duplicate work")
    results_path = out / "results.jsonl"
    prior = read_results(results_path)
    done = {r["id"] for r in prior if not r.get("error")}
    spent = sum(r.get("attempts", 1) for r in prior)
    rng = random.Random(seed)
    todo = []
    for name in families:
        fam = FAMILIES[name]
        for item in fam.items(rng, limit or None):
            body = fam.request(item, model)
            item["id"] = item_id(item, body)
            if item["id"] not in done:
                todo.append((item, body))
    budget = max(0, max_requests - spent)
    todo = todo[:budget]
    (out / "job.json").write_text(json.dumps({"families": families, "limit": limit, "model": model,
                                               "seed": seed, "max_requests": max_requests}, indent=1))
    write_lock = threading.Lock()
    stats = collections.Counter()

    def work(pair):
        item, body = pair
        rec = {"id": item["id"], "family": item["family"], "situation": item["situation"],
               "phrasing": item["phrasing"], "gold": item["gold"], "slots": item["slots"], "model_requested": model}
        try:
            resp, err, latency, attempts = call(body, key)
            err = err or valid_answer(resp, body["questions"]["op"]["criteria"])
            rec.update(latency_s=round(latency, 3), attempts=attempts, error=err,
                       input_tokens=(resp or {}).get("usage", {}).get("input_tokens") if isinstance(resp, dict) else None)
            if not err:
                ans = resp["answers"]
                op = ans["op"]
                rec.update(model=resp.get("model"), choice=op["choice"], confidence=op["confidence"],
                           probabilities=op.get("probabilities"),
                           probes={k: {"p": (ans.get(k) or {}).get("noul"), "truth": truth}
                                   for k, (_, truth) in item["probes"].items()})
        except Exception as e:  # never let one item abort the run unrecorded
            rec.update(error=f"exception: {type(e).__name__}: {e}", attempts=rec.get("attempts", 1))
        with write_lock:
            stats["sent"] += 1
            stats["errors"] += rec["error"] is not None
            stats["attempts"] += rec.get("attempts", 1)
            with open(results_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())

    t0 = time.monotonic()
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(work, todo))
    return {**stats, "already_done": len(done), "budget_remaining": budget - stats["attempts"],
            "seconds": round(time.monotonic() - t0, 1)}


def ece(pairs, bins=10):
    pairs = [(c, ok) for c, ok in pairs if isinstance(c, (int, float)) and math.isfinite(c) and 0 <= c <= 1]
    if not pairs:
        return None
    total, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            err += len(sel) / total * abs(sum(c for c, _ in sel) / len(sel) - sum(ok for _, ok in sel) / len(sel))
    return err


def reduce(out: Path) -> dict:
    rows = read_results(out / "results.jsonl")
    if not rows:
        raise SystemExit(f"no results in {out}")
    tokens_by_family = collections.Counter()
    latest = {}
    for r in rows:  # cost counts every recorded call, including superseded retries
        tokens_by_family[r["family"]] += r.get("input_tokens") or 0
        if not r.get("error") or r["id"] not in latest:
            latest[r["id"]] = r
    by_family = collections.defaultdict(list)
    for r in latest.values():
        by_family[r["family"]].append(r)
    report = {"families": {}, "total": {}}
    all_pairs, all_n, all_correct = [], 0, 0
    for fam, items in sorted(by_family.items()):
        n = len(items)
        good = [r for r in items if not r.get("error")]
        pairs = [(r["confidence"], r["choice"] == r["gold"]) for r in good]
        correct = sum(ok for _, ok in pairs)
        all_pairs += pairs
        all_n += n
        all_correct += correct
        selective = {}
        for t in (0.5, 0.7, 0.9):
            sel = [ok for c, ok in pairs if c >= t]
            selective[str(t)] = {"coverage_of_all": round(len(sel) / n, 3),
                                 "accuracy": round(sum(sel) / len(sel), 4) if sel else None}
        probe = [(v["p"] > 0.5) == v["truth"] for r in good for v in (r.get("probes") or {}).values()
                 if isinstance(v.get("p"), (int, float))]
        situations = collections.defaultdict(list)
        for r in good:
            situations[r.get("situation")].append(r["choice"])
        multi = [c for c in situations.values() if len(c) > 1]
        lat = sorted(r["latency_s"] for r in items if r.get("latency_s") is not None)
        conf = collections.Counter((r["gold"], r["choice"]) for r in good if r["choice"] != r["gold"])
        report["families"][fam] = {
            "n": n, "errors": n - len(good),
            "accuracy_all": round(correct / n, 4),
            "accuracy_answered": round(correct / len(good), 4) if good else None,
            "ece": None if not pairs else round(ece(pairs), 4),
            "selective": selective,
            "paraphrase_consistency": round(sum(len(set(c)) == 1 for c in multi) / len(multi), 4) if multi else None,
            "probe_accuracy": round(sum(probe) / len(probe), 4) if probe else None,
            "latency_p50_s": lat[(len(lat) - 1) // 2] if lat else None,
            "latency_p95_s": lat[min(len(lat) - 1, int(math.ceil(len(lat) * 0.95)) - 1)] if lat else None,
            "input_tokens": tokens_by_family[fam], "cost_usd": round(tokens_by_family[fam] / 1e6 * USD_PER_MTOK, 5),
            "top_confusions": [{"gold": g, "chose": c, "n": k} for (g, c), k in conf.most_common(5)],
        }
    report["total"] = {"n": all_n, "accuracy_all": round(all_correct / all_n, 4) if all_n else None,
                       "ece": None if not all_pairs else round(ece(all_pairs), 4),
                       "cost_usd": round(sum(f["cost_usd"] for f in report["families"].values()), 5)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def loop(jobs_dir: Path) -> None:
    """Run every job spec; finished jobs cost nothing because resume skips done
    items and the budget is cumulative. Meant for cron/launchd scheduling."""
    for spec_path in sorted(jobs_dir.glob("*.json")):
        spec = json.loads(spec_path.read_text())
        # "{date}" in out gives one fresh run per day, so model drift shows up as a dated series.
        out = Path(spec["out"].format(date=time.strftime("%Y-%m-%d")))
        stats = run(spec["families"], out, spec.get("limit", 0), spec.get("max_requests", 1000),
                    spec.get("concurrency", 16), spec.get("model", "jev-1.13.0"), spec.get("seed", 7))
        rep = reduce(out)
        print(json.dumps({"job": spec_path.name, **stats, "total": rep["total"]}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--families", default=",".join(FAMILIES))
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int, default=0, help="max situations per family (0 = all)")
    r.add_argument("--max-requests", type=int, default=500)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--model", default="jev-1.13.0")
    r.add_argument("--seed", type=int, default=7)
    d = sub.add_parser("reduce")
    d.add_argument("--out", type=Path, required=True)
    l = sub.add_parser("loop")
    l.add_argument("jobs", type=Path)
    a = ap.parse_args()
    if a.cmd == "run":
        stats = run(a.families.split(","), a.out, a.limit, a.max_requests, a.concurrency, a.model, a.seed)
        print(json.dumps(stats))
        print(json.dumps(reduce(a.out), indent=1))
    elif a.cmd == "reduce":
        print(json.dumps(reduce(a.out), indent=1))
    else:
        loop(a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

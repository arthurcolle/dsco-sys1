#!/usr/bin/env python3
"""Tiny client for the local System One endpoint, plus a self-test.

    python -m sys1.serve.client --selftest                 # in-process server on a free port
    python -m sys1.serve.client --url http://127.0.0.1:8777/v1/systemone --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8777/v1/systemone"


def systemone(body: dict, url: str = DEFAULT_URL, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def schema_errors(resp: dict, body: dict) -> list[str]:
    """Full response-shape check against the request, beyond jevmr.valid_answer()."""
    errs = []
    if not isinstance(resp.get("model"), str):
        errs.append("model missing")
    if not isinstance((resp.get("usage") or {}).get("input_tokens"), int):
        errs.append("usage.input_tokens missing")
    answers = resp.get("answers") or {}
    for qid, q in body["questions"].items():
        a = answers.get(qid)
        if not isinstance(a, dict) or a.get("type") != q["type"]:
            errs.append(f"{qid}: missing or wrong type")
            continue
        if q["type"] == "noul":
            if not (isinstance(a.get("noul"), float) and 0 <= a["noul"] <= 1):
                errs.append(f"{qid}: noul not a probability")
            continue
        names = list(q["criteria"])
        probs = a.get("probabilities") or {}
        if set(probs) != set(names) or not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-3):
            errs.append(f"{qid}: probabilities do not cover criteria or do not sum to 1")
        if not (isinstance(a.get("confidence"), float) and 0 <= a["confidence"] <= 1):
            errs.append(f"{qid}: confidence out of range")
        key = "choice" if q["type"] == "choice" else "score"
        if str(a.get(key)) not in names:
            errs.append(f"{qid}: {key} not among criteria")
    return errs


def _raw_status(url: str, data: bytes) -> int:
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def _oversize_status(url: str, n: int) -> int:
    """Declare an oversize body without sending it: the server must refuse on the header alone."""
    import http.client
    from urllib.parse import urlsplit
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    conn.putrequest("POST", u.path)
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(n))
    conn.endheaders()
    status = conn.getresponse().status
    conn.close()
    return status


def selftest(url: str | None) -> int:
    from sys1.jevmr.families import FAMILIES
    from sys1.jevmr.jevmr import valid_answer

    server = None
    if url is None:
        from sys1.serve.server import DEFAULT_CKPT, DEFAULT_TOKENIZER, Decider, build_server
        t0 = time.monotonic()
        decider = Decider(DEFAULT_CKPT, DEFAULT_TOKENIZER, "cpu")
        server = build_server(decider, port=0, timeout_s=30.0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/systemone"
        print(json.dumps({"server": url, "load_s": round(time.monotonic() - t0, 2)}))

    rng = random.Random(2026)
    picks = [("ctx_pressure", 0), ("tool_failure", 0), ("tool_failure", 1)]
    pools = {f: FAMILIES[f].items(random.Random(7)) for f in {f for f, _ in picks}}
    bodies = []
    for fam, _ in picks:
        item = rng.choice(pools[fam])
        body = FAMILIES[fam].request(item, "jev-1.13.0")  # choice "op" + noul probe(s)
        bodies.append((fam, item, body))
    # exercise the score type on the last request
    bodies[-1][2]["questions"]["severity"] = {
        "type": "score", "instructions": "How severe is the failure for task progress?",
        "criteria": {"1": "minor, the next attempt will likely work", "2": "moderate, needs a change of approach",
                     "3": "blocking, the agent cannot proceed on its own"}}

    ok = True
    lat = []
    for fam, item, body in bodies:
        t = time.perf_counter()
        resp = systemone(body, url)
        dt = (time.perf_counter() - t) * 1000
        lat.append(dt)
        va = valid_answer(resp, body["questions"]["op"]["criteria"])
        errs = schema_errors(resp, body)
        ok &= va is None and not errs
        op = resp["answers"]["op"]
        print(json.dumps({"family": fam, "situation": item["situation"], "gold": item["gold"],
                          "choice": op["choice"], "confidence": op["confidence"],
                          "probes": {k: {"p": resp["answers"][k]["noul"], "truth": tr}
                                     for k, (_, tr) in item["probes"].items()},
                          **({"score": resp["answers"]["severity"]["score"]} if "severity" in resp["answers"] else {}),
                          "input_tokens": resp["usage"]["input_tokens"], "valid_answer": va,
                          "schema_errors": errs, "latency_ms": round(dt, 1)}))
    # error paths: malformed JSON -> 400, bad type -> 400, oversize -> 413
    codes = {"bad_json": _raw_status(url, b"{not json"),
             "bad_type": _raw_status(url, json.dumps({"state": "x", "questions": {"q": {"type": "x"}}}).encode()),
             "too_large": _oversize_status(url, 2 << 20)}
    ok &= codes == {"bad_json": 400, "bad_type": 400, "too_large": 413}
    print(json.dumps({"error_paths": codes, "latency_ms": [round(x, 1) for x in lat],
                      "latency_ms_mean": round(sum(lat) / len(lat), 1), "pass": bool(ok)}))
    if server is not None:
        server.shutdown()
        server.server_close()
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help=f"existing server (default for requests: {DEFAULT_URL})")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--body", type=Path, help="send this JSON request file and print the response")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.url)
    if a.body:
        print(json.dumps(systemone(json.loads(a.body.read_text()), a.url or DEFAULT_URL), indent=1))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

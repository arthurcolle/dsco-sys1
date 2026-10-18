#!/usr/bin/env python3
"""Localhost System One endpoint backed by a Sys1Model checkpoint.

    python -m sys1.serve.server                      # 127.0.0.1:8777, CPU, runs/distill/c
    python -m sys1.serve.server --port 9000 --device mps

POST /v1/systemone takes the TypeSafe System One request shape
    {"model", "state", "questions": {id: {"type": "choice"|"noul"|"score", "instructions", "criteria"}}}
and answers with the same response shape
    {"model", "answers": {id: {...}}, "usage": {"input_tokens"}}.
GET /healthz reports the loaded checkpoint.

Encoding matches sys1/model/distill_jev.py: [GOAL] " "+state, then [OPTS], then
[OPT] " "+"<name>: <description>" per option, then [DECIDE]; option scores are read
at the last token of each option span.
  choice  pointer head over the criteria options
  score   pointer head over the level texts (criteria {level: text}); score = argmax
          level (numeric when the level key parses as a number)
  noul    noul head at [DECIDE] after [GOAL] " "+state+" Question: "+instructions.
          The distilled checkpoint was trained on choice only, so its noul head is
          inherited from the init checkpoint and is not calibrated on probe questions.
Instructions are not encoded for choice/score because distillation did not see them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from sys1.model.model import Sys1Config, Sys1Model

DEFAULT_CKPT = Path("runs/distill/c/model.pt")
DEFAULT_TOKENIZER = Path("runs/distill/c/tokenizer.json")
DEFAULT_PORT = 8777
MAX_BODY_BYTES = 1 << 20      # 1 MiB request cap
MAX_QUESTIONS = 64
MAX_OPTIONS = 64
QUESTION_TYPES = ("choice", "noul", "score")


class RequestError(ValueError):
    pass


def _criteria_items(criteria) -> list[tuple[str, str]]:
    """criteria as {name: description} or [name | {name, description}] -> [(name, description)]."""
    if isinstance(criteria, dict):
        items = [(str(k), "" if v is None else str(v)) for k, v in criteria.items()]
    elif isinstance(criteria, list):
        items = []
        for c in criteria:
            if isinstance(c, dict) and "name" in c:
                items.append((str(c["name"]), str(c.get("description", ""))))
            else:
                items.append((str(c), ""))
    else:
        raise RequestError("criteria must be an object or a list")
    if not items:
        raise RequestError("criteria must not be empty")
    if len(items) > MAX_OPTIONS:
        raise RequestError(f"at most {MAX_OPTIONS} criteria per question")
    if len({n for n, _ in items}) != len(items):
        raise RequestError("criteria names must be unique")
    return items


def validate(body) -> tuple[str, str, dict]:
    if not isinstance(body, dict):
        raise RequestError("body must be a JSON object")
    state = body.get("state")
    if not isinstance(state, str):
        raise RequestError("state must be a string")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise RequestError("questions must be a non-empty object")
    if len(qs) > MAX_QUESTIONS:
        raise RequestError(f"at most {MAX_QUESTIONS} questions")
    for qid, q in qs.items():
        if not isinstance(q, dict) or q.get("type") not in QUESTION_TYPES:
            raise RequestError(f"question {qid!r}: type must be one of {QUESTION_TYPES}")
        if q.get("instructions") is not None and not isinstance(q["instructions"], str):
            raise RequestError(f"question {qid!r}: instructions must be a string")
        if q["type"] in ("choice", "score"):
            try:
                _criteria_items(q.get("criteria"))
            except RequestError as e:
                raise RequestError(f"question {qid!r}: {e}") from None
    return str(body.get("model") or ""), state, qs


def _level_value(name: str):
    try:
        v = float(name)
    except ValueError:
        return name
    return int(v) if v.is_integer() else v


class Decider:
    """Owns the model; one forward at a time (the model is not re-entrant on MPS)."""

    def __init__(self, ckpt: Path, tokenizer: Path, device: str = "cpu", max_len: int = 512):
        self.tok = Tokenizer.from_file(str(tokenizer))
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = Sys1Config(**blob["config"])
        if cfg.vocab_size != self.tok.get_vocab_size():
            raise SystemExit(f"tokenizer vocab {self.tok.get_vocab_size()} != checkpoint vocab {cfg.vocab_size}")
        self.model = Sys1Model(cfg)
        self.model.load_state_dict(blob["model"])
        self.model.to(device).eval()
        self.device = device
        self.max_len = min(max_len, cfg.max_seq)
        self.sid = {t: self.tok.token_to_id(t) for t in ("[PAD]", "[GOAL]", "[OPTS]", "[OPT]", "[DECIDE]")}
        if any(v is None for v in self.sid.values()):
            raise SystemExit(f"tokenizer lacks special tokens: {self.sid}")
        self.name = f"sys1:{ckpt.parent.name}"
        self.ckpt = str(ckpt)
        self.lock = threading.Lock()

    def _encode(self, prefix: str, options: list[str]):
        ids = [self.sid["[GOAL]"]] + self.tok.encode(" " + prefix).ids
        tail, pos = [self.sid["[OPTS]"]], []
        for o in options:
            tail += [self.sid["[OPT]"]] + self.tok.encode(" " + o).ids
            pos.append(len(tail) - 1)
        tail.append(self.sid["[DECIDE]"])
        if len(tail) >= self.max_len:
            raise RequestError(f"options alone exceed the {self.max_len}-token context")
        ids = ids[: self.max_len - len(tail)]
        return ids + tail, [len(ids) + p for p in pos], len(ids) + len(tail) - 1

    @torch.no_grad()
    def answer(self, state: str, questions: dict) -> tuple[dict, int]:
        plan, enc = [], []
        for qid, q in questions.items():
            if q["type"] == "noul":
                prefix = state + (" Question: " + q["instructions"] if q.get("instructions") else "")
                enc.append(self._encode(prefix, []))
                plan.append((qid, "noul", None))
            else:
                items = _criteria_items(q["criteria"])
                enc.append(self._encode(state, [f"{n}: {d}" if d else n for n, d in items]))
                plan.append((qid, q["type"], [n for n, _ in items]))
        B, T = len(enc), max(len(e[0]) for e in enc)
        K = max(1, max(len(e[1]) for e in enc))
        ids = torch.full((B, T), self.sid["[PAD]"], dtype=torch.long)
        opt_pos = torch.zeros((B, K), dtype=torch.long)
        opt_mask = torch.zeros((B, K), dtype=torch.bool)
        decide = torch.zeros(B, dtype=torch.long)
        for i, (x, op, dp) in enumerate(enc):
            ids[i, : len(x)] = torch.tensor(x)
            if op:
                opt_pos[i, : len(op)] = torch.tensor(op)
                opt_mask[i, : len(op)] = True
            decide[i] = dp
        dev = self.device
        with self.lock:  # right padding + causal attention: padding never affects real positions
            scores, noul, _ = self.model.decide(self.model.hidden(ids.to(dev)), decide.to(dev),
                                                opt_pos.to(dev), opt_mask.to(dev))
            probs = F.softmax(scores.float().masked_fill(~opt_mask.to(dev), -1e9), -1).cpu()
            p_yes = torch.sigmoid(noul.float()).cpu()
        answers = {}
        for i, (qid, kind, names) in enumerate(plan):
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": round(float(p_yes[i]), 6)}
                continue
            p = [float(v) for v in probs[i, : len(names)]]
            top = max(range(len(names)), key=p.__getitem__)
            dist = {n: round(v, 6) for n, v in zip(names, p)}
            conf = min(1.0, max(0.0, round(p[top], 6)))
            if kind == "choice":
                answers[qid] = {"type": "choice", "choice": names[top], "probabilities": dist, "confidence": conf}
            else:
                answers[qid] = {"type": "score", "score": _level_value(names[top]), "probabilities": dist,
                                "confidence": conf}
        return answers, sum(len(e[0]) for e in enc)


def make_handler(decider: Decider, timeout_s: float, max_body: int, pool: ThreadPoolExecutor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "sys1-serve/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet by default; errors still reported via _send
            pass

        def _send(self, code: int, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, kind: str, msg: str):
            self._send(code, {"type": "error", "error": {"type": kind, "message": msg}})

        def do_GET(self):
            if self.path == "/healthz":
                return self._send(200, {"ok": True, "model": decider.name, "checkpoint": decider.ckpt,
                                        "device": decider.device})
            self._error(404, "not_found_error", "unknown path")

        def do_POST(self):
            if self.path.split("?")[0] != "/v1/systemone":
                return self._error(404, "not_found_error", "unknown path")
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._error(411, "invalid_request_error", "Content-Length required")
            if n < 0 or n > max_body:
                self.close_connection = True
                return self._error(413, "request_too_large", f"body exceeds {max_body} bytes")
            try:
                body = json.loads(self.rfile.read(n))
                model, state, qs = validate(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._error(400, "invalid_request_error", "body is not valid JSON")
            except RequestError as e:
                return self._error(400, "invalid_request_error", str(e))
            fut = pool.submit(decider.answer, state, qs)
            try:
                answers, ntok = fut.result(timeout=timeout_s)
            except FutureTimeout:
                fut.cancel()
                return self._error(504, "timeout_error", f"inference exceeded {timeout_s}s")
            except RequestError as e:
                return self._error(400, "invalid_request_error", str(e))
            except Exception as e:  # report, never kill the server
                return self._error(500, "api_error", f"{type(e).__name__}: {e}")
            self._send(200, {"model": decider.name,
                             "answers": answers, "usage": {"input_tokens": ntok}})

    return Handler


def build_server(decider: Decider, port: int = DEFAULT_PORT, timeout_s: float = 10.0,
                 max_body: int = MAX_BODY_BYTES, workers: int = 2) -> ThreadingHTTPServer:
    """Always binds 127.0.0.1. port=0 picks a free port (see server.server_address)."""
    pool = ThreadPoolExecutor(workers, thread_name_prefix="sys1-infer")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(decider, timeout_s, max_body, pool))
    httpd.daemon_threads = True
    httpd.timeout = timeout_s
    httpd._sys1_pool = pool
    return httpd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-body", type=int, default=MAX_BODY_BYTES)
    ap.add_argument("--timeout", type=float, default=10.0, help="per-request inference timeout, seconds")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    a = ap.parse_args()
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available")
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    decider = Decider(a.checkpoint, a.tokenizer, a.device, a.max_len)
    httpd = build_server(decider, a.port, a.timeout, a.max_body)
    host, port = httpd.server_address[:2]
    print(json.dumps({"listening": f"http://{host}:{port}/v1/systemone", "model": decider.name,
                      "device": a.device, "load_s": round(time.time() - t0, 2)}), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

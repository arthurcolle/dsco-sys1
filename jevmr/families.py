"""Prompt families: generated agent situations with rule-derived correct answers.

Each family enumerates a slot grammar, renders every combination in several
phrasings, and attaches:
  - one Choice question over op-codes (the decision under test), with the
    correct option derived by rule from the op-code definitions, and
  - Noul questions about facts stated in the situation (sanity probes).
All content is synthetic. Nothing from private logs is used.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

OPS = {o["mnemonic"]: o for o in json.loads((Path(__file__).resolve().parents[1] / "opcodes.json").read_text())["ops"]}


def option(mnemonic: str) -> tuple[str, str]:
    return mnemonic, OPS[mnemonic]["desc"]


class Family:
    name = ""
    question = "Which operation should the agent perform next?"
    options: list[str] = []
    slots: dict[str, list] = {}
    phrasings: list[str] = []
    policy = ""  # stated rules the oracle relies on; the answer must follow from the text

    def oracle(self, s: dict) -> str:
        raise NotImplementedError

    def valid(self, s: dict) -> bool:
        """Exclude slot combinations whose rendered text would contradict itself."""
        return True

    def probes(self, s: dict) -> dict[str, tuple[str, bool]]:
        return {}

    def items(self, rng: random.Random, limit: int | None = None):
        keys = list(self.slots)
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(self.slots[k] for k in keys))]
        combos = [c for c in combos if self.valid(c)]
        rng.shuffle(combos)
        if limit:
            combos = combos[:limit]
        out = []
        for si, s in enumerate(combos):
            for pi, phr in enumerate(self.phrasings):
                state = phr.format(**{k: self.render(k, v) for k, v in s.items()})
                if self.policy:
                    state += " " + self.policy
                out.append({"family": self.name, "slots": s, "situation": f"{self.name}:{si}", "phrasing": pi,
                            "state": state, "gold": self.oracle(s), "probes": dict(self.probes(s))})
        return out

    def render(self, key: str, value):
        return value

    def request(self, item: dict, model: str) -> dict:
        questions = {"op": {"type": "choice", "instructions": self.question,
                            "criteria": dict(option(m) for m in self.options)}}
        for k, (text, _) in item["probes"].items():
            questions[k] = {"type": "noul", "instructions": text}
        return {"model": model, "state": item["state"], "questions": questions}


class ContextPressure(Family):
    name = "ctx_pressure"
    options = ["CTX.MASK_BATCH", "CTRL.CONTINUE", "CTRL.FINAL", "CTX.UNMASK.OBSERVATION_BLOCK"]
    slots = {"pct": [22, 45, 63, 78, 91, 97], "done": [False, True], "needs_old": [False, True],
             "task": ["refactoring a parser", "triaging failing tests", "writing a migration", "researching an API"]}
    phrasings = [
        "The agent is {task}. Its context window is {pct}% full. {done} {needs_old}",
        "Status: task = {task}; context usage = {pct}% of the window. {done} {needs_old}",
        "While {task}, the agent notices the prompt now fills {pct}% of its context. {done} {needs_old}",
    ]
    policy = ("Policy: once context usage reaches 60%, old tool results outside the recent window are masked "
              "in one batch; below 60% the agent simply continues. Priorities: if the answer is ready, finish; "
              "otherwise, if a step needs a masked result, restore it; otherwise apply the 60% rule.")

    def render(self, k, v):
        if k == "done":
            return "All acceptance checks have passed and the answer is ready." if v else "The task is not finished yet."
        if k == "needs_old":
            return ("The next step needs the exact text of an earlier tool result that was replaced by a keyed placeholder."
                    if v else "Earlier tool results are no longer needed verbatim.")
        return v

    def valid(self, s):
        # "the answer is ready" and "the next step needs earlier text" contradict each other
        return not (s["done"] and s["needs_old"])

    def oracle(self, s):
        if s["done"]:
            return "CTRL.FINAL"
        if s["needs_old"]:
            return "CTX.UNMASK.OBSERVATION_BLOCK"
        return "CTX.MASK_BATCH" if s["pct"] >= 60 else "CTRL.CONTINUE"

    def probes(self, s):
        return {"over60": ("Is context usage at or above 60%?", s["pct"] >= 60)}


class ToolFailure(Family):
    name = "tool_failure"
    options = ["ACT.RETRY.TOOL_CALL", "ERR.BOUNDED_ALTERNATIVE", "GOV.NO_AUTHORITY", "CTRL.BACKOFF"]
    slots = {"failure": ["timed out", "returned HTTP 429 Too Many Requests", "was blocked because the agent's capability policy does not grant this operation",
                         "failed because the file does not exist", "rejected its arguments as invalid JSON",
                         "returned a connection reset"],
             "attempts": [1, 2, 4], "tool": ["bash", "http_request", "read_file", "git push"]}
    phrasings = [
        "The agent ran {tool}. The call {failure}. This has happened {attempts} time(s) in a row with the same input.",
        "Tool {tool} {failure} (attempt {attempts}; arguments unchanged).",
        "After {attempts} identical attempt(s), {tool} {failure}.",
    ]

    policy = ("Policy: a transient failure (timeout or connection reset) is retried once, backed off on the second "
              "failure, and replaced by a different approach after that. A rate limit always means back off. Invalid "
              "arguments are corrected and retried once, then the approach changes. A missing file means a different "
              "approach. A capability-policy block means the agent lacks authority. The call has no side effects, "
              "so repeating it is safe.")
    LOCAL = ("bash", "read_file")
    NETWORK = ("returned HTTP 429 Too Many Requests", "returned a connection reset")

    def valid(self, s):
        if s["tool"] in self.LOCAL and s["failure"] in self.NETWORK:
            return False  # local tools cannot return HTTP status or connection resets
        if s["tool"] in ("bash", "git push") and "invalid JSON" in s["failure"]:
            return False  # these take command strings, not JSON arguments
        return True

    def oracle(self, s):
        f, n = s["failure"], s["attempts"]
        if "capability policy" in f:
            return "GOV.NO_AUTHORITY"
        if "429" in f:
            return "CTRL.BACKOFF"
        if "does not exist" in f:
            return "ERR.BOUNDED_ALTERNATIVE"
        if "invalid JSON" in f:
            return "ACT.RETRY.TOOL_CALL" if n == 1 else "ERR.BOUNDED_ALTERNATIVE"
        return {1: "ACT.RETRY.TOOL_CALL", 2: "CTRL.BACKOFF"}.get(n, "ERR.BOUNDED_ALTERNATIVE")

    def probes(self, s):
        return {"repeated": ("Has the same call failed more than once?", s["attempts"] > 1)}


FAMILIES = {f.name: f for f in (ContextPressure(), ToolFailure())}

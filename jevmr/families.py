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


class ComposedFamily(Family):
    """Family whose phrasing fields are derived from several slots at once (fmt), not one slot each."""

    def fmt(self, s: dict) -> dict:
        raise NotImplementedError

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
                state = phr.format(**self.fmt(s))
                if self.policy:
                    state += " " + self.policy
                out.append({"family": self.name, "slots": s, "situation": f"{self.name}:{si}", "phrasing": pi,
                            "state": state, "gold": self.oracle(s), "probes": dict(self.probes(s))})
        return out


class DoneCheck(Family):
    name = "done_check"
    options = ["CTRL.FINAL", "VERIF.RUN_TESTS", "CTRL.CONTINUE"]
    slots = {"tests": ["pass", "fail", "not_run", "stale"], "items": [True, False],
             "task": ["adding a CSV export endpoint", "fixing a date-parsing bug", "renaming a config option",
                      "adding retry logic to an HTTP client"]}
    phrasings = [
        "The agent is {task}. {items} {tests}",
        "Task: {task}. Requirements: {items} Tests: {tests}",
        "While {task}, the agent reviews its state. {tests} {items}",
    ]
    policy = ("Policy, applied in this order: (1) if any test is failing, keep working on the fix as the next "
              "planned step; (2) otherwise, if the tests have not been run since the last code edit, run the test "
              "suite; (3) otherwise, if the tests pass and every requested item is implemented, emit the final "
              "answer; (4) otherwise, proceed to the next planned step to implement what is missing.")

    def render(self, k, v):
        if k == "items":
            return ("Every requested item is implemented." if v
                    else "At least one requested item is not implemented yet.")
        if k == "tests":
            return {"pass": "The full test suite was run after the last code edit and every test passed.",
                    "fail": "The full test suite was run after the last code edit and two tests failed.",
                    "not_run": "The test suite has not been run at all yet.",
                    "stale": "The tests passed earlier, but code has been edited since that run, so no test run "
                             "covers the current code."}[v]
        return v

    def oracle(self, s):
        if s["tests"] == "fail":
            return "CTRL.CONTINUE"
        if s["tests"] in ("not_run", "stale"):
            return "VERIF.RUN_TESTS"
        return "CTRL.FINAL" if s["items"] else "CTRL.CONTINUE"

    def probes(self, s):
        return {"current_tests": ("Has the test suite been run on the current code?", s["tests"] in ("pass", "fail")),
                "all_items": ("Does the situation say that every requested item has been written, regardless of test results?",
                              s["items"])}


class LoopBreak(Family):
    name = "loop_break"
    options = ["CTRL.BREAK_LOOP", "CTRL.CONTINUE", "CTRL.FINAL"]
    slots = {"repeats": [1, 2, 3, 5], "new_info": [True, False], "ready": [False, True],
             "tool": ["grep", "list_directory", "read_file", "a search query"]}
    phrasings = [
        "The agent has just issued {tool} with identical arguments {repeats} time(s) in a row. {new_info} {ready}",
        "Recent history: {repeats} consecutive identical call(s) to {tool}. {new_info} {ready}",
        "Looking back, the agent sees {tool} called {repeats} time(s) back to back with the same arguments. {new_info} {ready}",
    ]
    policy = ("Policy, applied in this order: (1) if the answer is ready, emit the final answer; (2) otherwise, if "
              "the same call with identical arguments has been made 3 or more times in a row and the latest result "
              "added no new information, change strategy to escape the loop; (3) otherwise, proceed to the next "
              "planned step.")

    def render(self, k, v):
        if k == "new_info":
            return ("The latest result contained information the agent had not seen before." if v
                    else "The latest result was identical to the previous one and added no new information.")
        if k == "ready":
            return ("The agent already has everything it needs and the answer is ready." if v
                    else "The answer is not ready yet.")
        return v

    def valid(self, s):
        # a single call has no "previous result" to compare against
        return not (s["repeats"] == 1 and not s["new_info"])

    def oracle(self, s):
        if s["ready"]:
            return "CTRL.FINAL"
        if s["repeats"] >= 3 and not s["new_info"]:
            return "CTRL.BREAK_LOOP"
        return "CTRL.CONTINUE"

    def probes(self, s):
        return {"three_plus": ("Has the identical call been made 3 or more times in a row?", s["repeats"] >= 3)}


class BudgetGuard(Family):
    name = "budget_guard"
    options = ["CTRL.CONTINUE", "BUD.CHEAP_PATH", "CTRL.STOP"]
    slots = {"pct": [25, 55, 85, 95, 100, 110], "fits": [True, False], "cheaper": [True, False],
             "task": ["summarizing a large log archive", "generating unit tests", "translating documentation",
                      "labeling a dataset"]}
    phrasings = [
        "The agent is {task}. It has spent {pct}% of its spending cap. {fits} {cheaper}",
        "Budget check while {task}: {pct}% of the cap used. {fits} {cheaper}",
        "Spend so far on {task} is {pct}% of the allowed cap. {fits} {cheaper}",
    ]
    policy = ("Policy, applied in this order: (1) at or above 100% of the cap, stop and return control; "
              "(2) if the estimated cost to finish exceeds the remaining budget, switch to the cheaper adequate "
              "approach if one exists, otherwise stop; (3) if spend is at or above 80% of the cap and a cheaper "
              "adequate approach exists, switch to it; (4) otherwise continue with the current plan.")

    def render(self, k, v):
        if k == "fits":
            return ("The estimated cost to finish fits within the remaining budget." if v
                    else "The estimated cost to finish with the current approach exceeds the remaining budget.")
        if k == "cheaper":
            return ("A cheaper approach that is still adequate for the task is available." if v
                    else "No cheaper approach would be adequate for the task.")
        return v

    def valid(self, s):
        # at or over the cap there is no remaining budget for anything to fit in
        return not (s["pct"] >= 100 and s["fits"])

    def oracle(self, s):
        if s["pct"] >= 100:
            return "CTRL.STOP"
        if not s["fits"]:
            return "BUD.CHEAP_PATH" if s["cheaper"] else "CTRL.STOP"
        if s["pct"] >= 80 and s["cheaper"]:
            return "BUD.CHEAP_PATH"
        return "CTRL.CONTINUE"

    def probes(self, s):
        return {"over80": ("Is spend at or above 80% of the cap?", s["pct"] >= 80)}


class Clarify(ComposedFamily):
    name = "clarify"
    options = ["COMM.ASK.USER", "GOAL.NEXT_BEST_ACTION", "CTRL.WAIT_USER"]
    REQUESTS = {
        "logs": ("clean up the logs", "deleting old log files or reformatting the log messages"),
        "fast": ("make the build faster", "caching dependencies or dropping slow optional checks"),
        "users": ("update the users table", "adding a column or backfilling existing rows"),
        "docs": ("fix the docs", "correcting typos or rewriting the outdated install guide"),
    }
    slots = {"req": list(REQUESTS), "ambiguous": [True, False], "default": [True, False], "asked": [False, True]}
    phrasings = [
        "The user asked the agent to \"{req}\". {ambiguous} {default} {asked}",
        "Instruction received: \"{req}\". {ambiguous} {default} {asked}",
        "The agent's current instruction is \"{req}\". {ambiguous} {default} {asked}",
    ]
    policy = ("Policy, applied in this order: (1) if the instruction has one clear meaning, pick the highest-value "
              "unblocked action and proceed; (2) if it is ambiguous but a safe, reversible default interpretation "
              "exists, proceed with that default by picking the highest-value unblocked action; (3) if it is "
              "ambiguous with no safe default and a clarifying question is already pending, block until the user "
              "responds; (4) if it is ambiguous with no safe default and nothing has been asked yet, ask the user "
              "a clarifying question.")

    def fmt(self, s):
        req, alts = self.REQUESTS[s["req"]]
        amb = (f"It is ambiguous: it could mean {alts}, and the two lead to different results." if s["ambiguous"]
               else f"Earlier in the conversation the user specified exactly which they meant, so the instruction "
                    f"has one clear meaning.")
        dflt = ("The project guide names a safe, easily reversible default interpretation." if s["default"]
                else "There is no safe default: either interpretation could be hard to undo.")
        ask = ("The agent already sent a clarifying question about this and the user has not replied yet."
               if s["asked"] else "No clarifying question has been asked yet.")
        return {"req": req, "ambiguous": amb, "default": dflt, "asked": ask}

    def valid(self, s):
        if not s["ambiguous"] and (s["asked"] or not s["default"]):
            return False  # no question pending on a clear instruction; "no safe default" only matters when ambiguous
        if s["default"] and s["asked"]:
            return False  # a safe default would have been used instead of asking
        return True

    def oracle(self, s):
        if not s["ambiguous"] or s["default"]:
            return "GOAL.NEXT_BEST_ACTION"
        return "CTRL.WAIT_USER" if s["asked"] else "COMM.ASK.USER"

    def probes(self, s):
        return {"ambiguous": ("Does the instruction have more than one plausible meaning?", s["ambiguous"])}


class RetrievalNeed(Family):
    name = "retrieval_need"
    options = ["KNOW.SEARCH.WEB", "KNOW.SEARCH.CODEBASE", "CTRL.FINAL"]
    FACTS = {
        "the latest released version of the requests library on PyPI": "web",
        "today's status of a public cloud provider's incident page": "web",
        "the current exchange rate between EUR and USD": "web",
        "which function in this repository parses the config file": "code",
        "where this project defines its database connection pool size": "code",
        "which modules in this repository import the retry helper": "code",
    }
    slots = {"fact": list(FACTS), "known": [False, True]}
    phrasings = [
        "To answer the user, the agent needs {fact}. {known}",
        "Missing piece for the answer: {fact}. {known}",
        "The user's question depends on {fact}. {known}",
    ]
    policy = ("Policy, applied in this order: (1) if the needed fact was already retrieved this session and the "
              "answer is complete, emit the final answer; (2) if the fact is about this repository's own code, "
              "search the code repository; (3) if the fact is public information that changes over time and lives "
              "outside the repository, search the web.")

    def render(self, k, v):
        if k == "known":
            return ("The agent already retrieved this fact earlier in the session, verified it, and the answer is "
                    "complete." if v else "The agent has not looked this up yet and does not know it.")
        return v

    def oracle(self, s):
        if s["known"]:
            return "CTRL.FINAL"
        return "KNOW.SEARCH.CODEBASE" if self.FACTS[s["fact"]] == "code" else "KNOW.SEARCH.WEB"

    def probes(self, s):
        return {"in_repo": ("Is the needed fact about this repository's own code?", self.FACTS[s["fact"]] == "code")}


class Delegation(ComposedFamily):
    name = "delegation"
    options = ["SWARM.FAN_OUT", "CTRL.CONTINUE", "SWARM.FAN_IN"]
    slots = {"n": [1, 2, 3, 5, 8], "dependent": [False, True], "returned": [False, True],
             "work": ["auditing service configs", "writing per-module docstrings", "benchmarking parsers",
                      "migrating test files"]}
    phrasings = [
        "The agent is {work}. {plan} {returned}",
        "Job: {work}. {plan} {returned}",
        "Planning {work}: {plan} {returned}",
    ]
    policy = ("Policy, applied in this order: (1) if a fan-out was already sent and every branch has returned, "
              "combine the results; (2) if there are 3 or more subtasks and none depends on another, send them to "
              "many agents at once; (3) otherwise (fewer than 3 subtasks, or subtasks that depend on each other), "
              "do them one by one by proceeding to the next planned step.")

    def fmt(self, s):
        n = s["n"]
        if n == 1:
            plan = "The work is a single subtask."
        elif s["dependent"]:
            plan = f"It splits into {n} subtasks, and each one needs the output of the one before it."
        else:
            plan = f"It splits into {n} subtasks that are fully independent of each other."
        ret = ("A fan-out to parallel agents was already sent, and every branch has now returned its result."
               if s["returned"] else "Nothing has been dispatched yet.")
        return {"work": s["work"], "plan": plan, "returned": ret}

    def valid(self, s):
        if s["n"] == 1 and s["dependent"]:
            return False  # a single subtask cannot depend on another
        if s["returned"] and (s["dependent"] or s["n"] < 3):
            return False  # a fan-out only happens for 3+ independent subtasks
        return True

    def oracle(self, s):
        if s["returned"]:
            return "SWARM.FAN_IN"
        if s["n"] >= 3 and not s["dependent"]:
            return "SWARM.FAN_OUT"
        return "CTRL.CONTINUE"

    def probes(self, s):
        return {"independent3": ("Are there 3 or more subtasks that do not depend on each other?",
                                 s["n"] >= 3 and not s["dependent"])}


FAMILIES = {f.name: f for f in (ContextPressure(), ToolFailure(), DoneCheck(), LoopBreak(), BudgetGuard(),
                                Clarify(), RetrievalNeed(), Delegation())}

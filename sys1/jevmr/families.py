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

from sys1.paths import OPCODES_JSON

OPS = {o["mnemonic"]: o for o in json.loads(OPCODES_JSON.read_text())["ops"]}


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


# ---------------------------------------------------------------------------
# Routine-engineering families over the newer op domains (codes >= 1500).
# They stay out of the fractal composition (its cross-family priority covers
# only the agent-mechanics families above): standalone = True.
# ---------------------------------------------------------------------------


class DbIncident(ComposedFamily):
    name = "db_incident"
    standalone = True
    options = ["DB.ROLLBACK.TRANSACTION", "DB.CREATE.SQL_QUERY", "DB.READ.QUERY_PLAN", "DB.CREATE.DB_INDEX"]
    slots = {"deadlock": [False, True], "nplus1": [False, True], "slow": [False, True],
             "plan": ["unread", "full_scan"], "table": ["orders", "invoices", "sessions", "audit_events"]}
    phrasings = [
        "The agent is investigating the {table} service database. {deadlock} {nplus1} {slow} {plan}",
        "Database incident notes for the {table} service: {slow} {deadlock} {plan} {nplus1}",
        "On-call review of the {table} database. {nplus1} {plan} {slow} {deadlock}",
    ]
    policy = ("Policy, applied in this order: (1) if the agent's own transaction was chosen as a deadlock victim "
              "and is still open, roll back that transaction; (2) otherwise, if a code loop issues one query per "
              "item, write one new batched SQL query to replace the loop; (3) otherwise, if a query exceeds its "
              "latency budget and its query plan has not been read yet, read the query plan; (4) otherwise, if the "
              "query plan has been read and shows a full table scan on the filtered column, create an index on "
              "that column.")

    def fmt(self, s):
        t = s["table"]
        dl = (f"The agent's own transaction on {t} was chosen as a deadlock victim and is still open."
              if s["deadlock"] else "No deadlock has been reported.")
        n1 = (f"The request handler loops over 300 {t} rows and issues one separate query per row."
              if s["nplus1"] else "Each request issues its queries in a single batch, with no per-row loop.")
        sl = (f"A report query on {t} takes 4.2 s against a 200 ms latency budget."
              if s["slow"] else "Every query is within its latency budget.")
        if not s["slow"]:
            pl = "No query plan needs reading."
        elif s["plan"] == "unread":
            pl = "The query plan for the slow query has not been read yet."
        else:
            pl = "The query plan for the slow query has been read: it shows a full table scan on the filtered column."
        return {"table": t, "deadlock": dl, "nplus1": n1, "slow": sl, "plan": pl}

    def valid(self, s):
        if not (s["deadlock"] or s["nplus1"] or s["slow"]):
            return False  # nothing to act on
        return s["slow"] or s["plan"] == "unread"  # a plan only exists for a slow query

    def oracle(self, s):
        if s["deadlock"]:
            return "DB.ROLLBACK.TRANSACTION"
        if s["nplus1"]:
            return "DB.CREATE.SQL_QUERY"
        return "DB.READ.QUERY_PLAN" if s["plan"] == "unread" else "DB.CREATE.DB_INDEX"

    def probes(self, s):
        return {"slow": ("Does a query exceed its latency budget?", s["slow"]),
                "deadlock": ("Is the agent's transaction an open deadlock victim?", s["deadlock"])}


class CiGate(ComposedFamily):
    name = "ci_gate"
    standalone = True
    options = ["CI.FREEZE", "CI.INSPECT.CI_JOB", "CI.ROLLBACK.ROLLOUT", "CI.PROMOTE.RELEASE", "CI.CANARY"]
    slots = {"freeze": [False, True], "build": ["green", "red"], "canary": ["none", "bad", "good"],
             "service": ["checkout-api", "search-indexer", "billing-worker", "auth-gateway"]}
    phrasings = [
        "The agent manages deploys for {service}. {build} {canary} {freeze}",
        "Deploy status for {service}: {freeze} {build} {canary}",
        "Release check on {service}. {canary} {freeze} {build}",
    ]
    policy = ("Policy, applied in this order: (1) if a deploy freeze is in effect, hold all deploys until it "
              "clears; (2) otherwise, if the main branch build is failing, inspect the failing CI job; "
              "(3) otherwise, if a canary is running and its error rate is above the baseline, roll back the "
              "staged rollout; (4) otherwise, if the canary finished its full watch window with an error rate at "
              "or below the baseline, promote the release; (5) otherwise, with a passing build and no canary "
              "started, ship to a canary first.")

    def fmt(self, s):
        fr = ("A deploy freeze is in effect until the quarterly audit finishes." if s["freeze"]
              else "No deploy freeze is in effect.")
        bd = ("The main branch build is failing: the integration-test job is red." if s["build"] == "red"
              else "The main branch build is passing.")
        cn = {"none": "No canary has been started for the new release yet.",
              "bad": "A canary of the new release is running on 5% of traffic, and its error rate is 3.1% "
                     "against a 0.4% baseline.",
              "good": "A canary of the new release finished its full 2-hour watch window on 5% of traffic with a "
                      "0.3% error rate against a 0.4% baseline."}[s["canary"]]
        return {"service": s["service"], "freeze": fr, "build": bd, "canary": cn}

    def valid(self, s):
        if s["build"] == "red" and s["canary"] != "none":
            return False  # a release only reaches canary from a passing build
        if s["freeze"] and s["canary"] == "bad":
            return False  # a live failing canary during a freeze puts two rules in tension
        return True

    def oracle(self, s):
        if s["freeze"]:
            return "CI.FREEZE"
        if s["build"] == "red":
            return "CI.INSPECT.CI_JOB"
        return {"bad": "CI.ROLLBACK.ROLLOUT", "good": "CI.PROMOTE.RELEASE", "none": "CI.CANARY"}[s["canary"]]

    def probes(self, s):
        return {"freeze": ("Is a deploy freeze in effect?", s["freeze"]),
                "red": ("Is the main branch build failing?", s["build"] == "red")}


class ApiPaging(ComposedFamily):
    name = "api_paging"
    standalone = True
    options = ["API.DIFF.API_SCHEMA", "API.THROTTLE.API_ENDPOINT", "API.IDEMPOTENCY_KEY", "API.PAGINATE_ALL",
               "API.INVOKE.API_ENDPOINT", "CTRL.FINAL"]
    slots = {"breaking": [False, True], "quota": [4, 9, 45, 80], "kind": ["list", "create"],
             "cursor": [False, True], "key": [False, True], "api": ["the payments API", "the CRM API",
                                                                   "the ticketing API"]}
    phrasings = [
        "The agent is working with {api}. {task} {breaking} {quota} {state}",
        "Integration log for {api}: {task} {quota} {state} {breaking}",
        "While calling {api}, the agent reviews the situation. {state} {breaking} {task} {quota}",
    ]
    policy = ("Policy, applied in this order: (1) if responses now fail schema validation after the provider "
              "announced an incompatible contract change, compute the difference between the old and new API "
              "schema; (2) otherwise, if less than 10% of the API quota remains, throttle calls to the endpoint; "
              "(3) otherwise, for a request that creates something: if it has no idempotency key, attach one, and "
              "if it already has one, send the request; (4) otherwise, for a listing: if the last response "
              "included a next-page cursor, follow cursors until every page is fetched, and if it had no cursor, "
              "every page is in hand, so emit the final answer.")

    def fmt(self, s):
        task = ("The task is to list every open record, page by page." if s["kind"] == "list"
                else "The task is to create one new record with a POST request.")
        br = ("The provider announced an incompatible contract change, and responses now fail schema validation."
              if s["breaking"] else "Responses pass schema validation and the contract is unchanged.")
        q = f"{s['quota']}% of the API quota remains for this window."
        if s["kind"] == "list":
            st = ("The last page returned included a next-page cursor." if s["cursor"]
                  else "The last page returned had no next-page cursor, so all pages have been fetched.")
        else:
            st = ("The POST request already carries a unique idempotency key." if s["key"]
                  else "The POST request has no idempotency key yet.")
        return {"api": s["api"], "task": task, "breaking": br, "quota": q, "state": st}

    def valid(self, s):
        if s["kind"] == "list" and s["key"]:
            return False  # the key slot only renders for creates
        if s["kind"] == "create" and s["cursor"]:
            return False  # the cursor slot only renders for listings
        if s["kind"] == "list" and not s["cursor"] and s["quota"] < 10 and not s["breaking"]:
            return False  # a finished listing makes no further calls, so there is nothing left to throttle
        return True

    def oracle(self, s):
        if s["breaking"]:
            return "API.DIFF.API_SCHEMA"
        if s["quota"] < 10:
            return "API.THROTTLE.API_ENDPOINT"
        if s["kind"] == "create":
            return "API.INVOKE.API_ENDPOINT" if s["key"] else "API.IDEMPOTENCY_KEY"
        return "API.PAGINATE_ALL" if s["cursor"] else "CTRL.FINAL"

    def probes(self, s):
        return {"low_quota": ("Does less than 10% of the API quota remain?", s["quota"] < 10)}


class QueueHealth(ComposedFamily):
    name = "queue_health"
    standalone = True
    options = ["QUEUE.TAG.MESSAGE", "QUEUE.ORDERING_REQUIRED", "QUEUE.BACKPRESSURE",
               "QUEUE.REPLAY.DEAD_LETTER_QUEUE", "QUEUE.MONITOR.CONSUMER"]
    slots = {"poison": [False, True], "order": [False, True], "lag": ["stable", "growing"],
             "dlq": ["empty", "fixed", "unfixed"], "topic": ["order-events", "email-jobs", "ledger-updates",
                                                             "thumbnail-requests"]}
    phrasings = [
        "The agent watches the {topic} queue. {poison} {order} {lag} {dlq}",
        "Queue health report for {topic}: {lag} {dlq} {poison} {order}",
        "Checking the {topic} consumers, the agent finds: {order} {poison} {dlq} {lag}",
    ]
    policy = ("Policy, applied in this order: (1) if one message has failed on every one of several processing "
              "attempts and is blocking the consumer, tag it as a poison message so it is skipped; (2) otherwise, "
              "if the events must be processed strictly in order but the consumer is handling them out of order, "
              "enforce strict ordering; (3) otherwise, if producers outpace consumers and the backlog keeps "
              "growing, slow intake; (4) otherwise, if the dead-letter queue holds messages and the bug that sent "
              "them there has been fixed and deployed, replay the dead-letter queue; (5) otherwise, keep "
              "monitoring the consumer.")

    def fmt(self, s):
        po = ("One message has failed on each of its 5 processing attempts and is blocking the consumer."
              if s["poison"] else "No message is failing repeatedly.")
        od = ("Events for the same account must be processed strictly in order, but the consumer is currently "
              "handling them out of order on parallel workers." if s["order"]
              else "Events are processed in the order the topic requires.")
        lg = ("Producers are publishing faster than consumers can process, and the backlog has grown for 20 "
              "minutes." if s["lag"] == "growing" else "The backlog is stable and consumers keep up with producers.")
        dq = {"empty": "The dead-letter queue is empty.",
              "fixed": "The dead-letter queue holds 140 messages, and the bug that sent them there has been fixed "
                       "and deployed.",
              "unfixed": "The dead-letter queue holds 140 messages, and the bug that sent them there is not fixed "
                         "yet."}[s["dlq"]]
        return {"topic": s["topic"], "poison": po, "order": od, "lag": lg, "dlq": dq}

    def oracle(self, s):
        if s["poison"]:
            return "QUEUE.TAG.MESSAGE"
        if s["order"]:
            return "QUEUE.ORDERING_REQUIRED"
        if s["lag"] == "growing":
            return "QUEUE.BACKPRESSURE"
        if s["dlq"] == "fixed":
            return "QUEUE.REPLAY.DEAD_LETTER_QUEUE"
        return "QUEUE.MONITOR.CONSUMER"

    def probes(self, s):
        return {"growing": ("Is the backlog growing?", s["lag"] == "growing")}


class PerfTriage(ComposedFamily):
    name = "perf_triage"
    standalone = True
    options = ["PERF.BENCHMARK.HOT_PATH", "PERF.COMPARE.LATENCY_PROFILE", "PERF.PROFILE.HOT_PATH",
               "PERF.UPDATE.CACHE_LAYER"]
    slots = {"optimized": [False, True], "regression": [False, True], "profile": ["none", "captured", "recompute"],
             "path": ["the image upload handler", "the search endpoint", "the nightly report job",
                      "the login flow"]}
    phrasings = [
        "Users say {path} feels slow. {regression} {profile} {optimized}",
        "Performance ticket for {path}: {profile} {optimized} {regression}",
        "The agent is looking at the speed of {path}. {optimized} {regression} {profile}",
    ]
    policy = ("Policy, applied in this order: (1) if an optimization was just applied and has not been benchmarked "
              "yet, benchmark the hot path; (2) otherwise, if a specific change made the path slower and latency "
              "profiles from before and after that change are both available, compare the two latency profiles; "
              "(3) otherwise, if no profile has been captured, profile the hot path before changing any code; "
              "(4) otherwise, if the profile shows most time spent recomputing identical results, update the "
              "cache layer to reuse them.")

    def fmt(self, s):
        rg = ("Response time doubled right after a specific commit was merged." if s["regression"]
              else "No single change is linked to the slowness; it has always been this way.")
        if s["profile"] == "none":
            pf = "No profile of the path has been captured yet."
        elif s["profile"] == "captured":
            pf = "Latency profiles from before and after that commit have both been captured."
        else:
            pf = ("A latency profile has been captured, and it shows 70% of the time spent recomputing the same "
                  "results for identical inputs." + (" Profiles from before and after the commit are both "
                                                     "available." if s["regression"] else ""))
        op = ("A caching change was just applied to the path and has not been benchmarked yet." if s["optimized"]
              else "No optimization has been applied yet.")
        return {"path": s["path"], "regression": rg, "profile": pf, "optimized": op}

    def valid(self, s):
        if s["profile"] == "captured" and not s["regression"]:
            return False  # before/after profiles only exist for a regression
        if s["optimized"] and s["profile"] == "none":
            return False  # an optimization without any profile contradicts measure-first
        return True

    def oracle(self, s):
        if s["optimized"]:
            return "PERF.BENCHMARK.HOT_PATH"
        if s["regression"] and s["profile"] != "none":
            return "PERF.COMPARE.LATENCY_PROFILE"
        if s["profile"] == "none":
            return "PERF.PROFILE.HOT_PATH"
        return "PERF.UPDATE.CACHE_LAYER"

    def probes(self, s):
        return {"profiled": ("Has a latency profile been captured?", s["profile"] != "none")}


class PkgConflict(ComposedFamily):
    name = "pkg_conflict"
    standalone = True
    options = ["PKG.INSPECT.PACKAGE_VERSION", "PKG.PIN_VERSION.PACKAGE_VERSION", "PKG.UPDATE.LOCKFILE",
               "PKG.REPRODUCIBLE_INSTALL"]
    slots = {"conflict": ["none", "uninspected", "overlap"], "lock": ["current", "stale"], "mismatch": [False, True],
             "project": ["a Python web service", "a Node.js CLI", "a Rust parser crate", "a Ruby admin app"]}
    phrasings = [
        "The agent maintains {project}. {conflict} {lock} {mismatch}",
        "Dependency report for {project}: {lock} {mismatch} {conflict}",
        "While setting up {project}, the agent notes: {mismatch} {conflict} {lock}",
    ]
    policy = ("Policy, applied in this order: (1) if two dependencies require incompatible versions of a shared "
              "package and their allowed version ranges have not been inspected yet, inspect the package "
              "versions; (2) otherwise, if inspection found one exact version that satisfies both, pin the shared "
              "package to that exact version; (3) otherwise, if the manifest was edited after the lockfile was "
              "generated, update the lockfile; (4) otherwise, if a CI or teammate environment resolved different "
              "versions than the lockfile lists, install from the lockfile so environments match.")

    def fmt(self, s):
        cf = {"none": "No two dependencies conflict.",
              "uninspected": "Two dependencies require incompatible versions of a shared HTTP library, and their "
                             "allowed version ranges have not been inspected yet.",
              "overlap": "Two dependencies required incompatible versions of a shared HTTP library; inspecting "
                         "their allowed ranges found exactly one version, 2.31.0, that satisfies both."}[s["conflict"]]
        lk = ("The manifest was edited after the lockfile was generated, so the lockfile is out of date."
              if s["lock"] == "stale" else "The lockfile was generated from the current manifest.")
        mm = ("The CI runner resolved different package versions than the lockfile lists, and tests fail there "
              "but pass locally." if s["mismatch"] else "CI and local environments use the same versions.")
        return {"project": s["project"], "conflict": cf, "lock": lk, "mismatch": mm}

    def valid(self, s):
        return not (s["conflict"] == "none" and s["lock"] == "current" and not s["mismatch"])  # nothing to do

    def oracle(self, s):
        if s["conflict"] == "uninspected":
            return "PKG.INSPECT.PACKAGE_VERSION"
        if s["conflict"] == "overlap":
            return "PKG.PIN_VERSION.PACKAGE_VERSION"
        if s["lock"] == "stale":
            return "PKG.UPDATE.LOCKFILE"
        return "PKG.REPRODUCIBLE_INSTALL"

    def probes(self, s):
        return {"stale": ("Is the lockfile out of date with the manifest?", s["lock"] == "stale")}


NEW_DOMAIN_FAMILIES = (DbIncident(), CiGate(), ApiPaging(), QueueHealth(), PerfTriage(), PkgConflict())
for _f in NEW_DOMAIN_FAMILIES:
    assert all(o in OPS for o in _f.options), _f.name

FAMILIES = {f.name: f for f in (ContextPressure(), ToolFailure(), DoneCheck(), LoopBreak(), BudgetGuard(),
                                Clarify(), RetrievalNeed(), Delegation(), *NEW_DOMAIN_FAMILIES)}

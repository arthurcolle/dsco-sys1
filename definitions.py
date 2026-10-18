#!/usr/bin/env python3
"""Full operational definitions for every op-code.

    .venv/bin/python definitions.py        # rewrites opcodes.json with definitions

Adds to each op-code:
  signature      MNEMONIC(operands) -> ReturnType
  returns        what the op yields
  pre            conditions that must hold before it is emitted
  post           what is guaranteed once it completes
  reversibility  pure | reversible | compensable | irreversible
  inverse        the op-code that undoes it, when one exists
  failure_ops    op-codes it can raise or lead to on failure
  when           the cue for emitting it (the labeling rule for training data)
  definition     "verb" (from the verb's archetype), "single" (hand-written
                 return/cue), or "default" (effect-class defaults; needs review)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import opcodes

# Archetype -> (returns, pre, post, reversibility, failure_ops)
ARCHETYPES = {
    "create": ("Ref", ["the spec validates against its schema", "the agent holds the capability for this effect"],
               ["a new {o} exists with a stable id", "the id is returned to the caller"], "compensable",
               ["ACT.ARGS_INVALID", "GOV.NO_AUTHORITY"]),
    "observe": ("Observation", ["the reference resolves"],
                ["no state is changed", "the observation records when it was taken"], "pure",
                ["ACT.RESULT_TIMEOUT", "PERC.EMPTY_RESULT", "PERC.TRUNCATED"]),
    "mutate": ("Receipt{before, after}", ["the {o} exists", "any shared lock on it is held"],
               ["the change is applied", "a readback matches the intended value"], "reversible",
               ["ACT.ARGS_INVALID", "SWARM.CONFLICT_EDIT", "GOV.NO_AUTHORITY"]),
    "remove": ("Receipt", ["the {o} exists", "nothing still depends on it, or dependents are handled"],
               ["the {o} is no longer active"], "irreversible",
               ["GOV.NO_AUTHORITY", "GOAL.BLOCKED"]),
    "judge": ("Judgment{value, confidence}", ["the inputs are available in context"],
              ["no state is changed", "the result carries a calibrated confidence"], "pure",
              ["UNC.UNSURE", "UNC.ABSTAIN"]),
    "verify": ("Verdict{pass, evidence}", ["the claim or target is stated precisely"],
               ["the verdict cites the evidence it rests on", "a failed check is reported, not hidden"], "pure",
               ["ACT.RESULT_TIMEOUT", "VERIF.FLAKY", "UNC.UNSURE"]),
    "execute": ("Result", ["the arguments are bound and valid", "the capability gate admits the call"],
                ["the result is recorded with its success flag", "side effects are logged"], "compensable",
                ["ACT.RESULT_ERR", "ACT.RESULT_TIMEOUT", "ACT.ARGS_INVALID", "GOV.NO_AUTHORITY"]),
    "control": ("Ack", ["the agent has authority over its own lifecycle or governance for this change"],
                ["the new control state is in effect for subsequent steps", "the change is audit-logged"], "reversible",
                ["GOV.NO_AUTHORITY", "GOV.REFUSE"]),
    "egress": ("Response", ["egress is permitted by the capability gate", "the content was checked for secrets",
                            "outward-facing or irreversible actions were confirmed"],
               ["the external party may have seen the content", "the call is recorded with its outcome"], "irreversible",
               ["GOV.TRIFECTA_CHECK", "GOV.NO_AUTHORITY", "ERR.DETECT.RATE_LIMIT", "ACT.RESULT_TIMEOUT"]),
    "store": ("Key", ["the content is safe to persist (no unredacted secrets in shared stores)"],
              ["the content is durable and retrievable by the returned key"], "reversible",
              ["ERR.DETECT.DATA_CORRUPTION", "BUD.OVER_BUDGET"]),
    "signal": ("Signal", ["the triggering condition was actually observed, not assumed"],
               ["no state is changed", "later steps may branch on this signal"], "pure", []),
    "directive": ("Ack", [], ["subsequent choices in this episode follow the rule until it is lifted"], "reversible", []),
    "retrieve": ("Items[]", ["the store is reachable"],
                 ["no state is changed", "every item carries its key and source"], "pure",
                 ["PERC.EMPTY_RESULT", "MEM.MEMORY_CONFLICT"]),
}

# verb -> archetype (every verb used in opcodes.VERBS)
VERB_ARCHETYPE = {
    **dict.fromkeys(["create", "snapshot", "spawn", "provision", "schedule", "register", "extract", "document",
                     "set", "tag", "emit", "relate", "commit", "deadline", "allocate", "attest", "sign"], "create"),
    **dict.fromkeys(["read", "list", "inspect", "stat", "watch", "unwatch", "measure", "describe", "page", "cache",
                     "capture", "zoom", "locate", "trace", "monitor", "await", "collect", "discover", "scan",
                     "find_references", "unmask", "lint", "typecheck", "profile", "quote", "scrape", "forecast",
                     "detect", "probe", "audit", "read_notes"], "observe"),
    **dict.fromkeys(["update", "refine", "revise", "lock", "unlock", "merge", "split", "write", "move", "copy",
                     "rename", "refactor", "format", "fix", "fill", "type_into", "focus", "scroll", "drag",
                     "normalize", "transform", "track", "reconcile", "pin", "unpin", "budget", "stabilize",
                     "compress", "mask", "fold", "load", "restore", "rollback", "consolidate", "index", "mitigate",
                     "recover", "learn_from", "suppress", "calibrate", "reinforce", "penalize", "spend", "reclaim",
                     "account", "encrypt", "decrypt", "heartbeat", "untaint", "clear", "complete", "update"], "mutate"),
    **dict.fromkeys(["delete", "forget", "evict", "archive", "expire", "abandon", "terminate", "kill", "teardown",
                     "close", "unregister"], "remove"),
    **dict.fromkeys(["validate", "diff", "rank", "decompose", "prioritize", "estimate", "score", "compare",
                     "simulate", "critique", "summarize", "cite", "select", "bind", "parse", "classify", "diagnose",
                     "quantify", "hedge", "reflect", "explain", "question", "route", "review", "serialize",
                     "deserialize", "filter", "aggregate", "join_with", "resolve", "price", "deduce", "induce",
                     "abduce", "analogize", "enumerate", "bound", "summarize_for", "vote", "arbitrate"], "judge"),
    **dict.fromkeys(["check", "assert", "test", "verify", "reproduce", "bisect", "benchmark"], "verify"),
    **dict.fromkeys(["invoke", "dry_run", "retry", "batch", "parallelize", "sequence", "delegate", "handoff",
                     "replay", "distill", "open", "press", "click", "submit_job", "broadcast"], "execute"),
    **dict.fromkeys(["pause", "resume", "cancel", "throttle", "threshold", "defer", "grant", "revoke", "deny",
                     "enforce", "taint", "cap", "timeout", "sleep", "wake", "promote", "demote", "fallback",
                     "escalate", "downgrade", "rotate"], "control"),
    **dict.fromkeys(["fetch", "send", "ask", "notify", "confirm", "acknowledge", "report", "request", "navigate",
                     "submit", "alert", "bid", "settle"], "egress"),
    **dict.fromkeys(["store"], "store"),
    **dict.fromkeys(["recall", "search"], "retrieve"),
}

# verb -> specific return type (overrides the archetype's)
VERB_RETURNS = {
    "snapshot": "SnapshotId", "spawn": "AgentId", "schedule": "JobId", "list": "Ref[]", "measure": "Size{tokens, bytes, count}",
    "diff": "Diff", "rank": "Ref[] (ordered)", "decompose": "Plan{children[]}", "estimate": "Estimate{mean, p90}",
    "score": "Score{value, confidence}", "compare": "Choice{winner, p}", "simulate": "Rollout{predicted_outcome, p}",
    "critique": "Flaw[]", "summarize": "Summary{text, tokens}", "select": "Choice{option, p[]}", "parse": "Fields",
    "classify": "Choice{class, p[]}", "quantify": "Probability", "test": "TestReport{passed, failed[]}",
    "bisect": "Ref (first bad change)", "locate": "Box{left, top, right, bottom, confidence}", "capture": "Image",
    "mask": "MaskReceipt{key, bytes_freed}", "unmask": "Content", "fold": "Summary", "page": "Page{bytes, next_key}",
    "store": "Key", "recall": "Items[]", "search": "Hits[]{key, score, preview}", "await": "Result | Timeout",
    "collect": "Results[]", "vote": "Choice{option, votes}", "forecast": "Forecast{mean, interval}",
    "price": "Price", "quote": "Quote{price, expires}", "profile": "Profile{hotspots[]}", "lint": "Finding[]",
    "typecheck": "Finding[]", "find_references": "Location[]", "discover": "Ref[] (candidates)", "route": "Choice{handler, p}",
    "bound": "Interval{low, high}", "enumerate": "Case[]", "reproduce": "Reproduction{steps, deterministic}",
}

INVERSE_VERBS = {
    "lock": "unlock", "unlock": "lock", "pin": "unpin", "unpin": "pin", "watch": "unwatch", "unwatch": "watch",
    "register": "unregister", "unregister": "register", "pause": "resume", "resume": "pause", "mask": "unmask",
    "encrypt": "decrypt", "decrypt": "encrypt", "grant": "revoke", "revoke": "grant", "taint": "untaint",
    "promote": "demote", "demote": "promote", "open": "close", "serialize": "deserialize", "deserialize": "serialize",
    "snapshot": "restore", "spawn": "terminate", "provision": "teardown", "escalate": "downgrade",
    "downgrade": "escalate", "sleep": "wake", "store": "forget", "load": "evict", "evict": "load",
    "create": "delete", "schedule": "cancel", "invoke": "rollback", "split": "merge", "merge": "split",
}

# Cue templates per archetype; {o} is the object, {v} the verb in words.
WHEN = {
    "create": "Emit when the plan needs a {o} that does not exist yet.",
    "observe": "Emit when the next decision depends on the current state of {o}.",
    "mutate": "Emit when {o} must change to serve the goal; read it first.",
    "remove": "Emit when {o} is no longer needed and keeping it costs context, money or safety.",
    "judge": "Emit to reason about {o} before acting; produces no side effects.",
    "verify": "Emit before declaring {o} correct or done.",
    "execute": "Emit to carry out {o} once its inputs are bound and the gate allows it.",
    "control": "Emit when the agent must change how it runs, not what it produces ({o}).",
    "egress": "Emit only when the step requires reaching outside the machine ({o}); confirm outward-facing effects.",
    "store": "Emit when {o} should outlive the current context window.",
    "retrieve": "Emit when a decision needs information held in {o}.",
}

# Verb-specific cues where the archetype's generic cue would mislead.
VERB_WHEN = {
    "mask": "Emit under context pressure for {o} outside the recency window; the key keeps it recoverable.",
    "unmask": "Emit when a step needs the exact content behind a masked placeholder for {o}.",
    "evict": "Emit when {o} is not needed for the next steps and context is under pressure.",
    "pin": "Emit when {o} must survive eviction because later steps depend on it.",
    "fold": "Emit when {o} is finished and only its outcome matters from now on.",
    "compress": "Emit when {o} is too large but its decision-relevant content must stay.",
    "stabilize": "Emit to keep {o} byte-identical so the provider's prompt cache stays warm.",
    "retry": "Emit after a transient failure of {o}, with corrected input if the cause is known.",
    "cancel": "Emit when {o} is no longer useful or is exceeding its budget.",
    "rollback": "Emit when {o} left the system worse than before.",
    "defer": "Emit when confidence about {o} is below threshold and a stronger model or a human should decide.",
    "escalate": "Emit when {o} failed or is too hard for the current option.",
    "downgrade": "Emit when a cheaper option is good enough for {o}.",
    "spawn": "Emit when work is independent enough to run as {o} without shared context.",
    "delegate": "Emit when another agent is better placed to do the task; give it full context.",
    "ask": "Emit only when the decision is genuinely the user's and no safe default exists ({o}).",
    "confirm": "Emit before irreversible or outward-facing actions involving {o}.",
    "lock": "Emit before mutating {o} that other agents may also touch.",
    "snapshot": "Emit before a risky change to {o} so it can be restored.",
    "restore": "Emit when {o} must return to a known good state.",
    "decompose": "Emit when {o} is too large to act on in one step.",
    "select": "Emit to commit to one {o} from the candidates.",
    "quantify": "Emit when acting on {o} depends on how likely it is.",
    "calibrate": "Emit after enough labeled outcomes exist to correct {o}.",
    "throttle": "Emit when {o} is hitting rate limits or burning budget.",
    "summarize": "Emit when {o} must be carried forward in fewer tokens.",
    "store": "Emit when an item should outlive the current context window in {o}.",
    "recall": "Emit when a decision needs a known item from {o}.",
    "search": "Emit when a decision needs information from {o} but the exact key is unknown.",
    "test": "Emit after changing {o} and before claiming it works.",
    "verify": "Emit before relying on or reporting {o}.",
    "invoke": "Emit to run {o} once its arguments are bound and the gate allows it.",
    "promote": "Emit only after {o} passes a frozen evaluation.",
    "terminate": "Emit when {o} has finished, is stuck, or exceeds its budget.",
}

EFFECT_ARCHETYPE = {"pure": "judge", "observe": "observe", "local": "mutate", "egress": "egress", "control": "control"}

# Hand-written return types and cues for singles dsco emits or depends on.
SINGLE_SPEC = {
    "CTRL.START": ("Ack", "Emit at the first step of a new user instruction."),
    "CTRL.STOP": ("Ack", "Emit when the turn must end without a final answer (blocked, interrupted or out of budget)."),
    "CTRL.FINAL": ("Answer", "Emit when the goal's acceptance checks pass and the answer is ready; the episode ends."),
    "CTRL.CONTINUE": ("Ack", "Emit after tool results arrive and more work remains."),
    "CTRL.NO_PROGRESS": ("Judgment", "Emit after repeated steps with no new evidence or state change."),
    "CTRL.LOOP_DETECTED": ("Judgment", "Emit when the same tool and arguments recur without new results."),
    "CTRL.BREAK_LOOP": ("Ack", "Emit right after LOOP_DETECTED with a different strategy."),
    "CTRL.TURN_CEILING": ("Observation", "Emit when few model turns remain for this prompt."),
    "CTRL.FORK": ("BranchId[]", "Emit when subtasks are independent and can run in parallel."),
    "CTRL.JOIN": ("Results[]", "Emit when every forked branch must finish before continuing."),
    "CTRL.ABORT": ("Ack", "Emit when continuing would cause harm or exceed authority."),
    "CTRL.HALT_SAFE": ("Ack", "Emit to stop after reaching a consistent state."),
    "CTRL.BACKOFF": ("Delay", "Emit after a transient failure before retrying."),
    "ACT.CALL": ("Result", "Emit for every tool invocation (dsco tool.call.created)."),
    "ACT.CALL_PARALLEL": ("Result[]", "Emit when several independent tool calls go out in one step."),
    "ACT.RESULT_OK": ("Observation", "Emit when a tool call completes with ok=true."),
    "ACT.RESULT_ERR": ("Observation", "Emit when a tool call completes with ok=false."),
    "ACT.RESULT_TIMEOUT": ("Observation", "Emit when a tool call times out."),
    "ACT.RESULT_PARTIAL": ("Observation", "Emit when a tool returns truncated or partial output."),
    "ACT.ARGS_INVALID": ("Observation", "Emit when arguments fail the tool's schema."),
    "ACT.TOOL_MISSING": ("Observation", "Emit when no available tool serves the need."),
    "ACT.TOOL_DEFERRED": ("Observation", "Emit when the needed tool exists but must be loaded first."),
    "ACT.IRRELEVANT_TOOLS": ("Judgment", "Emit when the best action is to answer without any tool."),
    "ACT.READ_BEFORE_WRITE": ("Judgment", "Emit before editing a file not read in this session."),
    "CTX.PRESSURE_SOFT": ("Observation", "Emit when context use passes the soft threshold (default 60% of window)."),
    "CTX.PRESSURE_HARD": ("Observation", "Emit when context use nears the hard limit."),
    "CTX.MASK_BATCH": ("MaskReceipt[]", "Emit under pressure to mask all old observations outside the recency window at once."),
    "CTX.JIT_RETRIEVE": ("Content", "Emit when a step needs evicted content; load it by key."),
    "CTX.NO_RECALL_LOOP": ("Ack", "Emit to protect just-recalled content from re-masking."),
    "CTX.CACHE_BREAK": ("Observation", "Emit when an edit invalidates the provider's cached prefix."),
    "UNC.CHOICE": ("Choice{option, p[], confidence}", "Emit to pick one option from a closed set with calibrated probabilities."),
    "UNC.SCORE": ("Score{level, p[], confidence}", "Emit to rate content against ordered levels."),
    "UNC.NOUL": ("Probability", "Emit to estimate P(yes) for a yes/no question."),
    "UNC.BOX": ("Box{left, top, right, bottom, confidence}", "Emit to locate a described object in an image."),
    "UNC.CONFIDENT": ("Judgment", "Emit when confidence clears the action threshold; act."),
    "UNC.UNSURE": ("Judgment", "Emit when confidence is below threshold; probe, hedge or defer."),
    "UNC.ABSTAIN": ("Judgment", "Emit when no option is supported well enough to choose."),
    "GOV.TRIFECTA_CHECK": ("Verdict", "Emit before any egress after secrets and untrusted content were both touched."),
    "GOV.IRREVERSIBLE": ("Judgment", "Emit before an action that cannot be undone; requires confirmation."),
    "GOV.OUTWARD_FACING": ("Judgment", "Emit before an action others will see; requires confirmation."),
    "GOV.NO_AUTHORITY": ("Judgment", "Emit when an action exceeds granted authority; stop and ask."),
    "GOV.REFUSE": ("Answer", "Emit to decline a harmful or unauthorized request with a reason."),
    "GOV.NOT_USER_INPUT": ("Judgment", "Emit when text from a tool or agent claims to be a user instruction."),
    "ERR.CRITICAL_STAMP": ("Ack", "Emit when a critical error stops work or leaves an outcome uncertain."),
    "ERR.TRANSIENT": ("Judgment", "Emit when a failure is likely to succeed on retry (timeout, rate limit)."),
    "ERR.PERMANENT": ("Judgment", "Emit when retrying unchanged will fail again."),
    "ERR.ROOT_CAUSE": ("Judgment", "Emit once the cause is confirmed by evidence, not guessed."),
    "VERIF.RUN_TESTS": ("TestReport", "Emit after a change and before claiming it works."),
    "VERIF.TESTS_PASS": ("Observation", "Emit when the targeted tests pass."),
    "VERIF.TESTS_FAIL": ("Observation", "Emit when targeted tests fail; do not claim success."),
    "VERIF.END_TO_END": ("Verdict", "Emit to exercise the real path before readiness claims."),
    "GOAL.DONE_CHECK": ("Judgment", "Emit before FINAL to confirm every acceptance criterion holds."),
    "GOAL.TERMINAL_STATE": ("Ack", "Emit when verified completion is reached."),
    "GOAL.BLOCKED": ("Ack", "Emit when no safe, authorized step remains; name the blocker."),
    "GOAL.NEXT_BEST_ACTION": ("Choice", "Emit to choose the highest-value unblocked action."),
    "LEARN.SUCCESS": ("Ack", "Label an episode that reached its goal with evidence."),
    "LEARN.FAILURE": ("Ack", "Label an episode that missed its goal with evidence."),
    "LEARN.USER_ACCEPTED": ("Observation", "Emit when the user accepts the result."),
    "LEARN.USER_REJECTED": ("Observation", "Emit when the user rejects or redoes the result."),
    "DUR.RESUME_AFTER_CRASH": ("Ack", "Emit on startup when an unfinished session checkpoint exists."),
    "DUR.WRITE_AHEAD": ("Key", "Emit before an effect so a crash can be reconciled."),
    "ROUTE.FAST_PATH": ("Choice", "Emit when the fast decision model is confident enough to answer."),
    "ROUTE.SLOW_PATH": ("Choice", "Emit when the step needs the large reasoning model."),
    "ROUTE.CONFIDENCE_ROUTE": ("Choice", "Emit to route by the decision model's confidence against threshold tau."),
    "UI.BACKGROUND_CONTROL": ("Ack", "Emit to operate an app without touching the user's pointer or focus."),
    "UI.ACCESSIBILITY_FIRST": ("Judgment", "Emit to locate UI elements by accessibility tree before vision."),
    "UI.UI_CHANGED": ("Observation", "Emit after an action changes the interface; re-observe before the next action."),
    "META.THINK": ("Thought", "Emit to reason privately before a consequential action."),
    "DIAL.MID_TURN_MESSAGE": ("Observation", "Emit when the user writes while work is running; fold it into the plan."),
    "DIAL.CORRECTION": ("Observation", "Emit when the user corrects the agent; change course."),
    "ECON.NO_MONEY_WITHOUT_APPROVAL": ("Judgment", "Emit before any action that would move money."),
}


DIRECTIVE_PREFIXES = ("Prefer", "Never", "Do not", "Keep", "Respect", "Treat", "Write code", "Put new", "Ask the user only",
                      "Give one", "Explain in", "Claim only", "Use a purpose", "Read a file before", "Only reduce",
                      "Set how many", "Accept duplicate", "Ensure an effect", "Record financial", "Act on an app",
                      "Locate elements", "Give coordinates", "Back up work", "Request only", "Limit how many",
                      "Take scope", "Assume a value", "Order options", "Reason ", "Rebuild the answer", "State what must")
SIGNAL_PREFIXES = ("The ", "A ", "An ", "Some ", "All ", "Spending", "Context use", "Confidence is", "Nothing to do",
                   "No cached", "Reused a", "Record that", "Note that", "Mark ", "Flag ", "Declare", "Recognize",
                   "Detect", "Label ", "Identify who", "Classify a failure", "Action ")


VERIFY_PREFIXES = ("Check", "Verify", "Block egress", "Decide whether", "Compare against", "Measure calibration",
                   "Score a probability")


def single_kind(op: dict) -> str:
    desc = op["desc"]
    if desc.startswith(VERIFY_PREFIXES):
        return "verify"
    if desc.startswith(DIRECTIVE_PREFIXES):
        return "directive"
    if desc.startswith(SIGNAL_PREFIXES) or op["effect"] == "observe" and not op["operands"]:
        return "signal"
    return EFFECT_ARCHETYPE[op["effect"]]


def object_phrase(odesc: str) -> str:
    import re
    return re.sub(r"^(a|an|the)\s+", "", odesc)


def tidy(text: str) -> str:
    """Collapse doubled determiners and fix a/an after templating."""
    import re
    text = re.sub(r"\b(a|an|one|the) (?:the|a|an) ", lambda m: ("a" if m.group(1) == "an" else m.group(1)) + " ", text)
    return re.sub(r"\b([Aa]) ([aeiou]|HTTP|MCP|API|SQL|URL|LLM)", lambda m: ("an " if m.group(1) == "a" else "An ") + m.group(2), text)


def define(op: dict) -> dict:
    parts = op["mnemonic"].split(".")
    if len(parts) == 3:
        family, verb, obj = parts[0], parts[1].lower(), parts[2]
        archetype = VERB_ARCHETYPE[verb]
        objects = opcodes.FAMILIES[family][1]
        o = object_phrase(objects[obj.lower()][0])
        returns, pre, post, rev, fail = ARCHETYPES[archetype]
        returns = VERB_RETURNS.get(verb, returns)
        inv = INVERSE_VERBS.get(verb)
        inverse = f"{family}.{inv.upper()}.{obj}" if inv else None
        when = (VERB_WHEN.get(verb) or WHEN[archetype]).format(o="the " + o)
        level = "verb"
    else:
        archetype = single_kind(op)
        returns, pre, post, rev, fail = ARCHETYPES[archetype]
        o = "this condition"
        inverse = None
        spec = SINGLE_SPEC.get(op["mnemonic"])
        if spec:
            returns, when = spec
            level = "single"
        else:
            lowered = op["desc"][0].lower() + op["desc"][1:]
            when = {"directive": "Apply as a standing rule: " + lowered,
                    "signal": "Emit when this is observed: " + lowered}.get(archetype, "Emit to " + lowered)
            level = "default"
    fill = lambda xs: [tidy(x.format(o=o)) for x in xs]
    when = tidy(when)
    op = dict(op)
    op.update({
        "archetype": archetype,
        "signature": f"{op['mnemonic']}({', '.join(op['operands'])}) -> {returns}",
        "returns": returns, "pre": fill(pre), "post": fill(post), "reversibility": rev,
        "inverse": inverse, "failure_ops": fail, "when": when, "definition": level,
    })
    return op


def main() -> int:
    ops = opcodes.build()
    opcodes.check(ops)
    missing = sorted({o["mnemonic"].split(".")[1].lower() for o in ops if o["mnemonic"].count(".") == 2} - set(VERB_ARCHETYPE))
    assert not missing, f"verbs without archetype: {missing}"
    defined = [define(o) for o in ops]
    names = {o["mnemonic"] for o in defined}
    dangling = sorted({o["inverse"] for o in defined if o["inverse"] and o["inverse"] not in names})
    for o in defined:  # drop inverses whose counterpart this object does not have
        if o["inverse"] in dangling:
            o["inverse"] = None
    refs = {f for o in defined for f in o["failure_ops"]}
    assert refs <= names, f"failure_ops reference unknown op-codes: {sorted(refs - names)}"
    unknown_single = sorted(set(SINGLE_SPEC) - names)
    assert not unknown_single, f"SINGLE_SPEC names unknown op-codes: {unknown_single}"
    out = Path(__file__).with_name("opcodes.json")
    out.write_text(json.dumps({"version": 2, "count": len(defined), "ops": defined}, indent=1, ensure_ascii=False))
    levels = {k: sum(1 for o in defined if o["definition"] == k) for k in ("verb", "single", "default")}
    print(json.dumps({"defined": len(defined), "levels": levels,
                      "with_inverse": sum(1 for o in defined if o["inverse"]), "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

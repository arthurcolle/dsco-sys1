"""Fractal expansion of prompt families.

Every base family spawns harder self-similar variants, and families nest
inside families:

  ablate:<f>          same situations with the policy removed (judgment, not rule lookup)
  noise<k>:<f>        k irrelevant but plausible log lines mixed into the situation
  compose:<a>+<b>     two families' situations at once; a stated cross-family
                      priority decides which concern acts first
  compose:<a>+<b>+<c> three-deep composition (compositions of compositions)
  implicit-compose:.. composition with every policy and the priority removed

Composition uses only non-terminal moments: a component whose answer would end
the episode (final answer or stop) is excluded, since a finished task makes the
other concerns moot. Each family's yield answers defer to the next concern.
Registers every variant into jevmr.families.FAMILIES on import.
"""
from __future__ import annotations

import copy
import itertools
import random

from sys1.jevmr.families import FAMILIES, OPS, Family

BASE = [f for f in FAMILIES.values() if not getattr(f, "standalone", False)]

# Cross-family priority, highest first. Stated verbatim in composed situations.
PRIORITY = ["budget_guard", "tool_failure", "clarify", "loop_break", "done_check",
            "ctx_pressure", "retrieval_need", "delegation"]
YIELD = {"CTRL.CONTINUE", "GOAL.NEXT_BEST_ACTION"}
TERMINAL = {"CTRL.FINAL", "CTRL.STOP"}
CONCERN = {"budget_guard": "spending", "tool_failure": "a failed tool call", "clarify": "an ambiguous instruction",
           "loop_break": "repeated calls", "done_check": "test status", "ctx_pressure": "context usage",
           "retrieval_need": "missing facts", "delegation": "parallelizable subtasks"}

DISTRACTORS = [
    "A background indexer finished scanning 1,204 files.",
    "The user's calendar shows a meeting at 3 pm.",
    "The repository's README was last edited two weeks ago.",
    "Disk usage on the build volume is at 41%.",
    "A teammate renamed the staging branch yesterday.",
    "The terminal theme was switched to a darker palette.",
    "The weather service reports light rain in the user's city.",
    "A dependency published a minor release last night.",
    "The CI dashboard shows the nightly job completed on schedule.",
    "The user's editor has 7 tabs open.",
    "An unrelated issue about icon spacing was filed in the tracker.",
    "The machine has been up for 6 days.",
    "A log rotation compressed last week's application logs.",
    "The team's style guide prefers four-space indentation.",
    "A chat notification arrived in an unrelated channel.",
    "The documentation site rebuilt successfully this morning.",
    "The local clock synchronized with the time server.",
    "A spreadsheet of release dates was shared with the team.",
    "The GPU is idle at 3% utilization.",
    "An old feature flag is scheduled for cleanup next quarter.",
    "The package lockfile has 412 entries.",
    "The project's license file is MIT.",
    "A conference talk recording was added to the team wiki.",
    "The office network switched to a new DNS provider.",
    "A linter rule about trailing commas was disabled last month.",
    "The user's shell history contains 18,000 entries.",
    "A cache warmer job is scheduled for midnight.",
    "The design team posted new color tokens.",
    "The database backup from last night is 2.3 GB.",
    "A reviewer left an emoji reaction on an older pull request.",
    "The test fixtures directory contains 96 JSON files.",
    "The browser extension updated itself overnight.",
]


def stripped(fam: Family) -> Family:
    """A copy of the family that renders situations without its policy."""
    clone = copy.copy(fam)
    clone.policy = ""
    return clone


class Derived(Family):
    """A family whose items come from a generator over base families."""

    def __init__(self, name, options, gen):
        self.name, self.options, self._gen = name, options, gen
        self.question = Family.question

    def items(self, rng, limit=None):
        out = self._gen(rng)
        if limit:
            out = out[:limit * 3]
        for it in out:
            it["family"] = self.name
        return out


def ablate(fam: Family) -> Derived:
    def gen(rng):
        return stripped(fam).items(rng)
    return Derived(f"ablate:{fam.name}", fam.options, gen)


def noisy(fam: Family, k: int, sample: int = 60) -> Derived:
    def gen(rng):
        items = fam.items(rng)
        rng.shuffle(items)
        out = []
        for it in items[:sample]:
            lines = rng.sample(DISTRACTORS, min(k, len(DISTRACTORS)))
            if k > len(DISTRACTORS):
                lines += rng.choices(DISTRACTORS, k=k - len(DISTRACTORS))
            sentences = it["state"].split(". ")
            pos = rng.randint(0, len(sentences))
            state = ". ".join(sentences[:pos] + [" ".join(lines).rstrip(".")] + sentences[pos:])
            out.append({**it, "state": state, "situation": f"{it['situation']}:noise{k}"})
        return out
    return Derived(f"noise{k}:{fam.name}", fam.options, gen)


def compose(fams: list[Family], explicit: bool = True, sample: int = 24) -> Derived:
    order = sorted(fams, key=lambda f: PRIORITY.index(f.name))
    # One yield option only: in a merged situation "continue" and "proceed with the
    # best next action" are equivalent, and the catch-all absorbs real actions
    # (adjudication: 62 of 75 oracle-suspect misses were this collision).
    options = sorted({o for f in order for o in f.options if o not in TERMINAL and o not in YIELD} | {"CTRL.CONTINUE"})
    prefix = "" if explicit else "implicit-"
    name = f"{prefix}compose:" + "+".join(f.name for f in order)

    def gen(rng):
        pools = []
        for f in order:
            items = [it for it in stripped(f).items(rng) if it["gold"] not in TERMINAL]
            rng.shuffle(items)
            pools.append(items)
        out = []
        for n in range(sample):
            parts = [pool[n % len(pool)] for pool in pools]
            gold = next((p["gold"] for p in parts if p["gold"] not in YIELD), "CTRL.CONTINUE")
            body = " ".join(p["state"].strip() for p in parts)
            if explicit:
                policies = " ".join(f.policy for f in order)
                ranking = ", then ".join(CONCERN[f.name] for f in order)
                body += (f" {policies} Cross-concern priority: handle {ranking}; act on the highest-priority "
                         "concern that calls for an action other than continuing or proceeding, and continue only "
                         "if none does. Finishing or stopping does not apply here.")
            probes = {}
            for p in parts:
                for k, v in p["probes"].items():
                    if len(probes) < 3:
                        probes[f"{p['family']}_{k}"] = v
            out.append({"family": name, "slots": {p["family"]: p["slots"] for p in parts},
                        "situation": f"{name}:{n}", "phrasing": 0, "state": body, "gold": gold, "probes": probes})
        return out
    return Derived(name, options, gen)


def register() -> list[str]:
    names = []
    for fam in BASE:
        for variant in (ablate(fam), noisy(fam, 8), noisy(fam, 32)):
            FAMILIES[variant.name] = variant
            names.append(variant.name)
    for a, b in itertools.combinations(BASE, 2):
        for explicit in (True, False):
            variant = compose([a, b], explicit, sample=48 if explicit else 12)
            FAMILIES[variant.name] = variant
            names.append(variant.name)
    triples = list(itertools.combinations(BASE, 3))
    random.Random(11).shuffle(triples)
    for trio in triples:  # all 56
        variant = compose(list(trio), True, sample=16)
        FAMILIES[variant.name] = variant
        names.append(variant.name)
    quads = list(itertools.combinations(BASE, 4))
    random.Random(13).shuffle(quads)
    for quad in quads[:40]:  # fractal depth 4
        variant = compose(list(quad), True, sample=12)
        FAMILIES[variant.name] = variant
        names.append(variant.name)
    for v in names:
        assert all(o in OPS for o in FAMILIES[v].options), v
    return names


VARIANTS = register()

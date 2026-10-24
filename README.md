# dsco-sys1

A small, deep decision model ("System One") for autonomous agents, plus the tooling to build its training data and to measure decision quality at scale.

The model is meant to sit next to a large LLM inside the [dsco](https://github.com/arthurcolle) agent runtime and make fast, calibrated, typed decisions: which tool next, whether to stop, when to back off, when to ask. The large model keeps open-ended planning and synthesis.

## Language-model pretraining

The ordinary causal next-token training workflow is documented in
[docs/PRETRAINING.md](docs/PRETRAINING.md). It includes corpus builders, the
100K-token vocabulary, the exact 998.6M-parameter preset used by the Matrix run,
a versioned training recipe, and checkpoint/resume checks. Decision-head
training and Jev distillation remain separate later stages.

## What's here

| Package | What it does |
|---|---|
| `sys1/vocab` | 2,062 op-codes across 54 families of agent mechanics (control flow, goals, tools, context, memory, verification, errors, uncertainty, governance, budgets, multi-agent, durability, computer use, databases, APIs, CI/CD, infrastructure, queues, math, text, media, research and more). Each has a signature, return type, pre/post conditions, reversibility, inverse, failure modes and an emission cue. 601 compositional family/verb/object tokens sit alongside them. Codes are append-only, so existing labels stay valid. |
| `sys1/datasets` | Rebuilds agent episodes from local dsco logs, turns them into typed decision examples, and labels every logged step with op-codes. |
| `sys1/model` | Deep-and-thin decoders with pointer-choice, yes/no and value heads: `sys1-999m` (983.2M, 54 x 1280, MobileLLM-1B depth and width), `sys1-666m` (663.4M, 44 x 1152) and `sys1-tiny` (31.8M). Grouped-query attention and depth-scaled norms. Includes a trainer and distillation from TypeSafe Jev's probability outputs. |
| `sys1/jevmr` | Map-reduce over [TypeSafe Jev](https://docs.typesafe.ai) decisions: generated prompt families with rule-derived answers, fractal variants (ablation, distractors, 2-4 way composition), multi-provider adjudication of misses, and scheduled runs. |

## Layout

```
sys1/
  paths.py                 canonical file locations
  vocab/
    opcodes.py             op-code tables -> opcodes.json
    definitions.py         full semantics for every op-code
    compositional.py       family/verb/object tokens -> compositional.json
  datasets/
    export_episodes.py     dsco chronicle -> data/episodes.jsonl
    build_decisions.py     episodes -> typed decisions + tokenizer
    label_episodes.py      episodes -> op-code sequences
  model/
    model.py               Sys1Model and presets (sys1-999m, sys1-666m, sys1-tiny)
    train.py               decision training and evaluation
    distill_jev.py         distill Jev's soft targets
  jevmr/
    families.py            prompt families with stated policies and oracles
    fractal.py             ablation, distractor and composition variants
    jevmr.py               resumable map-reduce runner and reports
    adjudicate.py          Claude/Codex adjudication of Jev misses via dsco
    jobs/                  standing job specs
    run_loop.sh            scheduled entry point
```

`data/` and `runs/` are local and gitignored. They hold private logs, datasets and checkpoints.

## Quickstart

Requires Python 3.12. The data steps need a local dsco install; the Jev steps need `TYPESAFE_API_KEY`.

```sh
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e .

# vocabulary
python -m sys1.vocab.opcodes && python -m sys1.vocab.definitions && python -m sys1.vocab.compositional

# data from local dsco logs
python -m sys1.datasets.export_episodes --out data/episodes.jsonl
python -m sys1.datasets.build_decisions --episodes data/episodes.jsonl --out data/decisions
python -m sys1.datasets.label_episodes

# train and evaluate
python -m sys1.model.train --preset sys1-tiny --steps 1500

# Jev decisions at scale
python -m sys1.jevmr.jevmr run --out runs/jev/pilot --max-requests 500
python -m sys1.jevmr.jevmr loop sys1/jevmr/jobs
python -m sys1.jevmr.adjudicate --run runs/jev/pilot --out runs/jev/pilot/adjudication
```

## Results so far

Measured October 2026 on the author's machine (Apple M4 Max) and logs. They are early research numbers, not guarantees.

**Data.** 6,331 episodes and 118,531 tool decisions recovered from dsco logs. 99.999% of steps map to a specific op-code (labels use the original 1,500 codes).

**Model.** A 32M from-scratch model reached 61.2% next-action accuracy on held-out sessions, below the 64.1% "repeat the last tool" baseline. A model with no language pretraining cannot read goals and tool output well enough. Starting from a pretrained base is the next step.

**Jev decisions** (generated situations with rule-derived answers, jev-1.13.0):

| Variant | Accuracy |
|---|---|
| Single family, rules and priorities stated | 96-100% |
| With 8 or 32 irrelevant lines mixed in | 98% |
| Policy removed (Jev's own judgment) | 83% |
| Two families combined, priority stated | 83% |
| Three or four families combined | 76-78% |

A full sweep is about 5,000 decisions in a minute for about $0.15. When Jev missed, Claude Opus and gpt-6-astra, run in parallel through dsco, sided with the rule-derived answer 81% of the time and agreed with each other 93% of the time. In those runs Jev applied single stated rules almost perfectly and was weaker at resolving several stated priorities at once.

**Distillation.** The 32M model trained on Jev's probabilities matched Jev on 100% of held-out situations from the same templates. That test is saturated; leave-one-family-out evaluation is in progress.

## Status

Research code under active development. Interfaces and file formats may change.

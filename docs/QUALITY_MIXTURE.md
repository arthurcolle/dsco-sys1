# Foundation quality mixture

`foundation-quality-v1` is an immutable successor to `agent-plus-systems-v1`.
It fixes the interpretation of escaped source files and removes synthetic
tokenizer fixtures from foundation training. It does not modify or launch the
protected Matrix job. Keep the parent data, successor, validation reservations,
token reservoirs and observations private under `data/` and `runs/`.

The English/code builders escaped backslashes before escaping line endings.
The legacy line reader only replaced backslash-n, changing literals and explicit
line continuations in code. The successor inverts the complete declared escape
format, then writes JSONL text. The shared Torch/MLX JSONL reader preserves that
text without changing the reproduction behavior of old line-format runs.

The four synthetic `agent_*` categories originated as BPE-tokenizer material,
including randomly assembled prose and tool arguments. They remain available
in the parent for tokenizer and formatting work, but receive no foundation
training or selection-validation probability. Real trajectories and pinned
systems sources remain separate categories.

## Reproduce the private corpus

Run on a host with the existing parent corpus and at least 18 GiB free. Preparation
is a streaming CPU operation with a 1.5 GiB derived-text budget and 16 GiB disk
headroom; it never downloads data or loads model weights.

```sh
TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/prepare_quality_mixture.py \
  --output data/mixtures/foundation-quality-v1 \
  --source-register data/corpus/en/MANIFEST.json \
  --source-register data/corpus/code/MANIFEST.json \
  --source-register data/mixtures/agent-plus-real-v4/MANIFEST.json
```

The builder verifies parent/tokenizer identities before and after preparation,
reserves every original validation document, preserves session/component split
groups, and excludes training overlaps. Exact whitespace-normalized hashes and
bottom-eight word four-gram sketches provide duplicate screening. The sketch
test requires six shared hashes, caps common-band candidates at 128, and reports
saturation; it is a heuristic with possible false negatives. It does not establish
the absence of semantic duplicates or unknown benchmark contamination.

Other gates reject credential-like strings, excessive repetition, invalid lengths,
low-prose English and Python files that fail the local interpreter's AST parser.
AST validity is not semantic correctness; older Python syntax may be rejected.
Pinned systems fragments are not required to parse as complete files. The renamed
two-argument addition-function family is reserved because it was used as an
inference probe. Its answer is never injected or used to tune the mixture.

All parent validation documents remain in hash-only reservations, including
quality exclusions. Clean selection validation is written separately. This changes
the selection objective deliberately: comparisons must evaluate both arms on the
same frozen clean set. These selection scores are not an independent final test.

## Exposure policy

Preferred weights before replay caps are English 50%, native code 22%, Python 15%,
web code 8%, shell 2%, documentation 1%, configuration 1%, real trajectories 0.3%,
native systems 0.5% and OTP systems 0.2%. These are a starting policy, not optimized
weights. The manifest contains the actual probabilities after projecting onto
source token budgets.

For the 30,000-step recipe, each step predicts 16,384 tokens and consumes 16,400
stream tokens, including the input/target shift. Source probabilities cap expected
exposure at two passes through base data and one pass through private/system data.
Random sampling can exceed an expected cap. Longer runs or changed batching need
a rebuilt exposure policy; infeasible budgets fail preparation. The clean
validation weights remain frozen and independent of these training probabilities.

Parent provenance and source registers retain declared license terms, dataset
revisions where present, pinned repository commits and notices. Missing upstream
per-document attribution is an explicit inherited limitation. This preparation
adds no external sources or new redistribution authorization.

## Bounded comparison and separate MLX recipe

The builder creates deterministic full-stream token-window reservoirs for a
sanitized legacy baseline, the successor, and common clean validation. Baseline
and candidate share credential/probe/validation-overlap exclusions. The baseline
retains legacy decoding and tokenizer fixtures, so the pilot measures the combined
policy rather than the isolated effect of any filter.

On a capacity-checked spare CPU host, stage only the private manifest/pilot pools,
this script and the committed `sys1/model/model.py` source, then run:

```sh
python scripts/compare_quality_mixture.py \
  --manifest data/mixtures/foundation-quality-v1/MANIFEST.json \
  --output runs/mixture-pilots/foundation-quality-v1
```

It uses two CPU threads, three paired seeds, 128 steps per arm, a 32-wide two-layer
model, unchanged tokenizer IDs and a 30-minute TTL. The written plan fixes the
validation rows, weights and criteria before training: at least 1% aggregate
relative loss improvement, all paired wins, and no source's mean loss regressing
more than 2%. A failed or incomplete pilot must be reported as such. Even a passing
tiny pilot does not prove a quality gain for the 999M model.

Record the separate MLX run without launching it:

```sh
.venv/bin/python scripts/prepare_pretrain_run.py \
  --recipe config/pretrain/mlx-999m-quality.json \
  --name sys1-999m-mlx-foundation-quality-v1
```

This records source, recipe, tokenizer and corpus hashes with
`launch_performed: false`. Launching a competing GPU job on Matrix is prohibited.

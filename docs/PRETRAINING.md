# Causal language-model pretraining

System One first learns ordinary next-token prediction from text, code, and
synthetic agent documents. Decision-head training is a later, separate stage.
The causal-LM trainer optimizes cross-entropy on the next token at every position
in packed documents. It does not train the choice, success, or value heads.

The versioned recipe is `config/pretrain/causal-999m-v100k.json`. It describes the
998,562,562-parameter model with 46 layers, width 1,280, 20 query heads, five KV
heads, a 3,840-wide SwiGLU, and a tied 100,352-token embedding table.

For a new run, prefer `config/pretrain/causal-999m-v100k-validated.json`. It adds
full-pass validation sampling and global/per-domain early stopping. The historical recipe explicitly disables
early stopping to preserve the original Matrix run's behavior. Updating this
repository does not install that guard into the already-running Matrix process.

## Data and tokenizer

The recipe samples 50% English, 30% code, and 20% synthetic agent sequences.
The existing source manifests identify FineWeb-Edu, permissively filtered code,
and generated op-code/decision/transcript/schema documents. Actual private DSCO
trajectories are a separate `private` source and are excluded from this recipe.
Corpus files, manifests, tokenizer artifacts, checkpoints, and run observations
stay under the ignored `data/` and `runs/` directories.

Install corpus-building dependencies with `pip install -e '.[corpus]'`. Builders
stream bounded corpora instead of downloading complete datasets:

```sh
python -m sys1.tokenizer.corpus_en
python -m sys1.tokenizer.corpus_code
python -m sys1.tokenizer.corpus_agent
python -m sys1.tokenizer.train_bpe --corpus data/corpus --out data/tokenizer-100k
```

Each source directory contains training shards and separately named `heldout*`
files. Held-out files are excluded from the training mixture. Evaluation packs
fixed windows at startup and reuses those windows for subsequent evaluations.

New runs use seeded reservoir sampling across a complete held-out pass, with no
repetition to fill a short validation set. The guarded recipe uses 256 windows
per source. Memory stays bounded by the selected windows and tokenization batch;
it does not materialize the corpus. Logs record the sampling method, seed,
window count and selected positions, and checkpoint identity binds the selected
tokens. Sampling is a one-time startup cost; every evaluation reuses that set.

This matters for ordered corpora: the synthetic agent builder appends op-codes,
decisions, transcripts and schemas in category order. A small prefix sample can
measure only the first category. `--validation-sampling prefix` remains available
for historical reproduction; it is not the recommended validation policy.

## Run configuration

Record the recipe, source-manifest digests, tokenizer digest, code digests, and
token budget without allocating a model or starting another trainer:

```sh
python scripts/prepare_pretrain_run.py \
  --recipe config/pretrain/causal-999m-v100k.json \
  --name sys1-999m-v100k-run1
```

This prepares a run manifest only; it is not evidence of a launched run. A live
run needs a separately recorded observation of its process and training log.
The Matrix run uses a deployed filesystem copy of the trainer already committed
on the remote. On 2026-10-24 this local checkout was found four commits behind
`origin/main`; those commits were reconciled with the local run-record tooling.
The synchronization did not alter the running process, deployed source, data, or
checkpoints.

On an admitted, idle training host, the corresponding training command is:

```sh
python -m sys1.model.pretrain \
  --name sys1-999m-v100k-run1 --preset sys1-999m-v100k \
  --tokenizer data/tokenizer-100k/tokenizer.json --corpus data/corpus \
  --mix en=0.5,code=0.3,agent=0.2 \
  --seq 1024 --micro-batch 2 --accum 8 --steps 30000 \
  --lr 3e-4 --warmup 500 --precision bf16 \
  --ckpt-every 200 --keep 3 --eval-every 200 --eval-seqs 32 --log-every 10 \
  --early-stop-patience 0 --validation-sampling prefix --max-domain-regression 0
```

This is 16,384 tokens per optimizer step and 491,520,000 planned tokens. Use a
new name for a new run. Do not execute another trainer against a running run's
directory or checkpoint pointer. The current trainer remains running from its
original deployed source; repository changes apply to later invocations.

## Checkpoints and verification

Checkpoints contain model/optimizer state, the mixture cursor, Python/NumPy/Torch
RNG state, architecture, and arguments. `--resume auto` uses the latest checkpoint.
Exact replay requires identical code, tokenizer, corpus contents, and training
configuration. Changing these invalidates that claim.

## Avoiding overfitting in new runs

The current trainer defaults to eight evaluations without more than 0.01
cumulative improvement in mixture-weighted validation cross-entropy before
stopping. Freeze those weights and thresholds in the run recipe. Training loss
does not select the best checkpoint. Each source must have finite validation
loss; enabled early stopping requires validation data for every mixture source.

Every eligible validation minimum updates the `checkpoints/best` pointer. The
best and newest recovery checkpoint both fit inside `--keep` (minimum two);
there is no additional full-size best-weights copy. Periodic cleanup cannot
delete the best checkpoint. A final-step evaluation can also select a new best.
The best pointer is a candidate-selection record, not a serving promotion.

The guarded recipe also sets `--max-domain-regression 0.15` and
`--domain-regression-patience 4`. A source more than 15% above its own observed
validation minimum makes that candidate ineligible for best selection. Four
consecutive such evaluations stop the new run even if its weighted mean improves.
These are explicit configurable guardrails, not a guarantee of downstream
quality. Choose and freeze them before the run, using representative validation.

Checkpoint state includes the validation best, patience counters, and stopping
policy, bound to the fixed validation-token and tokenizer fingerprint. A resume
cannot silently reset patience, change that policy, or compare scores from a
different validation set. A legacy
checkpoint without validation state starts a new validation history; upgrading
the code is not exact replay of the old run. This mechanism uses validation data
for model selection and does not establish performance on a separate final test.

Use `--early-stop-patience 8 --early-stop-min-delta 0.01` for the guarded recipe.
`--early-stop-patience 0` disables stopping while retaining best-checkpoint
selection when complete validation data is available.
To disable every stopping rule, also keep `--max-domain-regression 0`.
A stopped validation state is terminal: resuming that checkpoint logs the stop
without performing another optimization step. Version-1 guard state can migrate
with its checkpoint step while domain guarding remains disabled; enabling a new
domain policy requires a separately configured run.

Run the local CPU checks with:

```sh
python -m pytest -q tests/test_pretrain.py tests/test_tokenizer_spec.py
```

These checks cover causal masking, learning from next-token targets, held-out
separation, deterministic evaluation, and matching an uninterrupted run after
checkpoint/resume. They do not run on Matrix or alter its active training job.

## Adaptive training mixture

`config/pretrain/causal-999m-v100k-adaptive.json` enables an opt-in feedback
controller for a **new** run. Static sampling remains the default. The initial
English/code/agent training probabilities are 0.5/0.3/0.2. Checkpoint selection,
early stopping and monitoring always use those frozen objective weights, even
when training probabilities change. Lowering a difficult source's probability
cannot make the reported validation objective improve by itself.

Every evaluation interval records per-source mean training cross-entropy and
sample counts from the existing forward passes, without an additional model
evaluation or per-batch CPU synchronization. Validation examples never enter
the training stream. The controller requires full-pass reservoir validation,
every source's validation loss, and at least 32 training sequences per source
in an interval; otherwise it leaves probabilities unchanged. The first three
eligible intervals calibrate each source's own training and validation baseline.

After calibration it smooths the log of each loss relative to its baseline
with coefficient 0.3. Its score is the relative validation trend minus twice
the positive difference between validation and training trends. Sources whose
validation learns more slowly can receive additional probability. A normalized
validation/training gap above 5% blocks an increase for that source; this is a
heuristic warning about generalization, not proof of overfitting. Comparing
relative progress avoids favoring a source solely because its raw loss is high.

Targets are anchored to the initial mixture and projected onto probability
bounds: each source stays between half and twice its initial probability, with
at most 0.02 absolute change per evaluation. The controller cannot add sources,
write training examples, or bypass the existing validation stopping guards.
Each `mixture_update` event records old/new probabilities, counts, trends and
held-back sources. Checkpoints preserve the policy, calibration, smoothed
signals, weights, partial-interval training statistics, sampler RNG and data
cursors. A resume rejects changed policy/validation identity or enabling or
disabling adaptation; exact CPU resume is covered by an integration test.

This is a bounded source-level heuristic. It does not deduplicate examples,
rebalance categories within a source, or establish a causal quality advantage.
Validate it against a matched static run before choosing it for a large job.
Keep a separate final test set outside this feedback and checkpoint selection.

For a bounded pilot on the existing local corpus:

```sh
python scripts/pilot_adaptive_mixture.py --name mixture-feedback-pilot
```

This trains two fresh, small models for 96 steps each on CPU with two threads,
the same corpus, tokenizer, seed, validation set and token budget. It never
contacts Matrix or loads an existing checkpoint. Results and hashes remain
private under `runs/mixture-pilots/`; run names cannot be reused. The receipt
records both outcomes rather than asserting that adaptation improved quality.

## Interleaving agent categories

The original agent training source walks `decisions.txt`, `opcodes.txt`,
`schemas.txt` and `transcripts.txt` sequentially. Source-level mixing alone
cannot expose all four categories throughout training. For a new candidate run,
prepare an explicit six-source view:

```sh
python scripts/prepare_agent_mixture.py --output data/mixtures/agent-categories-v2
```

This preserves the parent corpus. It verifies the agent files against their
parent manifest, separates the existing validation file using its declared
category order and byte/line counts, and audits exact duplicates plus
whitespace-normalized duplicates and training/validation overlap. Duplicate and overlapping
training records are excluded from derived training files; validation is never
moved into training or changed to improve a score. Unaffected training files and
English/code sources are referenced by path and SHA-256. The actual corpus had
30 normalized overlaps in op-code training despite zero exact-line overlaps,
plus 393 further whitespace-equivalent repeats within training. The second
derived view excludes all 423 records. Two normalized repeats within validation
remain unchanged and are reported, so the comparison does not improve a score
by editing validation after seeing results.

The private manifest records counts, exclusions, provenance, transform identity
and file hashes, and reports unmeasured semantic contamination checks. Only
small validation files and affected training files are copied. It refuses an
existing output directory, checks referenced files before publication, and
publishes `MANIFEST.json` only after verification. It uses selection validation;
no separate final test set is consulted.

`config/pretrain/causal-999m-v100k-category-mixture.json` uses this view through
`--corpus-manifest`. The initial token-window weights are:

| Source | Probability |
| --- | ---: |
| English | 0.500 |
| Code | 0.300 |
| Agent op-codes | 0.022 |
| Agent decisions | 0.048 |
| Agent transcripts | 0.104 |
| Agent schemas | 0.026 |

The conditional agent proportions (11/24/52/13) inherit the generator's declared
starting shares; they are not claimed to be optimal. Every fixed-length training
window is drawn from one category, preventing file-order category blocks.
Each category gets its own full-pass reservoir validation, fixed objective
weight, checkpoint-selection eligibility check and regression patience. An
op-code regression cannot disappear inside the aggregate agent score.

Manifest mode requires exactly the declared source set. It verifies file size
and SHA-256 at startup, rejects training/validation file overlap, and binds the
manifest digest into sampler checkpoint state. Changing the manifest or a
referenced file prevents a matching resume. References must remain unchanged
during a run; this is a verified view, not an immutable copy of its parent.
Static sampling remains the candidate recipe's default. Adaptive sampling can
also operate over these six sources with its existing bounds; use a declared
minimum observation count appropriate to the smallest category.

For a controlled CPU pilot:

```sh
TOKENIZERS_PARALLELISM=false python scripts/compare_agent_mixtures.py \
  --name category-mixture-comparison \
  --manifest data/mixtures/agent-categories-v2/MANIFEST.json
```

The default is three paired seeds, 512 steps and 65,536 tokens per arm, two
Torch CPU threads, and a small two-layer model. The sequential control uses the
original corpus; the candidate combines category interleaving with overlap
removal. Both final checkpoints are evaluated on the same fixed, category-level
validation windows and objective. Thus this measures the combined data change,
not a causal attribution to either component. The plan is written before
training: at least 1% mean weighted improvement, a weighted win for every paired
seed, and no source's mean loss more than 2% worse. Those are pilot advancement
criteria, not a production promotion or evidence about the active 1B model.
Private plans, per-arm results and the final receipt live in
`runs/mixture-pilots/`. Matrix is never contacted by either data-preparation or
pilot scripts, and the active run is not changed.

## Keeping the repository current

Fetch and compare `origin/main` before making conclusions about missing code.
`scripts/source_watch.py --update --notify` checks Git and deployed Matrix source
hashes. A clean `main` may fast-forward; a dirty or diverged checkout requires
review. The same read-only probe now reports recent training-versus-validation
trends, per-source regression, stale metrics, missing best checkpoints and
validation-identity gaps. It checks the registered trainer PID without reading
its memory or loading weights. `source_status` is separate from the training status;
notifications are deduplicated by alert type, not by each new training step.
It never signals, pauses, restarts, or writes to the trainer. Matrix is only read.
`runs/source-sync/watch-config.json` can name `training_run` and `trainer_pid`;
otherwise the current run and its privately recorded live observation are used.
Its source baseline is initialized explicitly once using
`--accept-baseline`; a later code change produces an alert rather than silently
becoming the new baseline. Private status lives in `runs/source-sync/latest.json`.
The LaunchAgent template under `launchd/` runs every 15 minutes on the coordinator.

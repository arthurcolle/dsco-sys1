# Causal language-model pretraining

System One first learns ordinary next-token prediction from text, code, and
synthetic agent documents. Decision-head training is a later, separate stage.
The causal-LM trainer optimizes cross-entropy on the next token at every position
in packed documents. It does not train the choice, success, or value heads.

The versioned recipe is `config/pretrain/causal-999m-v100k.json`. It describes the
998,562,562-parameter model with 46 layers, width 1,280, 20 query heads, five KV
heads, a 3,840-wide SwiGLU, and a tied 100,352-token embedding table.

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
The original Matrix run was deployed as an unversioned filesystem copy. Its
trainer was recovered into this repository on 2026-10-24; recovery did not alter
that process, its source files, its data, or its checkpoints.

On an admitted, idle training host, the corresponding training command is:

```sh
python -m sys1.model.pretrain \
  --name sys1-999m-v100k-run1 --preset sys1-999m-v100k \
  --tokenizer data/tokenizer-100k/tokenizer.json --corpus data/corpus \
  --mix en=0.5,code=0.3,agent=0.2 \
  --seq 1024 --micro-batch 2 --accum 8 --steps 30000 \
  --lr 3e-4 --warmup 500 --precision bf16 \
  --ckpt-every 200 --keep 3 --eval-every 200 --eval-seqs 32 --log-every 10
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

Run the local CPU checks with:

```sh
python -m pytest -q tests/test_pretrain.py tests/test_tokenizer_spec.py
```

These checks cover causal masking, learning from next-token targets, held-out
separation, deterministic evaluation, and matching an uninterrupted run after
checkpoint/resume. They do not run on Matrix or alter its active training job.

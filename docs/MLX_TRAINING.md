# Native MLX System One training

`sys1.mlx.model.Sys1Model` implements System One directly in MLX. The training
entry point is `python -m sys1.mlx.train`. Its model, data, gradient and optimizer
path does not import PyTorch. GPU execution is the default and unavailable Metal
raises an error; CPU is an explicitly selected debugging option.

The architecture preserves grouped-query causal attention, adjacent-pair rotary
positions, depth-scaled RMSNorm, SwiGLU, tied input/output embeddings and typed
choice/noul/value heads. MLX attention consumes compact K/V heads through
[`mx.fast.scaled_dot_product_attention`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html).
The typed heads remain frozen during causal pretraining; training them needs a
separate decision objective.

Parameters and Adam moments stay FP32. `--precision bf16` casts compute operands
and activations to BF16 while retaining FP32 master updates and FP32 cross-
entropy. The train step is compiled by default, supports gradient accumulation
and global-norm clipping, and excludes norms, biases and (by default) embeddings
from weight decay. `--gradient-checkpointing` recomputes block activations during
backpropagation. `--no-compile` is available for debugging.

MLX arrays use the shared CPU/GPU memory pool described in the
[MLX unified-memory documentation](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst).
The input path builds MLX arrays from tokenized documents without a Torch
tensor or a `.to("mps")` staging step. Python/tokenizer ingestion still takes work;
this does not claim zero-copy construction from Python or NumPy buffers.

## Install and run

On Apple Silicon, install the tested MLX release in the project environment:

```sh
uv pip install --python .venv/bin/python 'mlx>=0.32.4,<0.33'
```

The package also declares a `mlx` extra. Existing package-wide Torch dependencies
are retained for the older backend; the native entry point does not load them.
The shared document reader is in `sys1/model/corpus.py` and has no tensor backend.

For a bounded run on the expanded nine-source corpus:

```sh
.venv/bin/python scripts/prepare_pretrain_run.py \
  --recipe config/pretrain/mlx-32m-systems.json --name mlx-sys1-small
```

The private manifest under `runs/mlx-pretrain/mlx-sys1-small/` records the command
but does not launch it. Execute its `command_argv`, or run:

```sh
TOKENIZERS_PARALLELISM=false .venv/bin/python -m sys1.mlx.train \
  --name mlx-sys1-small --preset sys1-mlx-small \
  --tokenizer data/tokenizer-100k/tokenizer.json \
  --corpus-manifest data/mixtures/agent-plus-systems-v1/MANIFEST.json \
  --device gpu --precision bf16 --seq 128 --micro-batch 8 --accum 1 \
  --steps 128 --warmup 16 --lr 0.0003 --log-every 16 --eval-seqs 16 \
  --eval-every 0 --early-stop-patience 0 --max-domain-regression 0 \
  --gradient-checkpointing --memory-limit-gib 8 --keep 2
```

This recipe defines 32,117,506 parameters with the 100,352-token vocabulary.
It has a fixed 131,072-token budget, final validation and one resumable checkpoint
to fit limited local disk. Its periodic stopping guards are explicitly disabled
for this bounded pilot. The trainer's normal defaults enable periodic evaluation
and the same declared global/per-source validation guards as the other backend.

`config/pretrain/mlx-999m-systems.json` defines the 998,562,562-parameter successor
architecture with activation recomputation, accumulation and a declared 32 GiB
MLX allocator guideline. It is a recipe, not evidence that this full-sized job launched.
FP32 masters/gradients/Adam state impose a parameter-memory minimum; checkpoints
also need sufficient disk. Both checks run before model allocation, with another
disk check before each atomic checkpoint write.
`--memory-limit-gib` uses MLX's allocation guideline, not an operating-system hard
cap; actual active and peak allocations are recorded in the log.

## Data, checkpoints and evidence

Training requires an explicit hash-verified corpus manifest. Validation uses a
fixed full-pass seeded reservoir for every source, bound to the selected tokens
and tokenizer identity. Source probabilities and fixed evaluation weights are
distinct when `--mixture-policy adaptive` is enabled. Per-source training losses,
counts, sampling probabilities and adaptation decisions are logged.

Checkpoints contain safetensors model/optimizer state and JSON sampler cursors,
buffers, RNG, source statistics, validation counters and adaptation state. Model
configuration, optimizer policy, precision, MLX version and data identities are
checked on resume. The model has no stochastic forward layers; initialization
is seeded and the sampler's random state is saved. GPU reductions need not be
bitwise identical across independent runs. Resume tests compare discrete state
exactly and numerical state within floating-point tolerance.

The newest recovery checkpoint and selected best both fit within `--keep`.
An early-stopped state is terminal on resume. Nonfinite loss/gradient aborts
without saving that step. Existing run names cannot silently start over; a
prepared-only directory must match its recorded source, data and settings.

Logs include actual backend/device, Apple hardware, precision, parameter count,
tokens, interval throughput, active/peak MLX allocation and source counts.
Compilation, validation and checkpoint time are distinguishable from steady
training intervals. A finite-loss pilot establishes execution, not downstream
quality or an advantage over the protected model.

The existing Matrix process is outside this path. No MLX command here connects
to it, changes its files, sends it a signal, or replaces its checkpoints. Use a
separate versioned release for any authorized successor.

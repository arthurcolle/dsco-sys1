# MLX streaming mixtures with bounded storage

This path streams the **existing cleaned corpus** from an owned source host. It
does not create dataset copies, token packs, trial datasets, or a persistent HTTP
cache. It does not modify a running trainer. Source files remain where they are;
the source publishes a small hash index and serves at most 256 KiB per request.

Use `sys1.mlx.train` for the next invocation. The existing protected Matrix run
uses its already-deployed backend and cannot acquire this input path by editing
files. No restart, replacement, or competing GPU work is part of preparation.

## Publish the data already present on the source host

```sh
python -m sys1.datasets.http_stream publish \
  --corpus data/mixtures/foundation-complete-v3/MANIFEST.json \
  --tokenizer data/tokenizer-100k/tokenizer.json \
  --output runs/stream-source-v1
python -m sys1.datasets.http_stream serve --index runs/stream-source-v1 --port 8955
```

Publishing verifies existing file hashes and scans bytes to compute chunk hashes.
It writes only `manifest.json`, `server-index.json`, and a receipt. The source
server binds **127.0.0.1 only**; it has no upload, execution, or modification API.
Keep its index private. Streaming is only as available as the source host and
authenticated route. No fallback corpus is silently substituted on failure.

Carry the stream to Matrix using a strict-host-key DSCO Fleet SSH tunnel to that
source host. The local port on Matrix forwards to the source's loopback 8955.
Do not open the source server publicly. The client also accepts HTTPS; cleartext
HTTP is accepted only at loopback. Never put credentials in a stream URL. A source
host and tunnel must remain available throughout training.

## Initialize from the completed current run, entirely without Torch imports

The existing checkpoint container is a legacy `.pt` zip, not an MLX checkpoint.
`scripts/convert_mlx_initialization.py` interprets a restricted allowlist of
metadata objects and streams raw model tensor bytes to safetensors. It does not
load the optimizer, instantiate a model, import Torch, or touch a GPU. Source
tensor bytes and converted tensors are hashed; config and tokenizer identities
are pinned. Unsupported objects or noncontiguous tensors fail closed.

Run conversion **on a capacity-checked storage node after the parent finishes**,
not on the interactive workstation. It requires the parent's `done` event and a
checkpoint matching that step, and refuses to overwrite an output directory.

```sh
python scripts/convert_mlx_initialization.py \
  --checkpoint "$PARENT_RUN/checkpoints/step_00030000.pt" \
  --parent-log "$PARENT_RUN/log.jsonl" --expected-step 30000 \
  --tokenizer "$TOKENIZER" --output "$INITIALIZATION"
```

This is a **model-only continuation with a fresh MLX optimizer**, not a claim of
exact optimizer-state resumption across frameworks. The receipt/config must match
the student's architecture and tokenizer. Configured max context may increase;
other architecture fields may not change. Model/checkpoint storage is separate
from the zero dataset cache: ~999M FP32 weights still need about 4GB, and resumable
Adam checkpoints need substantially more. No checkpoints are copied to the
interactive workstation by this streaming path.

## Start the next MLX job with a declared runtime budget

```sh
python -m sys1.mlx.train --name sys1-stream-next --preset sys1-999m-v100k \
  --tokenizer "$TOKENIZER" \
  --stream-manifest http://127.0.0.1:8955/manifest.json \
  --stream-manifest-sha256 "$STREAM_SHA256" \
  --stream-policy "$WEIGHTS_JSON" \
  --init-model "$INITIALIZATION/model.safetensors" \
  --init-receipt "$INITIALIZATION/initialization.json" \
  --device gpu --precision bf16 --seq 1024 --micro-batch 1 --accum 16 \
  --steps "${TRAIN_STEPS:?declare the next run budget}" \
  --lr 2e-5 --warmup 100 --gradient-checkpointing --loss-chunk-tokens 256 \
  --memory-limit-gib 32 --eval-every 200 --eval-seqs 256 --ckpt-every 200 --keep 2
```

These are preparation commands, not an automatically launched job. No 5M-token
trials or trial runner is included. Use a new release directory/run name. Preserve
the active run and all its checkpoint pointers. The learning rate above is a
conservative proposed continuation setting, not a measured optimum.

## Change mixture weights dynamically

```sh
python -m sys1.datasets.http_stream policy \
  --path "$WEIGHTS_JSON" --manifest-sha256 "$STREAM_SHA256" \
  --version 2 --weights '{"en":0.6,"code_python":0.06,"code_native":0.22,"code_web":0.05,"code_shell":0.02,"code_docs":0.02,"code_config":0.02,"agent_real":0.003,"systems_native":0.005,"systems_otp":0.002}'
```

Weights are normalized. All pinned source names must remain present; zero pauses
a source and at least one weight must be positive. The writer atomically replaces
a file <=64 KiB. The trainer checks it **once per optimizer batch**, logs each new
version/digest and its weights, and checkpoints policy version, RNG, source byte
cursors, token buffers, and realized draw counts. Invalid identities, edits that
reuse/roll back a version, corrupt chunks and network errors fail rather than
silently change data. A new source inventory requires a new pinned manifest/new
run; this version changes proportions of predeclared sources only.

External policy and the existing automatic adaptive-mixture controller cannot be
enabled together. Without an external file, the stream can use the existing
selection-validation adaptive controller. In both cases **validation sampling and
its objective weights remain frozen**; final-test data must never control weights.

## Representative held-out evaluation without copied packs

Startup streams the entire declared validation split, performs seeded full-pass
window reservoir sampling, and retains only the selected windows in RAM. A small
source contributes all its available complete windows; nothing is repeated to
pad a requested count. The tokenizer, stream manifest, selected indices, and
window bytes bind the validation identity. Historical validation exposure remains
historical exposure; this does not create a retrospectively untouched test set.

The data client retains one <=256 KiB chunk per source and bounded token buffers,
with a 1 MiB JSONL-record cap and 262,144-token document cap. It never writes HTTP
responses or token packs to disk. Reconnect/resume retrieves the pinned chunks
again from byte cursors. Checkpoints include bounded in-flight token buffers;
they are recovery state, not a dataset cache.

## 32K context: supported path, unmeasured full-model feasibility

The native model already uses compact GQA and MLX's fused causal attention. The
new chunked loss computes the tied vocabulary head in token chunks, rematerialized
for backward, rather than retaining a full `[batch, sequence, vocabulary]` logits
tensor. For vocabulary 100,352, a single 32,768-token FP32 logits tensor is
13,153,337,344 bytes (~13.15GB); a 256-token chunk is ~103MB. Hidden states,
attention compute, gradients and optimizer memory still remain.

For sequence lengths above 4096 the trainer requires both
`--gradient-checkpointing` and `--loss-chunk-tokens`. A future 32K invocation can
set `--seq 32768 --micro-batch 1 --accum 1 --loss-chunk-tokens 128`, under an
explicit memory/runtime budget. This is **not** evidence that the 999M model fits
or trains usefully at 32K on Matrix. Start with the next short-context run, then
extend 4K -> 8K -> 16K -> 32K as actual memory/throughput and long-context
validation permit. Accepting longer input does not mean useful long-range
behavior has been trained. Streaming addresses disk use, not attention FLOPs.

The tests compare full and chunked losses/gradients on a tiny MLX model, including
compiled updates; they do not run full-sized 32K training or competing Matrix work.

## Checks

```sh
python -m pytest tests/test_mlx_streaming.py tests/test_mlx_legacy_weights.py -q
```

Checks cover deterministic stream resume, atomic policy changes and paused
sources, fixed validation, changed-source rejection, UTF-8 chunk boundaries,
Torch-free imports/conversion, completed-parent guarding, and chunked-loss
gradient parity. No dataset or financial evidence belongs in Git.

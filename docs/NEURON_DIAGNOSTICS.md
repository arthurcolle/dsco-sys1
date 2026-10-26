# Offline SystemOne neuron diagnostics

This is an original SystemOne adapter inspired by the intervention-first method
in [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). It includes
Torch instrumentation for existing checkpoints, native MLX instrumentation, a
bounded CLI, and standalone interactive HTML. It does not vendor the upstream
code or its Qwen caches. No server, model download, embedded agent, shell tool,
training update or deployment is started.

Keep checkpoints, probe text and results private under `data/` and `runs/`.
The protected Matrix machine is excluded by the CLI. Use an offline copy on a
capacity-checked host; keep large checkpoint storage and inference off Chrysalis.

## Measurements

The adapter captures the natural `silu(gate) * up` activation before `mlp.down`.
It uses SystemOne's fused `gate_up`, its actual normalization, and its per-layer
scaling. Torch uses temporary hooks removed in `finally`; MLX uses the same
forward equations with explicit activation instrumentation and no Torch imports.
No model parameter is changed.

Each probe specifies a prompt and a continuation. The entire continuation is
scored under teacher forcing: `NLL = -mean(log p(target_i | prompt IDs, target_<i))`.
Prompt and continuation are tokenized separately, with an explicit EOS-prefix
policy. Tokens are never silently truncated. These scores measure sensitivity
of a fixed continuation; they are not generated-code correctness or general
model quality. Probe roles are diagnostic or selection validation. Final test
sets are rejected, and results are never fed into training or mixture adaptation.

For selected neurons the tool records:

- Natural per-token activation and baseline residual calibration.
- Zero ablation and the change in continuation NLL. Positive means removal hurt
  this continuation; negative means it helped.
- Seeded random output directions with the same per-token perturbation norm as
  removing that neuron's natural contribution. Small control counts are screens,
  not a significance estimate.
- Optional positive/negative additive doses, as fractions of the baseline
  residual norm along the neuron's down column. Default: ablation only.

Calibration and every measurement in one probe hold the same model lock.
Interventions cannot recalibrate themselves or interfere with another request.
The baseline's position-zero activation is visible but excluded from peak-based
selection and residual calibration when other positions exist.

## Run on an explicit checkpoint

Native MLX checkpoint directories contain `state.json` and `model.safetensors`:

```sh
.venv/bin/python scripts/inspect_neurons.py \
  --backend mlx --checkpoint runs/mlx-pretrain/MY-RUN/checkpoints/step_00000200 \
  --tokenizer data/tokenizer-100k/tokenizer.json \
  --probes config/neurons/probes.json \
  --layers 10,22 --units 10:12,22:34 \
  --output runs/neuron-diagnostics/MY-RUN-step200
```

MLX defaults to the Metal GPU; `--device cpu` must be selected explicitly for
debugging. For Torch state-dict checkpoints use `--backend torch`, which defaults
to CPU. The Torch loader retains weights-only deserialization, with a concrete
allowlist for the NumPy RNG/buffer types in original training checkpoints.

The built-in probes are handwritten synthetic diagnostics, including the
reserved addition-function family. The agent-format examples are **not real
trajectories**. Supply a private probe file for real source-stratified analysis:

```json
{
  "schema": "sys1-neuron-probes-v1",
  "role": "diagnostic",
  "probes": [{
    "id": "stable-probe-id", "stratum": "code", "source": "provenance reference",
    "prompt": "def first_or_none(items):\n    if not items:\n        ",
    "continuation": "return None\n"
  }]
}
```

Without `--units`, the tool selects `--neurons` units per layer by their largest
natural absolute activation. This is exploratory selection. Use the same fixed
`--units layer:neuron,...` panel across checkpoints for longitudinal comparisons.
Neuron indices are meaningful within the same initialized model lineage;
matching dimensions do not establish alignment between independently trained
models. The tool does not assign semantic neuron labels.

## Limits, identity and resume

Defaults: at most eight probes, one middle layer, two selected neurons, one
matched control per neuron, 96 input tokens, 64 forward passes, 300 seconds,
1.5 GiB peak resident memory, and 0.5 GiB checkpoint bytes. Each can only be
increased within the CLI's finite bounds. Torch uses two CPU threads. For native
MLX the runtime memory limit and cache limit are also set explicitly. The resident
guard is checked between forward passes; it does not prevent a transient
allocation from exceeding the guard inside a pass.

For `P` probes, `U` total selected units across layers, `C` controls, and `D`
positive dose sizes, planned forwards are `P * (1 + U * (1 + C + 2D))`.
Budget and sequence failures are checked before the first measurement.

`plan.json` freezes checkpoint bytes/SHA-256, tokenizer and probe digests, encoded
probe IDs, model implementation, adapter code, backend, precision, runtime,
options, seed and resource budget. Every case has its own digest and is written
atomically under an exclusive process lock. Completion is published only after
inputs and implementation hashes are checked again. Legacy checkpoints without
a training-tokenizer digest cannot authenticate that historical choice; the
supplied tokenizer is still frozen in the run identity.

An existing output requires `--resume`. Any changed identity requires a new run
directory; old results and their metadata are never relabeled. Interrupted runs
can reuse complete probe cases. A partially measured probe is rerun as a unit.
No completed case is overwritten.

## Owned int8 caches

Original locally produced caches can be inspected on the host that already
holds them, using `--trusted-pickle-source deployed_model.py` and
`--trusted-pickle-receipt comparison-receipt.json`, together with the checkpoint
and tokenizer paths. This is CPU-only. It explicitly executes that source and
unpickles an owned model, so it must never be used for untrusted files. Source,
tokenizer, step, config and original tensor identities must match the receipt.

Int8 results are labeled as such. Dynamic activation quantization means a
one-neuron change can alter the quantizer's scale, so these measurements can
include quantization effects. Use an independent FP32 control before interpreting
them as a property of the original FP32/BF16 model. Size resource budgets for the
actual cache and capacity; no large cache is transferred by this tool.

## Browse and compare

Open `runs/neuron-diagnostics/MY-RUN-step200/report.html` locally. It displays
continuation scores, ablation/control deltas, optional dose responses, and colored
per-token natural activation traces. It loads no weights and makes no network
requests. Probe and token strings are inserted as text, with embedded JSON
escaped against HTML/script injection.

To create a comparison from compatible completed records:

```sh
.venv/bin/python -m sys1.neurons.report \
  runs/neuron-diagnostics/MY-RUN-step200 \
  runs/neuron-diagnostics/MY-RUN-step400 \
  --output runs/neuron-diagnostics/comparison.html
```

The renderer verifies plan/result digests and requires identical tokenizer,
probes, encoded inputs, architecture, model implementation, adapter code,
precision, backend, device, framework and options. A row's cross-checkpoint delta
is only shown when the same unit was measured in the first run. Compare units
only within the same model lineage. The report is an analysis artifact; it is not
a model promotion or a training-health verdict.

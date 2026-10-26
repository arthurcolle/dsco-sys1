# Role-aware trajectory training and evaluation

This path tests whether concentrating loss on assistant calls and answers helps
System One learn from complete trajectories. It uses the custom native MLX model,
defaults to the Apple GPU, and never controls a production job or reads a model
checkpoint. It is a bounded diagnostic, not a replacement for foundation training.

## Source and validation integrity

`scripts/reconstruct_trajectory_selection.py` reconstructs the **existing**
selection episodes from pinned chronicle events and hash-verified blobs. It
requires a unique match of event identity, session, redacted goal and final text.
The old selection and training shards remain intact. Larger bounded payload
reads replace display truncation. If redacting serialized arguments breaks JSON,
the repair redacts decoded values and keys and serializes a valid object; merged
keys are rejected. Partial, missing or ambiguous reconstruction blocks the
diagnostic gate. New reserved and final data are never consumed.

`sys1.datasets.trajectory_windows` annotates roles using offsets from a single
tokenization of each complete source document. Each example is a contiguous
context slice ending at a chunk of assistant target tokens. A slice can start
mid-message; it is a model context, not a claim that a fragment is a complete
tool record. Original complete trajectories remain available in the parent.

Selection is deterministic and bounded per session, target role and tool class.
Calls and answers alternate before further windows from a role are selected.
Every validation anchor is reserved before subsampling, and exact/token-shingle
near overlaps are removed from derived training only. Session reservations remain
in force. Read-only memory maps store tokens, role labels, valid masks, target
masks and content-addressed item metadata. Loading verifies hashes, shapes,
vocabulary, mask semantics, padding, unique items and session separation.

## Controlled native MLX study

The two arms share initialization, session-balanced schedules, context tokens,
optimizer, learning-rate schedule and step budget. The uniform arm gives each
nonpadding token equal loss weight. The targeted arm gives assistant target
tokens weight 1 and context tokens the registered relative weight (default 0.1).
Both normalize by the sum of weights. Padding has zero weight. Context remains
visible to causal attention and can receive gradients through target prediction.

The primary panel uses complete reconstructions of the existing selection set.
An optional unchanged legacy panel is reported separately as a secondary
diagnostic. It never substitutes for the primary result or supports promotion.
Per-window observations retain item, role, task, context and session identities.
The primary NLL averages roles within a session and then averages sessions;
long sessions do not acquire more independent votes by producing more windows.

Uncertainty uses a paired crossed bootstrap over seeds and whole sessions.
Observed role, task and context guards use a Bonferroni correction. The default
gate requires at least three seeds and twenty sessions, a 1% practical gain,
a positive lower primary confidence bound, all paired seeds winning, complete
selection coverage, and no guarded regression beyond 2%. Thin slices fail their
support check. Small seed counts and a narrow local selection population still
limit inference; a passing diagnostic never establishes production model quality.

```sh
.venv/bin/python scripts/reconstruct_trajectory_selection.py \
  --parent-manifest data/mixtures/foundation-complete-v3/MANIFEST.json \
  --output data/mixtures/trajectory-selection-complete-new

RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python \
  -m sys1.datasets.trajectory_windows \
  --parent-manifest data/mixtures/trajectory-selection-complete-new/MANIFEST.json \
  --tokenizer data/tokenizer-100k/tokenizer.json \
  --output data/mixtures/trajectory-role-complete-new

.venv/bin/python -m sys1.mlx.trajectory \
  --manifest data/mixtures/trajectory-role-complete-new/MANIFEST.json \
  --output runs/mixture-pilots/trajectory-mlx-new \
  --steps 128 --seeds 1103,2207,3301 --memory-gib 2 --ttl-seconds 300
```

To add the legacy secondary panel, build a separate window view from the original
foundation-complete parent with the same context/tokenizer policy and pass its
manifest using `--legacy-panel`. All views, sources, plans, token files and
observations remain private under ignored `data/` and `runs/`. Use new output
names for new trials. The runner freezes code, data, schedules, objectives,
hardware and resource budgets before training, records actual GPU memory and
token exposure, checks identities afterward, and preserves failures. It saves no
production weights and excludes the protected Matrix host.

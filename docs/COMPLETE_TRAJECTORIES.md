# Complete real trajectories

`scripts/prepare_complete_trajectories.py` creates a new private mixture from a
foundation-quality parent and a bounded index of local DSCO chronicle sessions.
It replaces the real-trajectory **training** source and references the other
parent shards. It never launches training or contacts a fleet host.

The legacy episode exporter deliberately remains unchanged. Its small display
limits can cut JSON arguments and observations; those historical datasets must
remain reproducible. The new reader loads complete UTF-8 blobs, checks their
content-addressed SHA-256 identities, and rejects missing or oversized payloads.
Tool results must match the call's exact span ID and tool name, including when
several calls use the same tool concurrently. Unresolved calls, non-object or
ambiguous argument JSON, incomplete final responses and low tool-success rates
are excluded. Payloads containing the rendering format's control markers are
excluded to preserve role and argument boundaries.

Rendering retains the existing role format and redacts credentials and private
identifiers. Arguments must still parse after redaction. Completion and tool
status are screening signals; they do not verify that a task was solved.
Redaction counters include screening and rendering passes and excluded records;
they count transformations rather than distinct affected documents.

All parent validation documents and prior private/session reservations remain
reserved, including documents excluded from selection. Discovering a reserved
goal later in the scan removes earlier candidates from that entire session.
New hash-selected validation sessions are stored separately and excluded from
both training and pilot feedback. The existing selection files and validation
token pools remain byte-for-byte unchanged. Exact and shingle-near validation
overlaps and training duplicates are excluded; this is not semantic deduplication.

```sh
RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python \
  scripts/prepare_complete_trajectories.py \
  --parent-manifest data/mixtures/foundation-quality-v1/MANIFEST.json \
  --session-index data/exports/complete-session-index-20261026.txt \
  --output data/mixtures/foundation-complete-new
```

The index is a frozen, private newline-separated list of chronicle event paths.
It must fit the declared file budget. The default scan reads at most 7,500
indexed files, with a 1 GiB read budget and a 300 second time budget checked
between sessions. Each session has separate event and payload limits, so read
and time totals may exceed those scan limits by one bounded session. Encoded
candidate records have a 32 MiB combined cap. The manifest records actual
coverage, exclusions, redactions, event identities, input and code digests,
unchanged validation identities and supply-limited training probabilities.
It retains inherited license and attribution limits. All data, pools, manifests,
source indexes and receipts remain under ignored `data/` or `runs/` paths.

The candidate includes two baseline/candidate pool sets for a paired CPU pilot:
baseline uses the foundation parent's training source; candidate uses complete
real traces. Both use identical frozen selection pools. A source-only diagnostic
must be registered separately from a full-mixture experiment and must not be
presented as a full-mixture or production quality improvement. A structural
audit or prepared manifest is not a promoted model or a launched training run.
`scripts/compare_quality_mixture.py --validation-windows 512` can score the full
retained source reservoir for a registered diagnostic. The default remains 16
frozen windows per source; neither option pads or repeats validation windows.

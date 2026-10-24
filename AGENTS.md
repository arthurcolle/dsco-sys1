# System One development and source synchronization

This checkout is the canonical source repository for System One. Keep its
existing Git remote current with the reviewed, tested source changes made for
the user's requested System One work.

All agent commits must use `dsco <agent@distributed.systems>` for both author
and committer. Configure this identity locally and verify it with `git var
GIT_AUTHOR_IDENT` and `git var GIT_COMMITTER_IDENT` before committing.

Before developing or reporting repository state:

1. Fetch `origin` and compare local HEAD with `origin/main`. A stale local
   checkout is not evidence that work was never committed or published.
2. Preserve unrelated tracked edits and untracked files. Use an isolated
   integration worktree when branches have diverged; reconcile before pushing.
3. Keep the trainer, corpus/tokenizer builders, architecture presets, run
   recipes, tests, and reproduction instructions together in Git. Finish
   requested source changes with scoped commits and a normal fast-forward push.
   Never force-push or automatically commit unrelated work.
4. Run the appropriate checks and verify the remote commit after pushing.

## Running training jobs

The existing Matrix training job is protected. Read-only observation is allowed.
Do not signal, stop, restart, replace, redeploy into, or change that job's source,
inputs, configuration, checkpoints, or checkpoint pointer unless the user
explicitly authorizes that change. New deployment code belongs in a separate
versioned release directory. Do not start competing GPU work on Matrix.

Training data, conversations, credentials, weights, private manifests, and live
run observations remain outside Git under `data/` and `runs/`. Record dataset,
tokenizer, code, and checkpoint identities in private receipts. A prepared
recipe is not evidence of a launched training job.

## Periodic source check

`scripts/source_watch.py` compares Git state and Matrix source hashes against an
explicitly accepted private deployment baseline. It can fast-forward a clean
`main` checkout, but never overwrites local work or writes to Matrix. Source
drift requires review and a scoped integration; the watcher must not silently
publish arbitrary remote code or reset the deployment baseline.

The private status is `runs/source-sync/latest.json`; the macOS LaunchAgent
checks every 15 minutes while the coordinator is awake and connected.

The watcher also reports aggregate training-quality diagnostics. A historical
metric minimum or a single worsening source is a review signal, not proof of
broad model overfitting. Check validation coverage and identity first. New runs
should use the full-pass reservoir recipe and declared per-source guardrails;
the protected live run must not be hot-patched to adopt these changes.

# QVELT post-run improvement loop

Use after an owner-identified real training run. Read AGENTS, current STATE,
EXPERIMENTS and the supplied artifacts first. Keep a baseline; change one major
factor per experiment. Prefer checking evaluation and data quality before adding
architecture complexity at small scale. No paid run is authorized by this loop.

## 1. Intake

Supply run ID, exact command, resolved recipe/config, code revision/dirty status,
tokenizer/data/checkpoint manifest hashes, seeds, hardware/precision/world size,
raw training/validation metrics, loss curves, failures/restarts and eval outputs.
Include the baseline and the scaling-sanity configurations/results, plus held-out
split identities and contamination-audit coverage. If anything is absent, label
it missing; do not substitute old demo results for a real run.

From the matching checkout and existing environment, replace the two absolute
paths below with the completed run and a NEW archive filename:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/summarize_run.py \
  /absolute/path/to/completed-run \
  --archive /absolute/path/to/new-qvelt-run-logs.tar.gz
```

This is the existing log/metadata collector, not a training command. It excludes
tensor/optimizer payloads and omits files larger than64MiB. Eval outputs/receipts,
contamination reports, original launch command, baseline/scaling artifacts and
curve images may need to be supplied separately; the archive is not complete
evidence by itself. Review samples, paths and logs for secrets/personal data
before sharing. Retain originals and hashes; mark any redacted diagnostic copy.
Do not send credentials, private corpora or model weights for initial diagnosis.

## 2. Diagnose and attribute

- Verify identities, comparable objectives/token normalization, held-out splits,
  parser correctness, restarts and consumed versus applied-update counters first.
- Examine raw and smoothed train/validation curves, spikes, plateaus, divergence,
  domain losses, gradients/clipping, samples and failure/rollback records. Keep
  raw attempts; a merged last-attempt view can hide interruptions.
- Separate startup/compile/eval/checkpoint overhead from steady throughput and
  report end-to-end cost too. MFU requires the stated device/precision peak and
  FLOP convention; missing MFU stays unknown, not zero or an inferred GPU claim.
- Compare scaling only within justified data/tokenizer/objective/budget ranges.
  R018's four short synthetic runs are descriptive controls, not a prediction for
  a real corpus. Without a comparable fit, scaling expectations remain unverified.
- For each observed weakness consider data coverage, data quality, hyperparameters,
  architecture limits, post-training recipe and evaluation artifact. Distinguish
  measured facts, competing hypotheses and missing evidence. Propose the cheapest
  discriminating check before attributing causality to a preferred explanation.

## 3. Prioritize, implement and record

Append a ranked experiment queue to PLAN after intake. Each entry needs hypothesis,
baseline/control, one major config/data/code diff, exact reproducible command,
hardware, compute estimate with formula/assumptions, success metric, stop condition
and rollback. Rank expected gain/cost with explicit uncertainty; do not invent
numeric gains. Estimates include relevant attention/recompute/eval/seed costs.

Implement only evidence-required changes, run CPU/reference and appropriate
integration tests, and preserve immutable original artifacts. Tokenizer/data/
architecture/precision changes must declare lineage/replay consequences and pass
the existing named-human, legal, hardware and owner-budget gates. Stop for owner
review after a coherent change; this loop is not permission for unrelated Step B.

Append observations, failures, negative results and provenance to EXPERIMENTS;
append non-trivial decisions to DECISIONS. No experiment ID/result is created for
an intake with no run. Improvement needs a held-out, decontaminated evaluation
with comparable settings and multiple seeds where noise is plausible. A training
loss drop, format fix or one favorable sample is not general quality evidence.

## 4. Russian report

Give a plain-language summary; measured facts versus hypotheses; issue/cause/
discriminating-check table; ranked next experiments; exact server commands and
cost/stop gates; changed files/commit/test output; and the least certain conclusion.
If inputs are missing, request them and report the block instead of making up a
diagnosis, ranked trials or training commands. Do not launch the next run yourself.

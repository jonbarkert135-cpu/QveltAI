# Qvelt current state

Updated: 2026-10-31. Owner: Lead agent.

## Current phase and last completed task

Phase 3, M3.1 complete for the authored-fixture tokenizer workflow. Last completed:
strict YAML recipes, manifest-linked JSONL training, verified tokenizer bundles,
held-out segmentation and independent process replay. Code commit `2d1a05a`,
followed by documentation. Next is M3.2 deterministic corpus preparation.
No neural model, trainer, inference service or semantic checkpoint loader exists.

Checkout /workspace/newai_cddevel, main, jonbarkert135-cpu/QveltAI. Publication of
verified work/reports to main is authorized and active. No environment draft
change or paid computation. Ponytail full is required for all coding tasks;
409 skills remain installed (408 vendor plus the user attachment).

## Verified state

- CLI: tokenizer resolve/train/verify/evaluate; schema 1 recipe has no local paths
  and pins canonical raw-data identity. Strict bounded ordered text-only JSONL.
- Actual consumed file hashes/counts, safe regular-file reads, atomic manifest-last
  publication, full BPE/controls/vocabulary/recipe/data/provenance-link validation.
- 315 tests on each Linux/Python 3.11.14/3.12.14/3.13.5, no skips. Ruff/format and
  strict mypy green on 27 files. 3.11/3.13 tested installed wheel,3.12 editable.
- Wheel/source build and content checks pass; wheel Python module bytes match
  checkout. Base-only3.12 wheel needs no torch and ran train/verify/evaluate.
- Six clean-commit processes produce matching tokenizer JSON per policy across
  three Python versions and varied hash seeds/Rayon counts. Fixture artifacts
  contain actual observed code/environment; deliberate corruption exits1.
- Exact identities/results/commands: docs/development/validation_m3_1.md and
  tokenizer_fixture_results.json; experiment R005. Lock unchanged this session;
  tokenizers 0.22.2 and existing CPU PyTorch 2.9.1 remain pinned.
- CI includes tokenizer smoke; remote job status was not inspected this session.

## Limits and blockers

No external corpus approved or ingested. Registered 48/12 authored software
fixtures have development/test permission only; no representative corpus or
production 32768-token tokenizer is frozen. Model parameter counts stay analytical.
Streaming limits document/record size, not BPE pair-statistic memory. JSON byte
identity was measured on this Linux release/fixture, not promised universally.
Source/license strings are declarations, hashes are not signatures, provenance
is observational and dirty code is not snapshotted. Dataset payloads are checked
when consumed; a copied data manifest alone does not prove historical training.
Linux local filesystem only; no native Windows/network-store/power-loss/GPU test.
User's RTX3060 Laptop/Core i5/16GB RAM has not been profiled; exact VRAM/driver/
OS/server budget are unknown. Project-wide release license/language priorities
remain user decisions; they do not block authored-fixture development.

## Next three tasks

1. M3.2: implement bounded deterministic preparation of approved local documents,
   explicit transformation/source IDs and content hashes, with authored fixtures.
2. M3.2: exact/near dedup, document-level split isolation and decontamination
   contracts; retain audited source/license provenance. External data stays gated.
3. M3.2: checksummed token shards, packing/loss masks, deterministic rank/worker
   allocation and resume-order/corruption tests, before model/training work.

## Reproduce

```bash
cd /workspace/newai_cddevel
uv sync --frozen --extra cpu
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy
git diff --check
```

Expected: 315 tests, lint/format/types green. CLI runbook:
docs/development/tokenizer_workflow.md; choose a fresh output directory.
No language-model training/serving command exists. Earlier M2/R004 evidence
remains in the experiment log and validation reports, not current capability claims.

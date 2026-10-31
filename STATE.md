# Qvelt current state

Updated: 2026-10-31. Owner: Lead agent.

## Current phase and last completed task

Phase 2 is complete: M2.1 package/config/runtime tooling and M2.2 typed manifests,
observed provenance, bounded IO and atomic local artifacts. Next is M3.1 tokenizer
implementation/workflows. No tokenizer, model, trainer, inference service or
semantic model/optimizer checkpoint loader exists.

Checkout `/workspace/newai_cddevel`, branch `main`, repository
`jonbarkert135-cpu/QveltAI`. M2.2 code commits: `9027efe`, `f4fd2cc`, followed by
documentation. Publication remains paused: no push, environment draft edit,
republication or old setup-script replay occurred. User files/history preserved.

## Verified capabilities

- Schema-1 run/data/tokenizer/checkpoint records and canonical SHA-256 identity;
  strict bounds/unknown versions, lineage, file roles and actual config/lock checks.
- POSIX write-once publication, streaming/snapshot hashing/copy, manifest-last
  commit, symlink/nonregular/traversal checks, observed code/package provenance.
- Actual concurrent writers, process killed mid-write, injected callback/fsync/
  link failures and deliberate corruption tests pass. Directory-fsync failure
  after publication retains complete bytes and reports uncertain durability.
- 220 tests pass on each of Linux/Python 3.11.14, 3.12.14, 3.13.5; strict mypy
  passes on 21 files for all three; Ruff/format green. No skipped Linux tests.
- Built wheel/source archive and fresh installed-wheel matrix pass; a no-torch
  separate-process CLI init/inspect/verify flow passes. Corruption/wrong ID fail.
- Exact commands, code/wheel/manifest IDs and temporary evidence:
  `docs/development/validation_m2_2.md`; engineering experiment R003. Lock
  unchanged, no new dependencies. Remote CI has not run; no ML training ran.

## Limits and open prerequisites

- Artifacts are tested on Linux local storage. Native Windows, object/network
  stores, physical power loss and distributed transactions are not validated.
- Metadata/byte validity does not prove dataset rights, tokenizer semantics,
  valid neural state or training replay. Dirty flags/package inventory are
  observations, not code snapshots or execution/environment locks.
- Init declares one process and seed intent; it does not seed RNG, test CUDA,
  start training or create a checkpoint. Actual launch/replay contracts remain M5.
- Research baseline: dense pre-RMSNorm/RoPE/SwiGLU/GQA, tied embeddings, V=32768.
  Config counts remain analytical. User hardware RTX 3060 Laptop/Core i5/16 GB
  RAM has not been profiled; exact VRAM/OS/driver/server budget are unknown.
- No dataset is approved or ingested; source licenses in manifests are declarations.
  Project license/Russian-data priority are undecided, not blockers for core tests.

## Next three tasks

1. M3.1: verify and pin the byte-BPE dependency/API/license; implement tokenizer
   train/save/load using approved project-authored fixtures initially.
2. M3.1: validate stable IDs, control tokens, byte/Unicode/Cyrillic/LaTeX/numeral
   round trips and registered tokenizer/data manifests without quality claims.
3. M3.2: deterministic bounded ingestion, cross-source dedup/splits, checksummed
   token shards, packing/masks and rank/worker resume-order tests. External data
   remains gated by DATA_CARD.md; actual tensor training is not started here.

## Reproduce

```bash
cd /workspace/newai_cddevel
uv sync --frozen --extra cpu
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy
git diff --check
git status --short --branch
```

Expected: 220 passing tests, lint/format/types green, clean main with local
commits. Functional CLI example is in `docs/development/artifacts.md`; use a new
output directory under an existing parent. No train/serve command exists yet.

# Qvelt current state

Updated: 2026-10-31. Owner: Lead agent.

## Current phase and last completed task

Phase 2 is complete. M3.1 is in progress: M3.1a byte-BPE library API and 32
new tests are implemented. Tokenizer YAML/CLI recipes and data-manifest linkage
remain pending. No model, trainer, inference service or semantic checkpoint
loader exists. The user-attached ponytail skill is installed and must be read and applied
for all coding tasks, project-wide (full by default).

Checkout `/workspace/newai_cddevel`, branch `main`, repository
`jonbarkert135-cpu/QveltAI`. M2.2 code commits: `9027efe`, `f4fd2cc`, followed by
documentation. The user resumed publication to main after the pause. Push succeeded from
`db35482` through `213965e`, including Phase 1 research, M2.1/M2.2 implementation
and reports, and skill installation. Environment configuration remains unchanged.

## Latest supporting task

Installed 408 pinned agent skills with hashed source/assets and license
notices; six restrictive Anthropic skills excluded. Catalog/lock and universal
agent entry instructions are in docs/development/skills.md. No vendor installer,
job or helper script executed. Next remains M3.1. See ADR-0020. Integration checks: 220 tests passed in
4.35s; Ruff/format/mypy green; 3716 file hashes and 408 skill entries checked.

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

## Latest verification (M3.1a)

252 tests passed on Linux/Python 3.12.14 in 4.59s, no skips; Ruff/format and
strict mypy passed on 23 files. Tokenizers 0.22.2 and transitive dependencies
locked; installed library lacks py.typed, so a scoped external import override
is explicit. Small authored segmentation pilot and limitations: R004 and
`docs/development/tokenizer.md`. Earlier 220-test Python matrix remains historical;
new tokenizer behavior has not yet been tested across that matrix or on CUDA.
409 skill entries, with the new attachment retained byte-for-byte.

## Next three tasks

1. M3.1: add strict YAML/CLI training recipes and registered raw-data/tokenizer
   manifest linkage to the tested byte-BPE API.
2. M3.1: extend cross-process/version reproducibility and held-out segmentation
   measurements; stable IDs and round-trip baseline tests already pass.
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

Expected: 252 passing tests, lint/format/types green, clean main with local
commits. Functional CLI example is in `docs/development/artifacts.md`; use a new
output directory under an existing parent. No train/serve command exists yet.

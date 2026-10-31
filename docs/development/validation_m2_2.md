# M2.2 validation: typed provenance and atomic local artifacts

Date: 2026-10-31. Owner: Lead agent. Implementation commits: `9027efe` and
`f4fd2cc` (parent-directory concurrency/durability follow-up). No model training,
corpus ingestion, GPU run, remote CI or publication occurred.

## Actual final test matrix

Linux x86-64, pinned CPU environments from the unchanged lock:

| Python | Package mode | Result | Strict mypy |
| --- | --- | --- | --- |
| 3.11.14 | Built wheel in isolated environment | 220 passed in 5.89s; no skips | No issues in 21 source files |
| 3.12.14 | Project editable install | 220 passed in 5.64s; no skips | No issues in 21 source files |
| 3.13.5 | Built wheel in isolated environment | 220 passed in 5.84s; no skips | No issues in 21 source files |

The full suite includes the 121 M2.1/research cases and 99 new M2.2 cases.
Ruff and format checks passed for all 21 Python files. Dependency consistency,
lock freshness, wheel/source build and typed-package inclusion passed. CUDA
and native Windows were not executed; IO guarantees are for tested Linux local
storage, not object/network stores or physical power loss.

## What actually executed

- Typed round trips for run, raw/tokenized data, byte-BPE tokenizer and checkpoint
  metadata; strict types/ranges, source references, role requirements, immutable
  records, duplicate/unknown keys, incompatible versions and metadata size limits.
- Independent known SHA-256 `abc` vector and streaming copy/hash checks;
  regular-file-only reads, traversal/symlink/FIFO/directory rejection.
- Actual concurrent writers: one complete payload wins, existing targets survive.
  Actual child process killed mid-write: no final file; stale temporary ignored;
  a later new write succeeds. Injected callback, file/directory fsync and link
  failures test pre/postpublication states and manifest-last commit.
- Growing-file hashing is bounded to the initial size plus one; source changes
  after copy are rejected without publishing a bundle. New **and already-existing**
  nested parent entries are synced by each writer to cover concurrent creators.
- Corrupted/missing/truncated/same-size changed bytes and config/seed/precision/
  dependency-lock inconsistencies are rejected. Opaque checkpoint fixtures only
  exercise metadata/byte contracts, not optimizer or model-state replay.
- Actual local Git tests cover committed/dirty/untracked/unborn/missing code and
  lock cases. Actual package inventory and controlled error tests ensure no
  environment-value dump or fabricated revision/version fallback.

## Separate-process installed-wheel proof

A base-only Python 3.12 environment with **no torch installed** executed CLI
commands from `/tmp`, not through the editable checkout. Wheel Python module
bytes were compared with the implementation checkout. At bundle creation Git
was clean; the saved code identity was `f4fd2cc2fe0b130cd6c0df1f644fd1884281d24f`.

- Healthy temporary bundle: `/tmp/qvelt-m22-final-_j42a_un/run`.
- Manifest canonical identity: `e1b108b875a86045b630ebf834fbf02b67cc3576be6dd9a037173c5c393bcc3e`.
- Registered files: **2**, raw bytes checked: **107178** (`config.yaml`, `uv.lock`).
- `run init`, `manifest inspect` and `manifest verify --expected-sha256` exited 0.
- Corrupted clone: `/tmp/qvelt-m22-final-_j42a_un/corrupt`; one config byte flipped without changing size.
  Verification exited 1 with a hash mismatch.
- Wrong trusted metadata ID on the healthy bundle exited 1 with identity mismatch.

No trainer was launched or RNG policy applied by bundle initialization. It records
one declared process and observed config/code/package state only. These caches
and `/tmp/qvelt-m22-final-receipt.json` are temporary instance evidence; checked-in
source/tests and the commands below are the reusable validation artifacts.

## Commands and identities

From `/workspace/newai_cddevel`:

```bash
uv lock --check
uv pip check
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy
uv run --frozen --extra cpu python -m build --no-isolation
```

Fresh matrix environments used Python 3.11.14/3.13.5, the frozen CPU graph and
`uv pip install --no-deps --reinstall` of the built wheel, then their own Python
`-m pytest -q` and `-m mypy` with separate caches. Core/no-torch verification used
`/tmp/qvelt-config-only/bin/qvelt` with explicit config/repository/directory paths.
See [artifact examples](artifacts.md) for a new reproducible CLI probe. Fresh
runs have new UUID/time and therefore new manifest IDs; identical normalized
metadata retains the same identity.

- `uv.lock` SHA-256: `96183e14fe42283d3dafafb164340621d44bc59bb24191448e0ad3a6243c0aad`
  (unchanged; no new dependencies).
- Tested wheel SHA-256: `0342f05b42d484d9d06eb75f3c408c4dce69e072f8383dc7eaa947a086dfe77b`.
  Later documentation-only builds may change README metadata, not these source
  modules; the tested binary identity above refers to the actual matrix artifact.

## Diagnoses retained

An initial read-until-EOF hash loop could keep following an actively growing
file. Its targeted test run was interrupted; final code reads a bounded initial
snapshot plus one and rejects growth/observed changes. A regular-file check was
moved before binary wrapping to reject directories/FIFOs consistently. A later
review found that a writer must sync existing parent entries too, since another
creator may not have synced them yet; a dedicated test and `f4fd2cc` cover it.
Early helper typing/import/signature errors were corrected without type-ignore
suppression. The interrupted/failed intermediate checks were not counted as
passes; all final results above executed the corrected implementation.

Library/API evidence: installed Python 3.12 docs for JSON hooks, dataclasses,
SHA-256, file/temp/link/fsync/stat, metadata and subprocess; Linux link(2) and
fsync(2) pages read on 2026-10-31, with fetched HTML SHA-256 respectively
`9cb197a7abe32a459bb3dbe96f67d906ff199e0f9d5f19f41a3a05e46cea1ec9` and
`f0a986d21500222ecb69580fc7a910f346c6b1ff1005ca515501fb05bb5d91f5`.
References and supported limits are in [the design](artifacts.md).

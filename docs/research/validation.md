# Phase 1 validation record

Date: 2026-10-31. Scope: analytical calculator, source ledger and documents.
Baseline repository revision: `db35482531c77c198d93910bdb77ab8597da5369` plus
the locally committed Phase 1 changes. No ML model was instantiated or trained.

## Executed checks

Working directory: `/workspace/newai_cddevel`.

| Command | Actual outcome |
| --- | --- |
| `python3 docs/research/estimates.py --output /tmp/qvelt-estimates.json` | Exit 0; four configurations emitted |
| `cmp docs/research/estimates.json /tmp/qvelt-estimates.json` | Exit 0; byte-identical regeneration |
| `/workspace/qvelt-tools/venv/bin/python -m pytest -q docs/research/test_estimates.py` | 30 passed in 0.17s, exit 0; no skipped tests |
| `/workspace/qvelt-tools/venv/bin/ruff check docs/research/*.py` | All checks passed, exit 0 |
| `/workspace/qvelt-tools/venv/bin/ruff format --check docs/research/*.py` | Two files already formatted, exit 0 |
| `/workspace/qvelt-tools/venv/bin/mypy --strict docs/research/*.py` | Success: no issues found in two source files, exit 0 |
| `git diff --check` | Exit 0; no whitespace errors |

Versions: Python 3.12.14; pytest 8.3.5; Ruff 0.11.13; mypy 1.15.0.
Additional standard-library checks passed: 16 local Markdown links (before this
record/state added further links), newline/whitespace/fence checks, ten ordered
phases, M1.1/M1.2 status, 43 primary citations resolving to retrieved records,
13 ingestion-unapproved candidate cards, exact RFC size/dimension table parity.

An initial hand-case expected workspace term in a **test** was mis-added as 348
instead of 324 bytes. Recomputing 2*3*(8*4+3*6+2*2)=324 corrected that test;
the calculator's formula was unchanged. A strict-mypy failure in test argument
construction was fixed without suppression. Final results above include both
fixes and additional CLI/source/mix checks. These are not model failure results.

## Verified artifact hashes

- `estimates.py`: SHA-256 `31524f3b1b1f75f4132804c297a9acfb270db626103001dba8e6efaf884a43d1`
- `estimates.json`: SHA-256 `3855f505c6085fe2bb5c01f03a0608579699e8a886f902bf6c9cd9c0f637ed60`
- `sources.json`: SHA-256 `cf7d908daba9cb56d063b802e574dc67771c082153f8414f2101687750150572`

Primary-source text inspection supports only the attributed claims in the survey;
it is not experimental replication. No dataset shard, training checkpoint, GPU
profile, paid service, benchmark score, GitHub push, or environment-draft edit
was produced in this session. Local commits are the review artifacts.

# Phase 1 research package

Date: 2026-10-31. Owner: Lead agent. Scope: design and analytical validation,
not a model implementation or training result. Publication is paused by the
user's instruction following environment publication; commit locally only.

## Design and interfaces

Deliver a source-audited survey, an exact dense-decoder specification, an
assumption-based training recipe, a conditional data proposal, and a risk register.
The user supplied laptop RTX 3060/Core i5/16 GB RAM details during Phase 1.
Use the architecture/training RFCs for that planning scenario and distinguish
unallocated servers and unmeasured throughput/cost assumptions.

`estimates.py` is a standalone standard-library research calculator. Immutable
`ModelSpec` dataclasses describe four sizes. It enumerates parameter tensor
shapes, independently computes a closed-form count, estimates dense training
FLOPs and memory, and writes `estimates.json`. It contains no model/training code.
`test_estimates.py` checks counts against hand arithmetic, shape enumeration,
validation failures, resource formulas, and report consistency. These checks do
not prove that a future PyTorch implementation matches the specification.

## Reading order

1. [survey.md](survey.md): verified source claims and applicability limits.
2. [architecture_rfc.md](architecture_rfc.md): equations, sizes, alternatives.
3. [training_rfc.md](training_rfc.md): budgets, schedules, replay, evaluation.
4. [data_mix.md](data_mix.md): licenses, candidates, stage mixtures, restrictions.
5. [risks.md](risks.md): remaining risks and explicitly excluded work.
6. [sources.json](sources.json): exact retrieved URLs, revisions, content hashes,
   access failures, and a rejected bibliographic mismatch.

Paper PDFs and extracted text are research-session caches outside Git, under
`/tmp/qvelt-research`. No paper, dataset corpus, or third-party implementation
is redistributed. The manifest permits content verification if a source changes;
repository documents contain the reviewed claims and their locations.

## Reproduce calculations and tests

From the checkout, using the previously installed development tools:

```bash
python3 docs/research/estimates.py --output /tmp/qvelt-estimates.json
cmp docs/research/estimates.json /tmp/qvelt-estimates.json
/workspace/qvelt-tools/venv/bin/python -m pytest -q docs/research/test_estimates.py
/workspace/qvelt-tools/venv/bin/ruff check docs/research/*.py
/workspace/qvelt-tools/venv/bin/ruff format --check docs/research/*.py
/workspace/qvelt-tools/venv/bin/mypy --strict docs/research/*.py
```

Application dependency pins, YAML runtime configuration, model code, and actual
training remain later milestones. Bootstrap tooling versions are documented in
the root `SETUP.md`; these documents do not reapply environment settings.

## Subsequent implementation status

M2.1 now supplies pinned application/tooling profiles and validated YAML/runtime
infrastructure; see [development setup](../development/setup.md). The historical
Phase 1 validation above remains analytical; no model/training has been added.

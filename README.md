# Qvelt

Qvelt is an open, from-scratch language-model framework under development, with
an emphasis on mathematical reasoning. The roadmap covers tokenization, data,
pretraining, post-training, evaluation and inference.

**Current status: Phase 2 infrastructure complete; tokenizer/model work is next.**
The repository includes a typed Python package, strict YAML configuration,
reproducibility utilities, JSON logging, CPU diagnostics, typed provenance
manifests, atomic local artifact storage, pinned dependencies and
[source-audited research RFCs](docs/research/README.md). It has no tokenizer,
model, training loop, pretrained weights or benchmark results. No training ran.

The validated configuration targets are 19.34M, 125.85M, 493.55M and 1.510B
unique model parameters. They are analytical specifications, not trained models.
Current user hardware is an RTX 3060 Laptop/Core i5/16 GB RAM; server resources
and laptop VRAM/driver measurements are not supplied. CPU correctness and tiny
pilots come before larger user-operated training.

## Development quickstart

Use Python 3.11–3.13 and uv **0.12.23**. From this checkout:

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu qvelt config validate configs/tiny.yaml
uv run --frozen --extra cpu qvelt config resolve configs/tiny.yaml
uv run --frozen --extra cpu qvelt doctor --device cpu
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy
```

The config commands resolve defaults and verify types/shapes without allocating
a model. Doctor performs a small real tensor operation, not a training benchmark.
The full 220-test suite passed on Linux/Python 3.11.14, 3.12.14 and 3.13.5. The checked-in
GitHub CI workflow has not run remotely during the publication pause.

For a CUDA environment use `uv sync --frozen --extra cuda` **instead of** `cpu`.
This selects the matching PyTorch 2.9.1 CUDA 12.8 wheels. CUDA/Windows/laptop
execution is not validated here; see [development setup](docs/development/setup.md)
for driver checks, deterministic mode and limitations. Core config tools also
work with `uv sync --frozen --no-dev` and no PyTorch extra.

## Provenance and artifact checks

`qvelt run init` creates a new resolved-config/lock/provenance bundle;
`qvelt manifest inspect` reads metadata and `qvelt manifest verify` checks bytes
and identities. These commands work without torch and do not start training.
See [artifact contracts and examples](docs/development/artifacts.md).
Write-once/fsync publication is validated on Linux local storage only; neither
checkpoint metadata nor a file hash proves model-state replay or data rights.

## Project documents

| Document | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Working protocol and current publication pause |
| [PLAN.md](PLAN.md) | Phases, dependencies and acceptance gates |
| [STATE.md](STATE.md) | Current state, verification and next tasks |
| [DECISIONS.md](DECISIONS.md) | Choices, source evidence and revisit conditions |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Engineering checks and future ML experiments |
| [RISKS.md](RISKS.md) | Resource, stability, data and release risks |
| [DATA_CARD.md](DATA_CARD.md) | Dataset rights and provenance gate |
| [Scaffold design](docs/development/scaffold.md) | Implemented infrastructure interfaces |

Next: M3.1 tokenizer training/workflows on approved data. A project license
has not been selected; public visibility alone does not grant reuse rights.

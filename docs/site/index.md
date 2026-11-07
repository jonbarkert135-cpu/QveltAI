# Qvelt research framework

Qvelt is an original decoder-language-model framework for reproducible experiments,
with a focus on mathematical reasoning. Start with the [tiny tutorial](tutorial.md),
then read the [architecture](architecture.md) and [limitations](status.md).

The supported CPU workflows exercise real tokenization,packing,optimization,
checkpoint recovery,evaluation,export and authenticated serving. They establish
engineering behavior. They do not establish useful assistant quality,commercial
model parity or production readiness.

## Read in this order

1. [First tiny model](tutorial.md):actual RFC tiny dimensions,short original-data run.
2. [Data and tokenizer](subsystems/data.md):licenses,splits and contamination bounds.
3. [Training and recovery](subsystems/training.md):exact resume and hardware gates.
4. [Evaluation](subsystems/evaluation.md):scores,failures and provenance.
5. [Responsible release](responsible-release.md):separate framework/model decisions.

Canonical runbooks,cards and the generated results ledger appear under Reference
without a second editable score table. Source code,versioned configs and tests
remain in the repository. Build locally with:

```bash
make EXTRA_ARGS="--extra docs" setup
make docs
uv run --frozen --extra cpu --extra docs mkdocs serve --dev-addr 127.0.0.1:8002
```

The docs server binds loopback. A docs build does not publish GitHub Pages.

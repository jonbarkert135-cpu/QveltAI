# Development setup and supported scope

## Installation profiles

Use uv **0.12.23**; `.python-version` selects Python 3.12.14. The declared
Python range is >=3.11,<3.14. Linux CPU tests also ran on 3.11.14 and 3.13.5.
To keep a different interpreter across all uv commands, set `UV_PYTHON` explicitly
(CI does this per matrix job). Do not assume a single `--python` sync overrides
the version pin for later commands.

From the repository root:

```bash
uv sync --frozen --extra cpu
uv pip check
uv run --frozen --extra cpu qvelt --version
uv run --frozen --extra cpu qvelt doctor --device cpu
```

The base runtime pins NumPy 2.2.6, PyYAML 6.0.3 and tokenizers 0.22.2;
CPU adds PyTorch 2.9.1+cpu.
`uv.lock` pins transitive artifacts/hashes and environment markers, including
different compatible dependencies for Python 3.11 versus 3.12+. Do not rewrite
the lock during routine installation. An intentional upgrade is a tested change.

Base-only configuration tools require no torch:

```bash
uv sync --frozen --no-dev
uv run --frozen --no-dev qvelt config validate configs/tiny.yaml
```

Profiles change the project's environment; switching back to CPU requires the
CPU sync again. Package import and config commands work without torch; doctor
and tensor seeding fail with an actionable error rather than a silent fallback.

## Configuration contracts

```bash
uv run --frozen --extra cpu qvelt config validate configs/tiny.yaml
uv run --frozen --extra cpu qvelt config resolve configs/tiny.yaml
```

`validate` emits JSON with `valid`, schema version, canonical config SHA-256
and an **analytical** unique-parameter count. `resolve` emits complete YAML
including defaults on stdout. Redirect it to a chosen file if desired; the CLI
does not overwrite artifacts itself. These commands allocate no model.

Schema 1 has `name`, `schema_version`, `model`, and `runtime`. It intentionally
does not accept optimizer/dataset/training keys before those components exist.
Model dimensions are positive signed-32-bit integers; seed is uint32. A config
can still describe a model that does not fit your hardware. Shape validation
is not a capacity estimator or GPU compatibility check.

Use lowercase `true`/`false`, decimal integers, and finite positive floats
(including `1e-6`). Numeric fields never coerce quoted strings or booleans.
Only UTF-8, one document, <=128 KiB, <=8 collection levels and <=2048 parser
events. Duplicate/unknown keys, aliases/anchors, explicit tags and merges fail.
No environment expansion, include files or executable object constructors.

Canonical digest input is UTF-8 JSON of resolved dataclasses with sorted keys,
compact separators, Unicode preserved and NaN forbidden. Formatting/comments
do not change it; meaningful fields, including experiment name, do.

Presets: tiny/small/base/large match the research table; `laptop-debug.yaml`
uses tiny dimensions with context 128 and requests CUDA/BF16. It does not assert
that a GPU or enough VRAM is available. No trainer consumes these configs yet.

## RNG and deterministic policy

```python
from qvelt.config import RuntimeConfig
from qvelt.reproducibility import numpy_generator, seed_everything

report = seed_everything(RuntimeConfig(seed=20261031, device="cpu"))
worker_rng = numpy_generator(20261031, stream=0)
```

This seeds Python, legacy NumPy and torch and configures deterministic algorithms;
independent NumPy generators use explicit stream IDs. Strict mode raises on
known unsupported torch operations, disables cuDNN benchmarking and TF32, and
does not set `PYTHONHASHSEED` retroactively. Call it before allocating tensors.
It is not complete checkpoint replay, deterministic worker ordering, or a promise
of identical results across PyTorch releases, CPU/GPU or world sizes.

JSON logs use the `qvelt` logger on stderr. The framework replaces its reserved
`qvelt.console` handler without changing root handlers. Do not log credentials
or raw sensitive data; the formatter is not a redaction service.

## Optional CUDA path — not executed here

Linux or WSL2 is the immediate development target. Native Windows CUDA and
other device/platform combinations need separate validation. On the laptop,
first check actual VRAM, OS/driver and official CUDA wheel driver requirements:

```bash
nvidia-smi
uv sync --frozen --extra cuda
uv run --frozen --extra cuda qvelt doctor --device cuda
```

This installs CUDA 12.8 PyTorch wheels, not a GPU driver. CPU and CUDA extras
are mutually exclusive. Use uv profiles: wheel metadata alone does not encode
uv's index selection, so `pip install qvelt[cpu]` is not the documented CPU-only
installation route. No GPU dependencies were installed or tested in this session;
the CUDA dependency graph was resolved only.

For deterministic CUDA seeding set `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before**
process launch (PowerShell: `$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"`). The seed
utility requires a supported value and does not silently change it. Native
BF16 capability is queried with `including_emulation=False`; the doctor probe
itself tests FP32, not every dtype/kernel. Unsupported CUDA/BF16 fails explicitly.

## Validation and build

```bash
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy
uv run --frozen --extra cpu python -m build --no-isolation
```

The pinned development group includes Hatchling and build; `--no-isolation`
uses those locked build dependencies rather than resolving new ones. Build
outputs are ignored under `dist/`. Publishing a package or model release is a
separate milestone; source/report commits are published to main as authorized.

The CPU GitHub workflow uses commit-pinned checkout/setup-python actions and
Python 3.11/3.12/3.13 jobs. Local equivalents passed; remote status was not
inspected this session. No language-model training/export/serving command exists.
M2.2 provides [manifest and atomic artifact operations](artifacts.md).

The tokenizer workflow is verified with 315 tests on all three supported Linux
Python versions. See [its runbook](tokenizer_workflow.md) and
[validation matrix](validation_m3_1.md). M3.2 corpus preparation is next;
standalone tokenizer operations also work in the base no-torch profile.

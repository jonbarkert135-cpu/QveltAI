# M2.1 validation record

Date: 2026-10-31. Owner: Lead agent. Code commit: `9348442`.
Documentation is completed in the following local commit. No model training,
dataset ingestion, GPU execution, remote CI or publication occurred.

## Actual CPU test matrix

All environments were installed from the same frozen `uv.lock`. Linux x86-64;
PyTorch 2.9.1+cpu, NumPy 2.2.6, PyYAML 6.0.3, pytest 8.3.5, mypy 1.15.0.

| Python | Environment / package mode | Actual test result | Strict mypy |
| --- | --- | --- | --- |
| 3.11.14 | Fresh `/tmp/qvelt-cpu-py311`, built wheel installed | 121 passed in 2.49s; no skips | No issues in 12 source files |
| 3.12.14 | Project `.venv`, editable package | 121 passed in 2.28s; no skips | No issues in 12 source files |
| 3.13.5 | Fresh `/tmp/qvelt-cpu-py313`, built wheel installed | 121 passed in 2.13s; no skips | No issues in 12 source files |

The suite contains 30 prior research tests and 91 infrastructure tests/cases.
Tests cover strict config/constructor failures, YAML resource/syntax limits,
canonical digests, all size presets against research counts, real CPU RNG
references and tensor diagnostics, logger isolation, CLI subprocesses and
runtime failure handling. Mocked CUDA failure branches are not GPU tests.

Commands from the checkout:

```bash
uv sync --frozen --extra cpu
uv lock --check
uv pip check
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy
uv run --frozen --extra cpu python -m build --no-isolation
```

All final commands exited 0. Ruff passed; all 12 Python files were formatted.
The lock hash was unchanged by repeated frozen installation. CPU+CUDA extras
together were explicitly tested to fail with an incompatibility error, exit 2.

Fresh matrix installation used `UV_PROJECT_ENVIRONMENT=/tmp/qvelt-cpu-py311`
with `UV_PYTHON=3.11.14` (and corresponding 3.13 values), `uv sync --frozen
--extra cpu --no-editable`, then `uv pip install --no-deps --reinstall` of the
built wheel with each environment's `--python`. The fresh interpreters executed
`-m pytest -q` and `-m mypy` against this checkout; dependencies checked compatible.
Separate mypy cache paths were used for the final runs.

## Functional and packaging evidence

- Actual `qvelt doctor --device cpu`: checksum=8.0, requested_device=cpu,
  torch_version=2.9.1+cpu; no CUDA runtime in the tested build. The matrix
  includes no validation of the user's laptop, Windows or CUDA kernels.
- Actual `qvelt config validate configs/tiny.yaml`: valid=true, analytical
  parameters=19,338,560, resolved config digest
  `c6d028084262c19eb228aebb7bea5beff45263e4fa1b5197dffa101185f096f2`.
- All five presets are covered; resolve round-trips with defaults. These
  commands do not allocate model tensors or prove training memory fit.
- Wheel and source archive built successfully through the pinned Hatchling
  backend. The wheel includes `qvelt/py.typed`, the package and CLI.
- A clean base-only 3.12 environment at `/tmp/qvelt-config-only` has **no torch**;
  after installing the wheel, its CLI validated tiny from `/tmp` outside the
  checkout. Doctor returned exit 1 with the required-extra message. Imported
  module path was its own `site-packages`, not the editable checkout.
- GitHub CPU CI has commit-pinned actions and a 3.11/3.12/3.13 matrix, including
  `UV_PYTHON` per job so the project version pin cannot mask the tested version.
  It is configured only; no remote workflow run is claimed.

## Artifact identities

- `uv.lock`: SHA-256 `96183e14fe42283d3dafafb164340621d44bc59bb24191448e0ad3a6243c0aad`.
- `dist/qvelt-0.1.0.dev0-py3-none-any.whl`: SHA-256
  `bc32ff8b6cf5698cc516c7a724f3ff482026053baab7327f76b09ca5700a60f7`.

Build outputs are ignored, local artifacts; reproduce them using the commands
above. Temporary environments/caches are not a published snapshot or release.

## Diagnoses during implementation

Initial mypy failures exposed a SafeLoader return-type mismatch and a frozen-field
test that static checking correctly rejected. The constructor now matches the
library's Hashable-key return contract while enforcing string keys at runtime;
immutability tests exercise all dataclass fields through runtime reflection.
No type-ignore suppression was added. Other changes normalized imports and
protected the diagnostic CLI from native operation errors. Final counts above
reflect the corrected code. These failures were infrastructure issues, not
model instability or failed ML experiments.

M2.2 artifact schemas/atomic IO, dataset/tokenizer work, model reference tests
and actual training/replay remain future gates.

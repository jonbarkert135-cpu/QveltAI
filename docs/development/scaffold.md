# M2.1 design: package, configuration and reproducibility

Date: 2026-10-31. Owner: Lead agent. This milestone implements infrastructure,
not a model, dataset loader or trainer. Artifact manifests/atomic IO are M2.2.

## Interfaces and behavior before implementation

- `qvelt.config`: immutable `ModelConfig`, `RuntimeConfig`, `ExperimentConfig`;
  `load_config(path)`, `parse_config(text)`, `dump_config(config)` and a canonical
  JSON-derived `config_digest(config)`. Version 1 describes the selected model
  dimensions and runtime policy only; no unimplemented training flags.
- YAML is a bounded, single-document mapping. Reject duplicates, non-string
  keys, aliases/anchors, explicit tags, merges, unknown fields, excessive nesting,
  oversized input, booleans masquerading as integers and non-finite numbers.
  Use positive signed-32-bit dimensions, uint32 seeds, scientific/decimal floats
  and lowercase true/false. Bounds keep analytical outputs within a finite schema.
  No interpolation, includes or execution. Never echo raw config values on error.
- `qvelt.reproducibility`: explicitly seed Python, NumPy and PyTorch; offer
  independently indexed NumPy streams. Strict deterministic mode fails on known
  unsupported operations. CUDA initialization is an explicit requested action.
  Setting seeds is not complete checkpoint replay or cross-device equivalence.
- `qvelt.logging`: JSON records on stderr, no root logger changes, deterministic
  field schema and UTC timestamps. Repeated configuration must not duplicate
  framework handlers. User messages are not automatically secret-redacted.
- `qvelt.cli`: `config validate`, `config resolve`, and `doctor --device cpu|cuda`.
  No fake train/evaluate/serve command. Resolve emits YAML on stdout; validation
  returns a config digest and analytical parameter count. Doctor actually runs
  a small tensor operation, or fails clearly when a runtime is unavailable.
- CPU and CUDA PyTorch extras are mutually exclusive in uv. Base config tools
  work without torch. Source layout, pinned dependencies, a universal lock and
  Python 3.11–3.13 support range. Linux CPU/Python 3.11–3.13 is validated here; other
  platform/device combinations require CI/hardware evidence before readiness claims.

## Dependency choice and references

Use PyYAML 6.0.3 and NumPy 2.2.6 (supports Python 3.11), pinned development tools
and a pinned Hatchling backend. PyTorch 2.9.1 is a conservative matching CPU /
CUDA 12.8 wheel baseline; both indexes were checked. It is not the latest release
(PyPI reports 2.15.0); that release has no matching wheel in the inspected cu128
index. Upgrades are separate tested changes, not implied by this selection.
The laptop CUDA driver/OS are unknown; no GPU readiness claim is made.

Verified sources read for these choices:

- [uv 0.12.23 project configuration](https://github.com/astral-sh/uv/blob/0.12.23/docs/concepts/projects/config.md):
  conflicting extras and project resolution; `uv lock/sync --help` for frozen use.
- [uv 0.12.23 PyTorch integration](https://github.com/astral-sh/uv/blob/0.12.23/docs/guides/integration/pytorch.md):
  explicit CPU/CUDA indexes associated with mutually exclusive extras.
- [PyYAML 6.0.3 SafeLoader](https://github.com/yaml/pyyaml/blob/6.0.3/lib/yaml/loader.py),
  [constructors](https://github.com/yaml/pyyaml/blob/6.0.3/lib/yaml/constructor.py),
  [resolvers](https://github.com/yaml/pyyaml/blob/6.0.3/lib/yaml/resolver.py):
  explicit subclass/duplicate-key checks and local scalar resolver policy.
- [PyTorch 2.9.1 reproducibility notes](https://github.com/pytorch/pytorch/blob/v2.9.1/docs/source/notes/randomness.rst):
  seeds, deterministic algorithms and CUDA configuration limitations.
- [NVIDIA cuBLAS reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility):
  supported workspace settings. Installed PyTorch source confirms the
  `including_emulation=False` query for native BF16 capability.
- PyPI version metadata and official `download.pytorch.org/whl/cpu/torch/` and
  `download.pytorch.org/whl/cu128/torch/` indexes, inspected 2026-10-31.

## Acceptance

Unit tests for each module, actual CPU RNG/tensor checks, strict config failure
cases and preset parity with the research specification. Ruff/format/mypy green;
fresh frozen installation and built-wheel CLI operation. CPU CI is a checked-in
workflow, not a claim that remote CI has run. No environment draft changes or
publication during the user's pause.

# Cloud and project setup

The environment was published by the user before M2.1. No old script was
reapplied, no draft was edited and no new environment snapshot was published
in this development session.

## Current project workflow

Use the repository's `pyproject.toml` and `uv.lock`, not the earlier standalone
bootstrap-tool directory. From `/workspace/newai_cddevel`:

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu qvelt config validate configs/tiny.yaml
uv run --frozen --extra cpu qvelt doctor --device cpu
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy
```

See [development setup](docs/development/setup.md) for base-only/CUDA profiles,
Python pins, supported scope, build commands and deterministic policy. The
external `/workspace/qvelt-tools` directory is historical bootstrap tooling;
it is not the selected project ML environment. The project `.venv` has CPU
PyTorch; no GPU execution or final training has been validated.

## Repository and publication

Canonical repository: `jonbarkert135-cpu/QveltAI`; checkout keeps its original
`newai_cddevel` directory name. Preserve the uploaded brief and both histories.
The user's main-branch workflow resumes only after the later publication pause
is explicitly lifted. Keep current commits local; never force-push.

Previous-instance GitHub API denials are historical observations, not a current
readiness result. No GitHub API operation is needed for local development,
and API credentials/settings were not changed or retested in this session.

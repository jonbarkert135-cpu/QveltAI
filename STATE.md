# Qvelt current state

Updated: 2026-10-31. Owner: Lead agent.

## Current phase

M0 documentation bootstrap and development-tool setup are complete. Architecture
research M1.1 is next; no model code, training, or benchmark results exist.
The user authorizes direct development and publication on `main`, without PRs
as a prerequisite. Preserve history and user files; never force-push.

## Last completed task

Installed isolated hash-locked development tools; saved install/start cloud
instructions; preserved the uploaded brief and prepared direct main publication.

## Repository and tools

- Checkout: `/workspace/newai_cddevel`; branch `main`.
- Canonical origin: `https://github.com/jonbarkert135-cpu/QveltAI.git`.
- Both initial histories and `llm_build_prompts-2.md` are preserved.
- Python 3.12.14, uv 0.12.23, Git 2.52.0, gh 2.46.0.
- `/workspace/qvelt-tools/venv`: pytest 8.3.5, Ruff 0.11.13, mypy 1.15.0,
  PyYAML 6.0.3; nine total packages pinned with hashes in the external lock.
- PyTorch is not installed; no GPU capabilities have been profiled.

## Verification

- Documentation local links/structure and whitespace checked. The uploaded user
  brief was compared byte-for-byte with its original main version.
- Exact cloud install script ran twice; tool version/startup and dependency
  consistency checks passed. These are not application-test or ML-test passes.
- Feature branch push and remote SHA verification succeeded. Final main
  publication uses a normal non-forced push after fetching current main.
- GitHub API/PR attempts returned `Forbidden`; no PR was created. Git pushes
  work and PRs are optional under the latest instruction.
- Cloud draft saved: `install_script`, `start_skill`, and `api.github.com` custom
  domain, preserving presets. Draft application/publication remains unverified.

## Next three tasks

1. M1.1: review primary architecture sources and record verified findings.
2. M1.1: specify architecture, equations, shapes, masks, and parameter counts.
3. M1.2: establish resource/replay and mathematical evaluation contracts before
   implementing the M2.1 package scaffold.

## Blockers and limitations

No blocker prevents source review or direct Git publication. Servers/budget,
approved datasets, architecture, ML dependencies, and project license remain
unprovided or undecided. GitHub API remains unavailable; optional API operations
require saving the prepared environment settings and publishing the environment,
then retrying. Existing GH_TOKEN is present; its value was never inspected.

## Reproduce the tooling checks

```bash
cd /workspace/newai_cddevel
git status --short --branch
git log -3 --oneline
git diff --check
source /workspace/qvelt-tools/venv/bin/activate
python -m pytest --version
ruff --version
mypy --version
uv pip check --python /workspace/qvelt-tools/venv/bin/python
```

See `SETUP.md`. Expected: clean main and compatible tool dependencies. No train
command exists yet. No claim of hardware scalability or model quality is made.

# Cloud development environment

Verified on 2026-10-31: Python 3.12.14, uv 0.12.23, Git 2.52.0, gh 2.46.0.
The isolated `/workspace/qvelt-tools/venv` contains pytest 8.3.5, Ruff 0.11.13,
mypy 1.15.0, and PyYAML 6.0.3. All nine packages, including transitive
dependencies, are pinned with hashes in `/workspace/qvelt-tools/requirements.lock`.
These are cloud setup artifacts outside the checkout, not ML dependency choices.

The saved cloud draft embeds the complete lock in `install_script` and contains
`start_skill` activation instructions. The exact script ran twice successfully.
It enforces hashes and does not change global Python or tracked source files.

```bash
source /workspace/qvelt-tools/venv/bin/activate
cd /workspace/newai_cddevel
python -m pytest --version
ruff --version
mypy --version
uv pip check --python /workspace/qvelt-tools/venv/bin/python
```

Tool startup and dependency consistency are verified. No application tests or
model code exist yet; PyTorch is not installed. No training ran. There is no
service to start. Use the existing checkout, not a new worktree.

## GitHub

Canonical repository: `jonbarkert135-cpu/QveltAI`; GitHub reported the rename.
The checkout directory keeps its original name. The user authorizes routine
project pushes and PR creation. Preserve unrelated work; never force-push.
The uploaded brief from main was merged unchanged into `qvelt/environment-setup`
alongside the Qvelt documents. The feature branch was pushed successfully;
the latest user instruction now selects direct development and publication
on `main`. Publish tested changes with `git push origin HEAD:main`; never
force-push. Feature-branch PRs are optional and not a publication prerequisite.

Existing `GH_TOKEN` is present; its value has not been inspected. GitHub REST
and GraphQL/PR operations return `Forbidden` for `api.github.com`; no PR exists
from this session. This does not block direct Git pushes to main. Git push access does not imply API access. The cloud draft
adds that domain, preserving existing package-manager presets. Review/save the
prepared settings, then publish the environment. Saving the draft alone does
not apply the network change or publish a snapshot. Retry the API operation
before diagnosing additional credential needs; do not replace the existing token.

After configuration is applied, check access and existing PRs before creating one:

```bash
gh api repos/jonbarkert135-cpu/QveltAI --jq '{default_branch,permissions}'
gh pr list --repo jonbarkert135-cpu/QveltAI --head qvelt/environment-setup
```

Use `gh pr create --body-file` for the description. A branch compare link is
not evidence that a PR was created. ML scaffold and hardware validation remain
future roadmap milestones.

UV ?= uv
PROFILE ?= cpu
EXTRA_ARGS ?=
# Keep tool caches writable in managed environments with a read-only home.
export UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
export PRE_COMMIT_HOME ?= $(CURDIR)/.cache/pre-commit
RUN = $(UV) run --frozen --extra $(PROFILE) $(EXTRA_ARGS)

.PHONY: setup lint format typecheck test smoke hooks build
setup:
	$(UV) sync --frozen --extra $(PROFILE) $(EXTRA_ARGS)
lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
format:
	$(RUN) ruff check --fix .
	$(RUN) ruff format .
typecheck:
	$(RUN) mypy
test:
	$(RUN) pytest -q -m 'not gpu'
smoke:
	$(RUN) python scripts/smoke.py
hooks:
	$(RUN) pre-commit install
build:
	$(RUN) python -m build --no-isolation

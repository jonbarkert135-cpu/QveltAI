# Qvelt current state

Updated: 2026-10-31. Owner: Lead agent.

## Current phase and last completed task

Phase 1 design milestones M1.1/M1.2 are complete. `docs/research/` contains the
verified-source survey, architecture/training RFCs, conditional data proposal,
risks/exclusions and tested analytical estimates. This is design acceptance,
not acceptance of a model implementation, GPU support, training or benchmarks.

Current checkout: `/workspace/newai_cddevel`, branch `main`; canonical repository
`jonbarkert135-cpu/QveltAI`. Preserve the user's uploaded brief and all history.
**Publication is paused** by the user's environment-published instruction.
Research commits remain local; no push, cloud draft edit or republication occurred.

## Selected baseline

Dense pre-RMSNorm/RoPE/SwiGLU/GQA, tied embeddings, bias-free layers; V=32768.
Exact parameters: tiny 19,338,560; small 125,854,464; base 493,553,920;
large 1,510,082,560. AdamW/WSD and conditional stage mixtures are starting
hypotheses. Accepted choices/revisit criteria: ADR-0008 through ADR-0013.

User hardware: ASUS TUF Gaming F15, RTX 3060 Laptop, Core i5, 16 GB RAM.
Exact VRAM/power/OS are unverified; 6 GiB is only a planning assumption.
Tiny and short-context small pilots are the laptop path; full base/large training
requires separately assessed hardware. No server, budget or GPU profiling exists.

## Verification and limitations

- 43 scientific primary texts accepted in the survey, exact source hashes/versions
  recorded; one wrong-paper lookup excluded. Scientific claims are attributed,
  not independently replicated. Official source/API evidence is separate.
- 13 dataset candidates assessed; publisher license claims and unresolved upstream
  rights recorded. No dataset approved or corpus downloaded. Some legal-text
  endpoints returned 403, one gated dataset card returned 401.
- Calculator: formula, independent tensor enumeration, hand-counted control,
  exact four-size totals, memory/FLOP units and CLI/report consistency checked.
- 30 tests passed; Ruff check/format and strict mypy passed on the two research
  Python files. Exact JSON regeneration and documentation consistency passed.
- Commands, outcomes and hashes: `docs/research/validation.md`; engineering
  experiment R001 in `EXPERIMENTS.md`. No ML smoke test or training ran.
- PyTorch is not installed. Application dependency pins, YAML config loaders,
  numerical model modules and runtime/CI support remain unimplemented.

## Next three tasks

1. M2.1: select compatible pinned Python/PyTorch CPU/CUDA and tooling dependencies;
   establish package layout, config dataclasses/YAML validation and tests.
2. M2.2: implement versioned manifests/artifact identity and atomic/checksummed IO.
3. M3.1 preparation: resolve source rights/project license and train/test the
   tokenizer only on approved data, preserving the mathematical/Russian contract.

No external answer blocks M2.1 work. Before paid training or release, user input
matters most for server budget, Russian-language priority and project/data license
policy. These are not permission to consume unapproved data or run costly jobs.

## Reproduce the last verified state

```bash
cd /workspace/newai_cddevel
python3 docs/research/estimates.py --output /tmp/qvelt-estimates.json
cmp docs/research/estimates.json /tmp/qvelt-estimates.json
/workspace/qvelt-tools/venv/bin/python -m pytest -q docs/research/test_estimates.py
/workspace/qvelt-tools/venv/bin/ruff check docs/research/*.py
/workspace/qvelt-tools/venv/bin/ruff format --check docs/research/*.py
/workspace/qvelt-tools/venv/bin/mypy --strict docs/research/*.py
git diff --check
git status --short --branch
```

Expected: reproducible analytical report, 30 passing tests, lint/format/type
checks green, clean main with local Phase 1 commits. See research README for
source/evidence scope. No train/serve command exists yet.

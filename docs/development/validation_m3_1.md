# M3.1 validation: tokenizer recipes, data lineage and reproducibility

Date: 2026-10-31. Owner: Lead agent. Code commit: `2d1a05a8e43c74fe23a4f13f974de44f226c3e98`.
Scope: actual CPU byte-BPE training on authored software fixtures; no language
model training, external corpus ingestion, CUDA or reasoning-quality benchmark.

## Executed checks

| Linux Python | Package under test | pytest | Strict mypy |
| --- | --- | --- | --- |
| 3.11.14 | Installed wheel | 315 passed in 10.93s; no skips | 27 files, no issues |
| 3.12.14 | Project editable install | 315 passed in 9.24s; no skips | 27 files, no issues |
| 3.13.5 | Installed wheel | 315 passed in 11.11s; no skips | 27 files, no issues |

Ruff/format passed on 27 first-party Python files. The suite includes 252
previous tests and 63 new cases. Wheel and source archive built; wheel includes
py.typed and new modules; source archive includes fixture data/manifests/config.
Installed wheel module bytes match the checkout on all three Python versions.
Dependency/lock consistency passes; no dependency or lock change in this session.
The 3.12 base-only installed wheel has no torch and runs the tokenizer CLI.
Remote GitHub CI status was not inspected; its workflow includes the same smoke.

Coverage: strict YAML defaults/round trips, unknown/duplicate fields, aliases,
versions, bool/string/numeric errors and bounds; ordered multi-file JSONL,
duplicate fields, malformed Unicode/JSON, bounded lines/documents, empty inputs,
wrong document counts, observed file changes/growth, safe regular-file reads,
existing outputs, hash mismatches, recipe/metadata/digit-policy/provenance links,
explicit controls, semantic tokenizer mutation and independent CLI processes.
Source permission strings are metadata; tests do not establish external rights.

## Independent process proof

Six CLI training processes used the installed wheel from `/tmp`, with code Git
state clean and equal to the commit above. Three Python versions used hash seeds
1/42/20261031 and Rayon thread counts 1/2/4. For each version, train/verify/evaluate
ran with and without ASCII-digit isolation; TOKENIZERS_PARALLELISM was true for
the isolated variant and false for plain BPE. These are varied fixture probes,
not a factorial experiment separating every threading/environment effect.

Each policy produced identical tokenizer JSON bytes across the three processes:

| Policy | Actual vocabulary | Tokenizer JSON SHA-256 |
| --- | ---: | --- |
| Plain byte-BPE | 512 | `51d802a3e3e3881c9af7099f005700e38f22eb2b7f1d63ff5cdac280482eb3e2` |
| Isolated ASCII digits | 512 | `f7ee889d47afa402aabd464bf714ac5ea71a44da916aedda6ffde20ad18b90d6` |

Bundle IDs differ because timestamps/package/interpreter observations differ;
this does not change the learned-tokenizer JSON identity. Exact bundle IDs,
settings and results are in [the machine-readable record](tokenizer_fixture_results.json).
A same-size one-byte corruption in a copied tokenizer JSON made verify exit 1,
without a traceback. The unit suite also checks a wrong expected metadata ID.

These matching hashes establish only the tested fixture/version/platform result;
no universal bitwise promise across architectures, releases or larger corpora.

## Held-out segmentation probe

48 ordered authored training documents (16 distinct strings repeated three
times); 12 separate held-out text records, total 390 UTF-8 bytes/315 characters.
All whole-text round trips passed, with zero implicit control IDs. Both policies
were trained independently with vocabulary ceiling 512 and minimum frequency 2.
No held-out record was fed to these training calls. This is software-test split
separation, not a benchmark contamination audit.

| Fixture row | UTF-8 bytes | Plain BPE tokens | Isolated-digit tokens |
| --- | ---: | ---: | ---: |
| English | 42 | 35 | 35 |
| Russian | 74 | 46 | 45 |
| LaTeX limit | 34 | 29 | 29 |
| LaTeX symbols | 40 | 30 | 30 |
| ASCII digits | 40 | 37 | 40 |
| Numeral variants | 42 | 39 | 42 |
| Whitespace | 23 | 16 | 16 |
| Scripts and emoji | 52 | 51 | 51 |
| Control bytes | 5 | 5 | 5 |
| Unicode extremes | 11 | 7 | 7 |
| Literal controls | 27 | 21 | 21 |
| Empty | 0 | 0 | 0 |

Totals: plain 316 tokens /12 records =26.333333333333332 tokens/record;
isolated 321 /12 =26.75. Aggregate bytes/token are390/316 and390/321. Vocabulary
allocation can change segmentation even outside digits. The sample is tiny and
artificial; no language coverage, reasoning gain or production-compression claim.
The default remains the research RFC's isolated-digit baseline, pending real
approved-corpus measurements. No production 32768-token tokenizer is frozen.

## Reproduce

Follow [the workflow runbook](tokenizer_workflow.md), using fresh output paths.
The shipped recipe pins the train fixture's canonical data-manifest identity.
Full tests, lint, types and build:

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy
uv run --frozen --extra cpu python -m build --no-isolation
```

Matrix environments used the frozen offline CPU graph, then `uv pip install
--offline --no-deps --reinstall --python ENV/bin/python dist/qvelt-0.1.0.dev0-py3-none-any.whl`.
Each environment's own Python ran pytest/mypy with distinct mypy caches. No-torch
proof used the frozen no-dev base graph. These commands executed, not just CI text.
The temporary driver/receipt were `/tmp/qvelt_m31_probe.py` and
`/tmp/qvelt-m31-probe-receipt.json`; permanent fixtures/tests/runbook/results are
checked in and do not depend on retaining those temporary files.

- Lock SHA-256: `2800ecf775dcc761699a31795c2476abd42c9285145395db6edb546403388db6`.
- Tested wheel SHA-256: `600f4313fd36983743216a9693d52da95fee7296ba666b55c19f33d3f8b5627f`.
  Subsequent README-only builds can change wheel metadata; this is the actual
  tested binary, not a promise that all later rebuilds have that hash.

## Limits and diagnoses

No native Windows/network-store/power-loss test, large-corpus memory or GPU run.
BPE pair-statistic memory is unbounded by the document iterator. Source approval
requires DATA_CARD review; a matching hash is not permission or authenticity.
Provenance describes observations, not an execution lease or full dirty snapshot.

An early parametrized test used a 1 MiB byte value as its generated test ID.
A separate diagnostic reproduced the resulting inherited PYTEST_CURRENT_TEST
environment value exceeding exec limits: Git startup raised E2BIG before corpus
validation. Short explicit IDs fixed the harness; input limits/assertions were
preserved. The separate diagnostic passed once (not part of the 315-case suite).
Its temporary files are /tmp/test_qvelt_large_case_id.py and
/tmp/qvelt-large-id-diagnosis.log.
Initial one-line conditional lint issues were formatted. Failed intermediate
checks are not counted as passes. All final results above used corrected tests.

# Qvelt data register

Last updated: 2026-10-31. Owner: Lead agent.

**No external raw corpus has been downloaded or approved for ingestion.**
Small project-authored software fixtures are approved only as described below.
Phase 1 assessed 13 candidate dataset cards and proposed conditional stage
mixtures in [docs/research/data_mix.md](docs/research/data_mix.md). Exact immutable
card revisions, declared licenses, hashes, access failures, and approval=false
are in [docs/research/sources.json](docs/research/sources.json). Metadata/paper
retrieval is not corpus ingestion. A paper or wrapper license is not approval.

## Approval gate

Before downloading or consuming a source, record and verify:

1. Name, publisher, authoritative URL, immutable revision, artifact checksums,
   and access method. Public visibility alone does not establish permission.
2. Exact license text/version and authoritative location; distinguish upstream
   content rights, packaging license, and any separate usage terms.
3. Permission for intended research/commercial use, derivatives, redistribution,
   and model training; attribution and share-alike obligations where applicable.
   Record uncertainty and block use rather than guessing.
4. Source/collection provenance, personal-data risks, filtering, deletion policy,
   language/domain coverage, and mathematical solution-quality checks.
5. Split construction at document/problem-family level, deduplication policy,
   benchmark exclusion, and contamination audit limitations.
6. Local transformations, tokenizer identity, sampling weights, inclusion counts,
   artifact lineage, and the exact run manifests that consume the source.
7. For synthetic data: teacher model/version, license and terms, permission for
   output use, generation configuration, verification, and retained provenance.

Only verified permitted uses may be recommended. Unknown or incompatible rights
block ingestion. Do not scrape sites in violation of their terms. Do not put
raw corpora, personal information, or credentials in Git.

## Dataset inventory

| Source / revision | Role | License / evidence | Permitted intended use | Approval |
| --- | --- | --- | --- | --- |

The candidate inventory is D01–D13 in `docs/research/data_mix.md`, with publisher
license statements and all known restrictions per row. The approved external inventory
above still has zero entries. Promote a candidate only after the full gate,
including authoritative terms/upstream rights, actual shard access, transforms,
license obligations and exact corpus revisions, has passed.

## Locally authored test fixtures

M3.1 uses `tests/fixtures/tokenizer/`: 48 ordered training records (16 distinct
texts repeated three times) and12 held-out records, authored in this project.
No external corpus or teacher outputs were used. Permission text is included in
[PERMISSION.txt](tests/fixtures/tokenizer/PERMISSION.txt), registered with role
license in each immutable manifest. License ID: LicenseRef-Qvelt-DevelopmentFixture.
Permission covers Qvelt development/tests and their redistribution, not a broader
project license or a recommendation for production model training.

| Split | Canonical manifest SHA-256 | Scope |
| --- | --- | --- |
| train | `832c18b89e947c9953f0b79838b50223642c65c5fb66ed6672859552df42d57b` | local tokenizer correctness/smoke training |
| heldout | `0f84b8ef256117e11ca0dd6f94bef4d006fb78692a257c11bca10964f2258be4` | segmentation/round-trip checks only |

Each source revision is its raw JSONL SHA-256. Source declarations point to this
register; actual payload and permission hashes are recorded in the manifest.
Text includes English/Russian, LaTeX, numbers, whitespace and Unicode edge cases.
No personal information or scraped material is included. Held-out record content
is separate from training text, not a proof of benchmark decontamination or
problem-family independence. These are software fixtures with no semantic answer
labels and no reasoning-quality claim. Library unit-test literals remain authored
fixtures too. M3.2 must audit real corpora before ingestion and define filtering,
exact/near deduplication, splits and contamination controls.

## Evaluation separation

Approve benchmarks with the same license gate as training sources. Preserve
held-out identities, revisions, problem-family split rules, prompt templates,
and extraction/scoring policy. Document overlap searches and residual uncertainty;
do not claim decontamination merely because exact matches were not found.

## Synthetic numeric-symbol engineering fixture

The expanded foundation trainer uses the original `qvelt.data.synthetic` cyclic
sequence generator (Apache-2.0 code): sample identity = (step*world_size+rank)*B+i;
T+1 numeric symbols, IDs32+(sample_id+position) mod(V-32), shifted within that
sequence. No downloaded text, personal data, teacher outputs or evaluation labels.
This is approved only for code/learning/replay smoke tests, not reasoning quality.
Run artifacts save data-spec.json and tokenizer-spec.json descriptor hashes;
these intentionally describe numeric fixtures, not trained BPE or corpus manifests.
Real-data training still requires M3.2 and approval of actual source revisions.

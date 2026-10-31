# Qvelt data register

Last updated: 2026-10-31. Owner: Lead agent.

**No raw corpus has been downloaded and no dataset is approved for ingestion.**
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
license statements and all known restrictions per row. The approved inventory
above still has zero entries. Promote a candidate only after the full gate,
including authoritative terms/upstream rights, actual shard access, transforms,
license obligations and exact corpus revisions, has passed.

## Locally authored test fixtures

No fixtures exist yet. M3.2 and subsequent numerical/training tests should use
small project-authored deterministic fixtures when possible. Register authored
text/problems and any third-party fragments before treating them as a corpus.
Fixture learning is not evidence of general mathematical reasoning.

## Evaluation separation

Approve benchmarks with the same license gate as training sources. Preserve
held-out identities, revisions, problem-family split rules, prompt templates,
and extraction/scoring policy. Document overlap searches and residual uncertainty;
do not claim decontamination merely because exact matches were not found.

# Qvelt data register

Last updated: 2026-10-31. Owner: Lead agent.

**No datasets have been downloaded, recommended for ingestion, or approved.**
No tokenizer corpus, pretraining mixture, post-training source, or benchmark
revision has been selected. A paper in `DECISIONS.md` is not a dataset approval.

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

The inventory has zero entries. Populate real records rather than illustrative
dataset names whose licenses have not been checked.

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

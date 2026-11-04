# Qvelt data register

Last updated: 2026-10-31. Owner: Lead agent.

**No external training corpus is approved.** One GSM8K test snapshot is now
approved/acquired for evaluation and decontamination only (Phase3 entry below).
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
license statements and all known restrictions per row. The Phase1 inventory initially had zero approved external entries; the later
Phase3 GSM8K evaluation-only entry is the sole exception. Promote a candidate only after the full gate,
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

## Phase3 approved original math and offline fixtures

Approved before consumption, 2026-10-31: `qvelt.data.math` and `data.demo` generate
original EN/RU arithmetic, rational linear equations, gcd/lcm and combinations/
permutations. Apache-2.0 (this repository LICENSE); no teacher, scraped material,
external inputs or personal data. Rights permit development, training, evaluation
and redistribution with Apache notices. Seed/count/language/difficulty, generator
code revision and actual raw payload hashes are recorded in generated manifests.
Template-family holdouts share no families with training; these simple tasks do
not establish broad mathematical reasoning. Deliberate fake PII/toxic phrases and
contamination/duplicate canaries exist only to test removal. Authored holdout
questions are **not** GSM8K/MATH/AIME/MMLU replacements for production clearance.

## GSM8K evaluation snapshot approval

Approved **evaluation/decontamination indexing only**, not tokenizer fitting or
training-source selection: OpenAI `grade-school-math`, immutable revision
`3101c7d5072418e28b9008a6636bde82a006892c`, original
`grade_school_math/data/test.jsonl`. Before downloading data, read the publisher's
README (explicit dataset release and question/answer JSONL format) and repository
MIT LICENSE, copyright2021 OpenAI. No separate data restriction is stated in this
snapshot. Preserve the complete MIT text and immutable URL with any local derived
bundle. This is a documented publisher-license determination, not an audit of
all contributors' upstream rights. No teacher-generated socratic/model-solutions
files are approved. Original URL: `https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl`.

License URL: `https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE`;
SHA256 `86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b`.
README SHA256 `27e1676d572f2289210170ff6dbb8c2ef7d9f6950c0f08c978ea8ee6b39dca29`.
Exact acquired payload count/size/hash are recorded after acquisition in the
per-source spec and experiment receipt. The preparer copies only questions into
the exclusion index; gold answers never enter train text. Other mandatory public
benchmark rights/snapshots remain blocked; one permitted snapshot does not make
a production training corpus ready. No raw benchmark content is published in Git.

GSM8K acquisition executed: raw payload **749738 bytes /1319 rows**, SHA256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
Working checksum-pinned spec: `configs/data/gsm8k_test.yaml`; emitted local raw
question manifest identity
`5f5fb9b48439db1928555aa5e19eeff6e4b4d0efe96f9df78b578ac5b892ba84`.
The license text is retained verbatim in PERMISSION.txt; usage.json permits only
evaluation/decontamination. No answers or questions were used in the math-digit
experiment or tokenizer corpus. No public raw content is committed to Git.

Training sources D01–D13 remain unapproved. This evaluation approval supersedes
the earlier blanket zero-external-input status only for this narrow snapshot/use.
MinHash defaults preserve differing numeric/operator sequences to avoid dropping
distinct math problems just because their surrounding prose templates match.
NFC/casefold n-gram exclusion is lexical, not semantic; translated variants and
unknown copies are residual risks. PII/toxicity heuristics are incomplete; source
rights and audits are still required even after a successful offline demo.


Actual question-index audit: all1319 approved GSM8K questions were indexed
(55284 unique13-grams); a copied-question canary matched and an unrelated authored
control did not. Receipt: `docs/development/evidence/phase3-gsm8k-index.json`.
This adapter does not index gold solution text. It establishes neither complete
planned-benchmark coverage nor semantic/translated exclusion. Additional licensed
solution/derivative exclusion snapshots need explicit registration before promotion.
Original demo receipts, actual transformations/removals and tokenizer/loader
measurements are in EXPERIMENTS R007–R010 and the Phase3 evidence JSON files.

## Phase5 actual BPE training usage

The approved original Phase3 EN/RU math/code fixture now also drives a tiny
packed-data optimization/replay smoke (`scripts/pretraining_smoke.py`). It uses
actual BPE/shard identities and family-heldout LM loss, with explicit allow_demo
and scope=demo in outputs. No external training source/teacher or public benchmark
is consumed. Full/resumed runs share one immutable dataset; generated corpus,
tokenizer/provenance and per-run identities are retained next to checkpoints.
This is development training under the existing Apache-2.0 original-data approval,
not production data clearance, a model release or mathematical reasoning accuracy.

## D-SFT-01 — Original exact-answer conversational supervision

Approved 2026-11-04 for local development SFT and heldout diagnostics only.
Revision: `qvelt-sft-chat-v1/exact-math-20261104`; generated by
`qvelt.posttrain.smoke.math_supervision` using original `generate_problems`,
fixed seed20261104,32 examples per language EN/RU by default. License:Apache-2.0;
retain repository LICENSE/NOTICE. No external dataset, teacher model or model API.
Gold integers/rationals are independently checked by stdlib Fraction/math and
SymPy1.14.0 inside the generator before ingestion. Answers provide exact-answer
supervision, not verified chain-of-thought explanations. The generic chat reader
can retain licensed solution text but does not verify arbitrary imported proofs.

Each generated raw document stores versioned messages plus problem ID,family,
language,difficulty and exact answer. The raw manifest includes license and explicit
training/evaluation/decontamination uses; SFT preparation checks training permission.
Split whole families with the existing fixed-seed family hash, across languages;
retain the family lists and truncation/target counts in the report. Tokenizer is
reused from the base checkpoint, never refit on validation. The base demo contains
other instances of these broad mathematical families: SFT family isolation is not
an assertion that the base checkpoint never saw that task type. Actual raw and
packed manifest hashes are recorded with each run. No public benchmark or private
teacher data is consumed; production source clearance remains blocked.

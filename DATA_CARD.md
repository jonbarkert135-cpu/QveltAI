# Qvelt data register

Last updated: 2026-11-06. Owner: Lead agent.

## D-EVAL-PHASE7-01 — Expanded suite sources and original controls

Approved2026-11-06 before payload ingestion. Evaluation/decontamination only;
no tokenizer fitting,training,teacher use or production cleanliness approval.

- Original EN/RU math,choice,LM and bounded integer-function controls are authored
  Apache-2.0 fixtures. New seed20261106 namespace;no final800 examples consumed.
  Function checks interpret an allowlisted pure integer-expression AST,not arbitrary
  Python or official HumanEval. Register per-record language and test vectors.
- GSM8K original test at3101c7d5072418e28b9008a6636bde82a006892c uses the previously
  reviewed publisher MIT release;expand the existing evaluation approval to numeric
  gold labels needed for scoring. Exact payload749738 bytes/1319 rows,
  SHA256 and original license remain pinned by the existing acquisition record.
  Development imports<=64 (acceptance16);all test examples remain excluded from
  optimization. No Socratic/model-solutions file. Strict#### extraction follows
  installed lm-eval0.4.13's first regex capture/literal comparison;subset/zero-shot/
  context/completion differences are recorded and are not official full scores.
- MMLU abstract_algebra test only:publisher CAIS HF revision
  c30699e8356da336a370243923dbaf21066bb9fe,
  abstract_algebra/test-00000-of-00001.parquet,9964 bytes,SHA256
  92338cdcdb43d2871691ef9a0e125e633ca57335ed635a635fe6dead94f224c2.
  Immutable card explicitly attributes MIT to the original Hendrycks repository.
  Read that repository's LICENSE/README/evaluate.py at
  4450500f923c49f1fb1dd3d99108a0bd9717b660:MIT copyright2020 Dan Hendrycks,
  license SHA2565d841b9eee8bf0d721359b79fbae956a8b426d3139bfefee32149c7d4f2a1a9c.
  Preserve full MIT text/card provenance. Original evaluator uses5 exemplars,
  subject heading,A-D options and candidate-letter likelihood/accuracy. Our bounded
  acceptance uses an explicitly labeled zero-shot16-case subject prefix;no full
  MMLU or directly comparable paper score. Publisher-license determination is not
  exhaustive upstream question-rights verification. No other MMLU subject approved.

Blocked pending rights/access/execution evidence: MATH/MATH-500 (MATH repository
MIT is for loaders/evaluation;reviewed MATH-500 card lacks an explicit data license),
AMC/AIME (competition reuse rights not established),HellaSwag (publisher access
HTTP451;no alternative data mirror fetched),full HumanEval (MIT release reviewed,
arbitrary Python sandbox unimplemented). ARC legal-text review was initially
blocked by403,then completed from the official Creative Commons repository;the
approval/pins/conditions are recorded in the closure below. No raw external corpus
is committed to Git.

## D-GSM8K-TRAIN-BOUNDED-01 — Original human-written training problems

Approved 2026-11-06 before fetching the train payload, within the user's expanded
Phase6 instruction to use permissively licensed math datasets. Immutable publisher
revision `3101c7d5072418e28b9008a6636bde82a006892c`, file
`grade_school_math/data/train.jsonl`. Re-read immutable README and full MIT LICENSE
on this date: README explicitly releases the original human-written train/test
data; repository MIT copyright2021 OpenAI states no separate data restriction.
Preserve the full MIT text and immutable original URL. This is a documented
publisher-license determination, not an exhaustive upstream-rights/legal audit.

Scope: import at most64 (acceptance16) original train questions and numeric golds
as local reasoning problems; no Socratic/model-solutions, tokenizer fitting,
production decontamination approval or model optimization in this acceptance.
Payload checksum and actual imported counts must be measured before conversion
and retained in the source receipt. Golds are dataset labels, not independently
verified human reasoning steps. No difficulty annotations are invented: unrated.
All problems retain the source-family label gsm8k-original-train. Do not split
one source into fake independent template families. The separately approved
GSM8K test snapshot remains evaluation/decontamination-only. Apply the full
benchmark registry and overlap audit before a future actual training experiment.

Measured acquisition before conversion: train.jsonl4166206 bytes/7473 rows,
SHA256 `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`;
MIT LICENSE1062 bytes,SHA256
`86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b`.
Only a bounded original-train subset is converted; raw payload stays outside Git.

## D-POSTTRAIN-CONTROL-01 — Expanded Phase6 controls

Registered 2026-11-06 before new sampling/training. Original Apache-2.0 EN/RU
arithmetic a-a+b, b in {0,1}, a=2..33, 32 problems/language, eight operand
strata; IDs hash full prompts. This deliberately easy two-answer mechanism
fixture is not independent mathematical generalization evidence. Split whole
strata using the existing SFT/preference split policies; report both separately.
The frozen project final800 and GSM8K evaluation snapshot are never consumed.
Predeclared local control: base3, SFT20 at0.003, DPO4 at0.001/beta0.1
(base/SFT/DPO seed20261031), GRPO60 at0.001/KL0.01,4 prompts x8 completions,
max4 actions,RL seed20261106. The local specification preceded execution but
was not independently timestamped or committed before diagnostic outcomes.
Fixed 64-sample validation before/after uses seed20261107 and identical budgets.
Retain all outcomes even if reward does not rise; no forced gold rollouts.

Rejection sampling may consume the already approved D-SFT/D-RL original
generator, own checkpoints and tokenizer. Log every candidate, seeds, verifier,
difficulty, lexical dedup and selected counts. Correct final answers do not
verify steps. Unit-only synthetic external-teacher/terms fixtures are original
test data, not authorization for any real external teacher. No externally
generated synthetic source is newly approved. The bounded original GSM8K train
adapter above is a separate human-authored-source approval. The ingress
supports separately reviewed permissive raw bundles, explicit training permission
and a default-off external-synthetic switch; exact revisions/terms require their
own register entry before use. QLoRA/GPU/Lean datasets are not consumed here.

**No production external training corpus is approved.** A bounded original GSM8K
train-source adapter is approved above for Phase6 development; no optimization
on it is claimed. One GSM8K test snapshot is now
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

## D-TRAIN-200 — Original Phase5 process-recovery fixture

Approved2026-11-04 for development training/tokenizer fitting and heldout diagnostics
only,Apache-2.0;retain LICENSE/NOTICE. Revision:phase5-domains-nonce-v1,seed20261104.
Three authored domain labels:web-style prose,Python code and exact arithmetic.
Each has32 original documents with deterministic lowercase nonce text to supply
sufficient distinct byte-BPE merges for a genuine32768-token vocabulary. These
are authored engineering domains,not sampled web/code corpora or production math.
No external teacher,benchmark or private data. Arithmetic answers come from the
existing independent stdlib/SymPy-verified generator. The nonce is not a quality
signal and no language-model quality conclusion follows from this fixture.

Retain per-document IDs/families,source manifests/permissions,family-isolated
splits,train-only tokenizer and actual preparation checksums. Use an original
exclusion canary solely to exercise the demo decontamination path. All production
source/benchmark clearance remains blocked. The full RFC tiny model retains
V32768,L8,d320,Hq5,Hkv1,f896,context512;CPU acceptance uses shortened T16/B2 and
200 optimizer updates,not the RFC's full pretraining budget. Scaling/compile
controls may use smaller explicitly named model dimensions on the same fixture.

## D-PREF-01 — Original verifier-labeled preference fixtures

Registered2026-11-05 before ingestion for M6.2. Author:Qvelt project;Apache-2.0
with explicit training/evaluation permission. Generate32 seeded EN exact-answer
math problems using the existing independently checked stdlib/SymPy generator,
seed20261105;each preferred answer is verified gold and rejected answer is gold+1
as an exact Fraction,independently rejected by the bounded answer verifier. Label
meaning is verifier correctness,not human judgment or teacher-model preference.
Reuse the pinned SFT tokenizer;do not retrain on heldout answers. Split entire
problem families deterministically before optimization;test families remain unused.
Record source/generator revision,manifest hashes,tokenizer/masks and SFT reference
checkpoint identity. No external model/API/corpus or public benchmark is ingested.
These are demo-only pairs with simplistic negatives,not production or reasoning
quality evidence;length effects and cross-stage family overlap remain limitations.

## D-RL-01 — Original outcome-RL prompt fixtures

Registered2026-11-05 before RL sampling. Reuse D-SFT-01's original Apache-2.0
seed20261104 EN exact-answer generator and its independently checked gold values.
Only fixed prompts from the existing SFT-training family split enter RL rollouts;
validation/test families never select reward-positive training samples. Labels are
bounded exact-fraction correctness plus complete end-turn,not human preference or
proof checking. Sampling is on-policy with no gold forcing,rejection filtering or
external teacher/tool execution. Freeze prompt selection,tokenizer,SFT reference,
sampling distribution,seed and budgets in each run. This is demo-only engineering;
no production/public benchmark or expensive RL run is approved by this record.

## D-EVAL-01 — Frozen project-authored evaluation set v1

Registered2026-11-05 before generation/M7.1 execution. Original Qvelt mathematical
wording/code,Apache-2.0,explicit evaluation/decontamination permission only;training
is forbidden. Fixed seed20261105;1000 exact-answer cases,200 validation/800 final.
Five new template families:division remainder,arithmetic progression sum,two-equation
sum/difference,product of rational numbers,quadratic evaluation. Validation uses
small latent parameters2..25;final uses26..80. Namespace/family IDs distinguish
both templates and bands. Stdlib integer/Fraction answers are independently checked
with SymPy exact arithmetic before writing. No external source/teacher/benchmark.

Freeze generator version,all IDs/families,raw payload/manifest/license checksums
before new model learning. Raw corpus stays outside Git. Final800 are not scored
in M7.1 acceptance or used for tuning;their future use needs a declared protocol.
The set is created after historical checkpoints,so preregistration is prospective.
Template/parameter-band isolation does not prove semantic novelty or contamination
absence in unknown training histories. This is a project-specific set,not GSM8K,
MATH,AIME or MMLU;their exact revision/license gates remain blocked.

## D-EVAL-CONTROL-01 — Original evaluation/overlap calibration fixtures

Registered2026-11-05 before ingestion. Original Apache-2.0 numeric/choice/LM examples
and deliberate exact/near-overlap canaries;no external corpus or teacher. These are
engineering controls,not independent general-capability evidence. Evaluation source
permission is evaluation/decontamination only. Separate authored calibration source
may explicitly permit tokenizer fitting;never fit on D-EVAL-01 or its answers.
Known collisions must remain visible in reports rather than silently filtered.

## D-EVAL-ROBUST-01 — Original whitespace robustness controls

Registered2026-11-05 before generation/M7.2 learning. Original Apache-2.0
transformations of D-EVAL-01 validation prompts:leading spaces or trailing newline
on four cases per each of five families(40 variants,total20 parent problems).
Gold targets/families unchanged;parent identifiers preserved in variant IDs.
Evaluation/decontamination only,no training/tokenizer fitting. These correlated
variants test whitespace sensitivity,not independent reasoning/general capability.
The combined study suite contains200 original validation cases,these40 variants
and7 D-EVAL-CONTROL-01 sanity examples. No final case/public corpus/teacher added.

## D-INFER-CONTROL-01 — Original generation/export controls

Registered2026-11-06 before standalone M8.1 acceptance. Original short ASCII,
Russian,emoji and accented-text prompts,token-level stopping/cancellation oracles
and earlier approved original BPE fixture data;Apache-2.0,no external source or
teacher. Test-only examples are software fixtures,not a benchmark. The standalone
model remains untrained;tokenizer fitting uses only the preexisting approved demo
corpus. No final800 questions are generated/scored by inference acceptance.

### D-EVAL-PHASE7-01 ARC review closure (before corpus download)

The official Creative Commons GitHub legalcode repository was located after the
web legalcode403. Full unmodified CC-BY-SA4.0 text read at immutable
4d2de083a6dcc5095ed1a30885bc48f868b9bb55/by-sa_4.0.txt,20005 bytes,SHA256
 a0bd498de6b7991f563789715e3cf9f9a77860b6bb5b2c2b072fb3bb36361008.
Sections2/3/4 permit reproduction/technical conversion subject to attribution,
indication of modifications and ShareAlike for distributed adaptations/database
rights. Publisher AllenAI HF card210d026faf9955653af8916fad021475a3f00453 explicitly
states cc-by-sa-4.0;its unfinished body license field is disclosed. Approval is a
publisher-license determination,not exhaustive upstream exam-question-rights proof.

Approve only bounded evaluation/decontamination of ARC-Challenge test,first<=64
rows (acceptance16),never training or tokenizer fitting. Exact publisher Parquet:
ARC-Challenge/test-00000-of-00001.parquet,203808 bytes,SHA256
62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9.
Retain AllenAI/source/revision attribution,CC-BY-SA4.0 full text/URL and mark JSON
conversion/prompt formatting as changes. Converted data retains CC-BY-SA4.0,
not framework Apache. No raw/converted ARC text is published in Git;receipts omit
predictions/questions. Original score-only aggregate artifacts are separate.
ARC is no longer blocked solely by the legal-text fetch;other listed gates remain.

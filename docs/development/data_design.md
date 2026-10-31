# Phase3 data pipeline design

Date:2026-10-31. Owner: Lead agent. Status: implementation in progress.

Apply ponytail full, the installed tokenizer/SymPy skills and data-curation
principles. The NeMo skill is a checklist of stages, not a verified claim about
our CPU throughput or a requirement to install RAPIDS. No external corpus/teacher
or public evaluation set is approved by downloading a library.

## Interfaces and scope

- Byte-BPE remains tokenizers0.22.2 with vocab ceiling32768 by default. Version2
  controls assign formerly reserved IDs to think/end-think/tool-call boundaries;
  old version1 artifacts retain their original control mapping when loaded.
  Plain text cannot silently create control IDs. Measure both digit policies and
  EN/RU whitespace-word fertility on actual original verifiable math samples.
- Original math generators produce arithmetic/fractions, linear algebra problems,
  number theory and combinatorics with explicit difficulty bounds and exact
  SymPy/stdlib cross-checks. Reward parsing accepts only bounded integer/rational
  strings, never arbitrary CAS expressions or generated code.
- Raw sources use the already-verified text-only JSONL manifest contract. Optional
  parallel document_metadata JSONL carries IDs/families/language/difficulty/answer;
  all payload/permission hashes and row counts are checked. Downloader specs are
  per-source, HTTPS/checksum/license-evidence pinned and fail closed on approval.
  Unapproved research candidates remain catalog records, not working download URLs.
- Cleaning annotates seeded offline language ID and explicit heuristic quality,
  replaces common email/contextual phone/credential patterns, and quarantines
  configured toxic phrases. Preserve math/code/Unicode otherwise; bare long
  numerals are not automatically phone numbers. This is not comprehensive PII
  detection, a trained quality classifier or a universal toxicity judgment.
- Exact clean-byte SHA256 and MinHash/LSH candidate buckets live in SQLite on disk.
  Verify candidate Jaccard overlap before deleting; index only accepted records.
  Bound document size/candidate work; report all rejection/redaction counts.
  MinHash has probabilistic false negatives; thresholds need labeled calibration.
- Mandatory benchmark registry: production preparation requires approved/pinned
  GSM8K/MATH/AIME/MMLU and frozen project-final snapshots (and every additional planned set). Missing rights,
  missing payloads or wrong hashes block production; no bypass flag. Demo scope
  uses explicitly authored holdouts and is marked non-production. N-gram matching
  preserves mathematical symbols/numbers and reports every matching registry set;
  short questions use complete normalized phrase matches. This does not certify
  semantic/translation decontamination.
- Dedup precedes deterministic family/document hash splits; one family cannot
  cross partitions. Train BPE only on the train split. No tokenizer learns from
  heldout benchmark text. All source/config/tokenizer identities are retained.
- Binary token shards use little-endian uint16 for V<=65536, uint32 otherwise.
  Compact memmapped block/segment indices reconstruct inputs, shifted targets,
  reset positions, segment IDs and masks without storing six expanded arrays.
  Each original document has one BOS/EOS; windows overlap one token and invent no
  EOS. Every original adjacent-token target occurs exactly once before padding.
- Map-style PyTorch loading opens read-only memmaps lazily with bounded caches;
  ordered prefetch workers consume a pure batch plan. Consumer-committed cursor,
  source counts, seed, rank/world and stage schedule define resume; prefetched
  sampler progress is never the saved position. Changing worker count is tested
  independently of changing world size (which is rejected).
- Weighted pretrain/anneal stages choose sources deterministically; per-source
  epoch permutation uses a seed-keyed coprime affine bijection for bounded memory,
  explicitly not uniform over all possible permutations. Oversampling after an
  exhausted source is intentional and counted, not accidental duplicate delivery.

## Acceptance

Offline CLI demo: generated raw -> cleaning -> exact/near dedup -> decontam ->
family split -> tokenizer -> token/index shards -> batches -> committed-state
save -> simulated crash with prefetched work -> exact next-batch replay. Test
masks/labels/causality, uint16/32, malformed input, hashes, stages, rank isolation,
worker count and all registered benchmark checks. Test production gate failure
when any planned benchmark or license approval is missing. Never label authored
stand-ins as actual public benchmark scans.

Benchmark startup and steady-state valid-target throughput for workers0/2 on
this actual CPU host, report fixture disk/temporary storage. Laptop i5 estimates
remain assumptions until the user runs the same command. No GPU saturation,
public benchmark clearance, broad quality/PII recall or real-corpus training claim.

## Operational limits and classifier policy

SQLite bounds its page cache to32MiB; individual records are at most1MiB UTF-8,
cleaned text defaults to100000 characters/word-punctuation tokens, candidate lists
to1000. Disk and index construction time still grow with corpus size. The HF BPE
trainer's pair statistics are **not** bounded by streaming input: production
training must select an approved, representative train-only tokenizer corpus on
available RAM before attempting large-vocabulary fitting. This implementation
streams all approved train text; the laptop demo is deliberately small.

LID is langdetect1.0.9 with sorted shipped profiles and a local seeded factory;
redaction placeholders are excluded only from detector input. The declaration
in upstream package metadata says MIT, but installed LICENSE/NOTICE actually
contain Apache-2.0 (Danilak/Cybozu); retain and follow the installed legal texts.
Short equations/code may be mislabeled. The declared original source language and
model guess are separate annotations; no automatic language rejection is default.

The scalar quality score is an explicit alphanumeric/repeated-line heuristic,
not a trained classifier. A future learned quality classifier requires approved
and family-heldout human labels for educational/math/code quality, EN/RU slice
precision/recall and a false-removal audit, a separately pinned model/license,
calibration curves and logged version/threshold. It must not use evaluation test
questions or a hidden teacher to produce quality labels. No trained classifier
or toxicity-recall guarantee is shipped. Phrase quarantine is conservative and
configurable; preserve source rights, redact detected contact/credential patterns,
keep only hashed IDs in curation rejection logs and review ambiguous material in
a separately access-controlled source store (never publish raw rejected text).

Default packed attention isolates documents and resets fragment positions.
`LoaderConfig(attention_mode="causal")` explicitly permits earlier-document
context, uses contiguous block positions and preserves original IDs as segment_ids;
labels still exclude document-crossing EOS->BOS pairs. Loader-local Torch generator
keeps worker seeding from advancing model RNG. Loader cursors are immutable
write-once snapshots, so use step-specific paths. Source reuse after exhaustion is
intentional weighted oversampling; it is not additional unique training data.

Downloader currently supports checksum-pinned UTF-8 text JSONL and a GSM8K question
adapter. Approved GSM8K test is evaluation/indexing only, enforced by a raw-bundle
usage policy; it cannot be promoted to training by setting a recipe boolean.
Dataset-specific Parquet/WARC/code-license extraction is future work, blocked by
source rights and format-specific tests. No generic tool pretends to support these
formats. Production requires an explicit usage policy as well as recipe approval.

Math-aware near dedup defaults to preserving the ordered literal digits and
mathematical operators: a candidate with different atoms is kept even when its
surrounding template has high Jaccard overlap. This trades web dedup recall for
avoiding removal of distinct equations/answers. It is configurable and not symbolic
equivalence detection; equivalent reformulations can remain. Exact byte duplicates
are still removed. Calibrate the setting per approved source using labeled pairs.

# Expanded Phase7 contract

Date:2026-11-06. Extend the existing versioned evaluator without changing the
frozen project final corpus or claiming reasoning gains from engineering controls.

- Keep qvelt-eval-v1 behavior available. Add explicit v2 native generation with
  reasoning controls, majority-vote@k, length/failure diagnostics, language groups
  and bounded original code-function checks. Incorrect free text is unclassified
  unless a checked arithmetic witness, declared method mismatch or human label
  supports a more specific failure; never invent cognitive diagnoses.
- Pin optional lm-eval0.4.13 and implement its actual LM interface with numerical
  continuation/rolling likelihood and native generation references. Execute a real
  offline harness evaluation on original tasks; no service, downloaded model or
  implicit remote dataset/code execution. Read installed task definitions and
  publishers' metric/prompt/license sources before registering external protocols.
- Import only reviewed immutable benchmark bytes with license evidence. GSM8K,
  general multiple-choice and Russian/original synthetic coverage are independent
  from public MATH/competition/unsafe-code rights/execution gates. Block ambiguous
  sources explicitly, not through misleading toy benchmark names. A limited subset
  or adapted prompt is not an official full benchmark result.
- Extend lexical audit to actual tokenized training shards: decode document frames
  by stored coordinates, inspect the training split without cross-document joins,
  retain source/tokenizer hashes and byte/token coverage. Every eval has an audit;
  release ledger generation requires declared training-shard coverage for every
  declared benchmark and refuses incomplete release inputs. Unknown history and
  semantic leakage remain outside lexical cleanliness claims.
- Add an opt-in pinned regression recipe in shared training configuration. Evaluate
  each committed CPU single-rank checkpoint through the shared save path; support
  recovery of missing reports after a crash. Preserve RNG,model/optimizer/loader
  state; unsupported GPU/sharded/adapter tiers fail explicitly. Plot verified
  checkpoints with Matplotlib and retain checkpoint/report hashes for every point.
- Generate RESULTS.md deterministically from verified report artifacts, including
  exact resolved config,checkpoint/tokenizer/dataset hashes and reproduction argv.
  Bundle portable text reports/configs/provenance in Git,not model weights or raw
  corpora. README measured-result numbers must resolve to the ledger;operational
  version/command/config identifiers are not benchmark results. Compare only
  identical semantic evaluation settings/data/prompt protocols;external paper
  scores are not imported into model comparison tables.

Acceptance: independent hand-built metric/code/choice/adapter cases,brute-force
pass@k and vote references,real CPU native+harness parity,shard overlap canaries,
actual per-checkpoint training/regression plot with exact restart and automatic
ledger regeneration,full checks and source-bound CI evidence. Final800 untouched;
no paid,final-scale,GPU or unsafe arbitrary generated-code execution.

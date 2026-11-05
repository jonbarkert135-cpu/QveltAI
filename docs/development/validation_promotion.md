# M7.2 original CPU study acceptance

Date:2026-11-05. Design89a0350 was committed before study learning. See
[design](promotion_design.md),ADR0046 and[runbook08](../../runbooks/08_model_promotion.md).
This implements necessary comparison controls;production acceptance still needs
actual user-run model/hardware and independent licensed quality evidence.

## Implemented contracts

Strict study YAML binds evaluation recipe file bytes and checkpoint manifests.
All seeds require base/SFT/DPO/GRPO;optional half-LR DPO is matched. Full run/
checkpoint application provenance establishes actual SFT parents,initialization
seeds,shared tokenizer/model/data/source and one-factor LR settings. Distinct
base weights prevent relabeling identical deterministic updates as replications.
The same existing evaluation engine applies fixed prompts/budgets/scorers and
keeps failures in denominators. No new dependency/public corpus/teacher is used.

Original200-validation/five families plus40 whitespace transformations of20
parents and7 general/LM controls;all800 final remain unscored. Pairing verifies
task/gold/IDs,counts wins/losses and exact Fraction macro-family deltas. SymPy
independently verifies the unequal-family reference3/80. Known contingencies,
each regression check,missing gate/coverage,replication,seed/parent/role/tokenizer,
and final/sampling/cap guards are tested. Nineteen bounded positive/adversarial
verifier controls pass;no generated Python/CAS runs.

Fresh Linux CPU processes measure identical fixed16-action eager workloads:
2 warmups/5 durations,median/tokens-per-second,weights bytes and whole-process
peak RSS. Tests check median arithmetic,artifact identities and unchanged input
weights. This is not incremental tensor memory,end-to-end latency or GPU serving.
Top report publication commits child reports/resources and all identities;no
optimizer/RNG pickle is deserialized by comparison. No automatic model promotion.

## Developmental evidence

One-seed20261105 debug run trains all five roles with a shared tokenizer/data/
parent. SFT5/200,ordinary DPO0/200,half-LR5/200,GRPO5/200. Ordinary DPO loses5
matched remainder cases. Half-LR recovers that score but still regresses formatting/
invariance and shows no math gain. GRPO unchanged. All necessary candidate
decisions reject. This exploratory run is not a three-seed or production claim.

Focused35 pass in38.14s(18 promotion units,5 actual workflow cases,12 existing
evaluation config cases before final gate-schema strengthening). Full final-code
local CPU/optional-SDK regression:688 passed,1 old packed-DDP timeout,7 GPU
deselected in311.24s. No skip/weakening/local-full success claim;orphan worker
cleaned up after its parent failed. Ruff/format,mypy153 files,wheel/sdist and
frozen dependency checks pass(93 resolved,72 compatible installed packages).
Final-source three-seed receipts and clean13-job CI acceptance follow separately.

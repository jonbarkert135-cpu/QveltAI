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

## Final matched-seed study

Frozen package9ca8a8fd993965196dbaaaca1f1d8b4f22a13339644b3c73f784fa7ad5601195,
source6b0d4cc. All15 training/evaluation arms record clean Git provenance and
matching original tokenizer/data/source. Three actual runtime seeds20261105/6/7
change initialization **and sampler ordering**;this is not an initialization-only
variance decomposition. Within each paired LR arm,data order and parent agree.

All scores below use the same200 validation cases/greedy16 budget:

| Stage | Seed20261105 | Seed20261106 | Seed20261107 |
| --- | ---: | ---: | ---: |
| Base3 | 0 | 0 | 0 |
| SFT80 | 5 | 1 | 0 |
| DPO40,LR0.001 | 0 | 0 | 3 |
| DPO40,LR0.0005 | 5 | 0 | 0 |
| GRPO8 | 5 | 1 | 0 |
| RoPE scale2 SFT80 | 1 | 0 | 0 |

DPO's one positive seed does not repeat:macro-family delta ranges-0.025..0.015;
half-LR ranges-0.005..0;GRPO0..0. All candidates reject under the preregistered
necessary checks. No thresholds are changed to rescue one run. Full metadata/
per-seed family/format/robust/general/resource findings are in
[three-seed receipt](evidence/promotion-study-3.json). Base target counts706/706/698,
SFT2520 each,DPO1582 each;GRPO actual actions701/695/673,8 policy updates each,
constant groups15/14/13 out of32. These are not compute-matched stages or full FLOPs.

Analysis repeats without learning. All15 prediction,metric and audit files are
byte-exact;all comparisons/gates match. All15 input checkpoints still verify
against their original canonical manifests. Timing/RSS are measured separately,
not promised bitwise;see[repeat receipt](evidence/promotion-repeat.json).
The fixed16-action median durations span14.20..20.18ms in the first local study.
All process-RSS/latency necessary limits pass. A same-checkpoint shell probe
reports287260672 RSS bytes versus390721536 from the study-launched processes:
launch context matters;these whole-process peaks cannot estimate incremental
model memory or server/GPU resource requirements.

## Exploratory architecture control

The separate[RoPE design](rope_ablation_design.md) was committed in3d379fc before
its learning,but after validation had already been exposed. Sourcee2c628f binds
standalone script49b1d30d2b74baed6e0da1495ff88928cc96013180c0a38bf2a2ae4bb54743d6;
the model/trainer/evaluator package remains the frozen9ca8a8f digest. This avoids
changing executable training code between the paired original and variant arms.

Three new base3->SFT80 chains change only rope_scale1->2 and the derived SFT parent.
Initial weights are checked bit-exact before learning. Optimizer/schedule/data/
runtime seed/order and actual processed target counts match per stage. All use
the same247 cases/scorers/resource/gate protocol;all800 final remain unscored.
Macro-family deltas-0.02/-0.005/0;the first two seeds also fail domain-PPL and
robustness checks. Candidate rejects;no long-context or production gain is claimed.
See[three-seed architecture receipt](evidence/rope-ablation-3.json).
Two additional script recipe/initialization/RNG unit tests pass in1.80s;
Ruff/format and mypy155 files pass. The earlier full local688-pass/1-DDP-timeout
run covers the unchanged package;it is not relabeled as a new all-green full run.

Production promotion remains blocked on independent licensed/contamination/final
evidence and actual user-run small/GPU checkpoints. The next independent roadmap
implementation is M8.1 inference/export;negative CPU findings remain recorded.

## Final remote acceptance

Sourcee2c628f,[GitHub42100151962](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/42100151962):
**all13 jobs successful**. CPU3.11/3.12/3.13 each689 passed,2 optional SDK skips,
7 GPU deselected in223.44s/220.64s/185.26s;mypy155 files and packaging pass.
SDK event/offline tests pass separately. Dedicated matched2-seed and architecture
2-seed studies reject their candidates as expected;see
[study receipt](evidence/promotion-ci-final-study.json),
[architecture receipt](evidence/rope-ablation-ci.json) and
[matrix/recovery receipt](evidence/promotion-ci.json). Current-source200-step real
SIGKILL recovery has maximum loss delta0 and exact weights/Adam/RNG/application;
SFT,DPO,GRPO and evaluation replay jobs pass. This is portable CPU engineering
acceptance,not a diagnosis/fix of the old local packed-DDP timeout.

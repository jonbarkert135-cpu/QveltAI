# Expanded Phase6 CPU acceptance

Date:2026-11-06. Base:fbd7a3a. [Design](posttraining_phase6_design.md),ADR0048
and [consolidated runbook](../../runbooks/03_posttraining.md) define the supported
scope. Final executable source8e9b44c and package SHA256
28addb6ad1641aebd047a16931b4177db276022609a39eae9e63ca5e273a3bc3.
All15 [CI jobs](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/42253772192)
pass. This is portable framework acceptance;quality gates reject the control model.

## Final measured evidence

All five control-stage run manifests record clean source8e9b44c. SFT20,DPO4,
RL60 reproduce the diagnostics. RL60 optimizer steps,56 policy-signal windows,
4639 sampled actions,90 constant groups. Training reward first10=.3125,last10=
.709375. Fixed64-sample validation15->26 correct,59->62 complete. Offline replay
matches all60 losses,model,Adam and RNG exactly.

**Quality rejects:**validation golds are balanced32 zero/32 one,but all59 parseable
post-RL numeric responses are zero. Correctness for answer1 remains0/32;
answer0 goes15/32->26/32. Overall26/64 is below the32/64 always-zero baseline.
Training has12 zero/4 one problems (75% constant-zero baseline). Reward rises
with answer/termination collapse. These results do not demonstrate arithmetic learning or a reasoning
improvement. Original broader negative studies remain authoritative;no promotion.

Final rejection:16 training-only problems x4 samples,64 candidates,46 correct
before dedup,pass@1=.71875,pass@2/3/4=.75 (difficulty easy only). Selection keeps12
answer-only examples,all gold0. Numeric spelling variants no longer create trace
diversity. No actual free-text reasoning trace/process correctness is demonstrated.
Feeding this corpus back into SFT would reinforce imbalance without coverage review.

Final rank4/alpha8 LoRA20 independently repeats stop6/resume20 against continuous
weights/Adam/RNG/application exactly;9728 trainable parameters,127104 base unchanged,
training probe6.133254->5.143521. Full SFT's same-budget probe is0.755217;this is
adapter correctness,not superiority. Final original GSM8K import has16 distinct
unrated problems and exact pinned MIT/source bytes;no optimization consumes them.

Final focused88 tests pass in47.29s;Ruff/format,strict mypy179 files and build pass.
CI source8e9b44c:CPU3.11/3.12/3.13 each798 passed,2 optional SDK skips,7 GPU
deselected in324.22s/341.83s/212.55s. Optional SDK tests separately pass. All15
jobs pass,including native export,posttraining/replay,SFT,DPO,GRPO,studies and
full19.34M200-step pretraining with actual SIGKILL89/restore75;all200 losses have
delta0 and model/Adam/RNG/application match. These pretraining checks are not a
19.34M post-training quality result. Local pre-selector full798 pass/2 distributed
failures and prior CI3.13 timeout remain recorded below;no Gloo fix is claimed.
The orphan worker from the already-failed local packed test was explicitly cleaned
after results were recorded;test return codes/checkpoints were not changed.

Committed receipts and readable logs (no raw external corpus/model weights):

- [Complete control/validation audit](evidence/posttraining-control.json).
- [All60 RL metric windows](evidence/posttraining-metrics.json).
- [Question/gold/text sample audit](evidence/posttraining-sample-audit.json).
- [Derived answer-collapse/constant-baseline audit](evidence/posttraining-quality-audit.json).
- [LoRA20 replay](evidence/posttraining-lora.json).
- [Pinned GSM8K16 import](evidence/posttraining-gsm8k.json).
- [Fourteen fixed verifier controls](evidence/posttraining-verifier-audit.json).
- [All15 CI job identities/counts/recovery receipts](evidence/posttraining-ci.json).

## Implemented and tested interfaces

Chat-v2 explicit assistant reasoning,version-aware shifted masks,multi-turn and
existing packed isolation;unquantized LoRA in the shared full-SFT engine;bounded
exact/numeric/SymPy QQ polynomial verifiers;own-checkpoint native rejection with
all candidates/difficulty pass@k/lexical selection;immutable original GSM8K train
ingress;default-off external local trace ingestion with teacher terms evidence.
Original DPO reused. GRPO now uses native Phase8 KV-aware generation,separate
correctness/format/length/KL diagnostics,readable questions/golds/answers and14
public fixed controls excluded from optimization. Offline windows validate
policy/reference/step/prompts/seeds/actions/stops/text/rewards/logps before update.
Lean is [design only](../research/lean_verifier_design.md),not executed.

Independent tests cover dense LoRA values/gradients/zero-B initialization,frozen
base,merge parity and exact optimizer/RNG/cursor replay;explicit shifted target
offsets and zero non-assistant gradients;polynomial identities against trusted
SymPy plus exact numeric grid evaluation;adversarial AST/answer formats;pass@k
against exhaustive subset enumeration;real checkpoint sampling and deterministic
selection repeats;isolated forced-candidate selection oracles;default-off terms/
permission gates;offline stale-policy/action/logp/reward/text rejection;real full
SFT->DPO->RL control and exact online/offline updates. No forced gold generation
is used in actual training or acceptance sampling.

Pre-final focused83 passed in47.74s. An earlier full local regression had
796 passed,4 failed,7 GPU deselected in232.36s:one historical two-rank Gloo
SIGABRT at process exit and three new CLI failures from a branch-local import
shadowing the common manifest_digest function. The CLI bug is corrected at the
shared import;final repeat is recorded below when completed. No tests are skipped,
weakened or forced to exit to hide the distributed error. Ruff/format,mypy179
files and wheel/sdist build pass before the CLI correction;rechecked afterward.

## Observed control,not general reasoning quality

The original127104-parameter,V445,T128 control has only two numeric answers.
Base3/SFT20/DPO4 use seed20261031;RL60 seed20261106,4 prompts x8 samples,
temperature1,max4 actions,LR0.001,KL0.01. DPO descends from SFT,RL starts from
DPO and keeps SFT frozen as reference. DPO uses only SFT-training strata;the
64-sample before/after evaluation uses SFT-validation strata and seed20261107.

The local task/budget specification preceded diagnostics,but was not committed
or independently timestamped before seeing outcomes. Known diagnostic outcomes
precede the final source rerun;this is reproducibility evidence,not preregistered
independent hypothesis testing. The task was intentionally chosen to be easy.

Diagnostic v2:fixed SFT train probe6.133254->0.755217;DPO train .693147->.628261
but heldout .693147->.711595 (negative). Training reward first10=.3125,last10=
.709375. Fixed64-sample validation15->26 correct,59->62 complete. RL60 updates,
56 policy-signal windows,4639 actions,90 constant groups. Online/offline all
losses,weights,Adam,RNG exact. None of this supersedes the previous200-case
multi-seed negative studies or original GRPO1/8 unchanged result. Final800 remain
unscored;no general math gain or checkpoint promotion follows.

The source adapter's exact human-train payload is4166206 bytes/7473 rows,SHA256
17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465;MIT1062 bytes,
SHA25686bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b.
Actual16-row import passes with stable question IDs,unrated difficulty and full
license/provenance;no optimization consumes these external questions. No Socratic
or external teacher API/model-solution data is ingested. A first import exposed
duplicate commit-marker publication;fixed by one final root marker plus nested
source bundle,and covered by a separate offline publication test.

## Material boundaries

The first source49876b0 clean standalone reproduced the diagnostic gains,with
all five stage manifests clean/pinned to that commit. A20-step rank4/alpha8 LoRA
run stopped6/resumed20 exactly matches continuous weights/Adam/RNG/application;
9728 trainable adapter parameters,all127104 base parameters unchanged. The fixed
training probe6.133254->5.143521 is a mechanism result,not a gain versus full SFT.
The corrected CLI subset57 passes in23.58s and lint/mypy179/build pass. Repeat
full source49876b0:798 passed,2 historical distributed failures,7 GPU deselected
in352.91s (numeric-DDP SIGABRT;packed-DDP stop1 timeout120s). CLI failures are gone;
the local full run is not green. No distributed workaround or test suppression.

Audit review exposed equivalent final-number spellings being counted as different
lexical answers. Selection now compares reasoning bodies separately,retaining
one answer-only row/problem;the oracle uses2 versus4/2 with identical reasoning.
Pass@k still counts all raw sampled correct outcomes before dedup. This changes
selection,not the training/reward protocol. Final-source receipts/CI follow.

CPU FP32 LoRA/full-base checkpoints only;no QLoRA or adapter export compatibility.
GRPO is an original-environment,<1M-parameter CPU prototype,sequential cached
completion collection plus behavior-logp recomputation,not a rollout-performance
claim. Arbitrary external/public-problem RL,GPU/distributed/main-scale execution
and production data/model clearance remain open. Polynomial checks have explicit
grammar/degree/domain limits and verify outcomes only. Lexical trace diversity is
not semantic correctness. Fixed public verifier controls do not establish unknown
exploit coverage. Publisher-license assessment is not exhaustive upstream rights
clearance. Historical Gloo instability is not resolved by remote CI success.

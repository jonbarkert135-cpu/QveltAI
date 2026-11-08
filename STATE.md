# QVELT current state

Updated:2026-11-08. Implementation lead:Lead agent; named human reviewers pending.

## Current task: restore the critical coverage gate, 2026-11-08

Base main:`94f0551ec3b34d4cf5cfed96d6f28008da10c1f7`. [Verified by CPU execution]
the exact critical CI test selection passed114 tests in9.93s, but coverage report
failed with exit2 at88.082437% combined statement/branch coverage against89.
Add the existing test_tokenizer_v3.py suite to that selection; no new test
definitions or runtime changes are needed. The updated selection passed277 tests
in10.37s and coverage report exited0 at89.695341%. Threshold89, covered source
scope804 statements/312 branches and exclusions0 are unchanged;9 additional
statements and9 branches execute. Coverage7.17.2,pytest9.0.3,tokenizers0.22.2,
Torch2.13.0+cpu and FastAPI0.162.0 were read from the installed environment.

| Critical file | Combined before -> after | Missing lines before -> after |
| --- | --- | --- |
| tokenizer/core.py | 82.178218% ->91.089109% | 19 ->10; newly exercised94-99,182,231,244 |
| data/loader.py | 83.703704% ->83.703704% | 35 ->35 |
| data/shards.py | 93.846154% ->93.846154% | 4 ->4 |
| posttrain/verifiers.py | 93.518519% ->93.518519% | 7 ->7 |
| inference/batching.py | 94.478528% ->94.478528% | 4 ->4 |

The added gate cases assert v3 control IDs/literal isolation/Unicode, rejected
settings and corrupted policies, oversized non-publication, legacy hashes and
independent-process determinism. Makefile has no separate critical-coverage
target; its existing full CPU test discovery already includes v3, so it stays
unchanged. [ADR-0063 maintenance note](DECISIONS.md#adr-0063-explicit-v3-codec-selection-without-changing-legacy-identities)
records the selection fix. Existing model/training/serving behavior, tokenizer
artifacts, data identities, checkpoint formats, lockfiles and coverage exclusions
are untouched. Only CI selection and required project-memory documentation change.
No model-quality, GPU, remote-CI or complete-loader-coverage claim; C/D/E remain
unstarted and gated. Stop after this one local commit for owner review.

Additional validation: Ruff check/format passed240 files, strict mypy passed240
files; documentation-tool tests2 passed in0.12s, strict MkDocs passed in0.76s.
Checked118 local links/anchors, consecutive ADR0001..0066, unchanged threshold/
source denominator/exclusions, and that CI selection adds only the existing v3
suite. git diff --check passes. No full-suite, GPU or remote-CI rerun is claimed.

## Previous task: context-efficiency owner decisions, documentation only

Base main:`cfa0aa55f39aef6a79d0496e6e32e04ba88f3861`. On2026-11-08 the owner
accepted the context-efficiency audit and requested one documentation-only
commit, then STOP for review. [ADR-0064](DECISIONS.md#adr-0064-canonical-token-accounting-and-the-token-count-api),
[ADR-0065](DECISIONS.md#adr-0065-tenant-isolated-prefix-cache-identity-and-engine-acceptance)
and [ADR-0066](DECISIONS.md#adr-0066-versioned-compaction-with-source-linked-memory-and-drift-evaluation)
record contracts and acceptance requirements, not implemented capabilities.

Owner decisions received, not independently verified enforcement:

- Private KV cache is memory-only with a short finite TTL, deleted with its chat.
  Numeric TTL and idle/absolute expiry semantics are not chosen. Originals and
  summaries obey user retention/deletion; enterprise retention is configurable;
  legal review remains pending. These are requirements, not deployed controls.
- First cache is fully private. Shared public prefixes are deferred to a later
  explicit server allowlist. No custom block-store prefix cache is to be built
  now; a later pinned vLLM/SGLang selection must meet ADR-0065, not assumed defaults.
- Always pin user instructions/constraints, decisions, open tasks, unresolved
  errors, file paths/identifiers and permission/security decisions. If these
  cannot fit, report incomplete/insufficient_budget; do not silently discard them.
- Until a named ML reviewer exists, the owner reviews compact samples using a
  plain-language rubric. This is not named-expert production or legal sign-off.
- First evaluation sources:own synthetic data and the owner's own specifically
  consented conversations only; no other real-user data. Exact corpus permission,
  secret/PII filtering and retention still precede processing; private transcripts
  are not committed. External teacher outputs remain OFF.
- First GPU benchmark priority:repeated long prefixes, then multi-user load,
  then single-chat latency. No hardware, budget or paid run is authorized here.

Token accounting is first among the context-efficiency implementation follow-ups;
the previously agreed C -> D -> E order and review stops remain. C is unstarted:
typed tools/schema3/masks and schema2 report fix remain. D is unstarted:truthful
discovery/unsupported-feature tests remain. E is unstarted:multilingual measured-V
experiment design remains; no final V or corpus approval. [PLAN](PLAN.md) labels
each follow-up CPU-only, small rented GPU node, or cluster. No compaction training
now:only future task format/schema and CPU evaluator fixtures, separately reviewed.

Current runtime remains the audited baseline:request-local concatenation KV,
no shared prefix cache, token-count endpoint, compact task/evaluator or extended
capability discovery. Schema3 remains unimplemented and v3 chat is explicitly
rejected. No code/config/tokenizer/checkpoint/lockfile/artifact change or new ML
result occurs in this task. Existing replay/lineage guards remain authoritative;
a docs commit is not permission to override source/runtime compatibility checks.
The owner target is not verified scale/quality and establishes no comparison to
other models. Historical evidence below is retained; this section takes priority
over old next-step suggestions.

Documentation validation:the existing documentation-tool suite passed2 tests
in0.15s; strict MkDocs build passed in0.72s. Checked114 local links/anchors in
DECISIONS/STATE/PLAN, consecutive ADR0001..0066, append-only ADR history, C -> D
-> E ordering and the exact three-file docs-only scope. git diff --check passes.
These checks validate documents only; no new runtime/ML/GPU/security/privacy
enforcement result is claimed. Stop after this single commit for owner review;
no implementation or data processing starts.

## Previous task: Step A owner-spec amendment, documentation only

Base main:`f15b8274a9cf493acc640bc2515d078ce92483b4`. The owner supplied the full
Russian strategic specification (Q1-Q19 and principles1-5) in the conversation on
2026-11-07 after the attachment was unavailable. The entire supplied text was
read. This records owner requirements/statements, not independent verification.
[ADR-0052..0056 amendments](DECISIONS.md#adr-0052-frontier-horizon-and-gated-validation-ladder)
and [ADR-0057..0061](DECISIONS.md#adr-0057-effort-conditioned-training-and-enforced-compute-budgets)
record the resulting design changes. No implementation or model capability follows.

### Owner decisions and statements received 2026-11-07

| Spec | Owner-stated requirement or claim | Verification / operational status |
| --- | --- | --- |
| Q1/Q10, conclusion | A frontier general model/ecosystem for reasoning, software engineering, math/science and agents; first-year infrastructure/model/reasoning/platform/agent/limited-launch ambitions; small models are a validation/support ladder. | Goals and deadlines, not achieved results or delivery guarantees. |
| Q2 | Owner describes budget as practically unlimited; preliminary state-investment program USD50-100 billion over5-7 years, with USD10-15 billion proposed for12-18 months. | Owner statement only; no funding agreement, availability or spending authority independently verified. Numbers must not drive code, configs/defaults, procurement or paid runs. |
| Q3 | Plans for large owned NVIDIA H100/H200/B200-or-newer clusters, thousands/tens of thousands of GPUs, fast interconnect/storage, cooling/power/datacenters; cloud as needed and backend portability. | Plans, not verified ownership/access. Current environment reports Torch2.13.0+cpu, CUDA available=False, GPU count=0. |
| Q4/Q18/Q19 | International specialists, technical leads, independent review and qualified human accountability; agents provide clear evidence and explain disagreement simply. | Staffing/appointments unverified; no ML or legal reviewer named. Owner is not a programmer; each report requires a plain-language summary and a review stop. |
| Q5/Q6 | 100+ languages including low-resource; English/Ukrainian/Russian priorities; named programming languages; starting40/30/20/10 mix hypothesis. | Language scope/mix direction received, replacing EN/RU-only planning. Final registry, per-language weights, rights/data sufficiency and quality remain unverified. |
| Q7-Q9 | Source/license registry, legal review; lawful external outputs only; separate voluntary training opt-in, withdrawal/deletion, secret/PII filtering and enterprise rules. | No new source approved. Default candidates:own/permissive/explicitly licensed data only; external model outputs OFF pending terms-specific named legal review. Privacy controls unimplemented. |
| Q11 | File/project/archive/code/terminal/web/API/agent workflows with permission for critical actions. | First internal tools remain read_file, search, apply_patch, exec(tests/lint/build), git diff in an isolated runner; broader tools are later requirements. |
| Q12, principle5 | Separate retained reasoning with retention/access policy; concise explanations/summaries by default; effort none/instant, low, medium, high, extra high, maximum changes real compute. | Recommend effort field+reserved tokens+budget-conditioned training+enforced limits. No modes/summarizer implemented; effort_levels stays empty until supported. No retention duration assumed. |
| Q13 | Text/code/tools first, image understanding later; audio/video and visual creation further phases. | Separate later modality ADRs; no image capability or image tokens added. |
| Q14 | Up to1,000,000 TOTAL tokens including system, history, documents/files, tool results, service framing and generation; one response<=128,000. | Future target, not existing capacity. 128,000 supersedes the earlier131,072 design value. Hidden reasoning consumes output budget. |
| Q15 | QVELT is the product/model/ecosystem name; trademark/domain checks before launch. | Prose naming updated safely in this task; package paths, identifiers and existing response IDs unchanged. Legal name clearance unverified. |
| Q16/Q17 | Web first; accounts/auth/session/history/permissions and consumer/enterprise separation from day one; desktop/local agent later; payments before commercial launch, optional for the first closed version. | Product requirements, not features of the current single-key framework server. No account/payment/product implementation in this task. |
| Principles1-4 | Honesty without flattery, justified initiative/disagreement, sourced factual accuracy without partisan agenda, competence and explicit uncertainty. | Post-training/evaluation requirements; no compliance claim before probes and mandatory human review. Ukraine/Taiwan distinctions are included in the proposed factual-claims suite. |

**Claims rule:** no assertion that QVELT matches or exceeds another model may
appear as fact without independent measured evaluation with provenance and
comparable settings. Funding, team and competitive ambitions do not supply that
evidence. No architecture/parameter claims about other labs are inferred.

### Amendment scope and remaining gates

Tokenizer v3 still means a NEW lineage before real pretraining. V remains
configurable/undecided; candidates extend above131072/262144. The proposed
128-control block now reserves effort IDs22-27 and leaves100 spare slots.
This changes the unfrozen design only; v1/v2 and all existing artifacts remain
untouched. No language/data split or numeric effort budget is fixed in configs.

Named ML review is required for tokenizer table/V, architecture family, data
lineage/licenses and precision. Named legal review is required for the relevant
source/consent/release decisions. Until reviewers are appointed, every step
continues to stop for owner review; production freeze is not implied by a draft.
First small-GPU validation is a separate gated milestone before GPU-dependent
work. No paid runs, bookings, external teachers or ingestion are authorized here.

Remaining decisions:reviewer names, final100+ registry/per-language allocations,
approved corpus/sample/revisions, production V, per-run compute budget, effort
policy and retention/enterprise/jurisdiction details. Current native limits stay
context<=min(snapshot.context,2048), output<=256; planned capabilities stay false
until model/backend support is actually validated.

This pass changes DECISIONS/STATE/PLAN and the server-setup runbook only.
Amendment validation:2 documentation-tool tests passed in0.13s; strict MkDocs
build passed in0.73s;70 local links/anchors and consecutive ADRs0001..0061 checked.
The draft table has128 unique controls,6 effort controls and100 reserves; exact
arithmetic checked the context/output budget and40/30/20/10 mixture sum.
git diff --check and docs-only scope checks passed. These are document checks,
not ML/GPU/scale or legal acceptance. UN seed-document endpoints returned403;
they remain unverified, with human factual/legal adjudication still required.
Stop for review after this task. Step B and all later implementation remain
UNSTARTED; the schema2 reporting fix is still Step C.

## Environment readiness (2026-11-04)

Managed CPU development is restored: Python3.12.14/uv0.12.23/Torch2.9.1+cpu,
frozen optional metrics SDKs, writable Make caches and installed Git hooks.
Authenticated GitHub fetch/push dry-run works with explicit network permission;
local main tracks origin/main. CPU/optional-SDK suite:524 passed,5 GPU deselected,
no skips in94.75s. Lint/format,mypy97 files,lock/dependency checks,hook checks and
wheel/sdist build pass. Numeric100-step and packed BPE20-step exact replay smokes
pass. See docs/development/validation_environment.md and its receipts. Direct uv/
Git-hook invocations need the README cache exports in read-only-home environments;
network operations/local distributed sockets need execution network permission.
M6.1 CPU acceptance is recorded below; no GPU or paid/final-scale run was performed.

## Previous task: Phase9 portable framework acceptance

Accepted runtime8c082145201d6c27d2798ca5362945ddcc6390e3;all22 GitHub CI
42384342511 jobs successful,including complete CPU3.11/3.12/3.13 suites,fresh
RFC tiny tutorial,200-step actual-kill recovery,docs/browser,security and containers.
Official status evidence retained;hosted blob downloads hit session-proxy403,
so no uninspected hosted test counts/measurements are asserted.

Strict offline MkDocs site,original Mermaid,subsystem guides/FAQ/troubleshooting,
small safety SFT design/data,SemVer0.1.0/changelog/draft-only release plumbing and
README claim evidence complete. Security upgrades:Torch2.13.0,Transformers5.10.4,
pytest9.0.3. Restricted DCP metadata/extents and weights-only optimizer bytes tested.
Critical114 tests pass10.97s;coverage85.3186% ->89.2562%,no exclusions. Ruff,mypy238,
lock,16 historical/native receipts,wheel/sdist and actual installed-wheel native
stream parity pass. Original freeze remains byte-bound across version envelopes.

Fresh local clean7ef8bfd/new virtualenv with shared package cache completed full
RFC tiny19,338,560-param base6/stop3/resume6/SFT3/eval/HF/HTTP in477.817426s.
HF max error7.152557e-7,weights unchanged;one receipt added to RESULTS. Actual
current-source fresh hosted tutorial passes separately. Profiling fixed costly
per-token backend vocab queries;the same two-source audit now1.843169s with an
identical full report. This is engineering plumbing,not useful/safe model training.

Native+HF+serving advisory audit exit0;optional harness sqlitedict advisory remains
an all-extras failure.133 reviewed offline credential findings,zero unexpected;
hex entropy/history review scope explicit. Initial local full910pass/2timeouts/
5freeze setup errors retained;freeze repaired,local distributed recheck2pass/1DDP
timeout144.48s. Remote success does not establish a local Gloo root-cause fix.
CUDA13.0/GPU/production rights/complete contamination/model quality/safety gates
remain open;final800 unscored. No tag/public release/Hub upload. Next:reproduce
on intended hardware,approve production rights/exclusions,and run the bounded
small pilot in runbooks02 before a full horizon. See validation_release_phase9.md,
ADR0051 and release-phase9 evidence. Previous Phase8 acceptance follows.

## Previous task: Phase8 acceptance

Expanded Phase8 portable CPU reference accepted onb5deaf40d161a260be50cfe135b67dda479ccd7c;
all19 CI42356828424 jobs successful. CPU3.11/3.12/3.13 each846 passed,18 optional
skips,7 GPU deselected;actual HF/HQQ/API34 tests and complete source-clean smoke
pass separately. Local all-extras864 passed/7 deselected331.00s,no skips. Ruff,
mypy217,lock/package consistency and wheel/sdist pass. Final test-only extension
checks historical reasoning parity in HF/native chat;focused10 pass7.75s.

Shared Engine/sampling,private cached streaming,standard builtin HF Llama export
with Q/K conversion and direct dense indexed shards,safe numeric HQQ4/8,actual
FastAPI chat JSON/SSE/auth/microbatches/deadlines/failures,linear RoPE diagnostics,
three reproducible Docker targets,MODEL_CARD and runbooks05 implemented.
Clean runtime307139e86588377d9e01eded8d156b7534c0cea13722901e9b81f9a97fbaf061:
123904-param three-update debug model,all seven run/export artifacts and six eval
reports dirty=false. HF max error2.384e-7;sampled cached/full/stream exact. Matched
PPL374.932430/375.023875/374.914144 for FP32/HQQ4/HQQ8,56 original targets only;
all generation/choice0. Needles0/30 at factors1/2/4,context overflow12/6/0. This
cannot establish useful accuracy preservation or long-context quality. Final800
unscored,source weights unchanged. Six report-bound receipts added;RESULTS now
regenerates from15 native receipts,preserving Phase7 history.

Actual local CPU HTTP sample:109 completion tokens,8 requests/concurrency4,
374.008338tokens/s,P50.119225s/P95.233496s,server lifetime peak264736768 bytes;
EPYC9V74,Torch1 thread. SSE exact,real microbatches3/4,disconnect/expiry1,pending0.
CI builds all3 Docker targets,defaults UID10001,real container3-update training,
authenticated JSON/SSE and clean shutdown pass;quantized dependencies import.
Local vfs runtime ENOSPC remains a recorded failure,not a successful launch.
Current-source RFC tiny200-step actual SIGKILL89/resume75 keeps all losses/model/
Adam/RNG/application exact. Existing workflows remain green. See runbooks05,
validation_serving_phase8.md and serving-phase8-{acceptance,container,ci}.json.

Unrun/separate gates:GPU/vLLM deployment,large conversion/quantization,sharded
checkpoint consolidation,production serving and trained retrieval/benchmark
quality. Native/HQQ remain CPU FP32<=20M,batch8,context2048. vLLM source compatibility
is verified,not runtime acceptance. Next dependency-ready milestone:M9.1 clean
workflow documentation/release preparation,subject to existing rights/quality gates.

Previous Phase7 acceptance follows.

Expanded Phase7 portable evaluation accepted onacced9071be0896b2fa6e915c7ed9d1b0f73e9d9;
all17 CI42303284852 jobs successful. CPU3.11/3.12/3.13 each823 passed,6 optional
skips,7 GPU deselected;actual optional harness11 tests and metrics SDKs separately
verified. Final focused71 pass20.68s;additional preference-shard audit covered by
8-test ledger/source set4.28s. Lint/format,mypy195,lock/dependencies and build pass.
Native v2 metrics/majority/failure witnesses,actual lm-eval0.4.13 bridge,original
EN/RU/code controls,pinned bounded GSM8K/MMLU/ARC imports,train-only actual-shard
audits,per-checkpoint regression/repair/plots and verified generated RESULTS.
Clean source17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3:
123904-parameter debug model,three training updates,stop2/resume3 model/Adam/RNG
and curve values exact;sampled predictions/metrics/audits byte-equal,weights unchanged.
All three input runs and nine native report receipts dirty=false. Negative bounded
zero-shot prefixes:GSM8K0/16 (4 context overflows),MMLU algebra4/16 (5 failed
contexts),ARC raw2/16,normalized3/16 (5 failed contexts). Original generated math/
code all fail;regression0/0/0. No official full benchmark score or model promotion.
Nine receipts,acceptance and curve JSON/PNG committed;root RESULTS generated/checks.
MATH-500/AMC-AIME/HellaSwag/full HumanEval block release artifact coverage;rights,
access,LaTeX metric/full Python sandbox remain explicit gates. Final800 unscored.
Pre-final-ARC local full822 passed/2 two-rank120s timeouts/7 deselected505.11s;
no suppression. Initial f741151 CI caught missing runs/ parent;corrected with nested
parent/exclusive-output test. Remote green does not diagnose historical Gloo faults.
See runbooks/04_evaluation.md,validation_evaluation_phase7.md and source CI receipt.
Next dependency-ready implementation:M8.2 measured serving;large/GPU/DDP evaluation
and production quality/release gates remain separate.

Previous expanded Phase6 acceptance follows.



Expanded Phase6 portable extensions accepted on8e9b44c;all15 CI42253772192 jobs
successful. CPU3.11/3.12/3.13 each798 passed,2 optional SDK skips,7 GPU deselected;
SDKs verified separately. Final focused88 passed in47.29s;lint,mypy179/build pass.
Chat-v2 reasoning masks,shared full/LoRA SFT,bounded exact/numeric/SymPy polynomial
verifiers,own-model rejection/difficulty pass@k,default-off teacher/terms ingress,
pinned original GSM8K train adapter,native-engine GRPO/offline replay and Lean design.
Clean source28addb6ad1641aebd047a16931b4177db276022609a39eae9e63ca5e273a3bc3:
SFT20/DPO4/RL60 on127104params;training reward .3125->.709375,validation15/64->26/64.
Quality rejects:all59 parseable post-RL numeric outputs0,answer1 remains0/32;
26/64 is below32/64 always-zero baseline. DPO heldout loss worsens. Rejection12
of64 candidates (46 correct before dedup),all selected gold0;not reasoning traces
or a promotion. LoRA20 rank4/alpha8 stop6/resume20 exact weights/Adam/RNG/cursor,
9728 trainable parameters,base127104 unchanged. Actual16-row GSM8K train import,
no optimization on it. Native offline all60 losses/model/Adam/RNG exact. All five
control stage manifests clean on8e9b44c. Final800 unscored. See runbook03,
validation_posttraining_phase6.md and eight posttraining evidence receipts.
Pre-selector local full798 passed/2 historical distributed failures/7 deselected;
prior CI3.13 also had a numeric-DDP timeout. Remote success does not diagnose/fix
Gloo. No tests suppressed. GRPO stays original-environment CPU<1M;QLoRA/GPU/
distributed/arbitrary-problem RL,Lean execution and production source/model gates
remain open. Next dependency-ready implementation:M8.2 measured serving.

Historical M8.1 acceptance follows.

M8.1 native CPU generation/export is accepted on699ff0b;GitHub42239279899 has
all14 jobs successful. CPU3.11/3.12/3.13 each735 passed,2 optional SDK skips,
7 GPU deselected;SDKs separately verified. True left-padded batches,private KV/RNG,greedy/seeded sampling,stream
snapshots,stop/context/output/cache limits and cooperative cancellation/timeouts.
Native safetensors/tokenizer/config/source snapshots reload without original
directories;no optimizer/RNG pickle. Developmental123904-parameter V395 export
has logit delta0 and matching cached/uncached/CLI/stream outputs. Pre-final full
local733 passed,no skips,7 GPU deselected in236.44s;final guards focused53 pass
in19.80s,Ruff/format,mypy163 and build pass. Existing Gloo issue is not declared
fixed by this one green local run. Clean source
2adb1ec750a1e03d4f859597edd7e4d344d7a8b948fdd648e76cdaa143b77bba passes the full
standalone workflow. Historical19338560-parameter RFC tiny export/reload delta0,
cache/full max delta2.861e-6,unchanged weights and matching fresh CLI generation.
Extracted-wheel import/generation matches on the same locked dependencies.
Current-source200-step actual-SIGKILL and all training/evaluation replays pass CI.
See validation_inference.md/runbook09. No model promotion/GPU/HTTP/quantization.
Next:M8.2 measured serving interface;quantization/external runtime gates stay open.

Historical M7.2 acceptance follows.

M7.2 matched-lineage CPU study and necessary promotion controls are implemented.
Preregistered89a0350:shared training/tokenizer/SFT/pairs,initialization seeds
20261105/6/7,base3/SFT80/DPO40/GRPO8 and a DPO half-LR arm. New strict compare
CLI verifies actual parents,seeds,data/config/source and one-factor ablation;
paired family/robustness/general/verifier/resource gates never auto-promote.
Focused35 checks pass including actual one-seed training/analysis and fresh-process
CPU measurements. First one-seed diagnostic rejects all candidates;DPO regresses
5/200 to0/200. Full local688 passed,1 existing packed-DDP timeout,7 GPU deselected
in311.24s;Ruff/format,mypy155 files and existing build pass. Three-seed actual
results:base0/0/0,SFT5/1/0,DPO0/0/3,half-LR5/0/0,GRPO5/1/0 of200 each;all
candidates reject. Repeated15-model analysis gives exact predictions/metrics/
audits/decisions;inputs unchanged. Additional controlled RoPE1/2 ablation gives
SFT5/1/0 versus1/0/0,also reject. Initial weights/data order/target budgets match
within paired arms. Runtime seed couples initialization and sampler across runs.
Package9ca8a8fd993965196dbaaaca1f1d8b4f22a13339644b3c73f784fa7ad5601195;
sourcee2c628f GitHub42100151962 has all13 CI jobs successful. CPU3.11/3.12/3.13
each689 passed,2 optional SDK skips,7 GPU deselected;SDKs independently verified.
Matched2-seed/RoPE2-seed studies and200-step actual-SIGKILL/SFT/DPO/GRPO replays
all pass. Local packed-DDP timeout is not fixed by remote success. Production
acceptance still needs actual user-run small/GPU and independent licensed evidence.
Next independent implementation:M8.1 inference/export. See validation_promotion.md
and runbook08 for the necessary gates and observed seed/RSS limits.

Historical M7.1 acceptance follows.

M7.1 original CPU evaluation is accepted on source83a1adb;GitHub42092305795 has
all12 jobs successful. CPU3.11/3.12/3.13 each664 passed,2 optional SDK skips,
7 GPU deselected;SDKs separately verified. Versioned exact/numeric/choice/LM protocols,seeded sampling/pass@k,
manifest-bound reports and declared-source lexical audits. Prospective1000-case
freeze88db69e:200 validation/800 unscored final,seed20261105,SymPy gold verification.
Focused42 tests pass;Ruff/format,mypy146 files and build pass. Full local665 passed,
1 existing packed-DDP resume timeout,7 GPU deselected in281.85s. Historical SFT80
scores5/200 with one constant answer,not a reasoning gain;repeat outputs/metrics/
audits exact,weights unchanged. See validation_evaluation.md,runbook07 and ADR0045.
Paired evaluation outputs/metrics/audits exact on both local and CI hosts;weights
unchanged,0 final cases scored. Dedicated200-step actual-SIGKILL and SFT/DPO/GRPO
replays also pass. Source0f742e0d9c94a3165f9fc2205d894a9a58a0e2c0a659a5d149e2fa2ac09209fd.
Local Gloo timeout remains unresolved;remote success is not a local fix.
Next:M7.2 controlled comparisons/promotion and ablation gates;demo checkpoints
cannot justify production promotion. Public rights/GPU/user-server gates remain open.

Historical M6.3 acceptance follows.


M6.3 bounded CPU outcome-GRPO is implemented from the SFT branch. Actual8 real
policy-gradient/Adam updates,745 generated actions,11/32 constant groups;stop4/
resume8 all losses,rollouts,weights,Adam,RNG and prompt cursor exact. Reduced
127104-parameter model,V445/T128,P4/G4,max16 actions,temperature1,LR1e-6,KL0.01.
Validation1/8 before/after:engineering acceptance,not reasoning improvement.
Final-source combined DPO/GRPO33 tests pass in22.23s;Ruff/format,mypy135 files pass.
Pre-final full622 passed,1 numeric-DDP timeout,7 GPU deselected in258.63s;root cause
of the intermittent local Gloo problem remains unresolved. Current-source38a74bf GitHub42081751910:all11 jobs successful. CPU3.11/3.12/
3.13 each622 passed,2 optional SDK skips,7 GPU deselected;SDKs verified separately.
Dedicated GRPO,DPO,SFT and200-step SIGKILL replays pass. Source4775afbb264baa8c842cf40b67e0372162ee4e690f1c3738e227db42e7136b1b.
See validation_grpo.md and runbooks/06_reasoning_rl.md. Next:M7.1 evaluation harness,
with original fixture scope first;production/public benchmark rights still blocked.

Historical M6.2 acceptance follows.


M6.2 original CPU DPO is implemented after accepted Phase5. Full CPU/optional-SDK
614 passed,no skips,7 GPU deselected in158.37s;focused23 passed;Ruff/format,mypy128
files and build pass. Actual40 updates/80 pairs/1582 targets,127104 parameters,
V445/T128,beta0.1,LR0.001;stop13/resume40 all losses/weights/Adam/RNG/cursor exact.
Train DPO0.693147->0.352214,validation0.693147->0.843191:negative heldout result,
not a quality promotion. Frozen reference is byte-verified SFT and checked unchanged.
Source c3c18aa16e68110c5cb7e4b9750773a942d822984d537307e87f9b5f666dda08;
Current-source3b3ef4a GitHub42073726045:all10 jobs successful. CPU3.11/3.12/3.13
each612 passed,2 optional SDK skips,7 GPU deselections;SDKs separately green.
Both40-step DPO and200-step SIGKILL replays are exact independently on CI.
See validation_preferences.md and runbooks/05_preferences.md.
M6.3 readiness control is executed:base0/16,SFT3/16,DPO0/16 complete correct
answers at identical16-token greedy budget on original validation-only problems.
Use SFT for a future bounded engineering RL check;DPO is rejected for promotion.
The control is a necessary tiny check,not proof of general reasoning.
Reproduction:reasoning_rl_readiness.md and scripts/preference_generation_control.py.
Hardware/source gates remain separate;the subsequent GRPO prototype is recorded above.

Historical Phase5 acceptance follows.


Expanded Phase5 is implemented selectively on top of SFT. Actual full RFC tiny
19,338,560 parameters,V32768,T16/B2:200 optimizer updates/6386 valid targets.
SIGKILL89,restore75;all200 losses differ by0.0;weights/Adam/RNG/application match
bit-exactly. Final source f14b266,digest
ba3bb6d89d5aa74dbea7a80d3a1f43e1fdfab309ac8a0f04d0c635859de45e50.
Local full run586 passed,3 distributed-process failures/timeouts,7 GPU deselected
in365.00s;do not call it green. Final200-step repeat,compile and four-size measurements pass. Ruff/format,
mypy122 files,frozen lock/dependencies and build pass. GitHub42069685890 atf14b266:all9 jobs successful,CPU3.11/3.12/3.13 each
589 passed,2 optional SDK skips,7 GPU deselected;SDKs separately green. Dedicated
200-update hard-kill replay independently bit-exact. Final local restart run590
passed,1 packed-DDP timeout,7 GPU deselected in318.63s;Gloo root cause still open.
Portable Phase5 closed;M6.2 was the next milestone and is now accepted above.
See validation_phase5.md and runbooks/02_pretraining.md. No GPU/full small/paid run.

Historical Phase4 acceptance follows.


Selective Phase4 closure is implemented on top of M6.1:RoPE scale(default1),FP64
reference numerics,whole-model gradcheck,random-length scaled cache,compile/depth/
overfit/RFC-count tests and operator profiling.22 focused tests passed in63.36s;
final profiler subset3 passed in3.36s;Ruff/format,mypy105 files and build pass.
Default FP32/BF16 logits/gradients/weights match pre-import seeded values exactly.
Final local regression:562 passed,1 existing numeric-DDP exit SIGABRT,5 GPU
deselected in104.85s. Packed replay/model/SFT tests pass;root cause of the local
DDP process-exit fault remains unresolved,also observed before this import.
Corrected-source GitHub41909315717 atfee1e3d has all8 jobs successful:
CPU3.11/3.12/3.13 each561 passed,2 optional SDK skips,5 GPU deselections;optional
metrics separately3 passed. Numeric/data/packed/SFT smokes are green. CPU3.11's
initial missing setuptools was fixed by pinning already-locked84.2.1 in dev;bare
local3.11.17 Inductor without SDKs passes in44.92s. No dependency versions/hashes
upgraded. M4.3 closes for verified CPU scope;GPU remains unverified.
Do not call the local full suite green. See validation_phase4_closure.md,phase4_import_review.md and
new docs/model.md. Supplied old state/profile claims were not imported.


M6.1 at0abfcd3 is complete for original demo data/single-process CPU: strict chat formatting,
assistant-only masks,whole-dialog packing/truncation,licensed family-heldout data,
pinned base weights and atomic SFT replay. Full CPU/optional-SDK suite:543 passed,
5 GPU deselected,no skips in97.32s;Ruff/format,mypy103 files,build/lock/dependency
and hook checks pass. Fixed69-target train probe5.89794546->0.14195556;
16-target heldout probe5.81094980->4.76135039;80 updates/2520 assistant targets.
Stop26/resume80 weights/Adam/RNG/consumer match exactly. Final test execution uses
GLOO_SOCKET_IFNAME=lo,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1 after earlier distributed
process-exit stalls;root cause of those stalls is not established. See R016,
docs/development/validation_sft.md and runbooks/04_supervised_finetuning.md.
M6.1 remote GitHub41905361820 at0abfcd3 has all8 jobs successful.
M6.2 portable implementation is recorded above. No reasoning-accuracy claim,
production corpus,external teacher,paid run or GPU/distributed-SFT validation.


M4.1/available-CPU M4.2 and M5.1 are complete. Implementation557e4da; actual
GitHub run41290979913 has all7 jobs successful, including CPU3.11/3.12/3.13 and
real packed BPE optimization/replay. M5.2 CPU/DDP portion is verified; CUDA/FSDP/
shared-store/multi-node tiers remain blocked by hardware validation. Production
sources/benchmarks remain blocked in M3.3. Publication to main is active. No final
model/paid training or old cloud-environment configuration change was performed.

## Last completed work and evidence

- Portable RFC decoder:all four actual meta counts,tiny CPU backward,independent
  value/gradient/causality/mask oracles,seeded residual initialization and combined
  tied-embedding gradients. Constructors explicitly initialize FP32 parameters
  even under ambient float64. Non-finite losses stop before optimizer/checkpoint.
- Reference/SDPA runtime and non-reentrant block recomputation; compact explicit
  request-local KV-cache with packed/multi-token/full parity,stale-state checks,
  weak ownership and CPU BF16 coverage. CPU graph/storage profile recorded.
- Actual packed-data trainer with token-sum accumulation/partial windows,valid-token
  LR schedule,heldout probe,clipping/throughput and actual data/BPE canonical IDs.
  Atomic checkpoint stores model,Adam,RNG and each rank's committed loader state.
  Split-bound fingerprints and worker/prefetch/mount-safe replay are tested.
- Actual two-rank Gloo uneven-count gradient/global-batch reference,coordinated
  rank-local failure and SDPA+checkpoint optimizer/cursor/RNG replay. Numeric
  engineering trainer and CPU DCP/DTensor checks remain intact.
- Fresh clone:522 tests passed,2 optional SDK skips,5 GPU deselections in122.79s;
  Ruff/format and mypy97 files pass;frozen dependency check and wheel/sdist build
  succeed. Remote3.11/3.12/3.13 each522 passed;optional SDK job independently green.
- Clean actual-BPE smoke:127104 parameters,20 updates,40 microbatches,4959 valid
  targets,heldout256-target LM probe6.1267035->5.7565165. Stop6/resume20 has exact
  weights/Adam/consumer/RNG state. This is original demo language modeling,not a
  public benchmark or mathematical-answer-quality result.
- Existing tokenizer/curation/shards/weighted loading/exact math generators remain
  verified. No external training corpus/teacher;GSM8K is exclusion/evaluation only.

Evidence:docs/development/validation_pretraining.md and its committed receipts;
EXPERIMENTS R012–R014,runbooks/01_data_preparation.md,02_model_execution.md and
03_packed_pretraining.md. Earlier foundation/data evidence remains separate.

## Open limits

Five supplied GPU tests did not run. No CUDA/NCCL,packed FSDP,real laptop/shared-
store/multi-node validation or final-scale memory/throughput measurement. The
functional cache copies appended state;graph-storage bytes are not allocator peak.
Exact replay is limited to the documented source/software/topology/hardware scope;
unsafe .data/cache mutation and arbitrary uploaded optimizer files are unsupported.

Full checksum preflight requires input I/O budget. HF BPE pair statistics can grow
with corpus diversity. LID/PII/toxicity/LSH/lexical exclusion need real-source audits;
no trained quality classifier. MATH/AIME/MMLU and frozen project-final exclusions
are incomplete,so production preparation remains blocked. The heldout LM probe is
weighted/limited and may reuse source blocks. Reasoning RL,public evaluation and
serving/export beyond safetensors are still future milestones.

## Next three tasks

1. M7.1:build the versioned evaluation harness on original fixtures first;preserve
   fixed budgets,invalid/truncated counts and benchmark/source clearance gates.
2. M5.2–M5.3:actual CUDA/DDP/FSDP/shared-store/multi-node and resource validation
   when hardware is supplied;no costly/final-scale run without authorization.
3. M3.3:clear exact training/benchmark rights,freeze the distinct project final
   heldout and calibrate corpus-scale quality/dedup/PII/semantic-leakage audits.

## Reproduce

```bash
make setup
make lint
make typecheck
make test
mkdir -p runs
uv run --frozen --extra cpu python scripts/pretraining_smoke.py --directory runs/bpe-smoke-new --repository . --steps 20
```

Use a new output path,uv0.12.23,Linux/Python3.11–3.13. CPU is default. Operator
commands/actual-data identities/schedule/replay limitations are in runbook03.

## Latest session — Phase10 protocol intake, 2026-11-07

Base main:`fe5ee52ad8946b7f0c436d8f2e2e67a6ff5e87bf`. Owner requested recurring
analysis after each real training run. [ADR-0062](DECISIONS.md#adr-0062-evidence-gated-improvement-after-each-real-training-run)
and [runbook10](runbooks/10_continuous_improvement.md) record the protocol and
existing log-archive command. No new real-run artifacts/config/eval were supplied;
diagnosis, cause attribution, ranked experiments and exact next-training commands
remain blocked. Do not substitute historical demo results for owner training.

Only workflow documentation and intake status change. No ML experiment/result,
code/config/data/runtime change or paid run. Existing historical records are
preserved; pending Step A amendment review/Step B gates are not bypassed.
Documentation/link checks must pass before committing this protocol update.

Phase10 intake validation:2 documentation-tool tests passed in0.13s; strict
MkDocs build passed;81 local links/anchors and ADR0001..0062 sequence checked.
All four existing files are append-only; the sole new file is runbook10.
git diff --check and docs-only scope checks passed. The existing summarizer's
--help executed and confirmed directory/--archive arguments. No real-run archive,
loss analysis, training/eval execution, MFU measurement or scale claim was made.

## Previous task — Step B tokenizer v3 codec, 2026-11-08

Owner continuation selected the next reversible implementation step from main
09446b446830a2e8b38bb1a5a696db1eea8116eb. [ADR-0063](DECISIONS.md#adr-0063-explicit-v3-codec-selection-without-changing-legacy-identities)
records explicit v3 settings/recipe2, the128-control table, strict loading and
source/recipe linkage. V remains configurable/undecided; bounds384..1048576 are
request ceilings, not proof of full million-entry training or a production choice.
JSON/doc limits remain64MiB/1MiB; v3 oversized serialization fails before writing.
Existing v1/v2 control maps, minimal payload hashes, default settings and old
recipe serialization/digest are preserved. Existing artifacts/configs/checkpoint
formats/lockfile were not rewritten. Source changes still invalidate old exact
resume under the existing compatibility rules.

Approved local fixture:actual V512 and102720-parameter CPU decoder/Engine;
finite gradients, full/cache max logit difference3.5762786865234375e-7 and repeat
four-token plain generation. Optimizer updates0; no quality/model-release claim.
V3 JSON payload hash75635b1c4b439223e90e45ba7bf8e414c0251fdeea71db12be7056c55b5133bb
is stable across three independent hash-seed/thread-count settings. Creation-time
bundle manifest hashes are not promised equal. See EXPERIMENTS for scope/source.

Legacy chat/SFT/eval-chat/HF-chat and native snapshot export reject v3 explicitly;
packed-data version guards remain. No schema3, reporting fix, tool execution,
effort behaviour or capabilities endpoint was implemented. STOP for owner review
before Step C. Named ML/legal/production freeze and GPU gates remain in force;
the Phase10 loop still awaits real-run inputs.

Step B final validation:focused tokenizer/recipe/bundle/CLI/model set263 passed
in8.07s. All unit tests plus selected tokenizer/native/HF/SFT/safety/API/eval
integrations939 passed,2 GPU deselected in72.47s; this is not the full distributed
matrix. Ruff check/format pass240 files,strict mypy240 files passes. Strict MkDocs
build passes;91 local document links/anchors and ADR0001..0063 sequence verified;
documented recipe parses and git diff --check passes. No configured corpus,
tracked fixture, uv.lock, checkpoint/tensor artifact or final V was changed.
Native v3 snapshot/server deployment, large attained vocabularies, CUDA/Gloo/DDP
recovery and real model training were not executed for this step. Stop for review.

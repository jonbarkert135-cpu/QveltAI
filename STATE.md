# Qvelt current state

Updated:2026-11-04. Owner:Lead agent.

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

## Current task

Expanded Phase5 is implemented selectively on top of SFT. Actual full RFC tiny
19,338,560 parameters,V32768,T16/B2:200 optimizer updates/6386 valid targets.
SIGKILL89,restore75;all200 losses differ by0.0;weights/Adam/RNG/application match
bit-exactly. Source digest0c5afe9608d5c5df95cc0a308a80649ac936d8b4910050725e8f481cc3ddb114.
Local full run586 passed,3 distributed-process failures/timeouts,7 GPU deselected
in365.00s;do not call it green. Current clean GitHub CI and compile/scaling results
are pending. See validation_phase5.md and runbooks/02_pretraining.md. No GPU/full
small/paid run. Next dependency-ready portable task remains M6.2.

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
Next portable milestone:M6.2 preference optimization. No reasoning-accuracy claim,
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
weighted/limited and may reuse source blocks. DPO/RL,public evaluation and
serving/export beyond safetensors are still future milestones.

## Next three tasks

1. M6.2:select and verify a preference objective,register original licensed
   preference pairs,and implement objective/gradient and optimization/replay checks.
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

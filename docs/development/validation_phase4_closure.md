# Selective Phase4 closure: current acceptance

Date:2026-11-04. Starting revision8f16a9c. The user supplied a patch/archive
against260b431; selected original project source/tests/docs were adapted on top of
current M6.1. Selection, rejected items and input checksums are in
[import review](phase4_import_review.md); architecture remains RFC-001.

## Kept and adapted

- RoPE linear-interpolation hook with finite scale>=1, default1 exact baseline.
- At-least-FP32 RMSNorm/RoPE/reference attention, preserving FP64 for gradcheck.
  The training loss intentionally retains its existing FP32 contract.
- Whole-model FP64 parameter gradcheck for reference and SDPA, seeded training
  determinism,30 random-length/chunked/packed KV trials per backend/scale1 or3,
  actual19.34M tiny-model single-batch overfit, residual-depth variance/gradient
  scaling, AOT compilation parity across backends/recomputation, CPU Inductor
  parity and fullgraph block capture, RFC-derived parameter/FLOP table checks.
- Operator forward/backward profiler, valid finite peak-FLOP/context argument
  checks and math documentation in docs/model.md. No optimizer work is timed.

Imported tests now restore RNG/thread state. Non-default RoPE cache parity was
added. Statements about GPU kernel selection were narrowed to unverified dispatch.
Historical uploaded state/results/profile were not taken as current acceptance.
Existing memory,source licensing,SFT and exact-resume contracts are retained.

## Executed evidence

Focused model/compile/profile tests:22 passed in63.36s. Both AOT and CPU Inductor
actually ran with GCC/C++14.2.0; this was not an unavailable-compiler skip.
Ruff/format and strict mypy:pass,105 first-party files. Wheel/sdist build:pass.
Full CPU/SFT/optional-SDK regression:562 passed,1 failed,5 GPU deselected in104.85s.
The remaining failure is the existing numeric two-rank DDP process exiting with
SIGABRT after reporting completed optimization. The packed two-rank replay and all
model/SFT/profiler tests pass in that run. USE_LIBUV=0 was tried via its verified
installed PyTorch rendezvous setting; it did not eliminate the exit fault. Earlier
pre-import sessions had distributed exit stalls too. Root cause is unresolved;
this is not an all-green local full-suite claim. Remote acceptance is pending.

Before applying source changes, saved seeded weights/logits/gradients for both
backends in FP32 and CPU BF16 on a packed,padded B2/T10 case with hidden32/two
blocks. After applying the selected patch, all four combinations were tensor-exact.
This is a bounded default-behavior regression comparison, not proof for all shapes,
compilers or devices. The FP64 path intentionally differs from old FP32 narrowing.

Current CPU operator profiles:tiny19,338,560 parameters,B2/T128,one thread,
2 warmup/3 measured steps:SDPA488.39,reference468.05,SDPA+recompute404.89 targets/s.
SDPA trace:aten::mm377.5ms self time,then fill_16.8ms and add15.0ms.
These short non-isolated runs are not a speed ranking; a stalled diagnostic worker
was present. CUDA peak memory and estimated MFU are null without hardware/peak data. Raw commands, versions, arguments,
source identity and measurements are in evidence/phase4-current-profile.json.
The user's old profile receipt is not that measurement.

## RFC map and deviations

Token embeddings, pre-RMSNorm, interleaved RoPE, contiguous GQA, bias-free SwiGLU,
tied head and scaled residual initialization match the selected equations. Existing
component/reference/gradient/causality/packed tests remain part of full acceptance.
`Decoder(load_config(...).model, execution)` is the existing config-to-model factory
for tiny/small/base/large; adding a second wrapper would duplicate it. Counts for
all four actual meta models agree with the RFC table; no large model was trained.

`rope_scale>1` is an explicitly non-baseline interpolation extension. Default1
retains baseline values. Packed execution uses a block-diagonal causal boolean
mask, not a newly validated fused-varlen kernel. Whole Decoder compilation permits
host-validation graph breaks; block capture is tested with fullgraph=True. QK-norm,
MoE,MLA,sliding window and an untied head were not selected and are not added.

Adding a model-config field changes config identities; code/source identities also
change. Old exact resume remains rejected rather than migrated. This is compatible
default parameter/value behavior, not a claim of old-run exact resume. The M6.1
80-step receipt remains historical evidence for0abfcd3; current SFT regressions run
as part of the full suite under the new config.

## Limits

No GPU is present. CUDA/BF16 hardware behavior, FlashAttention dispatch, GPU peak
memory/throughput and compiler support remain unverified; five supplied GPU cases
are excluded from CPU acceptance. CPU Inductor/AOT results do not validate CUDA
Inductor. Low single-batch loss and one-seed depth profiles are engineering checks,
not reasoning quality or a general initialization-stability theorem. The profiler
omits Adam updates,communication and allocator peak on CPU; recompute adds actual
work absent from the analytical FLOP convention. No paid or final-scale run.


The earlier diagnostic full run had561 passed/2 failed/5 GPU deselected in237.07s:
one packed-DDP timeout plus a missing profiler metadata field caught by its new
test. The field was fixed; the profiler subset then passed3 tests in3.36s. An
instrumented standalone DDP attempt also aborted after completing training; it is
diagnostic evidence only. No failing test was removed,relaxed or marked xfail.


## Remote compiler-tooling correction

Run41908101017 at886ce61 passed7/8 jobs,including both complete CPU3.12/3.13
jobs and all data/pretraining/SFT smokes. CPU3.11 failed only the Inductor test
with ModuleNotFoundError:setuptools (560 passed,2 optional SDK skips,5 GPU
deselected in110.45s). Torch's locked dependency marker supplies setuptools only
on Python>=3.12;optional SDK installation masked this locally. Explicitly pin
already-locked setuptools84.2.1 in dev(MIT,Python>=3.10);the lock changes dependency
edges,not package versions/hashes. Compile tests remain enabled. Matrix acceptance
for the corrected dependency is pending.

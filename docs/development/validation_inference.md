# M8.1 native CPU inference acceptance

Date:2026-11-06. Base0390b60. See[design](inference_design.md),ADR0047 and
[runbook09](../../runbooks/09_inference_export.md). CPU FP32 native reference;
model release,HTTP serving,GPU,quantization and external-runtime gates are separate.

## Original implementation and independent references

Native snapshot export copies exact verified model/tokenizer bytes with original
config/checkpoint/tokenizer manifests,CPU config,lock and explicit protocol.
The loader enforces source/config/vocabulary/dtype/shape/role/size invariants;
optimizer/RNG pickle is neither loaded nor exported. Evaluation's checkpoint
loader API reuses the same bounded lower loader,with existing trust checks kept.
Snapshot construction preserves caller RNG and normal parameter version counters
even under an ambient inference_mode context.

True padded batch generation owns private RNG/cache per stream,uses logical
positions and explicit masks,and masks finished slots. Private caches are freed
on completion,error or close;model changes fail. The engine disables ambient CPU
autocast for FP32 calls and holds no grad/autocast context across yields. Token
requests,raw text,contexts,cache/token/output limits and source refs are validated;
timeouts/cancellation are cooperative. Streams explicitly carry cumulative text
snapshots,not append-only Unicode deltas.

Independent naive unpadded full-prefix decoding and scalar exponentials validate
greedy/categorical decisions and probabilities. Padded cached logits agree with
each unpadded reference within rtol3e-5/atol3e-6. Actual stored tensors and SymPy
verify FP32 KV plus metadata `B*T*(8*L*kv_width+17)`. Tests cover split stop strings,
partial Unicode replacement/revision,finished rows,cancellation/deadlines,nonfinite
outputs,output/context/cache limits,global RNG/grad/autocast preservation and weakref
cache release. Protocol/dtype/version/role corruption and incompatible inputs
fail even when a descriptor has a newly valid file/manifest hash.

## Executed portable workflow

Developmental standalone CLI run uses an untrained123904-parameter,V395/context256
real BPE checkpoint from earlier approved original demo text. For greedy and
categorical T0.8,three unequal-length ASCII/Russian/emoji/accented prompts each
produce48 sampled actions. Cached/uncached tokens/text/stops,source/export results,
CLI/API and stream events agree. Original checkpoint/data directories are hidden
during fresh-process CLI reload;the snapshot still works. Export/reload logit
maximum delta0 on shapes[2,T,395] for T1/3/17/61. No optimizer update,reasoning
quality benchmark or final800-case scoring is performed.

Pre-final full local CPU/optional-SDK suite733 passed,no skips,7 GPU deselected
in236.44s,including two-rank tests. Final batch-entry guard strengthening is
covered by focused53 tests in19.80s;Ruff/format,mypy163 files and build pass.
Lock93/pip72 consistency checks pass. This successful local run is not evidence
that historical intermittent Gloo failures were diagnosed or fixed.

Final-source standalone receipts,full RFC-tiny read-only export diagnostic and
current-source14-job CI are recorded below on completion. No CUDA/serving/model
quality promotion or external-format compatibility is claimed.

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

## Final source and full RFC tiny

Source699ff0b,package
2adb1ec750a1e03d4f859597edd7e4d344d7a8b948fdd648e76cdaa143b77bba.
The clean-source standalone workflow repeats all CLI/API/cache/stream/source-hidden
checks,with zero export/reload logit difference;see
[portable receipt](evidence/inference-smoke.json). It remains an untrained
123904-parameter engineering fixture,not a new training/quality result.

Separate read-only diagnostic exports the historical200-update **19338560**-
parameter RFC tiny checkpoint,V32768. Original weights stay unchanged. Export/
reload logits match exactly on[2,17,32768];cached prefix3/suffix14 versus full
forward maximum difference2.86102294921875e-6,within rtol3e-5/atol3e-6. Batched
greedy cached/full tokens/text/stops and fresh-process CLI/API agree,8 sampled
actions across2 requests. No new optimizer update or final case is scored. See
[full tiny receipt](evidence/inference-rfc-tiny.json). This validates native
correctness for that existing checkpoint,not useful generation or GPU performance.

The built wheel contains all four inference modules. A subprocess outside the
checkout imports them from an extracted wheel,observes the same package source
digest and generates identical CLI results on the same locked dependencies;see
[wheel receipt](evidence/inference-wheel.json). This is not a clean OS installation
or cross-hardware guarantee. Wheel SHA
0c5064f5065b0ab3008a1c4978809c68df2f3b332bf7579af81bf404f0f2ede9.

## Final remote acceptance

[GitHub42239279899](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/42239279899)
has **all14 jobs successful** on699ff0b. CPU3.11/3.12/3.13 each735 passed,
2 optional SDK skips,7 GPU deselected in175.50s/267.12s/298.76s. SDK event/offline
tests pass separately;mypy163 and packaging pass. Dedicated native inference
matches all cached/full/export/CLI/stream outputs and reloads with original paths
hidden;see[CI inference receipt](evidence/inference-ci-smoke.json).
Current-source200-step actual-SIGKILL recovery has maximum loss difference0 and
exact weights/Adam/RNG/application. SFT,DPO,GRPO,evaluation and matched/architecture
study jobs pass. See[matrix/recovery receipt](evidence/inference-ci.json).

M8.1 native CPU core is accepted. Quantization/external runtimes remain explicitly
deferred;M8.2 serving is the next independent implementation. No production/model
promotion or fix for historical local Gloo instability is implied.

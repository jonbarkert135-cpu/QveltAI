# Phase8 inference/export/serving acceptance

Date:2026-11-07. Portable CPU reference;final source-clean evidence and CI closure
are recorded below after execution. No GPU or final-scale training was launched.

## Implemented and independently checked

The native Engine adds top-k/top-p/min-p and repetition/frequency/presence controls
without changing default sampler semantics. Independent scalar filter oracles and
cached/full/stream tests cover it. HTTP and GRPO consume the shared protocol.
Inference requests project only the last prompt logit;default full model forward
and cached full-logit references remain available. Cache remains copy-on-append,
not a claimed paged or continuously scheduled runtime.

Real offline Transformers5.1 AutoModel/AutoTokenizer load builtin Llama safetensors
with no remote code. Q/K row permutation converts native adjacent-pair to split-
half rotary math. Full and cached logits,ordinary/chat tokenization,tied head and
extended-position interpolation agree. Direct dense checkpoint conversion writes
indexed bounded shards without instantiating a full model;dimension-only tests
cover all RFC tiers. Actual large/DDP export and DCP consolidation are unverified.

HQQ0.2.8.post1 int4/int8 use the library's actual packing and CPU PyTorch path.
Independent dense reconstruction and standalone reload agree. Only numeric tensor
state is accepted;encoded HQQ metadata is rejected. A test replaces its unsafe
dtype decoder with a function that raises,proving this path never invokes it.
Embedding,tied head and norm remain FP32. Quantized storage does not imply faster
CPU execution or compatibility with external quantized serving formats.

FastAPI JSON/SSE tests exercise auth-before-body,schema/body/control rejection,
stop/UTF-8 chunks,stream concatenation,real concurrent model microbatches,private
seeds,queue admission/deadline/worker errors and shutdown. Real loopback HTTP
acceptance also measures server process VmHWM,latencies,completion throughput,
disconnect and graceful SIGTERM. Metrics describe a small CPU workload only.

## Diagnostics before source freeze

Separate focused sets:HF3 passed7.79s;sampling/context/variant/API25 passed12.63s;
final HQQ/safe-decoder/policy9 passed9.58s. These sets overlap;they are not summed
into a total. Ruff/format and mypy217 pass;wheel/sdist build passes. Frozen lock
and all installed optional packages are compatible. Initial lock check without
UV_CACHE_DIR failed on read-only home;the documented writable cache passes.

`runs/phase8-diagnostic-2` executes three debug training updates,HF factor1/2,
native HQQ4/8,six full native evaluator reports and real HTTP. All assertions pass.
This is a dirty development run,not final publication provenance. Source-clean
receipts and measured values are published separately after the code commit.
The retrieval diagnostic has no baseline ability;all factors are at the accuracy
floor. Context overflows decrease with available context,but that is not evidence
of preserved long-context quality. Final800 remains unscored.

First HTTP diagnostic caught a real acceptance-script mismatch:Uvicorn0.54
re-raises SIGTERM after lifespan shutdown,so code after uvicorn.run is not reached.
The app now records sanitized shutdown counters during lifespan cleanup;the probe
requires those counters and accepts normal Unix SIGTERM status only with cleanup
evidence. No forced-success exit was introduced.

## Containers and environmental failures

Immutable Python/uv images,frozen Python dependencies,signed dated Debian Git,
nonroot targets and source-only build context are implemented. Native serving
image built locally (image3e2cefc9d25efb5c97e1f643c8da571de6dc89bab0a583b0fcd87e6081bb7a39).
Its runtime launch failed ENOSPC:local Docker vfs duplicates root layers on the
small7.9GiB filesystem. Only this task's scratch image,large identified BuildKit
cache and untrained container diagnostic were removed to recover free space.
No user checkpoint/data was deleted. Local image runtime is not accepted.
Dedicated CI builds all targets and runs actual tiny training and HTTP/SSE on a
runner with sufficient disk;its final outcome must be recorded before closure.

Local HTTP/asyncio tests require execution network permission;without it a
thread/socket wakeup can hang even without an external request. The accepted
runs use explicit permission,not skipped or mocked asyncio behavior.

## Compatibility and remaining gates

vLLM0.33.0 at4ba787ec9959e566d3b587a796dff07673324da5 registry,Llama,
rotary and FlashAttention predicates were read. The exported dense architecture
maps structurally:GQA,tied head,split-half linear RoPE and eligible head sizes.
No vLLM/GPU runtime was executed. Native/HQQ remain single-process CPU FP32<=20M,
batch<=8,context<=2048. GPU kernels,continuous/paged batching,large quantization,
real trained retrieval quality,production auth/TLS/tenant isolation and blocked
release benchmark coverage remain separate gates. See [runbook05](../../runbooks/05_serving.md)
and [model card](../../MODEL_CARD.md). No model-quality promotion is claimed.

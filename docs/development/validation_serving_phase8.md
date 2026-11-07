# Phase8 inference/export/serving acceptance

Date:2026-11-07. Portable CPU reference accepted;source-clean evidence and CI
closure are recorded below. No GPU or final-scale training was launched.

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

## Clean source-bound execution

Source commit:b5deaf40d161a260be50cfe135b67dda479ccd7c. Runtime source SHA:
307139e86588377d9e01eded8d156b7534c0cea13722901e9b81f9a97fbaf061.
Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/inference_phase8_smoke.py --directory runs/phase8-acceptance --repository .
```

[Acceptance](evidence/serving-phase8-acceptance.json) records actual identities,
config,HF errors,HQQ payload hashes and the HTTP workload. All seven source run/
export artifacts and six evaluator reports have dirty=false and the exact source
commit. Checkpoint weights are unchanged by evaluation;final_cases_scored=0.
Model:123904 parameters,vocab395,layers2/hidden64/head4/KV2/FF192,context256;
three original BPE optimizer updates. No trained assistant quality is implied.

HF full/cache maximum absolute errors are1.7881393432617188e-7/1.341104507446289e-7
at factor1,and2.384185791015625e-7/1.7881393432617188e-7 at factor2. Factor2
includes positions beyond the original context;all explicit numerical tolerances
pass. Native sampled cached/uncached/stream/repeated outputs agree.

Matched LM scores:FP32 PPL374.9324301437019;HQQ4 PPL375.0238753703861
(delta+0.09144522668424315);HQQ8 PPL374.9141437742822
(delta-0.01828636941968398). Two original LM documents/56 target tokens only.
Generation and choice accuracy are0 for each variant,so zero accuracy delta is
not evidence of preserved useful accuracy. Quantized numeric tensor payloads are
176128/225280 bytes for int4/int8;the FP32 parameter payload is495616 bytes.
These are storage payloads,not process RSS or acceleration claims. All values,
exact jobs/checkpoint hashes,audits and deltas are in generated [RESULTS](../../RESULTS.md)
and six native evaluation-phase8 receipts. Nine prior Phase7 receipts are retained;
the complete15-receipt ledger regenerates/checks exactly.

Needles:all factors1/2/4 score0/30. Context-overflow failures12/6/0 respectively
(length512+1024;length1024;none). Each executed attempt hits the eight-token limit.
Per-length/depth domain slices and failures remain in the receipts. This proves
expanded execution coverage,not retrieval quality or a degradation threshold.

Actual local loopback workload:AMD EPYC9V74 80-Core Processor,CPU FP32,Torch
threads1,one server worker,warmup1,requests8,concurrency4,max_new16,seed20261107.
109 completion tokens/0.291437353s =374.0083379082774tokens/s;request P50
0.119224917s,P95(nearest rank)0.233495560s. First SSE content0.020915994s;
isolated server lifetime peak VmHWM264736768 bytes,including startup. Microbatch
sizes include3 and4. Streaming text equals JSON;disconnect/expiry counter1,
shutdown pending0,SIGTERM return-15 after explicit stopped evidence. A tiny short
loopback sample is not a capacity SLA,and hardware/container results differ.

Full local installed-extras suite:864 passed,7 GPU deselected in331.00s,no skips
or failures. Final safe-decoder test extension is additionally covered by the
focused9-test run. Ruff/format,mypy217,lock/pip,build,inventory,ledger and changed
guide links pass. No repeated full suite was needed after docs/evidence changes.

## Actual container CI acceptance

[Source CI42356828424](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/42356828424)
container job128356624104 succeeds. All three targets build;quantized image imports
actual HQQ/Transformers/FastAPI. Training/serving defaults UID10001. Three real
training updates commit a verified checkpoint inside the image;authenticated
JSON/SSE agree,missing auth401,graceful shutdown pending0.

[Container receipt](evidence/serving-phase8-container.json) includes immutable image
IDs,source commit,checkpoint/lock hashes and GitHub archive identity. Downloaded
archive13567369659 was SHA256-checked against GitHub's registered digest before
reading its result. This is actual runtime acceptance on Ubuntu24.04 hosted
Docker;it does not erase the retained local vfs ENOSPC failure. The separate
inference-serving CI job passes34 tests19.05s and its entire source-clean smoke.
Final CPU matrix/recovery status follows below once completed.

## Source CI closure

All19 jobs in source CI42356828424 complete successfully,including every prior
SFT/DPO/GRPO/evaluation/promotion/native-inference workflow. [CI receipt](evidence/serving-phase8-ci.json)
records job IDs and actual test counts. CPU3.11/3.12/3.13 each846 pass,18 optional
skips,7 GPU deselected in216.69s/388.65s/332.56s. Optional skips are unavailable
SDKs/HF/HQQ/HTTP/data adapters in the default CPU install;dedicated jobs exercise
the actual optional implementations. No test suppression or forced-success exit.

The full RFC tiny19338560-parameter200-update recovery smoke remains exact after
this runtime source change:actual SIGKILL at89,resume checkpoint75,maximum loss
curve difference0,model/Adam/RNG/application exact. This is recovery correctness,
not an assistant quality score. Historical Gloo failures remain an open risk;
this successful source matrix is not a root-cause diagnosis.

An additional real CLI process serves the clean int4 artifact:authenticated JSON
and SSE text agree,shutdown pending0. The final test-only extension covers explicit
historical reasoning in both HF chat and native chat-prefix framing:10 tests pass
7.75s. Runtime source is unchanged from b5deaf4;docs/evidence/tests are the closure.

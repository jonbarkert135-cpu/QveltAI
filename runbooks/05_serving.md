# Inference,HF export,weight-only quantization and serving

The native server is a CPU FP32 research reference,not a high-throughput GPU
deployment. Native/HQQ loading is capped at20M parameters,context2048,batch8;
the direct dense-checkpoint HF exporter separately supports the RFC model tiers
through large without constructing a decoder. Read the [design](../docs/development/serving_phase8_design.md),
[native artifact contract](09_inference_export.md) and [model card](../MODEL_CARD.md).

## Reproduce the complete portable acceptance

From the repository root,on Linux with Python3.11–3.13 and pinned uv0.12.23:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
make EXTRA_ARGS="--extra hf --extra quantization --extra serving" setup
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu --extra hf \
  --extra quantization --extra serving python scripts/inference_phase8_smoke.py \
  --directory runs/serving-01 --repository .
```

Use a new output directory;exclusive artifacts are never overwritten. This
creates original approved data,three debug optimizer updates,verified native/HF/
HQQ snapshots,six native eval receipts,a local RESULTS and a real HTTP workload.
It checks HF full/cached logits,extended positions,tokenization,tied head,
sampled cached/uncached/stream replay,quantized payload identities,JSON/SSE text
parity,concurrent microbatches,client disconnect and SIGTERM cleanup. The HTTP
probe creates an ephemeral bearer key without logging its value. Final800 is
not read. No external data download,teacher call or GPU is needed.

The optional correctness set is also executable separately:

```bash
uv run --frozen --extra cpu --extra hf --extra quantization --extra serving \
  pytest -q tests/unit/test_sampling_controls.py tests/unit/test_long_context.py \
  tests/unit/test_serving_policy.py tests/integration/test_huggingface_export.py \
  tests/integration/test_quantization.py tests/integration/test_serving.py \
  tests/integration/test_inference_eval_variants.py
```

Managed sandboxes require execution network permission for loopback HTTP and
asyncio thread/socket wakeups,even when no internet is used. An unavailable
socket permission is an environment failure,not a passing API test.

## Run the local OpenAI-compatible API

After the smoke,create a relocatable server policy with the exact verified hash:

```bash
uv run --frozen --extra cpu python - <<'PY'
from dataclasses import asdict
from pathlib import Path
import yaml
from qvelt.eval.config import ArtifactRef
from qvelt.inference.serving_config import ServingConfig
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest
path = Path('runs/serving-01/snapshot/manifest.json').resolve()
manifest = ArtifactStore(path.parent).read_manifest()
policy = ServingConfig(ArtifactRef(str(path), manifest_digest(manifest)))
with Path('runs/serving-01/local-server.yaml').open('x') as stream:
    stream.write(yaml.safe_dump(asdict(policy)))
PY
# Bind QVELT_API_KEY privately from your secret store;never commit or paste it.
uv run --frozen --extra cpu --extra serving qvelt serve runs/serving-01/local-server.yaml
```

Default binding127.0.0.1:8000,one CPU model worker,one Torch thread,queue16,
microbatch wait5ms,post-validation queue+generation deadline30s,body64KiB.
`allow_unauthenticated:true` is an explicit loopback-only development option.
Binding0.0.0.0 requires bearer auth and an appropriate network/reverse-proxy
boundary. The app does not implement TLS,request-body read deadlines,rate
limiting,moderation or tenant isolation. Kernel calls cannot be preempted;cancel/
deadline checks run between model steps. Keep a reverse proxy body/read timeout.

```bash
curl --fail http://127.0.0.1:8000/readyz
curl --fail http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $QVELT_API_KEY"
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $QVELT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qvelt","messages":[{"role":"user","content":"2+2?"}],"max_tokens":16,"temperature":0,"seed":20261107}'
curl --no-buffer --fail http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $QVELT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qvelt","messages":[{"role":"user","content":"Привет!"}],"max_tokens":16,"stream":true,"stream_options":{"include_usage":true}}'
```

Supported:text messages,optional first system,multi-turn alternating user/
assistant,optional historical assistant `reasoning` with tokenizer-v2,n1,
max_tokens or max_completion_tokens,seed,temperature,top_k,top_p,min_p,
repetition_penalty,frequency_penalty,presence_penalty,up to four stop strings.
Tools,multimodal content,logprobs and unknown fields are rejected. Reserved
control spellings in chat content are rejected to preserve trained framing.

Sampling order:sign-aware repetition penalty on prompt+generated history;
frequency/presence penalties on generated IDs;temperature;top-k;nucleus;min-p
relative to retained peak probability;categorical draw with request-private RNG.
Greedy applies penalties then argmax;probabilistic filters do not affect greedy.
Top-k threshold ties are retained;nucleus keeps the first mass-crossing token.
Seeds are reproducible within the tested device/dtype/version/batch layout;
cross-hardware or arbitrary batch regrouping bitwise identity is not promised.

Responses use standard chat completion/usage objects. SSE emits role,append-only
text,finish,optional usage and `[DONE]`. Partial UTF-8 and possible stop prefixes
are withheld;concatenation equals nonstreaming text. Empty generated text is valid.
Error status400 invalid input,401 auth,404 model,413 body,429 capacity with Retry-After,
503 unavailable worker,504 deadline,500 sanitized worker failure. Streaming errors
after headers are SSE error events. Health/readiness are unauthenticated.
SIGTERM releases pending work and emits only bounded operational counters.
Pinned Uvicorn re-raises SIGTERM after graceful cleanup;Unix exit143/-15 alone
is not evidence of failed cleanup. Check the final stopped/pending summary.

## HF export and direct larger-checkpoint conversion

For the native smoke snapshot,use its canonical SHA from `qvelt manifest verify`:

```bash
uv run --frozen --extra cpu --extra hf qvelt export-hf NATIVE_MANIFEST \
  --sha256 NATIVE_CANONICAL_SHA --directory runs/hf-export-01 --repository .
```

For an actual small/base/large **dense** training checkpoint:

```bash
uv run --frozen --extra cpu --extra hf qvelt export-hf CHECKPOINT_MANIFEST \
  --sha256 CHECKPOINT_CANONICAL_SHA --tokenizer-path TOKENIZER_MANIFEST \
  --tokenizer-sha256 TOKENIZER_CANONICAL_SHA --directory runs/hf-small-01 --repository .
```

All registered bytes and config/tokenizer identities are verified. Optimizer/RNG
pickle is never deserialized. Exact dense keys,FP32 shapes/finiteness and vocab
are required;unmerged adapters and DCP/FSDP shards fail explicitly. Consolidate
sharded state into a separately verified dense artifact before conversion.
DDP replicated dense model payloads are accepted. Conversion iterates tensors and
writes HF-indexed shards,target128MiB;an individual larger tensor occupies one
shard. Memory is bounded by active tensors/shard rather than a full decoder plus
a full serialized checkpoint. Disk/RAM for real large conversion remain required;
only debug numerical execution and dimension-only tier counts are tested here.

Exports are builtin `LlamaForCausalLM`,safetensors,config,generation config,fast
tokenizer/chat template,source manifest,native config,lock and notices. Adjacent
native rotary pairs require Q/K row permutation for HF split-half RoPE. Head stays
tied. Standard offline loading needs no remote code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('runs/hf-small-01',local_files_only=True,trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained('runs/hf-small-01',local_files_only=True,trust_remote_code=False)
```

Ordinary `add_special_tokens=True` prepends BOS;chat already inserts explicit BOS/
roles/end-turn/EOS. Literal special spellings parse conventionally in HF plain text;
use the validated chat template for chat,and `add_special_tokens=False` for exact
native ordinary-text encoding. Native tokenizer bytes/behavior are unchanged.

## HQQ int4/int8 and measured degradation

```bash
uv run --frozen --extra cpu --extra quantization qvelt quantize NATIVE_MANIFEST \
  --sha256 NATIVE_CANONICAL_SHA --bits 4 --directory runs/hqq4-01 --repository .
uv run --frozen --extra cpu --extra quantization qvelt quantize NATIVE_MANIFEST \
  --sha256 NATIVE_CANONICAL_SHA --bits 8 --directory runs/hqq8-01 --repository .
```

Set serving `snapshot_format:hqq`,the corresponding manifest path/hash,and install
both serving and quantization extras. HQQ0.2.8.post1 quantizes linear projections
with group32/axis1;embedding,tied head and norm gains remain FP32. CPU PyTorch
dequantize/matmul is portable correctness,not a latency optimization. Native HQQ
format is deliberately **not** advertised as a standard HF/vLLM quantized format.
The loader restores numeric safetensors against trusted regenerated metadata;
it never invokes HQQ's encoded-dtype loader,which uses Python eval.

The smoke evaluates FP32/int4/int8 on identical source weights,prompts,LM targets,
settings and evaluator source;quantized report payload SHA must equal the exported
artifact's actual bytes. [RESULTS](../RESULTS.md) derives matched PPL/accuracy
deltas. Tiny fixture results cannot predict degradation on a trained small model.
Large/GPU quantization throughput and quality require separate measurements.

## Linear RoPE extension and needle evaluation

RFC linear interpolation maps position p to p/factor;derived model context and
rope_scale both multiply by factor. HF CLI `--context-extension 2` or4 performs
that explicit conversion. EvalJob v2 `context_extension:2` uses the same native
method. Native CPU context is capped2048;an HF export may describe larger contexts,
but that config alone establishes neither memory support nor model quality.

The smoke uses the **same unmodified three-update checkpoint** for factors1/2/4,
30 original digit needles:length64/128/256/512/1024,depth10/50/90%,two per cell,
eight generated tokens. `needles/construction.json` records actual tokenizer
budgets and needle offsets. Evaluation failures count in the denominator. Six
native receipts include the matched quantization and needle results,with per-cell
domain slices and contamination audits. The debug checkpoint has no baseline
retrieval ability;zero accuracy at all factors cannot show preserved quality or
locate an accuracy degradation threshold. Context-overflow counts separately show
which lengths execute. Repeat on an appropriate trained checkpoint before any
long-context capability claim;do not tune on the prospective final holdout.

## Established high-throughput engine:verified scope

**[Verified source compatibility,not executed GPU deployment]** vLLM0.33.0,
commit4ba787ec9959e566d3b587a796dff07673324da5:registry includes builtin
LlamaForCausalLM;implementation supports tied embedding,GQA,split-half RoPE and
linear scaling. Its FlashAttention backend accepts head sizes divisible by8 up
to256,FP16/BF16 and compute capability at least8.0. RFC tiers have head_dim64;
the debug head_dim16 also passes that source predicate. These checks do not
establish a working CUDA/driver/Torch/kernel combination on your server.

Use a separate pinned GPU environment;do not replace this repository's Torch lock
with vLLM dependencies. After checking hardware and exporting a real dense small
checkpoint,the candidate single-GPU command is:

```bash
# Separate vLLM0.33.0 GPU environment;runtime acceptance remains unverified here.
nvidia-smi
vllm serve /absolute/runs/hf-small-01 --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 2048
```

Keep that endpoint behind an authenticated private gateway. Verify actual startup,
tokenizer/template,greedy outputs and log probabilities against the exported HF
model before serving traffic. Capture engine version,hardware,CUDA/driver,dtype,
context and kernel logs. Do not pass the separate native HQQ directory. No vLLM,
CUDA,paged-cache or continuous-batching speed is measured in the CPU acceptance.

Source links:[registry](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/models/registry.py),
[Llama](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/models/llama.py),
[rotary](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/rotary_embedding/__init__.py),
[backend](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/attention/backends/flash_attn.py).

## Containers

```bash
docker build --target training --tag qvelt-training:phase8 .
docker build --target serving --tag qvelt-serving:phase8 .
docker build --target serving-quantized --tag qvelt-serving-quantized:phase8 .
uv run --frozen --extra cpu python scripts/container_smoke.py \
  --directory runs/container-01 --repository .
```

Python3.12.14-slim-bookworm and uv0.12.23 base digests are immutable;Python packages
use uv.lock;training Git uses a signed dated Debian snapshot. Build context is an
allowlist excluding Git,keys,data and weights. Default runtime UID10001;bind mounts
may use the owning nonroot UID. Mount model/config read-only,pass only the named
auth environment binding,publish the host port on loopback,provide writable /tmp.
The training target is CPU;no CUDA image is claimed. A transient BuildKit
`--secret id=proxy_ca,src=PUBLIC_CA_FILE` supports managed CA trust without retaining
the mounted CA. This does not grant network authorization.

Container CI builds all targets and executes nonroot doctor,three real training
updates and authenticated JSON/SSE server cleanup. Local managed Docker uses vfs:
the serving image built,but runtime root-copy allocation exhausted the small disk.
That local launch is recorded as failed,not accepted. Check the container CI
receipt for actual runtime acceptance on a runner with sufficient disk.

## What to send for analysis

Share the generated result.json,receipts,RESULTS,export manifests/conversion or
quantization descriptors,HTTP result and sanitized shutdown logs,container result,
and exact command/commit/lock/hardware identities. Keep model weights,raw prompts,
private datasets,environment dumps,authorization headers and secret values private.
Report failures as failures;attach context/stop counts and the negative baseline.

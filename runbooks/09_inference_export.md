# Native CPU generation and export

Scope:qvelt-inference-v1 snapshots and qvelt-generate-v1 requests for the original
Qvelt decoder. CPU FP32 eager,at most20M parameters. No HTTP service,quantization,
Hugging Face/GGUF/vLLM adapter,GPU performance or model-quality promotion is implied.
[Design](../docs/development/inference_design.md) and ADR0047 define the contract.

## Reproduce the portable acceptance

```bash
mkdir -p runs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/inference_smoke.py --directory runs/inference-01 --repository .
uv run --frozen --extra cpu qvelt manifest verify runs/inference-01/snapshot/manifest.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu qvelt generate \
  runs/inference-01/greedy.yaml --stream
```

Use a new directory. The script creates an **untrained** real BPE checkpoint from
the older approved original tokenizer fixture,executes CLI export,checks zero
export/reload logit difference,and compares greedy/categorical cached/uncached,
CLI/API and streaming outputs for three unequal-length prompts. It hides original
checkpoint/data paths and verifies fresh-process generation still works. No
optimizer update,external teacher or final800 scoring occurs. `result.json`
contains only summarized identities/check results,not a reasoning score.

## Export an existing local checkpoint

Keep the checkpoint and complete original tokenizer bundle intact. The exporter
checks every registered byte,recipe/config/tokenizer identity,single-rank layout,
finite FP32 state and strict model shapes before writing. Native snapshots copy
exact model/tokenizer bytes;they do not retrain the tokenizer or cast quantized
weights. Optimizer/RNG pickle is never deserialized or copied into the snapshot.

CLI syntax:

```bash
uv run --frozen --extra cpu qvelt export CHECKPOINT_MANIFEST \
  --checkpoint-sha256 CANONICAL_CHECKPOINT_SHA \
  --tokenizer-manifest TOKENIZER_MANIFEST --tokenizer-sha256 CANONICAL_TOKENIZER_SHA \
  --directory runs/native-01 --repository .
```

Canonical manifest SHA is not a hash of arbitrarily formatted JSON. For actual
paths,the Python API can derive trusted local manifest identities and create an
exclusive generation recipe without hand-copying hashes:

```bash
export QVELT_SOURCE_CP=/absolute/run/checkpoints/step-00000200/manifest.json
export QVELT_SOURCE_TOKENIZER=/absolute/data/tokenizer/manifest.json
uv run --frozen --extra cpu python - <<'PY'
import os
from pathlib import Path
from qvelt.eval.config import ArtifactRef
from qvelt.inference.artifacts import export_snapshot
from qvelt.inference.config import GenerationJob, GenerationOptions, TextRequest, dump_generation
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest

def ref(path):
    path = Path(path).resolve(strict=True)
    manifest = ArtifactStore(path.parent).read_manifest(path.name)
    return ArtifactRef(str(path), manifest_digest(manifest))

output = Path('runs/native-01').resolve()
manifest = export_snapshot(ref(os.environ['QVELT_SOURCE_CP']),
    ref(os.environ['QVELT_SOURCE_TOKENIZER']), output, Path.cwd())
snapshot = ArtifactRef(str(output/'manifest.json'), manifest_digest(manifest))
options = GenerationOptions(max_new_tokens=32)
job = GenerationJob(snapshot, (TextRequest('first', 'Привет!', options),
                              TextRequest('second', 'Return 2 + 3.', options)))
with Path('generate.yaml').open('x', encoding='utf-8') as stream:
    stream.write(dump_generation(job))
PY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu qvelt generate generate.yaml
```

Source checkpoints may describe a different training precision/device if their
stored weights are FP32. The snapshot normalizes runtime config to CPU FP32 and
retains original config/checkpoint/tokenizer manifests for identity checks.
It is self-contained for inference:copy the **entire snapshot directory**,then
update the recipe's snapshot path while retaining its canonical manifest hash.
No original corpus,optimizer,training checkout paths or remote code are resolved.

`inference.json` records exact format/schema,dtype/device,source identities,
parameter count and engineering-only release status. Root schema1 run manifest
binds model/tokenizer/config/lock/source metadata/framework LICENSE/NOTICE.
Framework code licensing does not clear model/data release rights. Native export
is not an approval to publish trained weights. Unknown formats/versions,dtype
claims,ambiguous roles,corruption,wrong vocabulary/config and missing commit
markers fail;do not edit/re-hash broken artifacts as a recovery shortcut.

## Request and stream semantics

The YAML recipe is strict:unknown fields,duplicate IDs,boolean numeric values,
NaN/inf,invalid stop lists and unsupported samplers fail. `prompt_format` is
chat-v1(default) or plain-v1;`use_cache:false` selects the full-prefix reference.
Chat prefix is BOS1/USER4/ordinary prompt/END_TURN6/ASSISTANT5;plain prefix is
BOS1/ordinary prompt. Literal control spellings in user text remain ordinary text.
Configured stop tokens(default EOS2 and END_TURN6) terminate output;other special
tokens are literal generated data and never parsed/executed as tools or roles.

Per-request options:

| Field | Meaning/default | Bound |
| --- | --- | --- |
| max_new_tokens | Sampled actions,including stop token;32 | 1..256 |
| strategy / temperature | greedy/T0 or full categorical | sample T1e-4..2 |
| seed | Private CPU RNG,20261106 | uint32 |
| stop_tokens | Defaults[2,6];empty permits count-only stop | <=8 unique in-vocab IDs |
| stop_strings | Omitted from displayed text | <=4 strings,1..64 codepoints |
| context_limit | Includes prompt/padded generation reservation;2048 | 4..2048 |
| max_cache_bytes | Logical FP32 KV/metadata reservation;64MiB | <=128MiB |
| max_seconds | Cooperative elapsed budget;30 | >0..300 |
| max_output_chars | Python Unicode codepoints;8192 | 1..16384 |

Engine limits can only reduce the supported maxima:batch8,context2048,padded
token slots8192,logical batch cache128MiB. Admission reserves
`batch*(max_prompt_length+max_new_tokens_in_batch)`,even if some rows finish early.
Each request's context/cache cap applies to its padded slot. A request may fit
alone and be rejected in a longer padded batch. No silent truncation,automatic
context extension or hidden retry occurs. Limits bound logical work/cache,
not total RSS,attention/logit intermediates or copy-on-append cache peaks.

Streaming emits one JSON object per event:identifier,token_id,generated_tokens,
text snapshot,done,reason and current logical cache bytes for that padded row.
**Replace displayed text with each snapshot;do not append it as a delta.** Partial
UTF-8 byte tokens can temporarily decode as replacement characters and revise
later. Stop strings are detected across tokens;possible stop-prefix suffixes are
withheld until resolved,and flushed on another terminal reason. Raw sampled IDs
retain stop actions even when their display text is omitted.

Terminal reasons:stop_token,stop_string,max_tokens,output_limit,cancelled,timeout,
nonfinite. Nonfinite is an explicit failed request,not a valid completion;inspect
reasons rather than treating all processed batch rows as healthy. Invalid batch
admission/stale model/IO/format exceptions abort the call. CLI --stream Ctrl-C
closes the iterator and returns130;already emitted events remain partial output.

The Python `GenerationEngine.stream` API accepts a trusted per-ID cancellation
callback;iterator.close() cancels remaining work and releases its private cache.
Timeout/cancellation checks occur between kernel calls,not inside a blocked
PyTorch operation. Finished slots remain masked while others decode;this is
static true batching,not continuous batching. The HTTP/concurrency scheduler is
M8.2. No admission/backpressure/authentication/server-readiness claim is made here.

## Cache and reproducibility boundaries

Each generator owns a fresh cache;no cross-request prefix cache,persistent KV
export or caller mutation of cache tensors is supported. Completion,exception or
close releases its references. Cache payload is FP32 K/V plus int64 position/
document and bool validity metadata: `B*T*(8*L*kv_width+17)` bytes. This is logical
payload,not a process allocator measurement. Events report bytes at that moment;
finished padded slots can remain physically present until the batch completes.

Model must remain CPU FP32/eval and read-only. Normal parameter changes or train
mode invalidate the engine;create a new engine after updates. Unsafe .data writes
or arbitrary internal-module mutation are outside the trusted cache interface.
Model loading should run during serialized setup;RNG preservation during loading
is not a guarantee against unrelated concurrent RNG mutations. Generation itself
uses private request RNGs,forces FP32 outside ambient autocast,and never holds a
grad/autocast context across a yielded event.

Reproduce with fixed weights,tokenizer,recipe,CPU/PyTorch/thread policy and batch
composition. Cached/full-prefix logits have documented FP32 tolerances;greedy
ties/categorical boundaries can amplify small floating-point differences.
Do not infer universal bit identity across batches,devices or releases. Current
acceptance compares actual fixed CPU cases and stream/CLI protocols. It does not
measure useful reasoning,GPU/VRAM,TTFT or a serving SLA. Final800 remain untouched.

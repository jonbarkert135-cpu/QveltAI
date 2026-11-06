# M8.1 bounded CPU generation and portable native snapshot

Date:2026-11-06. Base0390b60. Reuse the verified decoder/KV cache,tokenizer,
checkpoint checks,safetensors and atomic manifest store. Ponytail/full and pinned
Orchestra vLLM lifecycle/budget guidance inform review;no vLLM runtime,external
model code,quantization,GPU/server deployment or new dependency is introduced.

## Generation contract

Own CPU FP32 eager engine accepts immutable named token requests and strict YAML
text recipes. Greedy or temperature/categorical sampling uses private CPU RNGs
per request. No global RNG/thread/grad-mode mutation across a yielded event.
True batched forward supports unequal prompt lengths using left padding,explicit
logical positions and validity masks. Finished/cancelled slots stay masked;no
cross-request or prefix cache reuse. Cached and full-prefix paths share the same
sampling/stopping loop. Cache remains private and is released on completion,
exception or iterator.close();model eval mode/parameter stamps guard stale state.

Bound batch<=8,context<=2048,padded token slots<=8192,weights<=20M parameters,
per-request new tokens<=256 and output chars<=16384. Preflight worst padded KV
payload uses FP32 bytes8*B*T*L*kv_width plus17*B*T position/document/valid metadata.
Per-request and batch cache limits bound logical stored payload,not process RSS
or copy-on-append temporaries. Context/budget validation precedes model execution.
Cancellation and elapsed-time checks are cooperative between kernel calls;they
cannot preempt one PyTorch operation. No serving admission/concurrency SLA claim.

Version1 stream events contain request ID,new token ID,count,cumulative text
snapshot and terminal reason. Text snapshots may revise incomplete UTF-8 byte
decodes;they are deliberately not append-only deltas. Stop strings are detected
across token boundaries,withhold possible stop-prefix suffixes and are omitted
from displayed text. Configured stop tokens,count/output/context budgets and
nonfinite logits have explicit terminal/error outcomes. Raw sampled IDs retain
stop actions;non-stop special tokens are literal output data,never tool commands.

Reproducibility scope:fixed weights/tokenizer,PyTorch,CPU precision,thread/runtime
policy,batch composition,prompt positions and recipe. Independent RNG prevents
request scheduling from consuming another request's random stream;floating-point
batch/cache differences near ties do not justify universal cross-batch bit identity.

## Snapshot contract

qvelt-inference-v1 is a write-once native safetensors snapshot,committed by a
regular schema1 run manifest. Copy exact verified model/tokenizer bytes,normalized
CPU FP32 inference config,original config/checkpoint/tokenizer manifests,lock,
framework LICENSE/NOTICE and a strict versioned descriptor. Source model/tokenizer
hashes,shape,format and dtype must agree before allocation/use. Load only data;
never deserialize optimizer/RNG pickle or execute downloaded code. Bind original
tokenizer payload to its manifest rather than fitting/retraining it for export.

Reuse one lower checkpoint loader under the existing evaluation API without
changing validation/bounds. Exported bundles reload independently of the original
checkpoint/data directories,with strict shape/finite FP32/control-policy checks.
Unknown formats,quantization,shards,corruption,incompatible tokenizer/config,
duplicate roles or missing commit markers fail. Framework code license does not
grant unreviewed model/data release rights;this is a local engineering snapshot.

Acceptance:independent sampler/KV byte references,batched cached/full-prefix
logit and token comparisons,Unicode/stop/cancellation/timeout/RNG/cache-release
tests;real checkpoint->export->fresh-process reload parity after source relocation;
corruption/identity/schema guards;CLI/stream smoke,lint/types/build and clean CI.
Update runbook,state,decisions,experiments and risks. Final800 remain unscored;
no model quality promotion or paid/full-scale run is part of this milestone.

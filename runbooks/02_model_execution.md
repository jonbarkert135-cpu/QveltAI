# Portable model execution, cached decoding and recomputation

Scope:original RFC decoder, explicit runtime policies and CPU correctness.
No pretrained/final weights, GPU speedup or public reasoning score is supplied.
Follow runbook00 for environment/hardware; CPU uses the frozen default profile.

## CPU reference acceptance

```bash
make setup
make lint
make typecheck
make test
```

M4.1 covers component values/gradients, causal/document/padding masks, seeded
initialization/residual scaling, tied-embedding combined gradients, parameter
counts and non-finite training-stop behavior. M4.2 compares cached/full logits,
reference/SDPA gradients, CPU BF16 semantics and activation recomputation.
The default Make test excludes GPU cases; skips are never hardware-validation passes.

## Configure training execution

Learned dimensions stay in ModelConfig. Runtime ModelExecution is embedded at
training.execution in the resolved job. Default is reference attention, no block
recomputation. These flags are included in the job/checkpoint identity.

```bash
uv run --frozen --extra cpu qvelt train configs/train/smoke.yaml --directory /tmp/qvelt-sdpa-01 --set training.execution.attention_backend=sdpa --set training.execution.activation_checkpointing=true --stop-after 37
uv run --frozen --extra cpu qvelt train configs/train/smoke.yaml --directory /tmp/qvelt-sdpa-01 --set training.execution.attention_backend=sdpa --set training.execution.activation_checkpointing=true --resume latest
```

Choose a new directory. This100-update numerical fixture proves optimizer/replay
engineering, not real-corpus pretraining. Changing execution flags rejects exact
resume. Existing learned parameter/state-dict names are preserved, but old source/
job identities are not silently migrated into a new executable version.

Reference attention explicitly disables autocast for FP32 QK scores/softmax;
`.float()` alone would not prevent autocast lowering matmul. SDPA uses PyTorch's
actual native dispatcher, compact GQA heads, dropout0 and explicit packed causal
masks. It can choose a math fallback. Selecting sdpa **does not assert that a Flash
kernel ran or improved performance**. The portable backend remains available.

Checkpointing uses torch.utils.checkpoint with use_reentrant=false and RNG
preservation, only in train+grad mode. It recomputes each block during backward;
it is disabled during evaluation/no_grad. Model mode, weights and global state
must stay fixed between forward/backward, as required by PyTorch recomputation.

## Request-local KV cache

```python
import torch
from qvelt.config import ModelConfig
from qvelt.config.execution import ModelExecution
from qvelt.model.decoder import Decoder

config = ModelConfig(layers=2, hidden=32, heads=4, kv_heads=2,
                     ffn=64, vocab=288, context=16)
model = Decoder(config, ModelExecution(attention_backend="sdpa")).eval()
with torch.inference_mode():
    prefix_logits, cache = model.forward_cached(torch.tensor([[32, 33, 34]]))
    suffix_logits, extended = model.forward_cached(torch.tensor([[35, 36]]), cache)
```

These IDs are an untrained numerical example. Real inference needs actual compatible
model/tokenizer artifacts; no chat/answer quality is implied. The method returns
suffix logits and a new cache; the prior cache is unchanged. It automatically
runs under no_grad and requires eval mode. Multi-token prefixes/suffixes are valid.

For packed data pass positions/document_ids/valid_mask explicitly, with the same
semantics as ordinary forward; positions may reset between documents, whereas
causal time order uses the actual cache offset. Omitted metadata means ordinary
single-document defaults (document0, contiguous absolute positions), not implicit
inference of the previous nonzero document ID. Different batch/context/device/
compute dtype, execution or model weights rejects cache reuse. Context overflow
fails without eviction or advancing the original cache.

K/V retain only KV heads. The cache holds a weak model reference and parameter
identity/data_ptr/version stamps, so it cannot keep discarded model weights alive
or silently attach to a new model. Normal optimizer/in-place/load_state_dict changes
invalidate it. Torch2.13.0 Tensor._version is a pinned internal dependency; construct
model parameters outside inference_mode so counters exist. Cached calls themselves
support inference_mode. Unsafe .data writes, external cache tensor edits and
concurrent model mutation are outside this trusted in-process API. Do not serialize
or accept arbitrary uploaded cache objects. Drop the cache to free it.

The functional reference cache concatenates state per append; it is not a paged
serving allocator. Measure copy/temporary memory before server-scale inference.
KV reuse and training activation checkpointing have separate ownership/lifetimes.

## Measure the available CPU

```bash
uv run --frozen --extra cpu python scripts/profile_decoder.py --job configs/train/smoke.yaml --repeats 8 > /tmp/qvelt-decoder-profile.json
```

This compares reference/SDPA with and without block recomputation, using identical
seeded numeric fixtures. It reports actual forward/backward target throughput,
cached/uncached append timings (both include prefill), compact cache storage bytes,
and unique nonparameter storage saved for backward. Graph storage excludes model
weights and is **not allocator/process peak RAM**. No optimizer update occurs.
Default117056-parameter job is deliberately small; do not profile huge server
configs on the16GB laptop without resource planning. CPU BF16 value/gradient tests
do not establish native laptop GPU BF16 support. Timing is a small host measurement,
not a promise that a backend/cache is faster for every size or device.

On actual compatible CUDA hardware, after the explicit CUDA setup in runbook00:

```bash
uv run --frozen --extra cuda pytest tests/unit/test_model_execution.py -m gpu -q
```

The provided GPU FP32/native-BF16 comparisons require actual CUDA (BF16 capability
checked separately). They have not executed in this CPU environment. Kernel
selection, peak VRAM, laptop behavior, NCCL and multi-node efficiency remain
hardware validation gates; no forced fused kernel or fallback is called Flash.

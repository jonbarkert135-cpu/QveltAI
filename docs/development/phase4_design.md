# Phase4 portable correctness, cached decoding and recomputation

Date:2026-11-01. Owner:Lead agent. Status:implementation in progress.

First close M4.1 using independent fixed-seed/statistical initialization, split
input/output embedding-gradient and non-finite training-stop regressions. Existing
IEEE propagation plus trainer collective finite-loss/gradient checks remain the
policy: no layer-local exception/sanitization that could strand distributed peers.
Empty-mask zero loss assumes finite input logits. Tests must prove no optimizer
update/checkpoint follows a non-finite loss.

Then add opt-in runtime execution settings separately from learned model dimensions:
portable versus PyTorch SDPA attention, non-reentrant block activation checkpointing,
and bounded per-request KV caching in evaluation with gradients disabled. Preserve
the existing forward signature/default reference path and state-dict parameter names.
Cache state is explicit, stores compact KV heads, positions/document/valid metadata,
and rejects context, model, dtype/device or weight-version mismatches. Full and
incremental logits must agree, including multi-token suffixes and packed masks.
Checkpointing and attention backend settings belong in resolved training config and
checkpoint compatibility; recomputation must preserve autocast/RNG semantics.

Use installed torch2.9.1 source/docs as API authority. Verify SDPA math fallback on
CPU against the independent portable path (values and all gradients), including
padding/packed documents. GPU kernel selection/speed remains unverified without
hardware; keep the reference path. No vendor timing claim is a measured Qvelt result.

## Runtime/cache contract

ModelExecution is a frozen config dataclass (reference/sdpa backend, block
activation_checkpointing). TrainingConfig embeds it so resolved jobs and resume
identities include execution choices; ModelConfig and parameter/state-dict shapes
remain unchanged. SDPA receives an explicit boolean allowed mask, dropout0 and
compact GQA heads. Safe self-attention rows for padded queries prevent all-masked
softmax; output is zeroed for padding. The reference path remains default.

Decoder.forward_cached returns logits plus a new immutable request cache under
no_grad, requires eval mode, and accepts multi-token prefill/suffixes with explicit
packed positions/documents/validity. Cache stores unexpanded per-layer rotated K/V
and shared metadata; full-sequence causal order uses cache offsets, not reset RoPE
positions. Rejected calls leave the old cache unchanged. No eviction/context wrap,
cross-model, cross-device/dtype or silent stale-weight reuse. Parameter identity,
data_ptr and pinned PyTorch Tensor._version counters detect normal weight updates,
load_state_dict and replacement; unsafe .data mutation/external cache tensor edits
are outside the trusted cache contract. Construct model parameters outside
inference_mode so counters exist; cached calls themselves support inference_mode.

The functional cache uses concatenation, not paged/preallocated serving storage;
measure transient copying and memory before scaling inference. Block checkpointing
uses torch.utils.checkpoint(...,use_reentrant=False,preserve_rng_state=True), only
in train+grad mode. Evaluation caching and training recomputation are separate.
Keep optional GPU tests behind hardware markers and record them as unrun here.

## Executed CPU profile

Clean code efa6f304d3c9a6348c6a85f4640d944de66ab25b, Python/PyTorch2.9.1+cpu,
117056 parameters,B2,T32,one Torch thread,two warmup/eight repetitions.
`uv run --frozen --extra cpu python scripts/profile_decoder.py --job configs/train/smoke.yaml --repeats 8`.
Receipt:evidence/phase4-profile-efa6f30.json.

| Backend | Recompute | Saved nonparameter storage bytes | Forward/backward targets/s | Cached/uncached seconds (8 repetitions) |
| --- | --- | ---: | ---: | --- |
| reference | false |932236 |9571.36 |.135808/.163432 |
| reference | true |157836 |4494.59 |.137194/.179225 |
| sdpa | false |881036 |9459.94 |.127555/.155027 |
| sdpa | true |157836 |4975.82 |.134671/.299306 |

All cache variants store25392 bytes and pass logit comparison; prefill16 plus8
one-token appends. Inference timings include prefill. Saved graph storage is not
peak allocator/process RAM. These short untrained CPU measurements show a compute/
retention tradeoff, not a universal speedup, deployment benchmark or GPU result.
Full acceptance:487 passed/2 optional SDK skips/4 GPU deselections in81.70s,
Ruff/format and mypy89 files clean. Actual CUDA cases remain unrun.

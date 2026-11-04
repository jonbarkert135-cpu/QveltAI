# Qvelt decoder: block mathematics and implementation map

Status (2026-11-04): implemented per [RFC-001](research/architecture_rfc.md) and
verified on CPU (PyTorch 2.9.1). CUDA/BF16 kernels, FlashAttention dispatch and
GPU throughput are **unverified** until run on real hardware.

Notation: B batch, T time, d hidden, L layers, Hq query heads, Hkv KV heads,
dh=d/Hq, k=Hkv*dh, f FFN width, V vocabulary. Code lives in `src/qvelt/model/`.

## Embedding and tied head (`decoder.py`)

    x0 = E[token_id],                 E in R^(V x d)
    logits = RMSNorm(x_L; g_final) E^T

The same parameter E is used at input and output; there is no positional
embedding, embedding multiplier, output bias or separate `lm_head`.

## RMSNorm, pre-norm (`layers.RMSNorm`)

    RMSNorm(x; g) = g * x / sqrt(mean_j x_j^2 + eps),   eps = 1e-6

Statistics run in at least FP32 (`_wide`: BF16/FP16 -> FP32, FP64 kept for
gradcheck) and the result returns to the input dtype. Each block has two gains
(attention, FFN) plus one final gain, all initialized to 1.

## Block (`decoder.Block`)

    x = x + Attention(RMSNorm(x; g_attn))
    x = x + SwiGLU(RMSNorm(x; g_ffn))

## Rotary position embedding (`layers.RotaryEmbedding`)

For each adjacent even/odd pair j of Q and K at (document-reset) position t:

    omega_j = theta^(-2j/dh),  theta = 10000,  j = 0..dh/2-1
    phi = (t / s) * omega_j
    [z'_2j, z'_2j+1] = [[cos phi, -sin phi], [sin phi, cos phi]] [z_2j, z_2j+1]

`s = model.rope_scale` is the long-context hook: linear position interpolation
(Chen et al., 2023, "Extending Context Window of Large Language Models via
Positional Interpolation"). `s = 1` is the exact RFC baseline; raising `context`
with `s > 1` still requires fine-tuning and evaluation before any context claim.

## Grouped-query attention (`layers.Attention`)

    Q = u Wq^T (B,Hq,T,dh),  K = u Wk^T, V = u Wv^T (B,Hkv,T,dh)
    g(h) = floor(h / (Hq/Hkv))                       contiguous KV groups
    A_h = softmax((RoPE(Q_h) RoPE(K_g(h))^T) / sqrt(dh) + M)
    Attention(u) = concat_h(A_h V_g(h)) Wo^T

Mask: `M[t,s] = 0` iff `s <= t`, both tokens valid and `doc[t] == doc[s]`;
otherwise `-inf`. Padded queries get zero output (never an all-masked softmax).
Tiny uses Hkv=1 (MQA). No QK-norm, MLA, sliding window or MoE: RFC-001 rejected
them for the baseline; they are separate ablations.

Backends (`ModelExecution.attention_backend`):

- `reference`: explicit `repeat_interleave` head mapping, scores and softmax in
  >= FP32 with autocast disabled. This is the oracle path.
- `sdpa`: `F.scaled_dot_product_attention(..., attn_mask=allowed,
  enable_gqa=True)`. Padded query rows get a safe diagonal key and are zeroed
  afterward. The packed (varlen) path is the same block-diagonal boolean mask.
  Kernel selection depends on the installed PyTorch version, hardware, dtype
  and mask/shape support. GPU kernel dispatch has not been measured here.

## SwiGLU FFN (`layers.SwiGLU`)

    SwiGLU(v) = (SiLU(v Wgate^T) * (v Wup^T)) Wdown^T,   SiLU(a) = a sigmoid(a)

All projections are bias-free; there is no dropout.

## Loss (`loss.causal_loss`)

    loss_sum = -sum_valid log_softmax_FP32(logits)[target],  count = #valid

Targets are shifted within documents; PAD (0), BOS (1) and -100 are excluded.
The caller divides by the global valid-target count (not a mean of rank means).

## KV cache (`Decoder.forward_cached`, `cache.py`)

Each layer stores post-RoPE K and V of shape (B,Hkv,t,dh) plus position, document
and validity metadata. A suffix of n tokens attends to the cached keys with query
offset t. Returned cache containers are frozen; callers must not mutate their tensors. failed calls never advance
the input cache, and parameter updates invalidate it. The cache is FP32 or the
autocast compute dtype; it copies on append (no paged allocator yet).

## Initialization

    W ~ Normal(0, 0.02) for E, Wq, Wk, Wv, Wgate, Wup
    Wo, Wdown ~ Normal(0, 0.02 / sqrt(2L)),   norm gains = 1

Each block adds two residual branches of variance proportional to 1/(2L), so the
variance added to the residual stream is roughly independent of depth.
`test_residual_scaled_init_keeps_activation_and_gradient_scale_across_depth`
measures this at L=4 versus L=24 and checks that unscaled init grows instead.

## Counts and FLOPs

    P_block = 2d^2 + 2dk + 3df + 2d
    P_total = Vd + L P_block + d
    F/token ~= 6[Vd + L(2d^2 + 2dk + 3df)] + 12 L T d
    MFU = tokens_per_second * F/token / peak_FLOPs

`ModelConfig.parameter_count`, `train.optim.flops_per_token` and meta-device
enumeration are checked against the RFC table rows by
`test_parameter_and_flop_counts_match_rfc_table`.

| Config | L | d | Hq | Hkv | f | T | Params | GFLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 8 | 320 | 5 | 1 | 896 | 512 | 19,338,560 | 0.131727 |
| small | 16 | 768 | 12 | 4 | 2048 | 2048 | 125,854,464 | 1.056965 |
| base | 26 | 1280 | 20 | 5 | 3456 | 4096 | 493,553,920 | 4.596695 |
| large | 32 | 2048 | 32 | 8 | 5632 | 4096 | 1,510,082,560 | 12.280922 |

## Execution options

- Activation checkpointing: `ModelExecution(activation_checkpointing=True)`
  recomputes each block (non-reentrant, RNG-preserving) only in training mode.
- `torch.compile`: values/gradients match eager for both backends with and
  without recomputation (`aot_eager`; `inductor` CPU marked slow). Blocks trace
  with `fullgraph=True`; `Decoder` permits graph breaks for host-side input
  validation (`bool(tensor)` checks); those checks are preserved.
- Precision: parameters are FP32; BF16 autocast is supported. Norm statistics,
  RoPE angles and reference softmax use >= FP32; causal_loss deliberately computes
  FP32 even for FP64 logits. Whole-model FP64 gradcheck covers logits, not that
  deliberately narrowed loss.

## Profiling

`scripts/profile_model.py` reports tokens/s, analytical TFLOP/s, optional MFU
(`--peak-tflops`), CUDA peak memory and top self-time operators.
`scripts/profile_decoder.py` keeps the earlier CPU graph-storage profile.

Current operator measurements and their scope are recorded in
[Phase4 review acceptance](development/validation_phase4_closure.md). The uploaded
historical profile was not adopted as newly measured evidence. Profiling measures
forward/backward without Adam updates, so analytical throughput/MFU excludes
optimizer, communication, padding and checkpoint recomputation overhead.

Bottlenecks to measure on GPU (unverified):

1. Measure actual SDPA dispatch for the block-diagonal boolean mask. Specialized
   FlexAttention/varlen kernels require hardware capability and parity checks
   against `reference`; packed correctness alone is not a fused varlen speed claim.
2. Full-vocabulary FP32 logits (4*B*T*V bytes) dominate small-model memory.
   Chunked or fused cross-entropy is the candidate fix.
3. Copy-on-append KV cache costs O(t) per decode step; a preallocated or paged
   cache is needed for serving.

GPU command (unverified; replace the peak value with the actual measured/specification value):

```bash
uv run --frozen --extra cuda python scripts/profile_model.py \
  --model configs/model/small.yaml --device cuda --bf16 --recompute \
  --batch 1 --length 512 --peak-tflops "$DEVICE_PEAK_TFLOPS"
```

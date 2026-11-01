# RFC-001: Qvelt dense decoder baseline

Date: 2026-10-31. Status: selected baseline; portable decoder implemented/tested on CPU.
Optimized decode/KV cache and real-corpus/GPU validation remain separate gates.
Owner: Lead agent. Sources refer to the verified ledger in `survey.md`.

## Constraints and decisions

The user's current machine is an ASUS TUF Gaming F15 with RTX 3060 Laptop,
Core i5 (unspecified model), and 16 GB system RAM. Exact VRAM, GPU power limit,
driver, OS, and BF16/kernel support are unknown. Use **6 GiB VRAM as a planning
assumption only**, confirm with `nvidia-smi` on the laptop. The cloud machine is
not that laptop and has no validated CUDA device. Future servers are unspecified.

Select an original dense causal decoder following public component designs:
pre-RMSNorm [S02,S37], full causal GQA [S09], RoPE [S03], SwiGLU [S07], tied
embeddings [S12,S32], bias-free projections [S45]. This is a tractable reference
architecture, not a novel architecture or a claim of state-of-the-art quality.
Deep/narrow sizing is informed by MobileLLM [S32], with exact shapes chosen as
project hypotheses. Do not reuse any published weights or implementation code.

## Mathematical specification

Let x have shape (B,T,d). E is (V,d). Start x=E[token_id]; no learned positional
embedding or embedding multiplier. Two independently learned norm gains per
block, initialized to ones, and one final gain:

    RMSNorm(x; g) = g * x / sqrt(mean_j(x_j^2) + 1e-6).

Accumulate RMS and division in FP32, then return compute dtype. For each block:

    u = RMSNorm(x; g_attention)
    x = x + Attention(u)
    v = RMSNorm(x; g_ffn)
    x = x + (SiLU(v W_gate^T) * (v W_up^T)) W_down^T
    SiLU(a) = a * sigmoid(a)

Wq/Wo are (d,d); Wk/Wv are (k,d), k=Hkv*dh, dh=d/Hq.
Wgate/Wup are (f,d); Wdown is (d,f). No projection bias, norm bias, dropout,
QK normalization, learned attention scale, or extra residual gate in baseline.
Zero dropout is a large-data baseline choice, not established optimality on
small/repeated corpora; expose it as a later controlled regularization ablation.

Reshape Q to (B,Hq,T,dh) and K/V to (B,Hkv,T,dh). Query head h maps to
KV head floor(h/(Hq/Hkv)); heads are contiguous groups. For each even/odd pair
j of Q and K at position t:

    omega_j = 10000^(-2j/dh), j=0,...,dh/2-1
    [z'_(2j), z'_(2j+1)] = [[cos(t omega_j), -sin(t omega_j)],
                            [sin(t omega_j),  cos(t omega_j)]] [z_(2j),z_(2j+1)]
    A_h = softmax_FP32((Q'_h K'_group(h)^T)/sqrt(dh) + M)
    O = concat_h(A_h V_group(h)) W_o^T

M[t,s]=0 only when s<=t, both tokens are valid, and their packed document IDs
match; otherwise -infinity. Never compute an all-masked softmax row: padded
queries produce zero attention output and no loss. A portable reference uses
explicit masking and head mapping. Optimized implementations must preserve this
block-diagonal causal mask, including packed document boundaries; a generic
`is_causal=True` across the entire packed sequence is insufficient.

Positions restart at zero for each independent packed document segment. Split
long documents into self-contained windows of at most T input tokens plus the
next-token label, using one source-token overlap between adjacent windows so
each original adjacent-token training pair appears once. Reset positions at each
window; do not carry KV across windows or invent EOS at a chunk boundary. Keep
BOS/EOS only at the true document boundaries. Record window offsets and packing
in the manifest. Shift labels only within documents: predict each document's
next token including EOS, never the next document's BOS. Pretraining includes all valid target tokens except BOS/padding;
SFT selects assistant targets. Define empty-target batches as invalid/skip-all-
ranks together with no optimizer/scheduler update, not a divide-by-zero loss.

    logits = RMSNorm(x; g_final) E^T
    loss = -sum_valid log_softmax_FP32(logits)[target] / valid_target_count

E is **the same parameter object** at input and output. No output bias. An
optional vocabulary z-loss is separate and defaults to zero (see training RFC).
Gradients use the global sum of valid-token losses, not a mean of rank means.

Initialize unique linear/embedding weights with Normal(0,0.02); scale Wo and
Wdown standard deviation by 1/sqrt(2L), accounting for two residual branches
per layer. GPT-2 [S44] supports residual-depth scaling; the exact 0.02 base
value here is a tunable engineering choice, not a derived optimum. Initialize
tied E once; norm gains=1. FP32-seeded initialization and reference checks are
required before changing initialization or parameterization. No muP claim.

## Tokenizer and interfaces

Use a from-scratch byte-level BPE baseline [S43,S44], V=32768 **including**
256 byte symbols and 32 reserved control symbols. Retain whitespace and Unicode;
no lossy case folding, equation stripping, or untracked normalization. Treat
ASCII digits individually before BPE merges, following the LLaMA precedent
[S13], without claiming this improves math. M3.1 must compare compression and
numeral/LaTeX/Cyrillic round trips before freezing the tokenizer hash. No token
IDs beyond the size/reservation contract are frozen by this RFC. A tokenizer
change invalidates model/data compatibility, not merely a CLI default.

Portable interfaces: `ModelConfig` validates sizes, epsilon, RoPE base/context,
and tied embeddings; `forward(token_ids, positions, document_ids, valid_mask)`
returns logits; loss is separate for streaming/chunked vocabulary handling.
`Decoder.forward_cached` maintains explicit per-layer KV and request metadata
with a hard context bound (ADR-0034); it returns suffix logits and a new cache.
`ModelExecution` selects reference/SDPA and block recomputation separately from
learned dimensions. YAML/dataclass training execution settings are saved with
checkpoints. See runbooks/02_model_execution.md for the implemented CPU contract;
GPU speed/VRAM and paged serving allocation remain unverified.

## Exact size table

All dh=64; Hq divisible by Hkv; FFN dimensions aligned to 128 where possible.
Tiny is a genuine ~20M integration configuration; unit tests use smaller tensors
and CI smoke overrides T=32/64, not a full 512-token multi-step run by default.
The larger tiers retain the same equations; large-scale efficiency still needs
measurement. Context is a training/default maximum, not long-context evidence.

| Config | L | d | Hq | Hkv | f | V | T | Total P | Embedding P | Non-embedding P | Training GFLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 8 | 320 | 5 | 1 | 896 | 32768 | 512 | 19,338,560 | 10,485,760 | 8,852,800 | 0.131727 |
| small | 16 | 768 | 12 | 4 | 2048 | 32768 | 2048 | 125,854,464 | 25,165,824 | 100,688,640 | 1.056965 |
| base | 26 | 1280 | 20 | 5 | 3456 | 32768 | 4096 | 493,553,920 | 41,943,040 | 451,610,880 | 4.596695 |
| large | 32 | 2048 | 32 | 8 | 5632 | 32768 | 4096 | 1,510,082,560 | 67,108,864 | 1,442,973,696 | 12.280922 |

Tiny's Hkv=1 is MQA. Small's grouping=3 versus base/large=4 avoids wasting
width merely to standardize a ratio. The small label ~125M refers to total
unique parameters, unlike some publications' non-embedding labels [S16].
32768 reduces vocabulary-state cost on the laptop but may tokenize Russian
less efficiently than larger multilingual vocabularies; quantify that trade-off.

Count derivation, with k=Hkv*d/Hq:

    P_block = 2d^2 + 2dk + 3df + 2d
    P_total = Vd + L*P_block + d
    P_nonembedding = P_total - Vd

`estimates.py` computes this formula **and independently enumerates named tensor
shapes**. Its tests include a hand-counted 172-parameter case. These are analytical
checks; M4.1 must instantiate the actual model and compare unique parameter
storage/counts before these size labels become implementation claims.

Training FLOP convention (multiply-add=2, forward/backward ~3*forward):

    F/token ≈ 6[Vd + L(2d^2 + 2dk + 3df)] + 12LTd.

This includes full vocabulary projection and full-square QK/AV attention, but
omits norm, activation, softmax, optimizer, padding, communication, and checkpoint
recomputation. Causal triangular kernels may reduce the attention term toward
6LTd. Checkpoint recomputation increases actual work, so this is an accounting
convention, not a hardware-counter measurement or guaranteed runtime bound.

## Memory and hardware scenarios

Persistent FP32 weights + FP32 gradients + two FP32 Adam moments = **16P bytes**.
No additional master copy is needed when stored parameters are already FP32.
The GPU planning model adds 2P for BF16 compute-cache allowance, block input
checkpoints, one-block workspace, FP32 logits, and a 2 GiB fixed margin:

    A ≈ 2*B*T*L*d + 2*B*T*(8d + 3f + 2k) + 4*B*T*V
    M_DDP/rank ≈ 18P + A + 2 GiB
    M_FSDP/rank ≈ 18P/n + 4*max(P_block,Vd) + A + 2 GiB

FSDP's gather term assumes two BF16 gathered units and block/embedding wrapping;
prefetch, parameter/gradient dtypes, bucketing, and version can change it. These
activation/workspace coefficients are **project heuristics, not a certified
upper bound**. Include actual profiler peaks and safety margin before launch.
Unfused attention can add approximately 8*B*Hq*T^2 bytes for a block's FP32
score/probability workspace; block checkpointing avoids retaining every block's
attention map. CPU FP32 activations differ from this BF16 GPU model.

| Config | State GiB (16P) | Planned GPU GiB, B=1 | Extra unfused attention GiB | 8-way FSDP planned GiB/rank | Checkpoint GiB (12P) | BF16 KV GiB, B=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 0.288 | 2.394 | 0.010 | 2.150 | 0.216 | 0.001 |
| small | 1.875 | 4.456 | 0.375 | 2.703 | 1.407 | 0.031 |
| base | 7.355 | 11.190 | 2.500 | 4.106 | 5.516 | 0.127 |
| large | 22.502 | 28.576 | 4.000 | 6.676 | 16.876 | 0.250 |

KV estimate=4*B*T*L*k bytes, only cached inference, **not added again** to the
training table. Checkpoint estimate includes FP32 weights/two optimizer moments,
not gradients; allow metadata, transactional temporary copies and retention.

- **Actual laptop, assumed 6 GiB:** prioritize tiny and CPU fixtures. Small
  may fit at B=1,T=256–512 with activation checkpointing and bounded logits;
  full T=2048 estimates are close enough to require a dry run, not a fit promise.
  Base/large full-state GPU training do not fit this assumed device. CPU offload
  to 16 GB RAM is not a practical large-training plan; do not hide it as support.
  Keep loaders at 0–2 workers initially, stream/map shards, and avoid full corpora
  or multiple tokenization workers resident in RAM.
- **Hypothetical 24 GiB GPU:** small/base are planning candidates at B=1;
  large does not fit the baseline even before unusual allocator overhead.
- **Hypothetical 8x80 GiB server:** DDP is a reasonable first choice even for
  large if B=1 is enough per rank. FSDP becomes useful for larger microbatches,
  contexts or future sizes, not a mandatory source of speedup at these sizes.
  Interconnect and collective correctness remain untested.

## Alternatives rejected for baseline

1. **Same L,d,f,V but MHA + untied output head.** Hkv=Hq triples the KV width
   relative to small's grouping and quadruples it for base/large; an additional
   Vd head increases state. It is an important quality control, but does not
   meet our memory-first default. Reject as default, retain matched-budget
   ablations [S08,S09,S12,S32]. No assertion that its quality is worse.
2. **Eight-expert top-2 SwiGLU at every block, same d/f, GQA retained.** FFN
   parameter storage becomes 8x, active FFN compute roughly 2x plus routing;
   balancing/capacity and dispatch semantics are new failure modes. It is not
   a 1.5B-storage model merely because few experts activate. Reject at current
   laptop/unknown-server scale [S14,S29]. MLA, sparse/hybrid attention, layer
   sharing, and FP8 are separate later ablations, not silently bundled changes.

## Acceptance gates before model training

M2.1: dataclass/YAML validation and manifest schema. M3: licensed tokenizer/data,
known token IDs and boundaries. M4.1: independent forward/gradient oracles for
norm/RoPE/attention/MLP/loss, causal leakage tests, unique parameter counts,
padding/document masks and fixed-seed initialization. M4.2: cached/uncached
parity and dtype/backend checks. Only then can these estimates be replaced by
measured memory/throughput. No training or benchmark has executed in Phase 1.

# Portable decoder review against RFC-001

Date:2026-11-01 (Europe/Kiev). Owner:Lead agent.
Reviewed implementation:91e9dc85956815e04fd7d5032a084060e59e51bb.
Scope:the existing portable CPU decoder, not optimized inference or a new model.

## Review design and acceptance

Compare the selected architecture equations, tensor shapes, initialization and
packed-data interface in [RFC-001](../research/architecture_rfc.md) with
`src/qvelt/model/{layers,decoder,loss}.py` and shared ModelConfig/presets.
Rerun the existing independent numerical/model tests and the actual Phase3 BPE
packed-batch backward test. Record concrete implementation/test locations and
remaining coverage gaps. Do not change learned architecture, add a cache or
claim accelerator performance in this bounded review.

## Findings

| RFC requirement | Implementation inspected | Executed evidence |
| --- | --- | --- |
| Two pre-RMSNorm branches, FP32 norm statistics and final norm | Block.forward; RMSNorm.forward | test_rmsnorm_value_and_gradient; real tiny backward |
| Interleaved even/odd RoPE, theta10000 | RotaryEmbedding; ModelConfig | Independent rotation/value/gradient oracle |
| Contiguous GQA groups, scale1/sqrt(head_dim) | Attention.forward | Loop oracle compares outputs, input and all projection gradients |
| Causal/document/padding mask; zero padded attention queries | Attention.forward | Loop oracle and future-token/other-document perturbation tests |
| Bias-free SwiGLU and projections; no baseline dropout/QK norm | SwiGLU,Attention,Decoder | SwiGLU value/gradient oracle, parameter checks; remaining structure inspected |
| One input/output embedding parameter | Decoder uses F.linear(...,embedding.weight) | Four meta counts and no separate lm_head.weight; direct identity visible in code |
| Normal std.02, residual output/down std.02/sqrt(2L), norm gains1 | Decoder.__init__ | Source inspection only; dedicated initialization regression remains open |
| FP32 masked loss returns sum/count; ignore PAD/BOS/-100 | causal_loss | Independent logsumexp value, zero excluded gradients and empty-mask test |
| Shifted document targets and reset positions from Phase3 | PackedDataset -> Decoder.forward | Actual BPE packed-batch CPU forward/backward test |

No architecture mismatch was found in the inspected portable default path.
The public forward signature accepts token IDs, explicit positions/document IDs
and validity mask, returning B,T,V logits as specified. Padding attention is zero;
this does not require all padded residual activations/logits to be zero. Empty
loss masks return sum0/count0; optimizer/scheduler skipping belongs to the caller.

Actual model meta enumeration matches RFC totals:tiny19338560,small125854464,
base493553920,large1510082560. Meta counting allocates no full model weights;
only tiny and deliberately reduced configurations execute CPU forward/backward.
This review reuses existing code and tests; no architecture or numerical behavior
was changed. Sources:[verified] RFC-001/current implementation and the executed
reference tests, at the reviewed commit above. Component-paper citations remain
in RFC-001 and the verified research source ledger.

## Follow-up coverage

M4.1 remains in progress: add a dedicated fixed-seed initialization/residual-scale
regression, explicit non-finite numerical-input policy/tests, and a tied-embedding
combined-gradient oracle before marking the whole milestone complete. Existing
counts and functional output tying do not replace that gradient-specific oracle.
KV-cache, activation checkpointing and accelerator optimization remain M4.2.
This bounded review does not test GPU performance or mathematical reasoning quality.

## Reproduce this review

```bash
uv run --frozen --extra cpu pytest tests/unit/test_model.py tests/integration/test_data_pipeline.py::test_packed_batch_runs_real_decoder_backward -q
```

Executed result:18 passed in6.56s, CPU. These are existing tests rerun for the
review, not18 newly added tests or an additional training experiment.

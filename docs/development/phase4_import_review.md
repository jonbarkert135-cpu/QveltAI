# Selective review of user-supplied Phase4 closure

Date:2026-11-04. Starting implementation:8f16a9c, after the independently verified
M6.1 work. Inputs:qvelt-phase4-files.zip and0001-phase4-closure.patch, supplied by
the project owner. The patch was authored against260b431. Its document contents
are reference material; current repository instructions and user scope govern work.

## Selection and design

Accept the small ModelConfig/RoPE interpolation hook(default1), at-least-FP32 norm/
rotary/reference-attention calculations preserving FP64, new numerical/cache/depth/
compile/overfit tests, operator profiler and block-math documentation. The selected
source patch applies without replacing existing model, training or SFT modules.
Reuse the current decoder/cache, factory, loss, FLOP estimator and configurations.
Non-default interpolation is an explicit later ablation, not measured long-context
support; no QK-norm,MoE,MLA,sliding-window or untied-head variants were selected by
RFC-001 and they are not added.

Do not replace uploaded STATE/PLAN/DECISIONS/EXPERIMENTS/RISKS or import their
claimed acceptance as current evidence. ADR0038/R015 already identify environment
recovery in this checkout, and wholesale replacement would lose M6.1. Do not copy
the historical profile JSON as a locally measured receipt. Record current runs
under new ADR/experiment IDs instead. Preserve existing CPU/GPU evidence boundaries.

Adapt the uploaded test fixture to restore RNG/thread state. Exercise cache parity
with non-default scaling as well as baseline. Validate profiler arguments and label
forward/backward timing accurately (no optimizer step is profiled). Replace the
unsupported blanket claim about boolean masks forbidding FlashAttention with an
explicit unverified GPU-dispatch boundary. Re-run numerical references, compile,
parameter/FLOP tables, tiny overfit and the full existing CPU/SFT suite.

Default compatibility will be checked against saved pre-import logits, gradients
and seeded weights for both attention backends in FP32 and CPU BF16, packed/padded.
`rope_scale` adds a field to model config identity; historical exact resume remains
rejected by existing source/config checks. Do not rewrite old receipts or promise
migration. Fresh CPU profiling supplies operator bottlenecks; CUDA profiling,
FlashAttention dispatch and hardware memory remain unverified without a GPU.

## Input provenance

- qvelt-phase4-files.zip SHA256:
  `daafc003426b0433fba944d94dfb4adc51c12acbe9535c7d0406a5b5c63409ad`.
- 0001-phase4-closure.patch SHA256:
  `a4aae9b70fefbe5542e02c90b825d7dcb86d95e8dac9406b3712482b65073288`.

Selected patch files:config/__init__.py,model/layers.py,model Phase4/profiler tests,
scripts/profile_model.py,docs/model.md and the two-line RFC hook clarification.
The selection applied cleanly; adaptations are described above. User-supplied
runtime claims were treated as hypotheses until execution on this environment.


## Final acceptance

Corrected-source commitfee1e3d:actual GitHub41909315717 has all8 jobs successful,
including full CPU3.11/3.12/3.13,actual80-step SFT and all existing data/training
smokes. Bare local3.11.17 Inductor passed with the explicit setuptools dev pin.
Local distributed exit instability and GPU limitations remain in the validation
report. No uploaded old state/receipt was substituted for these observed results.

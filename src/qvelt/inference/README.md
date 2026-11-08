# Native CPU inference

## Current scope

Updated 2026-11-08 against code at334b4c6.

Original bounded CPU FP32 greedy/seeded generation,true padded batching,
request-local KV/RNG,cooperative cancellation/deadlines and cumulative snapshot
streaming. Native qvelt-inference-v1 exports are self-contained safetensors/
tokenizer/config/provenance bundles with strict source/payload checks.

The bounded CPU reference also implements [FastAPI chat JSON/SSE](api.py),
bearer auth, microbatching and post-validation queue/generation deadlines,
[builtin HF export](huggingface.py) and [HQQ weight-only quantization](quantization.py).
Body-read timeouts, TLS/rate limiting and tenant isolation require their own
deployment boundary; they are not supplied by queue deadlines or bearer auth.
GPU/high-throughput engine execution, large-model quantization and production
model-release acceptance remain unverified. See
[serving scope and commands](../../../runbooks/05_serving.md).

See [runbook09](../../../runbooks/09_inference_export.md) and
[design](../../../docs/development/inference_design.md).

## Archived scope note

Superseded on 2026-11-08 by [Current scope](#current-scope); the old readiness
wording did not distinguish implemented CPU interfaces from deployment acceptance:

> No HTTP/GPU/quantization/external-runtime/model-release readiness claim.

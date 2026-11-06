# Native CPU inference

Original bounded CPU FP32 greedy/seeded generation,true padded batching,
request-local KV/RNG,cooperative cancellation/deadlines and cumulative snapshot
streaming. Native qvelt-inference-v1 exports are self-contained safetensors/
tokenizer/config/provenance bundles with strict source/payload checks.

See [runbook09](../../../runbooks/09_inference_export.md) and
[design](../../../docs/development/inference_design.md).
No HTTP/GPU/quantization/external-runtime/model-release readiness claim.

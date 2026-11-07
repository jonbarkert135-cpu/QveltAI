# Expanded Phase8 design

Date:2026-11-07. Extend the accepted native engine;do not replace original model
math or promote an engineering snapshot as a trained reasoning model.

- Keep current CPU FP32/cache/request limits and exact default sampler behavior.
  Add top-k/top-p/min-p and documented repetition/frequency/presence controls,
  independent probability oracles,and an Engine protocol shared by HTTP and RL.
  Existing copy-on-append cache remains the bounded reference;no paged-cache speed
  claim or custom quantization kernel without measurements justifying one.
- Export reviewed checkpoint/tokenizer inputs as builtin HuggingFace Llama config,
  safetensors and fast-tokenizer/chat-template files,without trust_remote_code.
  Permute Q/K projection rows to convert adjacent-pair RoPE to split-half RoPE.
  Verify full/cached logits,tokenization/chat framing,tied head and linear scaling
  through the actually installed Transformers implementation. Conventional HF
  special-token parsing requires rejection of control spellings in chat contents.
- Use optional pinned HQQ pure PyTorch weight-only int8/int4,keeping embedding/head
  and norms FP32. Quantized storage/reload and scoring must use the real library,
  with logit/NLL/PPL/accuracy differences and memory payloads recorded. CPU fallback
  dequantization is not claimed to accelerate inference or prove GPU support.
- Minimal optional FastAPI app:strict chat schema,OpenAI JSON/SSE,actual queued
  microbatching with one model worker,capacity/deadline/disconnect/shutdown checks,
  bounded body/output/cache,health/readiness and constant-time bearer auth hook.
  Convert native cumulative Unicode snapshots to append-only SSE by withholding
  undecodable byte suffixes and pending stop prefixes;prove concatenation parity.
  No prompt/key/raw validation-body logging. One process only;TLS/rate limiting
  and high-throughput external deployment are explicitly separate.
- RFC linear position interpolation is an explicit derived model/config,never an
  undocumented context increase. Evaluate original synthetic needle controls across
  lengths/depths plus likelihood metrics;zero baseline cannot establish preserved
  long-context quality. Verify cached/full and HF equivalence at extended positions.
- Build reproducible training/serving Docker targets from immutable image digests
  and uv.lock,nonroot runtime,explicit model mounts,no weights/data/credentials in
  build context. Test actual local Docker capability/build/run when available.
- Reuse verified native evaluation receipts for explicit HQQ/context variants,
  preserving old recipe digests when new fields use defaults. Derive matched
  quantization deltas only from the same checkpoint/prompts/settings/source. Record all numbers/config/source/checkpoint/
  commands;README remains free of numeric result prose. Publish model-card template,
  runbooks/05_serving.md,state/decision/experiment/risk updates and CI evidence.

Acceptance:independent sampler references;decode seed/cache/stream regressions;
real AutoModel/AutoTokenizer export and actual quantization tests;OpenAI API/SSE,
concurrency/batching/auth/limits/timeouts/disconnect/shutdown tests;source-clean
standalone CPU experiment with honest quantization/needle outcomes;Docker smoke;
full applicable checks and remote CI. GPU/vLLM runtime is not assumed:read its
pinned implementation/registry and distinguish architectural support from an
executed deployment. Final800 remain unscored;no paid or final-scale training.

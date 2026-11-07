# Inference,export and serving

The native Engine owns private request RNG/KV state and bounded padded batches.
Sampling includes temperature,top-k,top-p,min-p,penalties,seeds and stop strings.
The HTTP worker forms real queued microbatches and returns OpenAI-style JSON/SSE,
with auth,body/admission limits,cooperative deadlines and shutdown cleanup.

Native/HQQ loading remains CPU FP32,bounded model/context size. HQQ uses actual
int4/int8 packing with safe numeric tensor reload;CPU dequantization is not a
speed claim. HF export uses builtin Llama with a verified Q/K rotary-row permutation,
indexed dense shards and standard config/tokenizer files. DCP consolidation is
required before direct dense export. External vLLM architectural support is read
from source;GPU runtime acceptance remains unrun.

Linear RoPE interpolation changes the explicit config. Debug needle accuracy was
at the floor before extension,so execution at a longer context establishes no
quality preservation. Read [serving commands and measured limitations](../reference/runbooks/05_serving.md).

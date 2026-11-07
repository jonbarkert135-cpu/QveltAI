# Qvelt model card template

This is a release template. Current checkpoints and workflows are engineering
references,not a validated general-purpose assistant. Complete the identity and
rights fields for the exact model before a model release.

## Identity and provenance

- Model identifier/revision:record the immutable release identifier.
- Checkpoint manifest/model payload,tokenizer,config and evaluator hashes:copy the
  exact identities from its verified RESULTS entry and receipt.
- Training data/run manifests and parent checkpoints:provide the complete lineage,
  including SFT,preferences,RL,adapter merging,context extension and quantization.
- Release status:engineering reference until source,quality and hardware gates pass.

## Intended use

Research and reproducible local experiments in original decoder language modeling,
math reasoning,verification and serving. State tested model/language/device/precision
and workload from the artifacts. The framework and small controls do not establish
parity with a commercial pretrained model.

## Architecture and interfaces

Original pre-RMSNorm,GQA,adjacent-pair RoPE,SwiGLU decoder with tied embedding/head.
Native generation owns per-request KV and RNG state. HF export uses builtin Llama
with an explicit Q/K row permutation;no remote Python code is needed. Describe
native,HF and HQQ formats separately. A native quantized snapshot is not automatically
a vLLM/HF quantized checkpoint.

## Training data and rights

Identify sources,revisions,licenses,permitted uses and preparation hashes from
DATA_CARD and actual training manifests. Describe tokenizer fitting,cleaning,
mixture,splits and benchmark exclusions. Existing debug runs use original generated
fixtures. Evaluation-only approval grants no training permission. External teacher
use remains a separate explicit gate. Code licensing does not clear data or weights.

## Evaluation

Use only the verified generated [RESULTS.md](RESULTS.md). Link entries/receipts
instead of maintaining a second manual score table. Record prompts,shots,budgets,
sampling,seeds,verifiers,failure counts,language/domain slices and audit coverage.
Limited development prefixes are not official full benchmark scores.

For quantization,link matched-source NLL/perplexity/accuracy differences,packed
payload hash,grouping and library/backend. For context extension,link the needle
length/depth grid and baseline. A zero retrieval baseline cannot establish preserved
long-context quality. RoPE/cache equivalence is separate from retrieval ability.

## Limitations and excluded uses

Debug training does not provide trustworthy mathematical reasoning. Correct final
numbers do not verify traces. Verifier/code controls cover a bounded subset,not
arbitrary symbolic math,Python or formal proofs. Template-set intervals do not
establish population accuracy. List blocked benchmarks,context failures and unknown
training history;lexical audits are not semantic cleanliness certificates.

Do not use an unvalidated model for high-stakes medical,legal or financial decisions.
Language coverage,bias,robustness,privacy and adversarial behavior require independent
release evaluation. An HTTP API implies no safety/alignment rating or human audit.

## Serving and safety

The native CPU reference has bounded requests,one model worker and cooperative
cancellation/deadlines. It provides no kernel preemption,tenant isolation,TLS,rate
limiting or moderation. Use bearer auth behind an appropriate private boundary/
reverse proxy. Keep credentials outside model/config artifacts and never log
prompts,authorization headers or raw validation bodies. Untrusted tool execution
is unsupported. Chat control spellings are rejected to preserve framing.

Validate the exact external engine,hardware,dtype and kernels before compatibility
or throughput claims. Source architecture support is not a deployment measurement.
Keep quantized storage and runtime latency claims separate.

## Reproduction

Provide exact ledger commands,locked environment,hardware/container digests and
[the serving runbook](runbooks/05_serving.md). Preserve negative findings and known
failures. Share sanitized artifacts through the repository,not private data or keys.

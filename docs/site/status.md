# Limitations and status

The authoritative current status is [STATE](reference/STATE.md),with experiments,
source commits and CI evidence. This page separates implemented interfaces from
unverified quality/hardware. It contains no second editable benchmark table.

| Area | Implemented evidence | Remaining acceptance |
| --- | --- | --- |
| Tokenizer/data | Byte BPE,packed masks,hashes,loader replay,property tests | Production source mix,rights and complete exclusion snapshots |
| Decoder | Original dense GQA/RMSNorm/RoPE/SwiGLU,component/gradient/cache tests | Main-run quality and hardware calibration |
| Training | CPU optimization,atomic resume,DDP references,DCP state checks | Intended GPU/NCCL/FSDP2 and large-run resources |
| Post-training | Full/LoRA SFT,DPO,bounded GRPO/verifiers | QLoRA,general-problem RL,independent math gains |
| Evaluation | Native metrics,actual optional harness,receipts/audits/regression | Blocked public sources,full Python isolation,model release |
| Serving | Native cache/sampling,HF,HQQ,HTTP/SSE,Docker tests | GPU/vLLM runtime,large quantization,production isolation |
| Safety | Small bilingual design fixture and exact masking | Independent red-team/human evaluation and trained refusal behavior |
| Release | Version policy,checks,workflow and templates | Explicit operator decision and artifact-specific gates |

Native inference/HQQ is limited to CPU FP32 and small models. The tutorial is a
fixture run. Earlier generated math/retrieval controls failed;this is not an
independently validated reasoning model. Final holdout data remains unscored.
No commercial-lab parity,universal compatibility,semantic decontamination,complete
security or useful safety guarantee is claimed.

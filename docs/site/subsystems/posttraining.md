# Post-training and safety data

Chat frames use explicit role/end-turn controls and shifted assistant-only masks.
Historical reasoning fields require tokenizer-v2. The same SFT loop supports
full fine-tuning and original frozen-base LoRA. QLoRA is not implemented.

DPO has a bounded reference-model policy and resume evidence. GRPO samples real
native-engine rollouts and uses correctness/format/length/KL controls,with strict
offline policy replay. Current RL environments and model-size limits are explicit.
The bounded Fraction/SymPy verifier is not a general mathematical proof checker.

The [post-training runbook](../reference/runbooks/03_posttraining.md) includes
rejection data,license/teacher switches,instability diagnosis and readable audits.
Earlier toy RL reward increased while outputs collapsed to one answer;the
[ledger](../reference/RESULTS.md) and state preserve that negative finding.

Phase9 adds a small original English/Russian refusal/helpfulness SFT design fixture,
including benign contrast cases. `scripts/make_safety_data.py` prepares it through
the real family-isolated assistant-mask pipeline. Its data test verifies exact
supervised response tokens. Preparing or briefly fitting these examples does not
demonstrate safe behavior. See the [independent safety plan](../responsible-release.md).

# M6.1 design: original packed assistant supervision

Status: single-process CPU/demo acceptance verified, 2026-11-04; see validation_sft.md.

Reuse the packed trainer, token-sum accumulation, validation and atomic
model/Adam/RNG/consumer checkpoints. Add a distinct SFT objective and pinned
weights-only initialization from a verified Qvelt checkpoint; start a new optimizer
and schedule, while ordinary resume retains the exact SFT state.

Chat schema v1 accepts optional system then alternating user/assistant turns,
ending with an assistant. Format explicit BOS/role/content/end-turn/EOS tokens.
Only assistant content and its end-turn tokens are targets; role headers, prompts,
outer EOS and padding are masked. Plain-text control spellings remain ordinary
text. Reject empty messages, unsupported roles/order and examples with no targets.

Right truncation retains a prefix of at most context+1 tokens, keeping only the masked outer EOS and never inventing
an end-turn when a response was cut; reject a cut before any assistant content.
Pack complete (possibly explicitly truncated) conversations as independent segments;
never split one conversation across blocks. Layout v2 declares the assistant
objective; layout v1 and pretraining behavior remain unchanged. Derive masks from
explicit role controls in each complete segment, including an unterminated final assistant response.

Supervision input is a licensed raw bundle of JSON conversational documents with
family metadata and explicit training permission. Split whole families by the
existing deterministic hash before packing, using the same tokenizer as the base
checkpoint. Preserve source manifests, licenses, tokenizer and preparation recipe.
Initial acceptance uses original exact-verified bilingual mathematical examples,
no external teacher or corpus. Real-source license/quality approval stays separate.

Acceptance: independent mask/loss/gradient and truncation/packing references;
invalid roles, policies, identity and corruption failures; tiny base checkpoint ->
SFT optimization with recorded train and heldout masked losses, exact weights/
Adam/loader/RNG replay, and all existing tests/lint/types. No GPU or reasoning-
quality claim follows from a CPU fixture overfit.

# Expanded Phase 6 contract

Date: 2026-11-06. This extends the accepted M6.1–M6.3 CPU implementation;
historical negative evaluations remain valid evidence, not acceptance targets.

1. Keep packed full SFT as the main path. Add an original FP32 LoRA option to
   the same packed optimizer/checkpoint loop, with frozen base weights and exact
   adapter replay. No bitsandbytes/QLoRA or external runtime support is implied.
2. Chat schema 1 stays unchanged. Schema 2 permits a separate assistant reasoning
   field encoded using explicit tokenizer-v2 think controls. Ordinary user text
   cannot inject those controls. Shifted loss covers assistant reasoning, answer
   and end-turn, excluding prompts/roles/outer EOS/padding. Prefix truncation
   never creates a completed response. Multi-turn and packed isolation stay tested.
3. Add bounded exact, numeric and polynomial verifiers. Never pass generated
   strings to eval, sympify or parse_expr. A small allowlisted AST constructs
   rational polynomial coefficients; SymPy compares those trusted coefficients.
   Symbolic scope is polynomial identity over rational coefficients, not arbitrary
   expression equivalence or step-by-step proof verification.
4. Rejection sampling uses the native Phase 8 GenerationEngine with pinned own
   checkpoint/tokenizer identities, private seeds and fixed budgets. Record every
   candidate, reason for rejection, difficulty pass@k, normalized trace diversity
   and selected chat data. Correct outcomes do not certify reasoning steps.
   Local source ingestion requires immutable provenance, a permissive license,
   usage permission and prior DATA_CARD approval. External synthetic traces are
   disabled by default and require a separate explicit terms acknowledgement.
5. Preserve sigmoid DPO and its independent value/gradient tests. Frozen reference
   costs are explicit; no unverified variant is added. GRPO uses the native engine
   and teacher-forced sampled-action log probabilities. Add configurable bounded
   format/length shaping; correctness, format, truncation and length are separately
   logged. No correctness reward for incomplete or malformed completions.
6. Offline GRPO replays bounded, checksum-bound rollout windows only when the
   current policy, prompt, reference, actions and recomputed log probabilities
   match. It is not arbitrary stale-data off-policy RL. Reject policy mismatch.
7. Add a fixed original arithmetic mechanism control running SFT → DPO → GRPO.
   Preregister the tiny task/budgets/seeds before running. Show actual sampled
   reward and a fixed-seed evaluation before/after; report failure honestly if
   reward does not rise. This is a deliberately learnable engineering fixture,
   not general math improvement, and does not touch the frozen final 800 cases.
8. Lean is a design document only: pinned Lean/mathlib, isolated processes,
   resource budgets, forbidden sorry/axioms, statement faithfulness and licensing.

Acceptance: adversarial verifiers, exact reasoning masks, independent LoRA math
and frozen-base/replay checks, native-rollout log-probability parity, offline
replay/rejection tests, full local checks and source-bound execution receipts.
Runbooks document genuine commands, supported CPU scope, reference-memory costs,
source/terms gates and realistic limitations. GPU/QLoRA/Lean execution and public
dataset/model quality require separate evidence; no final-scale runs.

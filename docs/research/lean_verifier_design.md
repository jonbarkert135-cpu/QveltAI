# Optional Lean verifier track (design only)

Date: 2026-11-06. No Lean runtime, theorem dataset, proof generation or theorem
accuracy has been implemented or measured. This is independent of answer GRPO.

Use a pinned Lean4 release and immutable mathlib commit with a recorded license
inventory. Compile a small approved statement set before any model rollout.
Record statement hash, imports, allowed axioms, toolchain and environment hash.
The model supplies proof text only; never permit it to replace the statement,
imports, compiler options or verifier executable. Reject `sorry`, `admit`, unsafe
axioms, alternative statement declarations and unreviewed extensions. Textual
filters alone are insufficient: inspect elaborated declarations and transitive
axiom dependencies, then kernel-check them under the allowlisted environment.

Run each check in a separate resource-limited process/container, without network,
credentials or writable project files, with a wall deadline, memory/CPU/output
limits and process-tree termination. Compiler success is only a reward after the
expected declaration and axiom audit pass. Preserve failure diagnostics and audit
samples; limit reward on timeouts, parser errors and truncated generations to zero.
A versioned verifier adapter would accept statement ID plus bounded proof bytes
and return status, runtime, resources, expected declaration identity and checked
axioms. It must not execute arbitrary Python in the trainer. Online scheduling,
parallel compiler workers and replay need their own tests and provenance contract.

Split by theorem dependency/family and deduplicate statement/proof sources before
training. Hold out statements and imports; a proof can simply invoke an already
imported theorem. Record licenses for mathlib, statements and synthetic teacher
outputs separately. Model-generated informal-to-formal statements need human
review: proving a different formal statement does not solve the original problem.

Realistic next step: a few elementary Nat/Int equalities with fixed imports and a
tested sandbox, followed by supervised syntax learning. Our 19.34M tiny model has
no demonstrated Lean syntax competence; the 125.85M small configuration is still a
resource plan, not an executed reasoning model. Small vocabulary/context and CPU
elaboration cost make broad theorem discovery unrealistic as an immediate goal.
Do not advertise miniF2F/competition proof capability or numerical gains without
a licensed independent evaluation and actual kernel logs.

Research basis: survey S28, Xin et al. (2024), *DeepSeek-Prover: Advancing Theorem
Proving in LLMs through Large-Scale Synthetic Data*, previously reviewed in the
survey. The paper motivates verification; its scale and results do not validate
this design. Current Lean APIs/toolchain/sandbox behavior remain **unverified**.

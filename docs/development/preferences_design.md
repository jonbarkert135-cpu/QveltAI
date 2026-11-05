# M6.2 preference optimization design

Date:2026-11-05. Implement original single-process CPU DPO on the existing decoder,
strict chat formatter and trusted manifested SFT checkpoint. Reuse artifact/RNG/
Adam/replay helpers and existing token/seed/config validation. Do not add TRL or
external teachers. Project ponytail and local TRL DPO/reference guide inform review;
primary Rafailov et al.2023 arXiv2305.18290v2 section4 equations7/gradient and
appendixB were read. Reference support is positive for finite-softmax decoder.

Choose baseline sigmoid DPO rather than new IPO/hinge variants:one verified
objective minimizes softplus(-beta*((chosen-rejected)policy-(chosen-rejected)ref)).
Use sequence SUM response log probabilities including end-turn,mask prompt/outerEOS/
padding and never average by response length. Reference initialized from the same
pinned SFT weights as policy,eval/no_grad/frozen. Recompute reference on demand;
no cached logits and therefore no ambiguous cache invalidation. Reject empty,
identical,malformed/contradictory pairs and sequences exceeding context;do not
truncate away a preference decision. Pair group/split and provenance are explicit.

Register original Apache-2.0 exact-answer math pairs before generation. Correct
answer versus deterministic verified wrong integer uses existing exact generator;
labels describe verifier correctness,not human preference. Family-heldout validation;
no benchmark/generalization or win-rate claim. Production/preference-scale input
clearance and CUDA/distributed preference training remain unsupported.

Small YAML job binds manifest/tokenizer/SFT hashes,beta,seed,batch,horizon,optimizer.
Deterministic cyclic pair order,cursor at complete updates;checkpoint model,Adam,
RNG and application state using existing atomic engine. Bind source/software and
pair/reference identities on resume;reject changes before mutating persistent state.
Use fixed train/heldout loss and reference-relative reward margin/accuracy with ties
counted separately. These are implicit offline pair scores,not sampled generation
quality or measured KL. Tests:independent NumPy loss/analytic gradient,mask and
length/padding reference,invalid pairs,no reference gradients,actual tiny updates,
exact continuous/resumed weights/Adam/RNG/cursor/loss curve. Report reference model
cost,length sensitivity,limitations. No paid/final-scale compute.

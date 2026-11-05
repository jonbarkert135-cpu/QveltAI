# M6.3 bounded original outcome-GRPO design

Date:2026-11-05. Original CPU FP32 only. Continue from accepted SFT candidate,
never promote the degraded DPO checkpoint. Reuse known licensed exact-answer SFT
prompt fixtures and bounded Fraction verifier;register RL use before sampling.
No public data,teacher,reward-model/tool execution or paid/GPU compute.

Primary source DeepSeekMath arXiv2402.03300v3 section4.1.1 Eq3/4 and4.1.2 read.
Select outcome-supervised GRPO:groups G>=2;mean-center/population-std normalize
scalar rewards;explicit epsilon,zero-variance groups have zero advantage. Average
masked objective per completion then per group/prompt (not globally by token count).
Min of ratio*A and clipped-ratio*A,plus beta*(exp(logpref-logp)-(logpref-logp)-1).
Detach old/reference logps and normalized advantages. Reject nonfinite values/
overflow,empty masks,invalid metadata. Verify numerical value and analytic gradient
against independent NumPy,including clip signs,zero groups and padding;SymPy checks
KL first/second derivatives and nonnegative minimum at0.

One policy update per fresh rollout batch. Policy weights remain frozen during
sampling;record all old logps,source policy/checkpoint hash,prompt/group IDs,masks,
stop reasons,reference identity,sampling temperature/seed/RNG. Temperature1 for the
first on-policy path so behavior logps equal raw policy logps;other temperatures
must use the corresponding distribution consistently or be rejected. No hidden
rejection sampling or forced gold completions. Hard token/context caps;include
actual generated stop actions in policy loss. Truncated/invalid/prematureEOS answers
score0;known gold must not leak into generation. KL estimator is a sampled penalty,
not a measured exact KL under stale rollouts. No multi-epoch stale policy reuse.

Checkpoint at whole rollout+update boundaries using existing atomic engine. Restore
model/Adam/RNG,prompt cursor,applied update count,reference/data/recipe/source/software
identity. Failed/zero-variance groups are logged;do not fake learning from zero
reward. If every group has zero variance and no KL signal,skip optimizer and record
that fact. Assert at least one actual nonzero policy-gradient update in acceptance.

Use a fixed seeded training-only subset from existing SFT family split and separate
original validation control. Bounded smoke verifies weights/Adam/RNG/cursor/rollout
and loss replay at a mid-run boundary. Report reward/format/truncation/group-variance
statistics and negative findings,not reasoning improvement. Broad reasoning,
production/hardware and model promotion require later independent evidence.

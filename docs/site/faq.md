# Frequently asked questions

## Does one hour produce a useful assistant?

No. The tutorial exercises an actual tiny model on original fixtures and records
engineering evidence. Meaningful pretraining,independent reasoning accuracy and
safety require substantially different data,compute and review.

## Can I train or release on every registered benchmark?

No. Evaluation approval does not grant training or redistribution rights. Some
planned sources remain blocked. Read the exact data-card record and usage policy.

## Why does a checkpoint reject my new version?

Exact replay binds runtime source,dependencies,config,data and topology. A security
upgrade changes that environment. Check out the old version to reproduce a
historical result;use explicit safetensors initialization for a new lineage.

## Are GPU,FSDP2 and vLLM verified here?

They have capability-gated code or architectural compatibility where documented.
Actual intended-server CUDA/NCCL/kernel/resource tests remain mandatory. CPU DCP
and read upstream source do not establish GPU acceptance.

## Does the model card certify safety or data rights?

No. It is a template. Actual trained artifacts need complete lineage,rights,
contamination reports,quality evaluation and independent safety review.

## Why retain negative scores and failures?

They prevent unsupported promotion and make regressions auditable. A rising toy
reward,zero quantized accuracy delta or successful context allocation can coexist
with an unusable model.

## Is a passing vulnerability scan a security proof?

No. Advisories are point-in-time data;optional dependencies and cluster trust still
matter. See the [security review](security.md) for scope and unresolved findings.

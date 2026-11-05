# Selective Phase5 import review

User uploads target260b431 and are explicitly WIP. See
[evidence/phase5-upload-identities.json](evidence/phase5-upload-identities.json)
for actual SHA256/byte identities. The existing repository had already advanced
through current SFT and Phase4 closure to b82131c. Uploaded instructions were
review material; project/user scope and current source remain authoritative.

Selected config,trainer,stability,evaluation,recipes,sweep/report and checkpoint
improvements; three-way adapted CLI,PretrainingJob and packed loop. Retained SFT
commands,objective,base-weight digest,assistant masks and train probes. Do not
import old STATE/PLAN/ADR/experiment claims or replace current source blindly.

Review corrections:complete backward synchronization on recoverable bad windows;
check persistent model/Adam state;initial finite rollback snapshot and correct
DCP Adam time0;strict byte-versus-semantic auto-resume;symlink-safe quarantine;
exclusive archive/output creation;allowlisted logs;interrupted final JSON handling;
exact final-target budget;batch overrides before accumulation derivation;nonzero
sweep child failures;bounded generation with collective error propagation.
FSDP CUDA hardware guards,mixed precision and unverified compile combinations are
explicit. Added original full-tiny hard-kill,compile and four-size controls absent
from the WIP. Source data,benchmark commands and results were freshly executed.

A real compile measurement exposed bundled setuptools metadata in environment
capture. Only its exact vendored path is excluded;independent installed-version
conflicts remain fatal. Existing SFT/model/numeric/distributed tests are preserved.
See validation_phase5.md for honest current test and hardware boundaries.

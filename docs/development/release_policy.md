# Versioning and release policy

Qvelt uses semantic framework versions. Version0.1.0 is an experimental pre-1.0
framework candidate,not a production or trained-model claim. Preparing its tag,
notes or workflow does not publish a release. The owner/operator chooses the
release action after checking its evidence.

## Version identities

`pyproject.toml` and `qvelt.__version__` must agree. Stable tags are `vMAJOR.MINOR.PATCH`.
PATCH fixes compatible framework behavior;MINOR may add or change pre-1.0 APIs;
MAJOR marks a declared stable compatibility boundary. Model/tokenizer/data artifacts
have independent hashes and schema versions;they are not identified by the package
version alone. Keep a dated CHANGELOG entry with source/evidence and known limits.

Exact resume binds source,lock,config,data,precision and world size. Even a PATCH
security update may invalidate exact replay. Reproduce historical results at their
recorded commit/lock;for a new environment initialize a new explicit lineage from
verified safetensors. Never relabel an old checkpoint or silently bypass a failed
compatibility check. A new schema needs reviewed migration and tests.

## Framework release

The release workflow validates version/tag alignment,builds wheel/sdist,installs
the wheel in a fresh environment,checks source licensing and produces checksums.
Required CI/security/docs/reproduction evidence must be reviewed. It can create
a **draft** release after an operator creates a stable version tag;no automatic
PyPI upload or public model release is configured.

The current candidate includes a substantial dependency security update:
Torch2.13.0 CPU/CUDA13.0 profile,Transformers5.10.4 and pytest9.0.3. The previous
Torch2.9.1/CUDA12.8 environment remains historical evidence,not the current supported
lock. CUDA13 driver/kernel/FSDP acceptance on the intended server is unrun;inspect
hardware before selecting that profile. No precise driver minimum is asserted here.

## Trained-model release

Before uploading any weights,complete the artifact-specific data/model cards,
source/data/parent lineage,redistribution rights,all required contamination reports,
independent quality evaluation,content-safety/human audits and deployment boundaries.
A framework release cannot grant these approvals. Current debug models fail useful
quality/safety gates and are not cleared for a public model release.

Optional Hub upload automation is deferred until an eligible model exists. The
existing HF exporter provides standard tooling-compatible local files,not upload
authorization. No Hub upload or release tag is created by this phase's developer
checks. Git main publication continues under AGENTS.md.

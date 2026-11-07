# Phase9 documentation,hardening and release readiness design

Date:2026-11-07. Framework readiness and trained-model release are separate gates.
No model release,Hub upload,paid compute or final-scale training is implied.

- Build a strict MkDocs site with Mermaid architecture diagrams,a bounded first-
  tiny tutorial,subsystem pages,FAQ,troubleshooting and explicit status. Reuse
  existing runbooks/cards/RESULTS through generated pages,not parallel score tables.
  Validate actual rendered navigation/diagrams with Chromium and inspect a preview.
- Add seeded Hypothesis tokenizer/packing invariants and bounded parser fuzzing,
  independent of the implementation. Measure critical-path branch coverage before
  and after meaningful failure tests;do not invent a coverage percentage or hide
  required paths with exclusions.
- Audit locked dependencies,Git-tracked secrets and deserialization sites. Pin real
  tool versions. Never deserialize external pickle;native weights remain safetensors
  and training-state torch.load must explicitly use weights_only=True. Review DCP
  and distributed-object serialization separately;record real residual risks.
  Keep full machine audit outputs ignored and publish sanitized findings/provenance.
- Reproduce documented commands in an isolated checkout/fresh virtual environment
  and a fresh hosted runner. Use the actual RFC tiny dimensions/vocabulary with
  approved original fixtures;show elapsed time/config/source/hashes,short real
  optimization,resume,evaluation,export and authenticated serving. The one-hour
  budget is an engineering tutorial,not a useful-model training guarantee.
- Review existing data licenses and native contamination artifacts,retain blocked
  benchmark/unknown-history gates. Design content-safety evaluation and implement
  a small original EN/RU safety/helpfulness SFT bundle with assistant masks and
  held-out families. Preparing examples is not evidence of trained refusal ability.
- Add a consistent pre-1.0 version/compatibility policy,CHANGELOG,release workflow,
  issue/PR templates and optional manifest-verified Hub upload preparation. Default
  actions build/review artifacts;publication remains an explicit operator action.
  Model artifacts need model/data rights,complete audits and quality/safety gates.
- Trace every README claim to a real command,test or verified ledger entry. Publish
  pass/fail readiness with remaining gaps and the ordered first-real-run procedure.

Acceptance:strict site and actual browser checks;property/fuzz/security tests and
audits;measured coverage;fresh documented end-to-end tiny reproduction;small safety
data/mask verification;release artifacts/workflow validation;all applicable CPU
checks and CI. GPU/large/production/model-quality acceptance stays explicit.

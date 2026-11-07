# Phase9 release-readiness validation

Date:2026-11-07. Scope:portable CPU framework candidate,not a trained-model release.
Design:release_phase9_design.md;decision:ADR0051;operator checklist:../site/responsible-release.md.

## Evidence already executed

- Updated Torch2.13.0,Transformers5.10.4,pytest9.0.3 after real dependency findings.
  Full installed-extras normalized audit retains one distinct unpatched optional
  sqlitedict advisory(CVE-2024-35515);no advisory ignore. Native-profile audit is
  a separate clean-environment/CI gate,not inferred from optional failure.
- Focused security/DCP/release/tutorial/eval/HTTP tests:61 passed6.82s. Real DCP
  tensor and optimizer-byte reload,malicious globals/byte payloads,relative paths,
  unregistered files and out-of-bounds extents are tested.
- Critical-path coverage:98 original tests versus113 expanded tests. Combined
  statement/branch coverage85.3186% ->89.2463%;statement89.1165% ->92.2194%;
  branches75.4967% ->81.5789%. No coverage exclusions. Three new constructor
  validation statements change the denominator. This covers five named modules,
  not the entire project. Baseline runtime2.9.1,updated runtime2.13.0.
- Property tests:200 Unicode tokenizer roundtrips,70 randomized disk-packing
  examples,1200 bounded adversarial parser strings and250 exact arithmetic/
  polynomial oracles. Hypothesis derandomized,deadline disabled,case bounds fixed.
- Strict MkDocs build and actual Chromium154.0.8037.57/Playwright1.64.1 navigation:
  five routes,two actual SVG images,one H1/page,zero page errors. Light/dark/mobile
  captures retained locally;architecture screenshot inspected. Original Mermaid
  diagrams have editable sources;no CDN needed to render the built site. The
  standard theme needed a local compatibility override for disabled highlighting
  in dark mode;external highlighting is disabled. This is not a screen-reader audit.
-132 offline tracked-tree credential findings reviewed as exact false positives,
  predominantly vendored examples,environment-binding names and test values.
  Baseline hashes bind path/detector/value;new findings fail. Bare hex-entropy
  excluded explicitly because of provenance hashes. No provider verification,
  no full-history audit,no claim that vendored helpers are comprehensively secure.
- Small original bilingual safety/helpfulness SFT construction verified against
  exact assistant labels and disjoint family splits. Its24 examples do not
  demonstrate trained safety or replace independent human review.

## Real failures retained

The first diagnostic tiny run was interrupted by the operator-agent before root
storage exhaustion;checkpoints were retained on the scratch filesystem. The
second completed base/resume/SFT/evaluation but receipt export exposed duplicate
multi-source audit-role rejection. The fix permits only repeat audit-source roles
and binds ordered path/hash pairs to the evaluation policy;other roles remain
unique. A negative test swaps source hashes rather than record order because
manifest parsing canonically sorts paths. Full final clean reproduction remains
required and is recorded separately once executed.

Initial archive validation rejected a vendored .env.example force-included through
Hatch's discovery of nested VCS-exclusion files. Sdist now uses an explicit source
allowlist with VCS discovery disabled. A subsequent actual wheel/sdist content check
passes;build outputs do not include vendored skills,caches,raw data or checkpoints.

## Remaining release gates

Current model quality,production data rights,complete public benchmark/sandbox
coverage,unknown parent contamination and independent safety evaluation are open.
CUDA13.0/NCCL/FSDP and high-throughput GPU serving are unrun. Optional harness audit
fails. No tag,public GitHub release,PyPI publication or model upload was performed.
Follow the tiny reproduction,hardware/data checks and short small pilot before
committing a real training horizon. Preserve historical receipts with their actual
source/library identities;security upgrades are not exact-resume compatible.

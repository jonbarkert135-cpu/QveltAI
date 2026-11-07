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

## Updated candidate checks and corrections

Initial full updated local suite:910 passed,2 distributed90/120-second timeouts,
5 promotion-fixture setup errors,7 GPU deselected543.60s. The setup errors share
one real version-envelope bug:0.1.0 differs from the historical0.1.0.dev0 producer.
The repair reconstructs the complete historical envelope and checks its original
hash after verifying every current file;it does not regenerate the committed
freeze or change generator/payloads/rights/case identities. The five affected
integration tests pass after repair;unit negative cases reject payload/count/rights
changes. An initial new negative test expected ValueError instead of the actual
ArtifactIntegrityError;the test expectation was corrected.

Fresh CPU+HF+serving profile dependency audit:exit0,no known vulnerabilities and
no skipped Torch local-version record. All-extras optional sqlitedict failure
remains separate. Offline secret gate:132 reviewed findings,zero unexpected.
The hash-only baseline is explicitly excluded from recursively scanning itself;
source/provider detectors remain active for tracked code and vendored references.


## Clean local reproduction and measured performance follow-up

Fresh local checkout7ef8bfd/new virtualenv passed every tutorial step,including
all verification/streaming commands. Package cache was shared;this is not a cold
machine proof. Full RFC tiny19,338,560 parameters,base6 updates,stop3/resume6,
full safety SFT3,original eval with both actual training-source audits,HF/native
parity and authenticated real loopback HTTP. Elapsed477.817426s;HF max absolute
logit error7.152557e-7. Weights unchanged by eval;final800 not model-scored;no
trained-safety assertion. Raw-result hash and environment/source/config identities
are in release-phase9-reproduction.json;one native receipt adds the sixteenth
RESULTS row group,with its actual older candidate identity preserved.

The full-size tokenizer exposed a real decoder audit bottleneck. A15-second
cProfile budget recorded3572 backend vocabulary-size queries,13.523s in that
method. Query once per decoded sequence;the ID check still covers every token.
The exact same two-source audit now takes1.843169s and its entire report equals
the original report. This is a diagnostic comparison,not a GPU/kernel benchmark.
A dedicated backend-call-count/reference-output test verifies the invariant.
Final expanded critical suite114 passed10.97s;combined coverage89.256198%,
statements92.229299%,branches81.578947%,zero exclusions. Initial113-test evidence
above is historical;the final denominator adds one decoding statement.

The focused distributed recheck passed real two-rank DCP/all-reduce and packed
replay,but replicated numeric-DDP resume timed out again(1 failed,2 passed144.48s).
Its initial failure remains unresolved locally. Hosted CI is checked independently;
a green remote run does not diagnose this environment's Gloo issue.

New optional/dev dependency license metadata: MkDocs1.6.1 BSD-2-Clause,
mkdocs-gen-files0.5.0 MIT,pymdown-extensions12.3 MIT,Hypothesis6.170.15 MPL-2.0,
coverage7.17.2 Apache-2.0,Playwright1.64.1 Apache-2.0. pip-audit2.10.1 and
 detect-secrets1.5.0 installed license files declare Apache2.0. Renderer
Mermaid12.0.0 is MIT;verified npm integrity and generated-source/SVG hashes are
recorded. This is a distribution notice review,not a legal opinion about an
unselected production corpus or weights.

## Final current-runtime CI and framework checklist

Source8c082145201d6c27d2798ca5362945ddcc6390e3:all22 jobs in
[CI42384342511](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/42384342511)
successful. This includes full CPU3.11/3.12/3.13 tests,actual optional HF/HQQ/API,
all containers,original training/posttraining/recovery smokes,source-clean first
model tutorial,strict docs/browser,native security/secret/coverage/archive gates.
Receipt records official job/step states. The session proxy blocks hosted blob
artifact/log downloads with403;do not infer unread hosted counts or numeric logs.
The independent local tutorial/evidence are retained and inspectable.

Actual current wheel/sdist content/version validation passes. Installed the wheel
into an isolated target over the same locked dependencies:package path/version,
CPU doctor and all eight actual native stream events match the documented editable
run. This is a wheel packaging check,not another fresh-dependency reproduction.
No tag or draft/public release was created;the operator-triggered draft action is
prepared but unexecuted. Optional Hub upload is deliberately deferred.

| Checklist | Result | Scope or remaining gap |
| --- | --- | --- |
| Site,diagrams,tutorial,subsystem docs,FAQ,status | PASS | Strict build and actual browser;not human screen-reader audit |
| Critical tests/properties/fuzz/coverage | PASS |114 focused;whole CPU suites pass on three hosted Python versions |
| Restricted loading and native dependency/secret audit | PASS | Explicit source/current-tree scope;not hostile-file sandbox/history scan |
| Optional all-extras advisory audit | FAIL | Unpatched sqlitedict CVE-2024-35515;native eval avoids it |
| Clean-machine tiny reproduction | PASS | Fresh hosted runner;independent fresh local venv uses warm cache |
| Rights/contamination release controls | PASS | Existing declared-source controls and blocked rights preserved |
| Production corpus/full benchmark/unknown-parent clearance | FAIL | No selected approved production mix or complete coverage |
| Safety design and small implementation | PASS |24 original examples,exact assistant masks,family split tests |
| Demonstrated useful/safe trained model | FAIL | No independent quality/safety/human acceptance |
| Version/changelog/archive/wheel/workflow/templates | PASS | Framework candidate;draft creation/model upload unexecuted |
| README provenance | PASS | Claim matrix and16 verified native receipts;no unbacked quality score |
| Intended GPU/production hardware | FAIL | CUDA13.0/NCCL/FSDP/large/high-throughput acceptance unrun |
| Local numeric-DDP timeout diagnosis | FAIL | Reproduced locally;hosted green is not a local root-cause fix |

Recommended first real run:tiny tutorial on the intended server;hardware/driver/
precision/recovery and representative-context resource checks;approved exact data
rights/mix/dedup/exclusions/tokenizer;fixed independent domain probes;small200-step
pilot and log review;only then commit the full horizon. Evaluate base before SFT,
preferences or RL. Keep final holdout untouched and stop on answer collapse or
invalid audit/recovery evidence. Exact commands are in runbooks02 and the tutorial.

Final evidence adds one reviewed Base64 false positive:the public npm Mermaid
SHA512 tarball integrity. The final baseline therefore has133 findings(previous132
were source/vendor examples). It is a public artifact digest,not a credential.
After successful clean reproduction,only owned failed diagnostic payload copies
were removed to free scratch space;their manifests/recipes/metrics/logs were retained.
The successful clean tutorial,weights,exports and full local logs remain in scratch.

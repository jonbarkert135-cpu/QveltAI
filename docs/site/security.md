# Security review scope and findings

Security is a point-in-time review,not a safety certificate. The current code
candidate uses Torch2.13.0,Transformers5.10.4 and pytest9.0.3 after the dependency
audit found vulnerable older versions. Current locked versions,actual tool outputs
and source identities must be retained with the Phase9 acceptance evidence.

## Deserialization and trust

Portable weights are safetensors. First-party torch.load calls explicitly require
weights_only=True;the source inventory also rejects direct unsafe pickle,dill,
joblib,eval,exec and unguarded numpy/YAML loads. This is a direct-call inventory,
not whole-program analysis. The original bounded verifier parses data with AST
allowlists and exact arithmetic;arbitrary generated strings never reach eval or
SymPy's expression parser.

Upstream Torch DCP still uses unrestricted pickle for metadata and
weights_only=False for optimizer byte fields. Qvelt replaces those two paths with
a bounded metadata-global allowlist and weights-only byte loader. Real DCP and
malicious-payload tests verify normal reload and absence of execution. Only data-
constructing classes/layouts are admitted;unknown globals/persistent IDs/trailing
payloads fail. The change does not sandbox resource denial of service or make
untrusted cluster ranks safe. Distributed object collectives still require trusted
peers and a protected network. Checksums need an externally trusted expected hash.

HQQ's encoded dtype loader uses eval. Qvelt never calls it:only numeric safetensors
are restored against trusted regenerated metadata. A test makes the unsafe decoder
raise if invoked. HF export uses the builtin Llama architecture and local safe
files;security updates remain necessary even with trust_remote_code=False.

## Dependency audit

```bash
uv run --frozen --extra cpu python scripts/security_review.py \
  --directory runs/security-native --dependencies
```

The tool audits installed pinned distributions through PyPI advisory data,normalizing
Torch's CPU/CUDA local version suffix so Torch is not silently skipped. The local
original Qvelt package is reviewed through source/tests rather than a nonexistent
public advisory record. Audits are not dependency resolution;the frozen lock is
the installed graph. Run each optional profile separately and retain failures.

The optional lm-evaluation-harness dependency sqlitedict2.1.0 has an unpatched
pickle-deserialization advisory,CVE-2024-35515/PYSEC-2026-1939. Native evaluation
does not depend on it. Qvelt's supported harness parity path does not open a
persistent cache;stock SQLite-cache use is unsupported. An all-extras audit remains
**failed for that optional dependency**. No global CVE ignore or fake clean result
is introduced. A model release must review its actual selected profile.

Torch2.13 emits one known JIT deprecation while its own mkldnn helper is imported
through HQQ compilation. The test configuration isolates only that exact upstream
message/module;other warnings remain errors and all numerical assertions still run.
This is a documented compatibility exception,not suppression of a failed test.

## Credentials and source artifacts

The offline detect-secrets scan covers Git-tracked files,including vendored skill
references. Provider/keyword/base64 detectors remain active. Bare hexadecimal entropy
is excluded because the repository intentionally contains thousands of provenance
SHA values;this means an unlabelled bare-hex credential is outside that detector
scope. Reviewed exact false positives are hash-bound in .secrets.baseline;new
findings fail without printing secret values. No guessed credential is tested
against an external service. This is a current-tree scan,not a full history audit.

```bash
uv run --frozen --extra cpu python scripts/check_secrets.py
```

Build contexts and framework archives exclude caches,raw data,checkpoints,credentials
and vendored helper collections. No automatic helper execution occurs. The project
does not claim an exhaustive security review of third-party skills or dependencies.
Use [SECURITY.md](reference/SECURITY.md) for sensitive reporting.

## Serving and release boundaries

Local serving provides bearer auth,body/admission limits and cooperative decode
deadlines. It does not implement TLS,slow-body protection,rate limiting,moderation,
kernel preemption or tenant isolation. Keep it private or behind an appropriate
gateway. A successful container smoke does not establish a secure production service.
Framework checks do not authorize a trained-model upload;see the
[responsible-release checklist](responsible-release.md).

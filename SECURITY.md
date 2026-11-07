# Security policy

Do not open a public issue containing an exploit payload,credential,private data
or unredacted server log. Use the repository's GitHub private vulnerability
reporting channel if enabled;otherwise contact the owner through an established
private channel before sharing sensitive details. No private channel availability
or response-time guarantee is asserted here.

Supported security scope is the current pinned framework candidate and documented
CPU workflows. Historical environments remain available for reproduction but may
contain known vulnerabilities. Do not load unknown checkpoints,join untrusted
distributed ranks,run downloaded skill helpers blindly or assume local HTTP auth
provides tenant isolation. See docs/site/security.md for audit scope and open risks.

Model weights use safetensors;training torch.load calls require weights_only=True.
DCP metadata uses a narrow data-class unpickler because upstream still pickles
its metadata. This blocks arbitrary globals but does not sandbox memory/CPU denial
of service or authenticate a publisher. Checksums require a trusted expected hash.

The optional lm-evaluation-harness dependency sqlitedict2.1.0 has a known unpatched
pickle-deserialization advisory. Do not open untrusted SQLite caches or use stock
persistent harness caching in the supported Qvelt workflow. Native evaluation
requires no such dependency. Release review must retain this finding rather than
globally ignoring the advisory or claiming every optional dependency is clean.

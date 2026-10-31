# M2.2 design: manifests, provenance and local artifact commits

Date: 2026-10-31. Owner: Lead agent. This is artifact infrastructure, not model
serialization, tokenizer training, corpus ingestion or training-state replay.

## Interfaces before implementation

- `qvelt.manifests`: immutable schema-1 dataclasses for run, raw/tokenized data,
  byte-BPE tokenizer and checkpoint metadata, artifact file records, code and
  environment identities. Strict JSON parsing rejects duplicate/unknown fields,
  unsupported versions, non-finite constants and invalid paths/types/counts.
- `manifest_bytes` is canonical UTF-8 sorted compact JSON plus one newline;
  `manifest_digest` is its SHA-256. Referenced manifest IDs use this canonical
  digest; artifact file hashes cover **raw bytes**, a different domain.
- `qvelt.artifacts.ArtifactStore`: bounded-memory file hashing/copying, write-once
  atomic file publication, manifest-last commit, expected-ID and file-integrity
  checks. No pickle/torch/model parsing. Config roles use the existing safe YAML
  loader and are checked against the canonical config digest.
- `qvelt.provenance.initialize_run`: create a new config/provenance bundle using
  actual Git HEAD/dirty status, lock hash, installed package names/versions and
  Python/OS/machine. Never copy the process environment, credentials, Git remote
  URLs or source patches. Record dirty state without claiming it is a snapshot.
- CLI: `qvelt run init CONFIG --directory DIR --repository REPO` creates only a
  new provenance directory; `qvelt manifest inspect PATH` reads metadata;
  `qvelt manifest verify PATH [--expected-sha256 HEX]` audits registered files.
  A different process must be able to inspect/verify without torch installed.

## Schema and trust boundaries

Each manifest has version, kind, owner, producer version, timezone-aware creation
time normalized to UTC, typed metadata, and a nonempty unique-path file list.
Run metadata includes config/code/environment/seed/world-size identities; data
includes declared source revisions/licenses and optional tokenizer linkage;
tokenizer metadata links raw training data; checkpoint metadata links run/config/
data/tokenizer IDs and counters/precision/world size. Required roles express a
contract, not proof that opaque checkpoint bytes can resume training.

JSON is limited to 1 MiB; file lists to 4096 entries, source lists to 1024 and
package lists to 2048. Only fixed typed fields are accepted, not arbitrary
metadata/env dictionaries. Publisher license metadata is not ingestion approval.

Artifact paths are canonical relative POSIX paths: no absolute/drive paths,
backslashes, empty/dot/parent components or symlink traversal. Hash/copy accepts
regular files, not FIFOs/devices. A root is an explicitly trusted local directory;
these checks do not create a sandbox against another actor rewriting that root
concurrently. Hashes detect accidental changes; authenticity requires a trusted
expected manifest digest or other external trust anchor.

## Atomic commit protocol

For Linux/POSIX local filesystems: create a private temporary file in the
destination directory, stream contents, flush and fsync, atomically hard-link
to the absent final name, fsync the directory, then remove the temporary name.
Hard-link publication fails if another writer already created the destination;
there is no overwrite option. Readers see absent or complete files.

Caught file/callback/fsync failures before publication leave no final name and
clean the owned temporary file under normal directory permissions. A killed
process can leave an uncommitted temporary name, which readers ignore. A directory-fsync
failure **after** publication leaves complete bytes visible and raises a distinct
durability-uncertain error; do not delete a file readers may already observe.
Hardware power-loss guarantees depend on the filesystem/storage honoring fsync.
Native Windows/object stores/network filesystems are not validated by this work.

Write artifacts first, verify the declared size/hash and config identity, publish
`manifest.json` last. Unreferenced files left after failure are not a committed
bundle. Do not silently reuse/delete an existing run directory. No distributed
multi-writer checkpoint transaction or topology migration is implemented here.

New artifact subdirectories have their parent entries fsynced too. Bundle init
creates one private directory under an **existing** parent. Regular-file hashing
reads at most its initial size plus one byte and compares observed stat metadata;
a growing file cannot extend the hash operation indefinitely. File copying uses
the initial source size as a ceiling, then rechecks source metadata. A source edit
detected after copying may leave a complete unreferenced copy, not a committed
bundle. Neither method is a filesystem snapshot against hostile concurrent edits.

## Working example

From the checkout with the locked development environment installed:

```bash
QVELT_BUNDLE_PARENT=$(mktemp -d)
uv run --frozen --extra cpu qvelt run init configs/tiny.yaml --directory "$QVELT_BUNDLE_PARENT/run" --repository .
uv run --frozen --extra cpu qvelt manifest inspect "$QVELT_BUNDLE_PARENT/run/manifest.json"
uv run --frozen --extra cpu qvelt manifest verify "$QVELT_BUNDLE_PARENT/run/manifest.json"
```

Init prints a canonical manifest ID. Verify can additionally require a trusted ID
with `--expected-sha256`. Inspect explicitly returns `files_verified=false`;
verification reports the number and total bytes of checked files. Base-only
installation also supports this workflow: no torch import is needed.

The bundle contains resolved `config.yaml`, the copied `uv.lock` and
`manifest.json`. Init records declared seed/world_size=1 and observed Git/package
state; it does not apply RNG settings, validate a GPU, allocate a model or start
training. Dirty code is flagged but its patch is not captured. Package inventory
is not a guarantee of agreement with the lock; no global code/filesystem lock
or installed-code-to-checkout equivalence is claimed. M5 must bind actual launch/
rank/checkpoint state. Run roots and metadata must not contain credentials.

## Acceptance

All metadata kinds round-trip, reject malformed/incompatible data, and expose
stable digests. Tests must cover corrupted/missing files, expected-ID mismatch,
path escapes/symlinks/nonregular inputs, partial writes, file and directory fsync
errors, a process killed while writing, concurrent writers and manifest-last
failure. Run a real CLI config/provenance/inspect/verify flow and demonstrate
corruption detection from another process. Run the full existing suite, lint,
format and strict mypy. No extra dependency, public push or environment edit.

## Verified primary API evidence

Python 3.12 installed standard-library documentation/source for `os.link`, `os.fsync`,
`tempfile.mkstemp`, `hashlib.sha256`, `json` hooks, `importlib.metadata` and
`subprocess.run`; Git's installed `rev-parse`/`status` behavior, inspected and
exercised on 2026-10-31. Linux manual pages [link(2)](https://man7.org/linux/man-pages/man2/link.2.html)
and [fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html) were read:
link does not replace an existing destination, and file fsync alone does not
ensure directory-entry durability. These references support the API/protocol
choice, not a power-loss test or a guarantee for network/object storage.

"""Reuse canonical repository documents without copying caches,weights or credentials."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/jonbarkert135-cpu/QveltAI/blob/main/"


def reference_files(repository: Path) -> set[Path]:
    roots = (
        "README.md",
        "STATE.md",
        "PLAN.md",
        "DECISIONS.md",
        "RESULTS.md",
        "DATA_CARD.md",
        "MODEL_CARD.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "LICENSE",
        "NOTICE",
        "SECURITY.md",
    )
    result = {Path(name) for name in roots if (repository / name).is_file()}
    for pattern in (
        "runbooks/*.md",
        "docs/model.md",
        "docs/research/*.md",
        "docs/development/*.md",
        "docs/development/evidence/*.json",
        "docs/development/evidence/*.png",
        "configs/**/*.yaml",
    ):
        result.update(
            path.relative_to(repository)
            for path in repository.glob(pattern)
            if path.is_file()
        )
    result.discard(Path("docs/development/skills.md"))
    # Legacy config aliases are symlinks;canonical model YAML is already selected.
    return {path for path in result if not (repository / path).is_symlink()}


def rewrite_links(text: str, source: Path, files: set[Path], repository: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        target = match[1]
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or not parts.path:
            return match[0]
        path = (repository / source.parent / unquote(parts.path)).resolve()
        if not path.is_relative_to(repository.resolve()):
            raise ValueError("Documentation link escapes repository")
        relative = path.relative_to(repository.resolve())
        if not path.exists():
            raise ValueError(f"Broken canonical document link:{source}:{target}")
        if relative in files:
            destination = Path(os.path.relpath(relative, source.parent)).as_posix()
        else:
            destination = SOURCE_URL + quote(relative.as_posix(), safe="/")
        if parts.fragment:
            destination += "#" + parts.fragment
        return "](" + destination + ")"

    return re.sub(r"\]\(([^)\s]+)\)", replace, text)


def generate() -> None:
    import mkdocs_gen_files

    files = reference_files(ROOT)
    for source in sorted(files):
        path = ROOT / source
        if path.is_symlink():
            raise ValueError("Documentation sources must be regular files")
        if source.suffix == ".md":
            text = rewrite_links(path.read_text(), source, files, ROOT)
            with mkdocs_gen_files.open(Path("reference") / source, "w") as output:
                output.write(text)
        else:
            with mkdocs_gen_files.open(Path("reference") / source, "wb") as output:
                output.write(path.read_bytes())


if __name__ in ("__main__", "<run_path>"):
    generate()

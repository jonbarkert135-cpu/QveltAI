"""Small first-party deserialization inventory and reproducible dependency-audit inputs.

This supplements dependency/secrets tools;it is not a complete static analyzer or
an attestation that arbitrary Python,third-party code or distributed peers are safe.
"""

from __future__ import annotations

import ast
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def deserialization_inventory(source: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({item.asname or item.name: item.name for item in node.names})
        elif isinstance(node, ast.ImportFrom) and node.module:
            aliases.update(
                {
                    item.asname or item.name: node.module + "." + item.name
                    for item in node.names
                }
            )

    def name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return name(node.value) + "." + node.attr
        return ""

    safe_loaders = {"yaml.SafeLoader", "yaml.CSafeLoader"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
            name(base) in safe_loaders for base in node.bases
        ):
            safe_loaders.add(node.name)

    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = name(node.func)
        safe, reason = False, ""
        if function == "torch.load":
            safe = any(
                item.arg == "weights_only"
                and isinstance(item.value, ast.Constant)
                and item.value.value is True
                for item in node.keywords
            )
            reason = "explicit weights_only=True required"
        elif function in (
            "pickle.load",
            "pickle.loads",
            "dill.load",
            "dill.loads",
            "joblib.load",
            "eval",
            "exec",
        ):
            reason = "arbitrary object/code loading is unsupported"
        elif function in ("numpy.load", "np.load"):
            safe = any(
                item.arg == "allow_pickle"
                and isinstance(item.value, ast.Constant)
                and item.value.value is False
                for item in node.keywords
            )
            reason = "explicit allow_pickle=False required"
        elif function in ("yaml.load", "yaml.unsafe_load"):
            safe = function == "yaml.load" and any(
                item.arg == "Loader" and name(item.value) in safe_loaders
                for item in node.keywords
            )
            reason = "explicit SafeLoader or reviewed subclass required;bound input separately"
        if reason:
            rows.append(
                {
                    "line": node.lineno,
                    "function": function,
                    "safe": safe,
                    "reason": reason,
                }
            )
    return sorted(rows, key=lambda row: row["line"])


def review_sources(repository: Path) -> dict[str, Any]:
    rows = []
    for prefix in ("src", "scripts"):
        for path in sorted((repository / prefix).rglob("*.py")):
            for row in deserialization_inventory(path.read_text()):
                rows.append({"path": path.relative_to(repository).as_posix(), **row})
    return {
        "scope": "first-party src/scripts direct calls;not whole-program analysis or third-party code",
        "calls": rows,
        "unsafe_calls": [row for row in rows if not row["safe"]],
        "dcp_metadata": "restricted data-class globals;reviewed separately in safe_dcp.py",
        "distributed_object_collectives": "trusted ranks/network required;upstream object collectives use pickle",
    }


def installed_requirements() -> tuple[str, ...]:
    """Normalize CPU/CUDA local suffixes so advisories for upstream Torch are not skipped."""
    return tuple(
        sorted(
            {
                distribution.metadata["Name"]
                + "=="
                + distribution.version.split("+", 1)[0]
                for distribution in distributions()
                if distribution.metadata["Name"].lower() != "qvelt"
            }
        )
    )

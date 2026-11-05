"""Config-driven LR/batch sweeps: cartesian grid -> validated pretrain commands."""

from __future__ import annotations

import itertools
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qvelt.config import ConfigError, parse_yaml_mapping
from qvelt.config.pretraining import PretrainingJob, load_pretraining_job
from qvelt.train.report import summarize_run

Scalar = int | float | str | bool
_BATCH_KEYS = ("job.training.batch_size", "accumulation_steps")


@dataclass(frozen=True)
class SweepSpec:
    name: str
    base: str
    output: str
    grid: dict[str, list[Scalar]]
    fixed: dict[str, Scalar] = field(default_factory=dict)
    stop_after: int | None = None
    preserve_tokens: bool = True
    launcher: list[str] = field(default_factory=lambda: ["qvelt"])
    repository: str = "."


@dataclass(frozen=True)
class SweepRun:
    name: str
    directory: str
    overrides: list[str]
    stop_after: int | None
    command: list[str]


def _format(value: Scalar) -> str:
    return json.dumps(value) if isinstance(value, str) else str(value).lower()


def load_sweep(path: Path) -> SweepSpec:
    raw = parse_yaml_mapping(path.read_text(encoding="utf-8"))
    allowed = set(SweepSpec.__dataclass_fields__)
    if not set(raw) <= allowed or not {"name", "base", "output", "grid"} <= set(raw):
        raise ConfigError("Sweep needs name/base/output/grid and known keys only")
    grid = raw["grid"]
    if (
        not isinstance(grid, dict)
        or not grid
        or any(not isinstance(v, list) or not v for v in grid.values())
    ):
        raise ConfigError("Sweep grid maps override paths to nonempty value lists")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", str(raw["name"])):
        raise ConfigError("Sweep name must be a short identifier")
    for name in ("base", "output", "repository"):
        value = raw.get(name, ".")
        if not isinstance(value, str) or not value or len(value) > 2048:
            raise ConfigError("Invalid sweep path")
    if type(raw.get("preserve_tokens", True)) is not bool:
        raise ConfigError("preserve_tokens must be boolean")
    stop = raw.get("stop_after")
    if stop is not None and (type(stop) is not int or stop < 1):
        raise ConfigError("stop_after must be positive")
    launcher = raw.get("launcher", ["qvelt"])
    if (
        not isinstance(launcher, list)
        or not 1 <= len(launcher) <= 32
        or any(not isinstance(x, str) or not x or len(x) > 2048 for x in launcher)
    ):
        raise ConfigError("launcher must be a bounded argv list")
    fixed = raw.get("fixed", {})
    if not isinstance(fixed, dict):
        raise ConfigError("fixed must be a scalar mapping")
    for key, values in [*grid.items(), *((k, [v]) for k, v in fixed.items())]:
        if not isinstance(key, str) or any(
            type(v) not in (int, float, str, bool)
            or isinstance(v, float)
            and not math.isfinite(v)
            for v in values
        ):
            raise ConfigError("Sweep overrides require finite scalar values")
    if math.prod(len(v) for v in grid.values()) > 256:
        raise ConfigError("Sweep exceeds 256 points")
    spec = SweepSpec(**raw)  # type: ignore[arg-type]
    root = path.parent
    return SweepSpec(
        **{
            **asdict(spec),
            "base": str((root / spec.base).resolve()),
            "output": str((root / spec.output).resolve()),
        }
    )


def _global_batch(recipe: PretrainingJob) -> int:
    return recipe.job.training.batch_size * recipe.accumulation_steps


def expand(spec: SweepSpec) -> list[SweepRun]:
    """Validate every point by actually parsing the overridden recipe."""
    base_path = Path(spec.base)
    base = load_pretraining_job(base_path)
    keys = sorted(spec.grid)
    runs = []
    for index, values in enumerate(itertools.product(*(spec.grid[k] for k in keys))):
        point = dict(zip(keys, values, strict=True))
        settings = {**spec.fixed, **point}
        overrides = [f"{k}={_format(v)}" for k, v in sorted(settings.items())]
        candidate = load_pretraining_job(base_path, overrides)
        stop_after = spec.stop_after
        if spec.preserve_tokens and any(k in settings for k in _BATCH_KEYS):
            # Same token budget at a different global batch: rescale the step
            # horizon (and the short-run stop) instead of training more tokens.
            if base.total_batches is not None:
                raise ConfigError("preserve_tokens needs total_batches unset")
            numerator, denominator = _global_batch(base), _global_batch(candidate)
            steps, remainder = divmod(base.job.training.steps * numerator, denominator)
            ratio = numerator / denominator
            if remainder or steps < 1:
                raise ConfigError("Global batch must divide the base step horizon")
            overrides.append(f"job.training.steps={int(steps)}")
            if "job.training.checkpoint_every" not in settings:
                overrides.append(
                    "job.training.checkpoint_every="
                    f"{max(1, int(base.job.training.checkpoint_every * ratio))}"
                )
            if stop_after is not None:
                stop_after = max(1, round(stop_after * ratio))
            candidate = load_pretraining_job(base_path, overrides)
        if stop_after is not None and stop_after > candidate.job.training.steps:
            raise ConfigError("stop_after exceeds a sweep point's step horizon")
        label = "_".join(
            f"{k.rsplit('.', 1)[-1]}-{_format(v)}" for k, v in sorted(point.items())
        )
        name = f"{spec.name}-{index:02d}-{re.sub(r'[^A-Za-z0-9_.-]', '', label)}"
        directory = str(Path(spec.output) / name)
        command = [
            *spec.launcher,
            "pretrain",
            spec.base,
            "--directory",
            directory,
            "--repository",
            spec.repository,
            "--resume",
            "auto",
            *itertools.chain.from_iterable(("--set", o) for o in overrides),
            *(["--stop-after", str(stop_after)] if stop_after is not None else []),
        ]
        runs.append(SweepRun(name, directory, overrides, stop_after, command))
    return runs


def execute(spec: SweepSpec, runs: list[SweepRun]) -> list[dict[str, Any]]:
    """Run sequentially (resumable: rerunning skips completed points)."""
    results = []
    for run in runs:
        code = subprocess.run(run.command, check=False).returncode
        results.append({"name": run.name, "returncode": code})
    return results


def summarize(runs: list[SweepRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        directory = Path(run.directory)
        if not (directory / "manifest.json").exists():
            rows.append({"name": run.name, "status": "missing"})
            continue
        summary = summarize_run(directory)
        rows.append(
            {
                "name": run.name,
                "overrides": run.overrides,
                "last_step": summary["last_step"],
                "final_validation_loss": summary["final_validation_loss"],
                "last10_mean_loss": summary["last10_mean_loss"],
                "red_flags": summary["red_flags"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row.get("final_validation_loss") is None,
            row.get("final_validation_loss") or 0.0,
        ),
    )

"""Summarize a pretraining run directory and pack its logs (never weights)."""

from __future__ import annotations

import io
import json
import logging
import math
import statistics
import tarfile
from pathlib import Path
from typing import Any

from qvelt.utils.checkpoints import committed_steps
from qvelt.utils.manifests import _reject_constant, _unique_mapping

# Small text artifacts worth sending for diagnosis; checkpoint payloads excluded.
_TOP_LEVEL = (
    "manifest.json",
    "config.yaml",
    "job.yaml",
    "application.yaml",
    "environment.json",
    "compatibility.json",
    "pip-freeze.txt",
    "uv.lock",
    "data.manifest.json",
    "tokenizer.manifest.json",
    "metrics-*.jsonl",
    "samples-*.jsonl",
    "attempt-*.yaml",
    "failure-*.json",
)
_CHECKPOINT_META = (
    "manifest.json",
    "scheduler.json",
    "sampler.json",
    "application-rank-*.json",
)
_MAX_FILE = 64 * 1024 * 1024


def merged_metrics(directory: Path) -> tuple[dict[int, dict[str, Any]], int]:
    """Per-step records across attempts; later attempts win replayed steps."""
    files = []
    for path in directory.glob("metrics-*.jsonl"):
        records = []
        lines = path.read_text().splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=_unique_mapping,
                    parse_constant=_reject_constant,
                )
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith("\n"):
                    logging.getLogger("qvelt").warning(
                        "Ignoring interrupted final metrics record in %s", path.name
                    )
                    continue
                raise
            if (
                not isinstance(record, dict)
                or type(record.get("step")) is not int
                or record["step"] < 0
            ):
                raise ValueError("Invalid metric step record")
            records.append(record)
        if records:
            files.append((path.stat().st_mtime_ns, records))
    merged: dict[int, dict[str, Any]] = {}
    for _, records in sorted(files, key=lambda item: item[0]):
        for record in records:
            step = int(record["step"])
            merged[step] = {**merged.get(step, {}), **record}
    return merged, len(files)


def _series(records: dict[int, dict[str, Any]], key: str) -> list[tuple[int, float]]:
    return [
        (step, float(values[key]))
        for step, values in sorted(records.items())
        if values.get(key) is not None
    ]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize_run(directory: Path) -> dict[str, Any]:
    records, attempts = merged_metrics(directory)
    losses = _series(records, "loss")
    norms = [v for _, v in _series(records, "gradient_norm")]
    clipped = [v for _, v in _series(records, "clipped")]
    speed = [v for _, v in _series(records, "tokens_per_second_per_device")]
    mfu = [v for _, v in _series(records, "estimated_mfu")]
    probe = _series(records, "validation_probe_loss")
    domains = sorted(
        {
            key[len("validation_") : -len("_loss")]
            for values in records.values()
            for key in values
            if key.startswith("validation_")
            and key.endswith("_loss")
            and key != "validation_probe_loss"
        }
    )
    spikes = [step for step, v in _series(records, "loss_spike") if v > 0]
    nonfinite = [step for step, v in _series(records, "nonfinite_window") if v > 0]
    rollbacks = _series(records, "rollback_to_step")
    last = [v for _, v in losses[-10:]]
    summary: dict[str, Any] = {
        "directory": str(directory),
        "attempts": attempts,
        "last_step": max(records) if records else None,
        "processed_tokens": records[max(records)].get("processed_tokens")
        if records
        else None,
        "first_loss": losses[0][1] if losses else None,
        "last10_mean_loss": statistics.fmean(last) if last else None,
        "min_loss": min(v for _, v in losses) if losses else None,
        "initial_validation_loss": probe[0][1] if probe else None,
        "final_validation_loss": probe[-1][1] if probe else None,
        "validation_points": len(probe),
        "domains": {
            name: {
                "first": (series := _series(records, f"validation_{name}_loss"))[0][1],
                "last": series[-1][1],
                "last_ppl": math.exp(series[-1][1]) if series[-1][1] < 700 else None,
            }
            for name in domains
        },
        "median_tokens_per_second_per_device": _median(speed),
        "median_estimated_mfu": _median(mfu),
        "clip_fraction_last100": statistics.fmean(clipped[-100:]) if clipped else None,
        "median_gradient_norm": _median(norms),
        "last_gradient_norm": norms[-1] if norms else None,
        "loss_spike_steps": spikes,
        "nonfinite_steps": nonfinite,
        "rollbacks": [{"step": s, "to_step": int(v)} for s, v in rollbacks],
        "checkpoints": [step for step, _ in committed_steps(directory)],
        "quarantined": sorted(
            p.name for p in (directory / "checkpoints/quarantine").glob("*")
        ),
    }
    summary["red_flags"] = red_flags(summary, speed)
    return summary


def red_flags(summary: dict[str, Any], speed: list[float]) -> list[str]:
    """Heuristic alarms (project defaults, not validated thresholds)."""
    flags = []
    if summary["nonfinite_steps"]:
        flags.append(f"nonfinite windows at steps {summary['nonfinite_steps'][:10]}")
    if summary["rollbacks"]:
        flags.append(f"{len(summary['rollbacks'])} automatic rollback(s)")
    if summary["loss_spike_steps"]:
        flags.append(f"loss spikes at steps {summary['loss_spike_steps'][:10]}")
    if summary["quarantined"]:
        flags.append(f"quarantined checkpoints: {summary['quarantined']}")
    initial, final = (
        summary["initial_validation_loss"],
        summary["final_validation_loss"],
    )
    if summary["validation_points"] >= 2 and final >= initial:
        flags.append("validation loss did not improve over the attempt's start")
    if (summary["clip_fraction_last100"] or 0) > 0.5:
        flags.append("gradient clipped in >50% of the last 100 updates")
    median, last = summary["median_gradient_norm"], summary["last_gradient_norm"]
    if median and last and last > 10 * median:
        flags.append("last gradient norm >10x the run median")
    if len(speed) >= 20 and statistics.median(speed[-10:]) < 0.7 * statistics.median(
        speed
    ):
        flags.append("recent throughput <70% of the run median")
    for name, values in summary["domains"].items():
        if values["last"] > values["first"]:
            flags.append(f"domain {name} validation loss increased")
    return flags


def archive_run(directory: Path, output: Path) -> list[str]:
    """tar.gz of run logs/manifests and checkpoint metadata; no tensors."""
    selected: list[Path] = []
    for pattern in _TOP_LEVEL:
        selected.extend(directory.glob(pattern))
    for checkpoint in (directory / "checkpoints").glob("step-*"):
        if checkpoint.is_symlink():
            raise ValueError("Checkpoint symlinks cannot be archived")
        for pattern in _CHECKPOINT_META:
            selected.extend(checkpoint.glob(pattern))
    names = []
    with tarfile.open(output, "x:gz") as archive:
        summary = json.dumps(summarize_run(directory), indent=2, sort_keys=True)
        info = tarfile.TarInfo("summary.json")
        data = summary.encode()
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        names.append("summary.json")
        for path in sorted(set(selected)):
            if path.is_file() and not path.is_symlink():
                if path.stat().st_size <= _MAX_FILE:
                    name = path.relative_to(directory).as_posix()
                    archive.add(path, arcname=name)
                    names.append(name)
    return names

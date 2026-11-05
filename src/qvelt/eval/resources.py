"""Fresh-process Linux CPU inference measurements;no stochastic or server SLA claim."""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from qvelt.eval.config import EvalJob, eval_digest, load_eval
from qvelt.eval.runner import load_model
from qvelt.utils.checkpoints import source_digest


def measure_resources(job: EvalJob) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise ValueError("Resource protocol v1 requires Linux ru_maxrss units")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, _, manifest = load_model(job)
    if min(job.context_limit, model.config.context) < 24:
        raise ValueError("Resource probe needs context>=24")
    durations = []
    with torch.inference_mode():
        for trial in range(7):
            ids = torch.tensor(
                [[1, 32, 33, 34, 35, 36, 37, 38]], dtype=torch.int64, device="cpu"
            )
            start = time.perf_counter()
            for _ in range(16):
                logits = model(ids)[:, -1]
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite fixed-budget inference logits")
                ids = torch.cat((ids, logits.argmax(-1, keepdim=True)), dim=1)
            elapsed = time.perf_counter() - start
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("Invalid measured inference duration")
            if trial >= 2:
                durations.append(elapsed)
    median = statistics.median(durations)
    return {
        "protocol": "linux-cpu-fixed16-v1",
        "checkpoint_sha256": job.checkpoint.sha256,
        "evaluation_sha256": eval_digest(job),
        "source_sha256": source_digest(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "precision": "fp32",
        "threads": 1,
        "prefix_tokens": 8,
        "generated_steps": 16,
        "warmups": 2,
        "repetitions": 5,
        "seconds": durations,
        "median_seconds": median,
        "tokens_per_second": 16 / median,
        "parameters": model.config.parameter_count,
        "weights_bytes": sum(f.size_bytes for f in manifest.files if f.role == "model"),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rss_scope": "whole fresh process including imports,checkpoint load and model;not incremental tensor memory",
        "workload": "eager full-prefix,fixed actions,ignore stop controls,no decode;not end-to-end latency",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure_resources(load_eval(args.recipe))
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

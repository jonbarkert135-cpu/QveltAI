"""Operator-level forward/backward profile with throughput, peak memory and estimated MFU.

CPU runs are executable evidence for the harness only; the CUDA/BF16 branch is
unverified until it runs on real hardware (see docs/model.md, Profiling).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from qvelt.config import load_config
from qvelt.config.execution import ModelExecution
from qvelt.model.decoder import Decoder
from qvelt.model.loss import causal_loss
from qvelt.train.optim import flops_per_token


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("configs/model/tiny.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend", choices=("reference", "sdpa"), default="sdpa")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--peak-tflops", type=float, default=None)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device cpu")
    if min(args.batch, args.length, args.steps) < 1:
        raise SystemExit("batch, length and steps must be positive")
    if args.peak_tflops is not None and (
        not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0
    ):
        raise SystemExit("peak-tflops must be finite and positive")
    config = load_config(args.model).model
    if args.length > config.context:
        raise SystemExit("length exceeds configured model context")
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = Decoder(config, ModelExecution(args.backend, args.recompute)).to(device)
    step_model = torch.compile(model) if args.compile else model
    tokens = torch.randint(
        32, config.vocab, (args.batch, args.length + 1), device=device
    )
    autocast = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if args.bf16
        else nullcontext()
    )

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with autocast:
            logits = step_model(tokens[:, :-1])
        total, count = causal_loss(logits, tokens[:, 1:])
        torch.autograd.backward(total / count)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(2):  # warmup, including any compilation
        step()
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        step()
    sync()
    seconds = time.perf_counter() - started
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as trace:
        step()
        sync()
    sort = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    tokens_per_second = args.steps * args.batch * args.length / seconds
    achieved = tokens_per_second * flops_per_token(config, args.length)
    d = config.hidden
    head_share = 6 * config.vocab * d / flops_per_token(config, args.length)
    report: dict[str, object] = {
        "verified_scope": "cpu harness"
        if device.type == "cpu"
        else "cuda harness execution",
        "profile_scope": "forward/backward only; no optimizer updates; analytical FLOPs exclude recomputation",
        "cuda_kernel_dispatch_verified": False,
        "args": {k: str(v) for k, v in vars(args).items()},
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokens_per_second": tokens_per_second,
        "achieved_tflops_estimate": achieved / 1e12,
        "vocabulary_head_flop_share": head_share,
        "estimated_mfu": achieved / (args.peak_tflops * 1e12)
        if args.peak_tflops
        else None,
        "peak_memory_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else None,
        "top_operators": [
            {"name": e.key, "self_us": float(getattr(e, sort)), "calls": e.count}
            for e in sorted(
                trace.key_averages(),
                key=lambda e: float(getattr(e, sort)),
                reverse=True,
            )[:10]
        ],
    }
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()

"""RFC linear position interpolation as an explicit derived CPU model,not a quality claim."""

from __future__ import annotations

from dataclasses import replace

import torch

from qvelt.model.decoder import Decoder


def extend_context(model: Decoder, factor: int) -> Decoder:
    if (
        type(factor) is not int
        or factor not in (1, 2, 4)
        or model.config.context * factor > 2048
    ):
        raise ValueError("CPU interpolation supports factors1/2/4 up to2048 tokens")
    if model.training or any(
        p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()
    ):
        raise ValueError("Interpolation requires an FP32 CPU eval model")
    config = replace(
        model.config,
        context=model.config.context * factor,
        rope_scale=model.config.rope_scale * factor,
    )
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        derived = Decoder(config, model.execution).eval()
        derived.load_state_dict(model.state_dict(), strict=True)
    return derived

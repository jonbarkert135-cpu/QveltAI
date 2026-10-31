"""Seed control and explicit PyTorch diagnostics, without import-time CUDA work."""

from __future__ import annotations

import os
import platform
import random
from dataclasses import dataclass

import numpy as np

from qvelt.config import Device, RuntimeConfig


class RuntimeUnavailableError(RuntimeError):
    """The requested ML runtime/device cannot execute the operation."""


@dataclass(frozen=True)
class SeedReport:
    """Seed policy applied in this process, not a complete replay checkpoint."""

    seed: int
    device: Device
    deterministic: bool
    torch_version: str


@dataclass(frozen=True)
class RuntimeReport:
    """Observed installed runtime plus the device used for a real tensor probe."""

    python_version: str
    numpy_version: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    requested_device: Device
    bf16_supported: bool | None
    test_checksum: float


def numpy_generator(seed: int, stream: int = 0) -> np.random.Generator:
    """Independent repeatable streams keyed by explicit uint32 seed/stream IDs."""
    for name, value in (("seed", seed), ("stream", stream)):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError(f"{name} must be a uint32 integer")
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))


def seed_everything(config: RuntimeConfig) -> SeedReport:
    """Seed Python/legacy NumPy/PyTorch and set deterministic operation policy.

    Independent NumPy generators must use ``numpy_generator`` explicitly.
    PYTHONHASHSEED and complete data/optimizer replay are not controlled here.
    Configure before allocating model tensors. CUDA policy additionally requires
    a supported deterministic cuBLAS workspace setting before initialization.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeUnavailableError(
            "PyTorch is required; install the cpu or cuda extra"
        ) from error

    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeUnavailableError("Requested CUDA device is unavailable")
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            raise RuntimeUnavailableError("Requested CUDA BF16 compute is unsupported")
        if config.deterministic:
            workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace not in (":4096:8", ":16:8"):
                raise RuntimeUnavailableError(
                    "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting deterministic CUDA"
                )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return SeedReport(
        config.seed, config.device, config.deterministic, str(torch.__version__)
    )


def inspect_runtime(device: Device = "cpu") -> RuntimeReport:
    """Run an actual 2x2 FP32 matrix multiply on the requested device.

    CPU success does not validate CUDA or BF16. No fallback is performed.
    No RNG is consumed, data downloaded, or model allocated by this probe.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda")
    try:
        import torch
    except ImportError as error:
        raise RuntimeUnavailableError(
            "PyTorch is required; install the cpu or cuda extra"
        ) from error
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise RuntimeUnavailableError("Requested CUDA device is unavailable")
    try:
        value = torch.ones((2, 2), dtype=torch.float32, device=device)
        checksum = float((value @ value).sum().item())
    except RuntimeError as error:
        raise RuntimeUnavailableError(
            f"Tensor diagnostic failed on {device}: {error}"
        ) from error
    if checksum != 8.0:
        raise RuntimeUnavailableError("Tensor diagnostic produced an unexpected result")
    return RuntimeReport(
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        torch_version=str(torch.__version__),
        torch_cuda_version=torch.version.cuda,
        cuda_available=available,
        requested_device=device,
        bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False)
        if device == "cuda"
        else None,
        test_checksum=checksum,
    )

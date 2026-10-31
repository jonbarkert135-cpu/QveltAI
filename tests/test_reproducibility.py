"""Real CPU RNG reference tests; mocked failure tests do not validate CUDA."""

from __future__ import annotations

import random
from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pytest
import torch

from qvelt.config import RuntimeConfig
from qvelt.reproducibility import (
    RuntimeUnavailableError,
    inspect_runtime,
    numpy_generator,
    seed_everything,
)


@pytest.fixture(autouse=True)
def restore_rng_and_policy() -> Iterator[None]:
    py, legacy_numpy, tensor = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    deterministic = torch.are_deterministic_algorithms_enabled()
    benchmark, cudnn_deterministic = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    cudnn_tf32, matmul_tf32 = (
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    yield
    random.setstate(py)
    np.random.set_state(legacy_numpy)
    torch.set_rng_state(tensor)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (
        benchmark,
        cudnn_deterministic,
    )
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = (
        cudnn_tf32,
        matmul_tf32,
    )


def test_seeded_cpu_draws_match_library_references() -> None:
    seed = 19
    random.seed(seed)
    py_reference = [random.random() for _ in range(4)]
    np.random.seed(seed)
    np_reference = np.random.standard_normal(4)
    torch.manual_seed(seed)
    torch_reference = torch.randn(4)
    report = seed_everything(RuntimeConfig(seed=seed))
    assert [random.random() for _ in range(4)] == py_reference
    np.testing.assert_array_equal(np.random.standard_normal(4), np_reference)
    torch.testing.assert_close(torch.randn(4), torch_reference, rtol=0, atol=0)
    assert report.seed == seed and report.device == "cpu" and report.deterministic
    assert report.torch_version == torch.__version__
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32


def test_same_seed_repeats_and_different_seed_changes_draws() -> None:
    seed_everything(RuntimeConfig(seed=7))
    first = torch.randn(10)
    seed_everything(RuntimeConfig(seed=7))
    torch.testing.assert_close(first, torch.randn(10), rtol=0, atol=0)
    seed_everything(RuntimeConfig(seed=8, deterministic=False))
    assert not torch.equal(first, torch.randn(10))
    assert not torch.are_deterministic_algorithms_enabled()


def test_independent_numpy_streams_do_not_consume_global_rng() -> None:
    np.random.seed(13)
    reference = np.random.random(4)
    np.random.seed(13)
    a = numpy_generator(5, 0).integers(0, 2**31, 20)
    b = numpy_generator(5, 1).integers(0, 2**31, 20)
    np.testing.assert_array_equal(a, numpy_generator(5, 0).integers(0, 2**31, 20))
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(np.random.random(4), reference)


@pytest.mark.parametrize(
    "seed,stream", [(-1, 0), (2**32, 0), (1, -1), (1, 2**32), (True, 0)]
)
def test_reject_invalid_stream_ids(seed: int, stream: int) -> None:
    with pytest.raises(ValueError):
        numpy_generator(seed, stream)


def test_actual_cpu_probe_has_no_rng_effect() -> None:
    state = torch.get_rng_state()
    report = inspect_runtime("cpu")
    assert report.test_checksum == 8.0 and report.requested_device == "cpu"
    assert report.bf16_supported is None
    assert report.numpy_version == np.__version__
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)


def test_cuda_failure_has_no_silent_cpu_fallback_or_seed_change() -> None:
    before = random.getstate()
    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeUnavailableError, match="unavailable"):
            seed_everything(RuntimeConfig(device="cuda"))
        with pytest.raises(RuntimeUnavailableError, match="unavailable"):
            inspect_runtime("cuda")
    assert random.getstate() == before


def test_bf16_rejection_before_tensor_allocation() -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "is_bf16_supported", return_value=False),
    ):
        with pytest.raises(RuntimeUnavailableError, match="BF16"):
            seed_everything(RuntimeConfig(device="cuda", precision="bf16"))


def test_deterministic_cuda_requires_explicit_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with patch.object(torch.cuda, "is_available", return_value=True):
        with pytest.raises(RuntimeUnavailableError, match="CUBLAS_WORKSPACE_CONFIG"):
            seed_everything(RuntimeConfig(device="cuda"))


def test_missing_torch_error_is_actionable() -> None:
    with patch.dict("sys.modules", {"torch": None}):
        with pytest.raises(RuntimeUnavailableError, match="cpu or cuda extra"):
            seed_everything(RuntimeConfig())
        with pytest.raises(RuntimeUnavailableError, match="cpu or cuda extra"):
            inspect_runtime()


def test_tensor_diagnostic_failure_is_actionable() -> None:
    with patch.object(
        torch, "ones", side_effect=RuntimeError("fixture kernel failure")
    ):
        with pytest.raises(RuntimeUnavailableError, match="Tensor diagnostic failed"):
            inspect_runtime("cpu")
    wrong_values = torch.full((2, 2), 2.0)
    with patch.object(torch, "ones", return_value=wrong_values):
        with pytest.raises(RuntimeUnavailableError, match="unexpected result"):
            inspect_runtime("cpu")

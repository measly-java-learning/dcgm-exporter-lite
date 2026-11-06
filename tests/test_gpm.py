from __future__ import annotations

import pytest
from fakes import FakeDevice

from dcgm_exporter_lite.gpm import GpmSampler
from dcgm_exporter_lite.nvml import DeviceIdentity, ReadError


def _device(**values) -> FakeDevice:
    identity = DeviceIdentity(0, "GPU-test", "00000000:01:00.0", 0, "NVIDIA Test")
    return FakeDevice(identity, **values)


def test_first_step_returns_none() -> None:
    device = _device(gpm_supported=True, gpm_sample=["a", "b"])
    sampler = GpmSampler([device])

    assert sampler.step(0) is None


def test_second_step_returns_named_values() -> None:
    device = _device(
        gpm_supported=True,
        gpm_sample=["a", "b"],
        gpm_metrics=[{1: 55.0, 7: 12.5, 10: 30.0, 20: 1.125, 21: 0.5}],
    )
    sampler = GpmSampler([device])

    assert sampler.step(0) is None
    assert sampler.step(0) == {
        "GRAPHICS_UTIL": 55.0,
        "HMMA_TENSOR_UTIL": 12.5,
        "DRAM_BW_UTIL": 30.0,
        "PCIE_TX_PER_SEC": 1.125,
        "PCIE_RX_PER_SEC": 0.5,
    }


def test_unsupported_device_always_none_and_never_samples() -> None:
    device = _device(gpm_supported=False, gpm_sample=["a", "b"])
    sampler = GpmSampler([device])

    assert sampler.step(0) is None
    assert sampler.step(0) is None
    assert sampler.step(0) is None
    assert device.calls["gpm_sample"] == 0


def test_missing_metric_id_omitted() -> None:
    device = _device(
        gpm_supported=True,
        gpm_sample=["a", "b"],
        gpm_metrics=[{1: 55.0}],
    )
    sampler = GpmSampler([device])

    sampler.step(0)
    assert sampler.step(0) == {"GRAPHICS_UTIL": 55.0}


def test_sample_error_resets_baseline() -> None:
    device = _device(
        gpm_supported=True,
        gpm_sample=["a", ReadError("boom"), "b", "c"],
        gpm_metrics=[{1: 55.0}],
    )
    sampler = GpmSampler([device])

    assert sampler.step(0) is None
    with pytest.raises(ReadError):
        sampler.step(0)
    assert sampler.step(0) is None
    assert sampler.step(0) == {"GRAPHICS_UTIL": 55.0}

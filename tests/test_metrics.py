from __future__ import annotations

import pytest
from fakes import FakeDevice

from dcgm_exporter_lite.counters import load_counters
from dcgm_exporter_lite.metrics import METRICS, NoValueYet, ReadContext, _nvlink_rate
from dcgm_exporter_lite.nvml import DeviceIdentity, MemoryInfo, RemappedRows


def _device(**values) -> FakeDevice:
    identity = DeviceIdentity(0, "GPU-test", "00000000:01:00.0", 0, "NVIDIA Test")
    return FakeDevice(identity, **values)


def _ctx(device: FakeDevice, **overrides) -> ReadContext:
    values = {
        "device": device,
        "gpm": None,
        "xid_last": None,
        "now": 0.0,
        "state": {},
    }
    values.update(overrides)
    return ReadContext(**values)


def _read(name: str, ctx: ReadContext) -> float:
    return METRICS[name].read(ctx)


def test_clock_hz() -> None:
    device = _device(clock_mhz={"sm": 1980, "mem": 10501})

    assert _read("DCGM_FI_DEV_SM_CLOCK_HERTZ", _ctx(device)) == 1_980_000_000.0
    assert _read("DCGM_FI_DEV_MEMORY_CLOCK_HERTZ", _ctx(device)) == 10_501_000_000.0


def test_power_watts() -> None:
    device = _device(power_mw=350_123)

    assert _read("DCGM_FI_DEV_BOARD_POWER_WATTS", _ctx(device)) == 350.123


def test_energy_joules() -> None:
    device = _device(energy_mj=1_032_632)

    assert _read("DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL", _ctx(device)) == 1032.632


def test_util_ratios() -> None:
    device = _device(utilization_pct=(87, 42), encoder_pct=5, decoder_pct=2)

    assert _read("DCGM_FI_DEV_GPU_UTIL_RATIO", _ctx(device)) == 0.87
    assert _read("DCGM_FI_DEV_MEMORY_UTIL_RATIO", _ctx(device)) == 0.42
    assert _read("DCGM_FI_DEV_ENC_UTIL_RATIO", _ctx(device)) == 0.05
    assert _read("DCGM_FI_DEV_DEC_UTIL_RATIO", _ctx(device)) == 0.02


def test_fb_bytes() -> None:
    device = _device(memory=MemoryInfo(17103323136, 16461987840, 250347520, 390987776))

    assert _read("DCGM_FI_DEV_FB_CAPACITY_BYTES", _ctx(device)) == 17103323136
    assert _read("DCGM_FI_DEV_FB_FREE_BYTES", _ctx(device)) == 16461987840
    assert _read("DCGM_FI_DEV_FB_USED_BYTES", _ctx(device)) == 250347520
    assert _read("DCGM_FI_DEV_FB_RESERVED_BYTES", _ctx(device)) == 390987776


def test_row_remap() -> None:
    device = _device(remapped_rows=RemappedRows(3, 1, False, True))

    assert _read("DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL", _ctx(device)) == 3
    assert _read("DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL", _ctx(device)) == 1
    assert _read("DCGM_FI_DEV_ROW_REMAP_FAILED", _ctx(device)) == 1.0


def test_prof_from_gpm() -> None:
    device = _device()
    gpm = {"GRAPHICS_UTIL": 55.0, "HMMA_TENSOR_UTIL": 12.5, "DRAM_BW_UTIL": 30.0}

    assert _read("DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO", _ctx(device, gpm=gpm)) == 0.55
    assert _read("DCGM_FI_PROF_TENSOR_UTIL_RATIO", _ctx(device, gpm=gpm)) == 0.125
    assert _read("DCGM_FI_PROF_DRAM_UTIL_RATIO", _ctx(device, gpm=gpm)) == 0.30


def test_prof_pcie_bytes_per_second() -> None:
    device = _device()
    gpm = {"PCIE_TX_PER_SEC": 1.125, "PCIE_RX_PER_SEC": 0.5}

    assert _read("DCGM_FI_PROF_PCIE_TX_BYTES", _ctx(device, gpm=gpm)) == 1_179_648.0
    assert _read("DCGM_FI_PROF_PCIE_RX_BYTES", _ctx(device, gpm=gpm)) == 524_288.0


def test_prof_absent_when_no_gpm() -> None:
    device = _device()

    with pytest.raises(NoValueYet):
        _read("DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO", _ctx(device, gpm=None))


def test_xid() -> None:
    device = _device()

    assert _read("DCGM_FI_DEV_XID_ERROR", _ctx(device, xid_last=79)) == 79.0
    assert _read("DCGM_FI_DEV_XID_ERROR", _ctx(device, xid_last=None)) == 0.0


def test_nvlink() -> None:
    device = _device(nvlink_raw=(1_000_000, 2_000_000))
    state: dict = {}

    with pytest.raises(NoValueYet):
        _nvlink_rate(_ctx(device, now=100.0, state=state))

    device.values["nvlink_raw"] = (1_030_000, 2_060_000)
    assert _nvlink_rate(_ctx(device, now=130.0, state=state)) == 3_072_000.0


def test_nvlink_uses_elapsed_time_not_interval() -> None:
    device = _device(nvlink_raw=(0, 0))
    state: dict = {}

    with pytest.raises(NoValueYet):
        _nvlink_rate(_ctx(device, now=100.0, state=state))

    # 10240 KiB in 10 ms is 10_485_760 B / 0.01 s.
    device.values["nvlink_raw"] = (10_240, 0)
    assert _nvlink_rate(_ctx(device, now=100.01, state=state)) == pytest.approx(1_048_576_000)


def test_nvlink_non_positive_elapsed_has_no_value() -> None:
    device = _device(nvlink_raw=(0, 0))
    state: dict = {}

    with pytest.raises(NoValueYet):
        _nvlink_rate(_ctx(device, now=100.0, state=state))
    device.values["nvlink_raw"] = (10, 0)
    with pytest.raises(NoValueYet):
        _nvlink_rate(_ctx(device, now=100.0, state=state))


def test_registry_matches_default_csv() -> None:
    expected = {row.name for row in load_counters(None) if row.type != "label"} - {
        "DCGM_FI_DEV_VGPU_LICENSE_STATUS"
    }

    assert set(METRICS) == expected
    assert len(METRICS) == 25

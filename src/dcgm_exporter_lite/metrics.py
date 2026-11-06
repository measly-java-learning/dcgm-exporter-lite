"""DCGM 4.9.0 metric definitions and NVML unit conversions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import pynvml

from .nvml import GpuDevice

_MIB = 1024 * 1024


class NoValueYet(Exception):
    """A metric needs a second sample (GPM or a cumulative counter)."""


@dataclass(frozen=True)
class ReadContext:
    device: GpuDevice
    gpm: Mapping[str, float] | None
    xid_last: int | None
    now: float  # monotonic seconds at which this cycle's readings were taken
    state: dict


@dataclass(frozen=True)
class MetricDef:
    name: str
    gpu_wide: bool
    source: Literal["nvml", "gpm", "xid"]
    read: Callable[[ReadContext], float]


GPM_IDS: dict[str, int] = {
    "GRAPHICS_UTIL": pynvml.NVML_GPM_METRIC_GRAPHICS_UTIL,
    "HMMA_TENSOR_UTIL": pynvml.NVML_GPM_METRIC_HMMA_TENSOR_UTIL,
    "DRAM_BW_UTIL": pynvml.NVML_GPM_METRIC_DRAM_BW_UTIL,
    "PCIE_TX_PER_SEC": pynvml.NVML_GPM_METRIC_PCIE_TX_PER_SEC,
    "PCIE_RX_PER_SEC": pynvml.NVML_GPM_METRIC_PCIE_RX_PER_SEC,
}


def _clock_hz(kind: Literal["sm", "mem"]) -> Callable[[ReadContext], float]:
    def read(ctx: ReadContext) -> float:
        return float(ctx.device.clock_mhz(kind)) * 1_000_000.0

    return read


def _memory_field(
    field: Literal["total", "free", "used", "reserved"],
) -> Callable[[ReadContext], float]:
    def read(ctx: ReadContext) -> float:
        return float(getattr(ctx.device.memory(), field))

    return read


def _util(index: int) -> Callable[[ReadContext], float]:
    def read(ctx: ReadContext) -> float:
        return float(ctx.device.utilization_pct()[index]) / 100.0

    return read


def _gpm_ratio(name: str) -> Callable[[ReadContext], float]:
    def read(ctx: ReadContext) -> float:
        if ctx.gpm is None or name not in ctx.gpm:
            raise NoValueYet(name)
        return float(ctx.gpm[name]) / 100.0

    return read


def _gpm_bytes_per_second(name: str) -> Callable[[ReadContext], float]:
    def read(ctx: ReadContext) -> float:
        if ctx.gpm is None or name not in ctx.gpm:
            raise NoValueYet(name)
        return float(ctx.gpm[name]) * _MIB

    return read


def _nvlink_rate(ctx: ReadContext) -> float:
    tx, rx = ctx.device.nvlink_raw()
    total = tx + rx
    previous = ctx.state.get("nvlink_prev")

    ctx.state["nvlink_prev"] = (total, ctx.now)
    if previous is None:
        raise NoValueYet("nvlink")

    previous_total, previous_now = previous
    elapsed = ctx.now - previous_now
    if elapsed <= 0:
        raise NoValueYet("nvlink")

    # The NVML fields are cumulative KiB; divide by the real time between
    # readings, not the nominal interval, so skipped cycles stay correct.
    delta = max(total - previous_total, 0)
    return delta * 1024.0 / elapsed


def _xid(ctx: ReadContext) -> float:
    # 4.9.0 reports 0 with err_code="0" before the first XID.
    return 0.0 if ctx.xid_last is None else float(ctx.xid_last)


# Rows of the 4.9.0 default CSV that lite omits on purpose (spec §4.3).
OMITTED: frozenset[str] = frozenset({"DCGM_FI_DEV_VGPU_LICENSE_STATUS"})

METRICS: dict[str, MetricDef] = {
    "DCGM_FI_DEV_SM_CLOCK_HERTZ": MetricDef(
        "DCGM_FI_DEV_SM_CLOCK_HERTZ", True, "nvml", _clock_hz("sm")
    ),
    "DCGM_FI_DEV_MEMORY_CLOCK_HERTZ": MetricDef(
        "DCGM_FI_DEV_MEMORY_CLOCK_HERTZ", True, "nvml", _clock_hz("mem")
    ),
    "DCGM_FI_DEV_MEMORY_TEMP_CELSIUS": MetricDef(
        "DCGM_FI_DEV_MEMORY_TEMP_CELSIUS",
        True,
        "nvml",
        lambda ctx: float(ctx.device.memory_temp_c()),
    ),
    "DCGM_FI_DEV_GPU_TEMP_CELSIUS": MetricDef(
        "DCGM_FI_DEV_GPU_TEMP_CELSIUS", True, "nvml", lambda ctx: float(ctx.device.gpu_temp_c())
    ),
    "DCGM_FI_DEV_BOARD_POWER_WATTS": MetricDef(
        "DCGM_FI_DEV_BOARD_POWER_WATTS", True, "nvml", lambda ctx: ctx.device.power_mw() / 1000.0
    ),
    "DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL": MetricDef(
        "DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL",
        True,
        "nvml",
        lambda ctx: ctx.device.energy_mj() / 1000.0,
    ),
    "DCGM_FI_DEV_PCIE_REPLAY_TOTAL": MetricDef(
        "DCGM_FI_DEV_PCIE_REPLAY_TOTAL", True, "nvml", lambda ctx: float(ctx.device.pcie_replay())
    ),
    "DCGM_FI_DEV_GPU_UTIL_RATIO": MetricDef("DCGM_FI_DEV_GPU_UTIL_RATIO", False, "nvml", _util(0)),
    "DCGM_FI_DEV_MEMORY_UTIL_RATIO": MetricDef(
        "DCGM_FI_DEV_MEMORY_UTIL_RATIO", False, "nvml", _util(1)
    ),
    "DCGM_FI_DEV_ENC_UTIL_RATIO": MetricDef(
        "DCGM_FI_DEV_ENC_UTIL_RATIO", False, "nvml", lambda ctx: ctx.device.encoder_pct() / 100.0
    ),
    "DCGM_FI_DEV_DEC_UTIL_RATIO": MetricDef(
        "DCGM_FI_DEV_DEC_UTIL_RATIO", False, "nvml", lambda ctx: ctx.device.decoder_pct() / 100.0
    ),
    "DCGM_FI_DEV_XID_ERROR": MetricDef("DCGM_FI_DEV_XID_ERROR", True, "xid", _xid),
    "DCGM_FI_DEV_FB_CAPACITY_BYTES": MetricDef(
        "DCGM_FI_DEV_FB_CAPACITY_BYTES", True, "nvml", _memory_field("total")
    ),
    "DCGM_FI_DEV_FB_FREE_BYTES": MetricDef(
        "DCGM_FI_DEV_FB_FREE_BYTES", True, "nvml", _memory_field("free")
    ),
    "DCGM_FI_DEV_FB_USED_BYTES": MetricDef(
        "DCGM_FI_DEV_FB_USED_BYTES", True, "nvml", _memory_field("used")
    ),
    "DCGM_FI_DEV_FB_RESERVED_BYTES": MetricDef(
        "DCGM_FI_DEV_FB_RESERVED_BYTES", True, "nvml", _memory_field("reserved")
    ),
    "DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND": MetricDef(
        "DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND", True, "nvml", _nvlink_rate
    ),
    "DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL": MetricDef(
        "DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL",
        True,
        "nvml",
        lambda ctx: float(ctx.device.remapped_rows().uncorrectable),
    ),
    "DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL": MetricDef(
        "DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL",
        True,
        "nvml",
        lambda ctx: float(ctx.device.remapped_rows().correctable),
    ),
    "DCGM_FI_DEV_ROW_REMAP_FAILED": MetricDef(
        "DCGM_FI_DEV_ROW_REMAP_FAILED",
        True,
        "nvml",
        lambda ctx: 1.0 if ctx.device.remapped_rows().failed else 0.0,
    ),
    "DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO": MetricDef(
        "DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO", False, "gpm", _gpm_ratio("GRAPHICS_UTIL")
    ),
    "DCGM_FI_PROF_TENSOR_UTIL_RATIO": MetricDef(
        "DCGM_FI_PROF_TENSOR_UTIL_RATIO", False, "gpm", _gpm_ratio("HMMA_TENSOR_UTIL")
    ),
    "DCGM_FI_PROF_DRAM_UTIL_RATIO": MetricDef(
        "DCGM_FI_PROF_DRAM_UTIL_RATIO", False, "gpm", _gpm_ratio("DRAM_BW_UTIL")
    ),
    "DCGM_FI_PROF_PCIE_TX_BYTES": MetricDef(
        "DCGM_FI_PROF_PCIE_TX_BYTES", False, "gpm", _gpm_bytes_per_second("PCIE_TX_PER_SEC")
    ),
    "DCGM_FI_PROF_PCIE_RX_BYTES": MetricDef(
        "DCGM_FI_PROF_PCIE_RX_BYTES", False, "gpm", _gpm_bytes_per_second("PCIE_RX_PER_SEC")
    ),
}

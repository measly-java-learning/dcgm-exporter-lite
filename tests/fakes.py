"""Pure-Python test doubles for the NVML facade."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import pynvml

from dcgm_exporter_lite.nvml import (
    DeviceIdentity,
    GpuLost,
    MemoryInfo,
    ReadError,
    RemappedRows,
    Unsupported,
)

_MISSING = object()


def _raise_if_exception(value: Any) -> None:
    if isinstance(value, BaseException):
        raise value
    if isinstance(value, type) and issubclass(value, BaseException):
        raise value


class FakeDevice:
    """A ``GpuDevice`` whose readings come from a mutable ``values`` mapping."""

    def __init__(self, identity: DeviceIdentity, **values: Any) -> None:
        self.identity = identity
        self.values = values
        self.calls: Counter[str] = Counter()

    def _read(self, name: str) -> Any:
        self.calls[name] += 1
        value = self.values.get(name, _MISSING)
        if value is _MISSING:
            raise Unsupported(f"{name} is not configured")
        _raise_if_exception(value)
        return value

    def mig_enabled(self) -> bool:
        return bool(self._read("mig_enabled"))

    def clock_mhz(self, kind: str) -> int:
        return int(self._read("clock_mhz")[kind])

    def gpu_temp_c(self) -> int:
        return int(self._read("gpu_temp_c"))

    def memory_temp_c(self) -> int:
        return int(self._read("memory_temp_c"))

    def power_mw(self) -> int:
        return int(self._read("power_mw"))

    def energy_mj(self) -> int:
        return int(self._read("energy_mj"))

    def pcie_replay(self) -> int:
        return int(self._read("pcie_replay"))

    def utilization_pct(self) -> tuple[int, int]:
        gpu, memory = self._read("utilization_pct")
        return int(gpu), int(memory)

    def encoder_pct(self) -> int:
        return int(self._read("encoder_pct"))

    def decoder_pct(self) -> int:
        return int(self._read("decoder_pct"))

    def memory(self) -> MemoryInfo:
        return self._read("memory")

    def remapped_rows(self) -> RemappedRows:
        return self._read("remapped_rows")

    def nvlink_raw(self) -> tuple[int, int]:
        tx, rx = self._read("nvlink_raw")
        return int(tx), int(rx)

    def gpm_supported(self) -> bool:
        self.calls["gpm_supported"] += 1
        return bool(self.values.get("gpm_supported", False))

    def gpm_sample(self) -> object:
        self.calls["gpm_sample"] += 1
        script = self.values.get("gpm_sample")
        if isinstance(script, list):
            if not script:
                return object()
            value = script.pop(0)
        else:
            value = script
        _raise_if_exception(value)
        return value

    def gpm_free(self, sample: object) -> None:
        self.calls["gpm_free"] += 1

    def gpm_metrics(self, prev: object, cur: object, metric_ids: Sequence[int]) -> dict[int, float]:
        self.calls["gpm_metrics"] += 1
        script = self.values.get("gpm_metrics")
        value = (script.pop(0) if script else {}) if isinstance(script, list) else script or {}
        _raise_if_exception(value)
        return {int(key): float(val) for key, val in value.items() if int(key) in metric_ids}


class FakeEventSource:
    """Return queued ``(gpu_index, xid_code)`` events, then ``None``."""

    def __init__(self, events: list[tuple[int, int] | BaseException]) -> None:
        self._events = list(events)
        self.closed = False
        self.available = True
        self.waits = 0

    def wait(self, timeout_ms: int) -> tuple[int, int] | None:
        self.waits += 1
        if not self._events:
            return None
        event = self._events.pop(0)
        _raise_if_exception(event)
        assert isinstance(event, tuple)
        return event

    def close(self) -> None:
        self.closed = True


def _identity(index: int, uuid_suffix: str, pci: str) -> DeviceIdentity:
    return DeviceIdentity(
        index=index,
        uuid=f"GPU-00000000-0000-0000-0000-00000000000{uuid_suffix}",
        pci_bus_id=pci,
        minor=index,
        name="NVIDIA H100 80GB HBM3",
    )


def canonical_pair() -> list[FakeDevice]:
    """Two datacenter-like GPUs with every reading populated.

    The values are also the source for ``tests/golden/default.prom``.
    """
    return [
        FakeDevice(
            _identity(0, "0", "00000000:17:00.0"),
            mig_enabled=False,
            clock_mhz={"sm": 1980, "mem": 2619},
            gpu_temp_c=41,
            memory_temp_c=39,
            power_mw=350123,
            energy_mj=1_032_632,
            pcie_replay=3,
            utilization_pct=(87, 42),
            encoder_pct=5,
            decoder_pct=2,
            memory=MemoryInfo(17103323136, 16461987840, 250347520, 390987776),
            remapped_rows=RemappedRows(3, 1, False, True),
            nvlink_raw=(1_000_000, 2_000_000),
            gpm_supported=True,
            gpm_sample=["gpu0-sample-1", "gpu0-sample-2", "gpu0-sample-3"],
            gpm_metrics=[
                {
                    pynvml.NVML_GPM_METRIC_GRAPHICS_UTIL: 55.0,
                    pynvml.NVML_GPM_METRIC_HMMA_TENSOR_UTIL: 12.5,
                    pynvml.NVML_GPM_METRIC_DRAM_BW_UTIL: 30.0,
                    pynvml.NVML_GPM_METRIC_PCIE_TX_PER_SEC: 1.125,
                    pynvml.NVML_GPM_METRIC_PCIE_RX_PER_SEC: 0.5,
                }
            ],
        ),
        FakeDevice(
            _identity(1, "1", "00000000:2A:00.0"),
            mig_enabled=False,
            clock_mhz={"sm": 1965, "mem": 2619},
            gpu_temp_c=43,
            memory_temp_c=41,
            power_mw=345000,
            energy_mj=987_654_321,
            pcie_replay=0,
            utilization_pct=(63, 30),
            encoder_pct=0,
            decoder_pct=0,
            memory=MemoryInfo(17103323136, 16000000000, 712335360, 390987776),
            remapped_rows=RemappedRows(0, 0, False, False),
            nvlink_raw=(500_000, 600_000),
            gpm_supported=True,
            gpm_sample=["gpu1-sample-1", "gpu1-sample-2", "gpu1-sample-3"],
            gpm_metrics=[
                {
                    pynvml.NVML_GPM_METRIC_GRAPHICS_UTIL: 40.0,
                    pynvml.NVML_GPM_METRIC_HMMA_TENSOR_UTIL: 10.0,
                    pynvml.NVML_GPM_METRIC_DRAM_BW_UTIL: 20.0,
                    pynvml.NVML_GPM_METRIC_PCIE_TX_PER_SEC: 0.25,
                    pynvml.NVML_GPM_METRIC_PCIE_RX_PER_SEC: 0.125,
                }
            ],
        ),
    ]


__all__ = [
    "FakeDevice",
    "FakeEventSource",
    "GpuLost",
    "ReadError",
    "Unsupported",
    "canonical_pair",
]

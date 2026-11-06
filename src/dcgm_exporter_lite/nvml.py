"""Thin ``pynvml`` facade with typed errors and a testable device protocol."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import pynvml

logger = logging.getLogger(__name__)


class Unsupported(Exception):
    """The driver, GPU, or function does not support a request."""


class GpuLost(Exception):
    """The GPU disappeared or returned ``NVML_ERROR_GPU_IS_LOST``."""


class ReadError(Exception):
    """Any other NVML failure."""


def _map_exception(exc: pynvml.NVMLError) -> Exception:
    if isinstance(exc, (pynvml.NVMLError_FunctionNotFound, pynvml.NVMLError_NotSupported)):
        return Unsupported(str(exc))
    if isinstance(exc, pynvml.NVMLError_GpuIsLost):
        return GpuLost(str(exc))
    return ReadError(str(exc))


def _call(fn, *args, **kwargs):
    """Call one NVML binding and map its error to the facade exceptions."""
    try:
        return fn(*args, **kwargs)
    except pynvml.NVMLError as exc:
        raise _map_exception(exc) from exc


def _raise_for_code(code: int, context: str) -> None:
    if code == pynvml.NVML_SUCCESS:
        return
    if code in (pynvml.NVML_ERROR_NOT_SUPPORTED, pynvml.NVML_ERROR_FUNCTION_NOT_FOUND):
        raise Unsupported(f"{context}: NVML error {code}")
    if code == pynvml.NVML_ERROR_GPU_IS_LOST:
        raise GpuLost(f"{context}: NVML error {code}")
    raise ReadError(f"{context}: NVML error {code}")


def _text(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def driver_major(version: str) -> int:
    """Return the leading integer of a driver version string, or 0."""
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class DeviceIdentity:
    index: int
    uuid: str
    pci_bus_id: str
    minor: int
    name: str


@dataclass(frozen=True)
class MemoryInfo:
    total: int
    free: int
    used: int
    reserved: int


@dataclass(frozen=True)
class RemappedRows:
    correctable: int
    uncorrectable: int
    pending: bool
    failed: bool


class GpuDevice(Protocol):
    identity: DeviceIdentity

    def mig_enabled(self) -> bool: ...

    def clock_mhz(self, kind: Literal["sm", "mem"]) -> int: ...

    def gpu_temp_c(self) -> int: ...

    def memory_temp_c(self) -> int: ...

    def power_mw(self) -> int: ...

    def energy_mj(self) -> int: ...

    def pcie_replay(self) -> int: ...

    def utilization_pct(self) -> tuple[int, int]: ...

    def encoder_pct(self) -> int: ...

    def decoder_pct(self) -> int: ...

    def memory(self) -> MemoryInfo: ...

    def remapped_rows(self) -> RemappedRows: ...

    def nvlink_raw(self) -> tuple[int, int]: ...

    def gpm_supported(self) -> bool: ...

    def gpm_sample(self) -> object: ...

    def gpm_free(self, sample: object) -> None: ...

    def gpm_metrics(
        self, prev: object, cur: object, metric_ids: Sequence[int]
    ) -> dict[int, float]: ...


class NvmlDevice:
    """``GpuDevice`` implementation over one pynvml handle."""

    def __init__(self, handle, index: int) -> None:
        self._handle = handle
        self.identity = DeviceIdentity(
            index=index,
            uuid=_text(_call(pynvml.nvmlDeviceGetUUID, handle)),
            pci_bus_id=_text(_call(pynvml.nvmlDeviceGetPciInfo_v3, handle).busId).upper(),
            minor=int(_call(pynvml.nvmlDeviceGetMinorNumber, handle)),
            name=_text(_call(pynvml.nvmlDeviceGetName, handle)),
        )

    def mig_enabled(self) -> bool:
        try:
            mode = _call(pynvml.nvmlDeviceGetMigMode, self._handle)[0]
        except Unsupported:
            return False
        return mode == pynvml.NVML_DEVICE_MIG_ENABLE

    def clock_mhz(self, kind: Literal["sm", "mem"]) -> int:
        clock_type = pynvml.NVML_CLOCK_SM if kind == "sm" else pynvml.NVML_CLOCK_MEM
        return int(_call(pynvml.nvmlDeviceGetClockInfo, self._handle, clock_type))

    def gpu_temp_c(self) -> int:
        return int(
            _call(pynvml.nvmlDeviceGetTemperature, self._handle, pynvml.NVML_TEMPERATURE_GPU)
        )

    def memory_temp_c(self) -> int:
        return self._field_int(pynvml.NVML_FI_DEV_MEMORY_TEMP)

    def power_mw(self) -> int:
        return int(_call(pynvml.nvmlDeviceGetPowerUsage, self._handle))

    def energy_mj(self) -> int:
        return int(_call(pynvml.nvmlDeviceGetTotalEnergyConsumption, self._handle))

    def pcie_replay(self) -> int:
        return int(_call(pynvml.nvmlDeviceGetPcieReplayCounter, self._handle))

    def utilization_pct(self) -> tuple[int, int]:
        rates = _call(pynvml.nvmlDeviceGetUtilizationRates, self._handle)
        return int(rates.gpu), int(rates.memory)

    def encoder_pct(self) -> int:
        return int(_call(pynvml.nvmlDeviceGetEncoderUtilization, self._handle)[0])

    def decoder_pct(self) -> int:
        return int(_call(pynvml.nvmlDeviceGetDecoderUtilization, self._handle)[0])

    def memory(self) -> MemoryInfo:
        info = _call(pynvml.nvmlDeviceGetMemoryInfo, self._handle, version=pynvml.nvmlMemory_v2)
        return MemoryInfo(
            total=int(info.total),
            free=int(info.free),
            used=int(info.used),
            reserved=int(info.reserved),
        )

    def remapped_rows(self) -> RemappedRows:
        correctable, uncorrectable, pending, failed = _call(
            pynvml.nvmlDeviceGetRemappedRows, self._handle
        )
        return RemappedRows(
            correctable=int(correctable),
            uncorrectable=int(uncorrectable),
            pending=bool(pending),
            failed=bool(failed),
        )

    def nvlink_raw(self) -> tuple[int, int]:
        return (
            self._field_int(pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX),
            self._field_int(pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX),
        )

    def gpm_supported(self) -> bool:
        try:
            support = _call(pynvml.nvmlGpmQueryDeviceSupport, self._handle)
        except Unsupported:
            return False
        return bool(support.isSupportedDevice)

    def gpm_sample(self) -> object:
        sample = _call(pynvml.nvmlGpmSampleAlloc)
        try:
            _call(pynvml.nvmlGpmSampleGet, self._handle, sample)
        except Exception:
            with contextlib.suppress(pynvml.NVMLError):
                pynvml.nvmlGpmSampleFree(sample)
            raise
        return sample

    def gpm_free(self, sample: object) -> None:
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlGpmSampleFree(sample)

    def gpm_metrics(self, prev: object, cur: object, metric_ids: Sequence[int]) -> dict[int, float]:
        request = pynvml.c_nvmlGpmMetricsGet_t()
        request.version = pynvml.NVML_GPM_METRICS_GET_VERSION
        request.numMetrics = len(metric_ids)
        request.sample1 = prev
        request.sample2 = cur
        for i, metric_id in enumerate(metric_ids):
            request.metrics[i].metricId = metric_id

        _call(pynvml.nvmlGpmMetricsGet, request)

        values: dict[int, float] = {}
        for i, metric_id in enumerate(metric_ids):
            metric = request.metrics[i]
            code = int(metric.nvmlReturn)
            if code in (
                pynvml.NVML_ERROR_NOT_SUPPORTED,
                pynvml.NVML_ERROR_FUNCTION_NOT_FOUND,
            ):
                continue
            _raise_for_code(code, f"GPM metric {metric_id}")
            values[metric_id] = float(metric.value)
        return values

    def _field_int(self, field_id: int) -> int:
        values = _call(pynvml.nvmlDeviceGetFieldValues, self._handle, [field_id])
        value = values[0]
        _raise_for_code(int(value.nvmlReturn), f"field {field_id}")
        if value.valueType == pynvml.NVML_VALUE_TYPE_DOUBLE:
            return int(value.value.dVal)
        return int(value.value.sllVal)


class XidEventSource(Protocol):
    def wait(self, timeout_ms: int) -> tuple[int, int] | None: ...

    def close(self) -> None: ...


def _handle_key(handle) -> int:
    value = getattr(handle, "value", handle)
    return int(value or 0)


class NvmlXidEventSource:
    """Event-set source for ``nvmlEventTypeXidCriticalError``."""

    def __init__(self, devices: Sequence[NvmlDevice]) -> None:
        self._event_set = _call(pynvml.nvmlEventSetCreate)
        self._index_by_handle: dict[int, int] = {}

        for device in devices:
            try:
                _call(
                    pynvml.nvmlDeviceRegisterEvents,
                    device._handle,
                    pynvml.nvmlEventTypeXidCriticalError,
                    self._event_set,
                )
            except Unsupported:
                logger.info("XID events are not supported on GPU %d", device.identity.index)
                continue
            self._index_by_handle[_handle_key(device._handle)] = device.identity.index

        self.available = bool(self._index_by_handle)

    def wait(self, timeout_ms: int) -> tuple[int, int] | None:
        try:
            event = pynvml.nvmlEventSetWait(self._event_set, timeout_ms)
        except pynvml.NVMLError as exc:
            if isinstance(exc, pynvml.NVMLError_Timeout):
                return None
            raise _map_exception(exc) from exc

        if not int(event.eventType) & pynvml.nvmlEventTypeXidCriticalError:
            return None
        index = self._index_by_handle.get(_handle_key(event.device))
        if index is None:
            return None
        return index, int(event.eventData)

    def close(self) -> None:
        if self._event_set is None:
            return
        try:
            pynvml.nvmlEventSetFree(self._event_set)
        except pynvml.NVMLError:
            logger.debug("failed to free the XID event set", exc_info=True)
        self._event_set = None


def init() -> str:
    """Initialize NVML and return the driver version."""
    _call(pynvml.nvmlInit)
    return _text(_call(pynvml.nvmlSystemGetDriverVersion))


def shutdown() -> None:
    _call(pynvml.nvmlShutdown)


def open_devices() -> list[NvmlDevice]:
    count = int(_call(pynvml.nvmlDeviceGetCount))
    devices: list[NvmlDevice] = []
    for index in range(count):
        handle = _call(pynvml.nvmlDeviceGetHandleByIndex, index)
        devices.append(NvmlDevice(handle, index))
    return devices

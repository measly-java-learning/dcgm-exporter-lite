from __future__ import annotations

from types import SimpleNamespace

import pynvml
import pytest

from dcgm_exporter_lite import nvml


def _raise(exc: BaseException):
    def boom(*args, **kwargs):
        raise exc

    return boom


def test_not_supported_maps_to_unsupported() -> None:
    with pytest.raises(nvml.Unsupported):
        nvml._call(_raise(pynvml.NVMLError_NotSupported(3)))


def test_function_not_found_maps_to_unsupported() -> None:
    with pytest.raises(nvml.Unsupported):
        nvml._call(_raise(pynvml.NVMLError_FunctionNotFound(13)))


def test_gpu_lost_maps() -> None:
    with pytest.raises(nvml.GpuLost):
        nvml._call(_raise(pynvml.NVMLError_GpuIsLost(15)))


def test_other_error_maps_to_read_error() -> None:
    with pytest.raises(nvml.ReadError):
        nvml._call(_raise(pynvml.NVMLError_Unknown(999)))


def _patch_identity(monkeypatch) -> None:
    monkeypatch.setattr(pynvml, "nvmlDeviceGetUUID", lambda handle: b"GPU-uuid")
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetPciInfo_v3",
        lambda handle: SimpleNamespace(busId="00000000:1b:00.0"),
    )
    monkeypatch.setattr(pynvml, "nvmlDeviceGetMinorNumber", lambda handle: 3)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetName", lambda handle: b"NVIDIA Test")


def test_identity_formats(monkeypatch) -> None:
    _patch_identity(monkeypatch)

    device = nvml.NvmlDevice("handle", 5)

    assert device.identity.index == 5
    assert device.identity.uuid == "GPU-uuid"
    assert device.identity.pci_bus_id == "00000000:1B:00.0"
    assert device.identity.minor == 3
    assert device.identity.name == "NVIDIA Test"


def test_identity_accepts_str_name(monkeypatch) -> None:
    _patch_identity(monkeypatch)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetUUID", lambda handle: "GPU-str")
    monkeypatch.setattr(pynvml, "nvmlDeviceGetName", lambda handle: "NVIDIA String")

    device = nvml.NvmlDevice("handle", 0)

    assert device.identity.uuid == "GPU-str"
    assert device.identity.name == "NVIDIA String"


def test_driver_major() -> None:
    assert nvml.driver_major("580.65.06") == 580
    assert nvml.driver_major("615.71.10") == 615
    assert nvml.driver_major("not-a-version") == 0


def test_memory_uses_v2(monkeypatch) -> None:
    _patch_identity(monkeypatch)
    versions: list[int | None] = []

    def memory_info(handle, version=None):
        versions.append(version)
        return SimpleNamespace(total=1, free=2, used=3, reserved=4)

    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo", memory_info)
    device = nvml.NvmlDevice("handle", 0)

    assert device.memory() == nvml.MemoryInfo(1, 2, 3, 4)
    assert versions == [pynvml.nvmlMemory_v2]


def test_xid_source_records_registered_gpus(monkeypatch) -> None:
    _patch_identity(monkeypatch)
    monkeypatch.setattr(pynvml, "nvmlEventSetCreate", lambda: "event-set")

    def register(handle, event_types, event_set):
        if handle == 2:
            raise pynvml.NVMLError_NotSupported(3)

    monkeypatch.setattr(pynvml, "nvmlDeviceRegisterEvents", register)
    devices = [nvml.NvmlDevice(1, 0), nvml.NvmlDevice(2, 1)]

    source = nvml.NvmlXidEventSource(devices)

    assert source.available is True
    assert source.registered == frozenset({0})

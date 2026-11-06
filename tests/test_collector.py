from __future__ import annotations

import logging
import time

import pynvml
from fakes import FakeDevice, canonical_pair

from dcgm_exporter_lite.collector import CollectionLoop, Collector, Snapshot
from dcgm_exporter_lite.counters import CounterSpec, load_counters
from dcgm_exporter_lite.gpm import GpmSampler
from dcgm_exporter_lite.nvml import GpuLost, ReadError, Unsupported


def _collector(
    devices: list[FakeDevice],
    *,
    counters: list[CounterSpec] | None = None,
    hostname: str | None = "host-a",
    replace_blanks: bool = False,
    interval_s: float = 30.0,
    xid=None,
    clock=None,
) -> Collector:
    kwargs = {} if clock is None else {"clock": clock}
    return Collector(
        devices,
        counters if counters is not None else load_counters(None),
        driver_version="580.65.06",
        hostname=hostname,
        replace_blanks=replace_blanks,
        gpm=GpmSampler(devices),
        xid=xid,
        interval_s=interval_s,
        **kwargs,
    )


def _gpu_names(snapshot: Snapshot, gpu: str) -> set[str]:
    return {sample.name for sample in snapshot.samples if dict(sample.labels)["gpu"] == gpu}


def _wait_for(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_labels_order_and_values() -> None:
    devices = canonical_pair()
    collector = _collector(devices)

    assert collector.labels_for(devices[0]) == (
        ("gpu", "0"),
        ("UUID", "GPU-00000000-0000-0000-0000-000000000000"),
        ("pci_bus_id", "00000000:17:00.0"),
        ("device", "nvidia0"),
        ("modelName", "NVIDIA H100 80GB HBM3"),
        ("hostname", "host-a"),
        ("DCGM_FI_SYSTEM_DRIVER_VERSION", "580.65.06"),
    )


def test_no_hostname_and_replace_blanks() -> None:
    devices = canonical_pair()
    collector = _collector(devices, hostname=None, replace_blanks=True)

    labels = dict(collector.labels_for(devices[0]))

    assert "hostname" not in labels
    assert labels["modelName"] == "NVIDIA-H100-80GB-HBM3"


def test_mixed_fleet_unsupported_omitted() -> None:
    devices = canonical_pair()
    devices[1].values["memory_temp_c"] = Unsupported("not on this GPU")
    collector = _collector(devices)

    collector.probe()
    snapshot = collector.collect_once()

    temps = [
        sample for sample in snapshot.samples if sample.name == "DCGM_FI_DEV_MEMORY_TEMP_CELSIUS"
    ]
    assert [dict(sample.labels)["gpu"] for sample in temps] == ["0"]
    assert snapshot.ok_gpus == 2


def test_function_not_found_treated_as_unsupported() -> None:
    devices = canonical_pair()
    devices[1].values["memory_temp_c"] = pynvml.NVMLError_FunctionNotFound(13)
    collector = _collector(devices)

    collector.probe()
    snapshot = collector.collect_once()

    temps = [
        sample for sample in snapshot.samples if sample.name == "DCGM_FI_DEV_MEMORY_TEMP_CELSIUS"
    ]
    assert [dict(sample.labels)["gpu"] for sample in temps] == ["0"]


def test_unknown_counter_warns_once(caplog) -> None:
    devices = canonical_pair()
    collector = _collector(
        devices,
        counters=[
            CounterSpec("BOGUS_COUNTER", "gauge", "not implemented"),
            CounterSpec("DCGM_FI_DEV_GPU_TEMP_CELSIUS", "gauge", "temperature"),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="dcgm_exporter_lite.collector"):
        collector.probe()
        collector.probe()

    assert caplog.text.count("BOGUS_COUNTER") == 1


def test_mig_gpu_only_gpu_wide() -> None:
    devices = canonical_pair()
    devices[0].values["mig_enabled"] = True
    collector = _collector(devices)

    collector.probe()
    snapshot = collector.collect_once()

    gpu0 = _gpu_names(snapshot, "0")
    assert not any(name.endswith("_UTIL_RATIO") for name in gpu0)
    assert not any(name.startswith("DCGM_FI_PROF_") for name in gpu0)
    assert "DCGM_FI_DEV_GPU_TEMP_CELSIUS" in gpu0
    assert "DCGM_FI_DEV_BOARD_POWER_WATTS" in gpu0
    assert "DCGM_FI_DEV_GPU_UTIL_RATIO" in _gpu_names(snapshot, "1")


def test_read_error_skips_sample_then_recovers() -> None:
    devices = canonical_pair()
    devices[0].values["gpu_temp_c"] = ReadError("transient")
    collector = _collector(devices)
    collector.probe()

    first = collector.collect_once()
    assert not any(
        sample.name == "DCGM_FI_DEV_GPU_TEMP_CELSIUS" and dict(sample.labels)["gpu"] == "0"
        for sample in first.samples
    )

    devices[0].values["gpu_temp_c"] = 42
    second = collector.collect_once()
    temps = [
        sample
        for sample in second.samples
        if sample.name == "DCGM_FI_DEV_GPU_TEMP_CELSIUS" and dict(sample.labels)["gpu"] == "0"
    ]
    assert [sample.value for sample in temps] == [42.0]


def test_gpu_lost_mid_run() -> None:
    devices = canonical_pair()
    collector = _collector(devices)
    collector.probe()

    first = collector.collect_once()
    assert first.ok_gpus == 2

    devices[1].values["gpu_temp_c"] = GpuLost("gone")
    second = collector.collect_once()

    assert second.ok_gpus == 1
    assert {dict(sample.labels)["gpu"] for sample in second.samples} == {"0"}

    calls_after_second = dict(devices[1].calls)
    third = collector.collect_once()

    assert third.ok_gpus == 1
    assert dict(devices[1].calls) == calls_after_second


def test_first_cycle_has_no_prof_second_has() -> None:
    devices = canonical_pair()
    collector = _collector(devices)
    collector.probe()

    first = collector.collect_once()
    assert not any(sample.name.startswith("DCGM_FI_PROF_") for sample in first.samples)

    second = collector.collect_once()
    assert any(sample.name.startswith("DCGM_FI_PROF_") for sample in second.samples)


def test_loop_runs_and_stops() -> None:
    devices = canonical_pair()
    collector = _collector(devices, interval_s=0.05)
    collector.probe()
    loop = CollectionLoop(collector, 0.05)

    loop.start()
    try:
        assert _wait_for(lambda: collector.latest is not None)
    finally:
        loop.stop()

    assert not loop.is_alive()


class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class _PartialXid:
    """XID watcher that only watches GPU 0."""

    available = True

    def watches(self, index: int) -> bool:
        return index == 0

    def last(self, index: int) -> int | None:
        return None


def _nvlink_values(snapshot: Snapshot) -> dict[str, float]:
    return {
        dict(sample.labels)["gpu"]: sample.value
        for sample in snapshot.samples
        if sample.name == "DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND"
    }


def test_probe_does_not_prime_nvlink_baseline() -> None:
    devices = canonical_pair()
    clock = _Clock()
    collector = _collector(devices, clock=clock)
    collector.probe()

    # Traffic between the probe and the first cycle must not become a rate.
    devices[0].values["nvlink_raw"] = (1_010_240, 2_000_000)
    clock.now += 0.01
    first = collector.collect_once()
    assert _nvlink_values(first) == {}

    devices[0].values["nvlink_raw"] = (1_040_240, 2_060_000)
    clock.now += 30.0
    second = collector.collect_once()
    assert _nvlink_values(second)["0"] == 3_072_000.0


def test_nvlink_rate_after_skipped_cycle_uses_elapsed_time() -> None:
    devices = canonical_pair()
    clock = _Clock()
    collector = _collector(devices, clock=clock)
    collector.probe()
    collector.collect_once()

    devices[0].values["nvlink_raw"] = ReadError("transient")
    clock.now += 30.0
    collector.collect_once()

    devices[0].values["nvlink_raw"] = (1_060_000, 2_120_000)
    clock.now += 30.0
    third = collector.collect_once()
    assert _nvlink_values(third)["0"] == 3_072_000.0


def test_xid_omitted_for_gpu_without_event_registration() -> None:
    collector = _collector(canonical_pair(), xid=_PartialXid())
    collector.probe()
    snapshot = collector.collect_once()

    assert "DCGM_FI_DEV_XID_ERROR" in _gpu_names(snapshot, "0")
    assert "DCGM_FI_DEV_XID_ERROR" not in _gpu_names(snapshot, "1")


def test_healthy_requires_recent_snapshot() -> None:
    clock = _Clock()
    collector = _collector(canonical_pair(), clock=clock)
    collector.probe()

    assert collector.healthy(max_age_s=90.0) is False
    collector.collect_once()
    assert collector.healthy(max_age_s=90.0) is True

    clock.now += 91.0
    assert collector.healthy(max_age_s=90.0) is False


def test_gpm_unavailable_logged_once_per_gpu(caplog) -> None:
    devices = canonical_pair()
    for device in devices:
        device.values["gpm_supported"] = False
    collector = _collector(devices)

    with caplog.at_level(logging.INFO, logger="dcgm_exporter_lite.collector"):
        collector.probe()

    assert caplog.text.count("GPM is not available on GPU 0") == 1
    assert caplog.text.count("GPM is not available on GPU 1") == 1


def test_deliberately_omitted_counter_logs_info_not_warning(caplog) -> None:
    collector = _collector(canonical_pair())

    with caplog.at_level(logging.INFO, logger="dcgm_exporter_lite.collector"):
        collector.probe()

    records = [r for r in caplog.records if "DCGM_FI_DEV_VGPU_LICENSE_STATUS" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO

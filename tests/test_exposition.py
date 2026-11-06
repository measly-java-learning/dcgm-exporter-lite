from __future__ import annotations

from pathlib import Path

from fakes import FakeDevice, canonical_pair

from dcgm_exporter_lite.collector import Collector, Snapshot
from dcgm_exporter_lite.counters import CounterSpec, load_counters
from dcgm_exporter_lite.exposition import make_registry, render
from dcgm_exporter_lite.gpm import GpmSampler
from dcgm_exporter_lite.nvml import DeviceIdentity

GOLDEN = Path(__file__).parent / "golden" / "default.prom"


class _StaticXid:
    available = True

    def watches(self, index: int) -> bool:
        return index in (0, 1)

    def last(self, index: int) -> int | None:
        return {0: 79, 1: 0}.get(index)


def _second_cycle_snapshot() -> Snapshot:
    devices = canonical_pair()
    clock = [0.0]
    collector = Collector(
        devices,
        load_counters(None),
        driver_version="580.65.06",
        hostname="host-a",
        replace_blanks=False,
        gpm=GpmSampler(devices),
        xid=_StaticXid(),  # type: ignore[arg-type]
        interval_s=30.0,
        clock=lambda: clock[0],
    )
    collector.probe()
    collector.collect_once()
    clock[0] = 30.0
    devices[0].values["nvlink_raw"] = (1_030_000, 2_060_000)
    devices[1].values["nvlink_raw"] = (530_000, 660_000)
    return collector.collect_once()


def test_golden_default() -> None:
    snapshot = _second_cycle_snapshot()
    registry = make_registry(lambda: snapshot, load_counters(None))

    assert render(registry) == GOLDEN.read_bytes()


def test_counter_names_match_dcgm() -> None:
    snapshot = _second_cycle_snapshot()
    registry = make_registry(lambda: snapshot, load_counters(None))
    text = render(registry).decode()

    assert "# TYPE DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL counter" in text
    assert 'DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL{gpu="0",' in text
    assert "_TOTAL_total" not in text
    assert "_created" not in text


def test_no_process_or_python_metrics() -> None:
    snapshot = _second_cycle_snapshot()
    registry = make_registry(lambda: snapshot, load_counters(None))
    text = render(registry).decode()

    assert "process_" not in text
    assert "python_" not in text


def test_empty_snapshot_renders_valid_empty() -> None:
    registry = make_registry(lambda: None, load_counters(None))

    assert render(registry) == b""


def test_label_value_escaping() -> None:
    identity = DeviceIdentity(0, "GPU-x", "00000000:01:00.0", 0, 'ACME "GPU" \\ Pro')
    device = FakeDevice(identity, gpu_temp_c=41)
    counters = [CounterSpec("DCGM_FI_DEV_GPU_TEMP_CELSIUS", "gauge", "GPU temperature (in C).")]
    collector = Collector(
        [device],
        counters,
        driver_version="580.65.06",
        hostname=None,
        replace_blanks=False,
        gpm=GpmSampler([device]),
        xid=None,
        interval_s=30.0,
    )
    collector.probe()
    snapshot = collector.collect_once()
    text = render(make_registry(lambda: snapshot, counters)).decode()

    assert 'modelName="ACME \\"GPU\\" \\\\ Pro"' in text

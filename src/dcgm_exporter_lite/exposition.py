"""Render snapshots in DCGM Exporter 4.9.0's Prometheus text format."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from prometheus_client import CollectorRegistry
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.exposition import floatToGoString
from prometheus_client.registry import Collector

from .collector import Sample, Snapshot
from .counters import CounterSpec

# DCGM emits the fixed GPU labels in this order, then extra labels sorted by name.
_LABEL_ORDER = {
    "gpu": 0,
    "UUID": 1,
    "pci_bus_id": 2,
    "device": 3,
    "modelName": 4,
    "hostname": 5,
}


class SnapshotCollector(Collector):
    """A prometheus_client collector over one immutable snapshot."""

    def __init__(
        self,
        get_snapshot: Callable[[], Snapshot | None],
        counters: Sequence[CounterSpec],
    ) -> None:
        self._get_snapshot = get_snapshot
        self._counters = [counter for counter in counters if counter.type != "label"]

    def collect(self):
        snapshot = self._get_snapshot()
        if snapshot is None:
            return

        by_name: dict[str, list[Sample]] = {}
        for sample in snapshot.samples:
            by_name.setdefault(sample.name, []).append(sample)

        for counter in self._counters:
            samples = by_name.get(counter.name)
            if not samples:
                continue

            label_names = [name for name, _ in samples[0].labels]
            if counter.type == "counter":
                # CounterMetricFamily would append "_total" to the already
                # suffixed DCGM name; a raw counter Metric keeps it exact.
                metric = Metric(counter.name, counter.help, typ="counter")
                for sample in samples:
                    metric.add_sample(sample.name, dict(sample.labels), sample.value)
                yield metric
            else:
                family = GaugeMetricFamily(counter.name, counter.help, labels=label_names)
                for sample in samples:
                    values = dict(sample.labels)
                    family.add_metric([values[name] for name in label_names], sample.value)
                yield family


def make_registry(
    get_snapshot: Callable[[], Snapshot | None],
    counters: Sequence[CounterSpec],
) -> CollectorRegistry:
    """Build a registry containing only the GPU snapshot collector."""
    registry = CollectorRegistry(auto_describe=False)
    registry.register(SnapshotCollector(get_snapshot, counters))
    return registry


def render(registry: CollectorRegistry) -> bytes:
    """Serialize a registry in DCGM's exact text exposition shape."""
    lines: list[str] = []

    for family in registry.collect():
        lines.append(f"# HELP {family.name} {_escape_help(family.documentation)}\n")
        lines.append(f"# TYPE {family.name} {family.type}\n")
        for sample in family.samples:
            label_text = _format_labels(sample.labels)
            lines.append(f"{sample.name}{label_text} {_go_float(sample.value)}\n")

    return "".join(lines).encode("utf-8")


def _format_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    ordered = sorted(
        labels.items(),
        key=lambda item: (_LABEL_ORDER.get(item[0], len(_LABEL_ORDER)), item[0]),
    )
    body = ",".join(f'{name}="{_escape_label(value)}"' for name, value in ordered)
    return "{" + body + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _escape_help(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def _go_float(value: float) -> str:
    text = floatToGoString(value)
    if text.endswith(".0"):
        return text[:-2]
    return text

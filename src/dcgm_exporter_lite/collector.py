"""Support probing, snapshot building, and the background collection loop."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

from .counters import CounterSpec
from .gpm import GpmSampler
from .metrics import METRICS, MetricDef, NoValueYet, ReadContext
from .nvml import GpuDevice, GpuLost, ReadError, Unsupported
from .xid import XidWatcher, xid_message

logger = logging.getLogger(__name__)

_READ_ERROR_LOG_INTERVAL_S = 300.0


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


@dataclass(frozen=True)
class Snapshot:
    samples: tuple[Sample, ...]
    ok_gpus: int
    taken_at: float


class Collector:
    """Build immutable snapshots of the latest NVML readings."""

    def __init__(
        self,
        devices: Sequence[GpuDevice],
        counters: Sequence[CounterSpec],
        *,
        driver_version: str,
        hostname: str | None,
        replace_blanks: bool,
        gpm: GpmSampler,
        xid: XidWatcher | None,
        interval_s: float,
    ) -> None:
        self._devices = list(devices)
        self._counters = list(counters)
        self._driver_version = driver_version
        self._hostname = hostname
        self._replace_blanks = replace_blanks
        self._gpm = gpm
        self._xid = xid
        self._interval_s = interval_s

        self._metrics: list[tuple[CounterSpec, MetricDef]] = []
        self._label_counters: list[CounterSpec] = [
            spec
            for spec in self._counters
            if spec.type == "label"
            and spec.name in ("DCGM_FI_SYSTEM_DRIVER_VERSION", "DCGM_FI_DRIVER_VERSION")
        ]
        self._warned_unknown: set[str] = set()
        self._read_error_logged: dict[tuple[int, str], float] = {}
        self._enabled: dict[int, set[str]] = {}
        self._state: dict[int, dict] = {}
        self._lost: set[int] = set()

        self.latest: Snapshot | None = None

    def probe(self) -> None:
        """Compute the per-GPU enabled metric set."""
        self._prepare_metrics()
        self._enabled = {}
        self._lost = set()
        gpm_support: dict[int, bool] = {}

        for position, device in enumerate(self._devices):
            try:
                mig = device.mig_enabled()
            except Unsupported:
                mig = False
            except GpuLost:
                self._mark_lost(position, device)
                continue
            except ReadError as exc:
                self._log_read_error(position, "mig_mode", exc)
                mig = False

            enabled = {definition.name for _, definition in self._metrics}
            if mig:
                enabled = {name for name in enabled if METRICS[name].gpu_wide}
                logger.info(
                    "GPU %d has MIG enabled; only GPU-wide metrics are exported",
                    device.identity.index,
                )

            for name in list(enabled):
                definition = METRICS[name]
                if definition.source == "gpm":
                    if not self._gpm_available(position, device, gpm_support):
                        enabled.discard(name)
                    continue
                if definition.source == "xid":
                    if not self._xid_available():
                        enabled.discard(name)
                    continue

                context = self._context(position, device, None, None)
                try:
                    definition.read(context)
                except NoValueYet:
                    continue
                except Unsupported:
                    enabled.discard(name)
                    logger.info(
                        "metric %s is not supported on GPU %d; omitting it",
                        name,
                        device.identity.index,
                    )
                except GpuLost:
                    self._mark_lost(position, device)
                    enabled = set()
                    break
                except ReadError as exc:
                    self._log_read_error(position, name, exc)

            self._enabled[position] = enabled

    def collect_once(self) -> Snapshot:
        """Read every enabled metric once and publish a snapshot."""
        taken_at = time.time()
        samples: list[Sample] = []
        ok_gpus = 0

        for position, device in enumerate(self._devices):
            if position in self._lost:
                continue

            labels = self.labels_for(device)
            state = self._state.setdefault(position, {})
            gpm_values = self._sample_gpm(position, device)
            xid_last = self._last_xid(device)
            local_samples: list[Sample] = []
            lost = False

            for spec, definition in self._metrics:
                if spec.name not in self._enabled.get(position, set()):
                    continue

                context = ReadContext(
                    device=device,
                    gpm=gpm_values,
                    xid_last=xid_last,
                    interval_s=self._interval_s,
                    state=state,
                )
                try:
                    value = definition.read(context)
                except NoValueYet:
                    continue
                except Unsupported:
                    self._enabled[position].discard(spec.name)
                    logger.info(
                        "metric %s became unsupported on GPU %d; omitting it",
                        spec.name,
                        device.identity.index,
                    )
                    continue
                except GpuLost:
                    self._mark_lost(position, device)
                    lost = True
                    break
                except ReadError as exc:
                    self._log_read_error(position, spec.name, exc)
                    continue

                sample_labels = labels
                if spec.name == "DCGM_FI_DEV_XID_ERROR":
                    code = int(value)
                    sample_labels = labels + (
                        ("err_code", str(code)),
                        ("err_msg", xid_message(code)),
                    )
                local_samples.append(Sample(spec.name, sample_labels, float(value)))

            if lost:
                continue

            if local_samples:
                ok_gpus += 1
            samples.extend(local_samples)

        snapshot = Snapshot(tuple(samples), ok_gpus, taken_at)
        self.latest = snapshot
        return snapshot

    def labels_for(self, device: GpuDevice) -> tuple[tuple[str, str], ...]:
        identity = device.identity
        model = identity.name
        if self._replace_blanks:
            model = " ".join(model.split()).replace(" ", "-")

        labels: list[tuple[str, str]] = [
            ("gpu", str(identity.index)),
            ("UUID", identity.uuid),
            ("pci_bus_id", identity.pci_bus_id),
            ("device", f"nvidia{identity.minor}"),
            ("modelName", model),
        ]
        if self._hostname is not None:
            labels.append(("hostname", self._hostname))
        labels.extend((spec.name, self._driver_version) for spec in self._label_counters)
        return tuple(labels)

    def _prepare_metrics(self) -> None:
        self._metrics = []
        self._label_counters = []

        for spec in self._counters:
            if spec.type == "label":
                if spec.name in ("DCGM_FI_SYSTEM_DRIVER_VERSION", "DCGM_FI_DRIVER_VERSION"):
                    self._label_counters.append(spec)
                elif spec.name not in self._warned_unknown:
                    self._warned_unknown.add(spec.name)
                    logger.warning("label counter %s is not implemented; ignoring", spec.name)
                continue

            definition = METRICS.get(spec.name)
            if definition is None:
                if spec.name not in self._warned_unknown:
                    self._warned_unknown.add(spec.name)
                    logger.warning("counter %s is not implemented; ignoring", spec.name)
                continue
            self._metrics.append((spec, definition))

    def _gpm_available(self, position: int, device: GpuDevice, cache: dict[int, bool]) -> bool:
        if position not in cache:
            try:
                cache[position] = device.gpm_supported()
            except Unsupported:
                cache[position] = False
            except GpuLost:
                self._mark_lost(position, device)
                cache[position] = False
            except ReadError as exc:
                self._log_read_error(position, "gpm_support", exc)
                cache[position] = False
        if not cache[position]:
            logger.info(
                "GPM is not available on GPU %d; omitting PROF_* metrics",
                device.identity.index,
            )
        return cache[position]

    def _xid_available(self) -> bool:
        return self._xid is not None and bool(getattr(self._xid, "available", True))

    def _sample_gpm(self, position: int, device: GpuDevice) -> dict[str, float] | None:
        if not any(METRICS[name].source == "gpm" for name in self._enabled.get(position, set())):
            return None
        try:
            values = self._gpm.step(position)
        except Unsupported:
            self._enabled[position] = {
                name for name in self._enabled.get(position, set()) if METRICS[name].source != "gpm"
            }
            logger.info(
                "GPM became unavailable on GPU %d; omitting PROF_* metrics",
                device.identity.index,
            )
            return None
        except GpuLost:
            self._mark_lost(position, device)
            return None
        except ReadError as exc:
            self._log_read_error(position, "gpm", exc)
            return None
        except Exception:
            logger.exception("unexpected GPM failure on GPU %d", device.identity.index)
            return None
        return dict(values) if values is not None else None

    def _last_xid(self, device: GpuDevice) -> int | None:
        if not self._xid_available():
            return None
        return self._xid.last(device.identity.index)  # type: ignore[union-attr]

    def _context(
        self,
        position: int,
        device: GpuDevice,
        gpm: dict[str, float] | None,
        xid_last: int | None,
    ) -> ReadContext:
        return ReadContext(
            device=device,
            gpm=gpm,
            xid_last=xid_last,
            interval_s=self._interval_s,
            state=self._state.setdefault(position, {}),
        )

    def _mark_lost(self, position: int, device: GpuDevice) -> None:
        if position in self._lost:
            return
        self._lost.add(position)
        self._enabled[position] = set()
        logger.error(
            "GPU %d (%s) was lost; its series will no longer be exported",
            device.identity.index,
            device.identity.uuid,
        )

    def _log_read_error(self, position: int, name: str, exc: Exception) -> None:
        key = (position, name)
        now = time.monotonic()
        last = self._read_error_logged.get(key)
        if last is not None and now - last < _READ_ERROR_LOG_INTERVAL_S:
            return
        self._read_error_logged[key] = now
        logger.warning(
            "reading %s on GPU %d failed: %s",
            name,
            self._devices[position].identity.index,
            exc,
        )


class CollectionLoop(threading.Thread):
    """Collect immediately, then every ``interval_s`` until stopped."""

    def __init__(self, collector: Collector, interval_s: float) -> None:
        super().__init__(name="dcgm-collection-loop", daemon=True)
        self._collector = collector
        self._interval_s = interval_s
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._collector.collect_once()
            except Exception:
                logger.exception("collection cycle failed")
            self._stop_event.wait(self._interval_s)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self.join(timeout)

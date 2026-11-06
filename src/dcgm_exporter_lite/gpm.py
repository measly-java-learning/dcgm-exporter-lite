"""Two-sample GPM sampler, one baseline per GPU."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence

from .metrics import GPM_IDS
from .nvml import GpuDevice, Unsupported


class GpmSampler:
    """Keep the previous GPM sample per GPU and compute rates against it."""

    def __init__(self, devices: Sequence[GpuDevice]) -> None:
        self._devices = list(devices)
        self._supported: dict[int, bool] = {}
        self._previous: dict[int, object] = {}

    def step(self, index: int) -> Mapping[str, float] | None:
        """Take a sample for ``index``.

        Returns ``None`` until a baseline exists, or whenever GPM is
        unavailable for the device.
        """
        if not self._is_supported(index):
            return None

        device = self._devices[index]
        previous = self._previous.get(index)

        try:
            current = device.gpm_sample()
        except Unsupported:
            self._supported[index] = False
            self._discard(index, previous)
            return None
        except Exception:
            self._discard(index, previous)
            raise

        if previous is None:
            self._previous[index] = current
            return None

        metric_ids = tuple(GPM_IDS.values())
        try:
            raw = device.gpm_metrics(previous, current, metric_ids)
        except Exception:
            self._discard(index, previous)
            self._free(index, current)
            raise

        self._previous[index] = current
        self._free(index, previous)
        return {
            name: float(raw[metric_id]) for name, metric_id in GPM_IDS.items() if metric_id in raw
        }

    def _is_supported(self, index: int) -> bool:
        if index not in self._supported:
            try:
                supported = self._devices[index].gpm_supported()
            except Unsupported:
                supported = False
            self._supported[index] = supported
        return self._supported[index]

    def _discard(self, index: int, sample: object | None) -> None:
        self._previous.pop(index, None)
        self._free(index, sample)

    def _free(self, index: int, sample: object | None) -> None:
        if sample is None:
            return
        # Freeing is best-effort; never mask the original failure.
        with contextlib.suppress(Exception):
            self._devices[index].gpm_free(sample)

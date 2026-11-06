from __future__ import annotations

from collections.abc import Sequence

import pytest
from fakes import FakeEventSource, canonical_pair

from dcgm_exporter_lite.nvml import GpuDevice


@pytest.fixture
def devices() -> Sequence[GpuDevice]:
    return canonical_pair()


@pytest.fixture
def event_source() -> FakeEventSource:
    return FakeEventSource([(0, 13), (1, 79), (0, 31)])

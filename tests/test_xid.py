from __future__ import annotations

import logging
import time

from fakes import FakeEventSource

from dcgm_exporter_lite.nvml import ReadError
from dcgm_exporter_lite.xid import XidWatcher


def _wait_for(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_records_last_code_per_gpu() -> None:
    watcher = XidWatcher(FakeEventSource([(0, 13), (1, 79), (0, 31)]))
    watcher.start()
    try:
        assert _wait_for(lambda: watcher.last(0) == 31 and watcher.last(1) == 79)
    finally:
        watcher.stop()


def test_none_before_event() -> None:
    watcher = XidWatcher(FakeEventSource([]))

    assert watcher.last(0) is None
    assert watcher.available is True


def test_stop_returns_promptly() -> None:
    watcher = XidWatcher(FakeEventSource([]))
    watcher.start()

    started = time.monotonic()
    watcher.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 1.5
    assert not watcher.is_alive()


def test_source_error_logged_and_thread_survives(caplog) -> None:
    watcher = XidWatcher(FakeEventSource([ReadError("boom"), (0, 42)]))
    with caplog.at_level(logging.WARNING, logger="dcgm_exporter_lite.xid"):
        watcher.start()
        try:
            assert _wait_for(lambda: watcher.last(0) == 42)
        finally:
            watcher.stop()

    assert "XID event source failed" in caplog.text

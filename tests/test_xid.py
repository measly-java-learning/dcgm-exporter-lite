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
    watcher = XidWatcher(FakeEventSource([ReadError("boom"), (0, 42)]), error_backoff_s=0.01)
    with caplog.at_level(logging.WARNING, logger="dcgm_exporter_lite.xid"):
        watcher.start()
        try:
            assert _wait_for(lambda: watcher.last(0) == 42)
        finally:
            watcher.stop()

    assert "XID event source failed" in caplog.text


class _AlwaysFails:
    available = True

    def __init__(self) -> None:
        self.waits = 0

    def wait(self, timeout_ms: int):
        self.waits += 1
        raise ReadError("gpu fell off the bus")

    def close(self) -> None:
        pass


def test_persistent_source_error_backs_off(caplog) -> None:
    source = _AlwaysFails()
    watcher = XidWatcher(source, error_backoff_s=0.2)
    with caplog.at_level(logging.WARNING, logger="dcgm_exporter_lite.xid"):
        watcher.start()
        time.sleep(0.5)
        watcher.stop()

    assert source.waits <= 4
    assert caplog.text.count("XID event source failed") == 1


def test_watches_only_registered_gpus() -> None:
    assert XidWatcher(FakeEventSource([], registered={0})).watches(0) is True
    assert XidWatcher(FakeEventSource([], registered={0})).watches(1) is False
    assert XidWatcher(FakeEventSource([])).watches(5) is True

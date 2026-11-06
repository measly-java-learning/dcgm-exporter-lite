from __future__ import annotations

import logging
import socket
import threading
import time
import urllib.request

import pytest
from fakes import canonical_pair

from dcgm_exporter_lite.cli import Config, parse_config, run


class FakeNvml:
    NvmlXidEventSource = None

    def __init__(self, devices, driver="580.65.06", init_error=None):
        self.devices = devices
        self.driver = driver
        self.init_error = init_error
        self.shutdown_calls = 0

    def init(self):
        if self.init_error is not None:
            raise self.init_error
        return self.driver

    def open_devices(self):
        return self.devices

    def shutdown(self):
        self.shutdown_calls += 1


class FakeXidSource:
    def __init__(self, devices):
        self.available = False

    def wait(self, timeout_ms):
        return None

    def close(self):
        pass


FakeNvml.NvmlXidEventSource = FakeXidSource


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for_metrics(url: str, needle: bytes, timeout: float = 3.0) -> bytes:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                body = response.read()
            if needle in body:
                return body
        except Exception as exc:  # server may not be listening yet
            last_error = exc
        time.sleep(0.05)
    raise AssertionError(f"metrics did not contain {needle!r}: {last_error}")


def test_defaults() -> None:
    assert parse_config([], {}) == Config(":9400", None, 30000, None, False, False, False, "text")


def test_short_aliases() -> None:
    config = parse_config(["-a", ":9500", "-f", "x.csv", "-c", "1000"], {})

    assert config.address == ":9500"
    assert config.collectors == "x.csv"
    assert config.interval_ms == 1000


def test_env_vars() -> None:
    env = {
        "DCGM_EXPORTER_LISTEN": ":9600",
        "DCGM_EXPORTER_INTERVAL": "5000",
        "DCGM_EXPORTER_COLLECTORS": "/tmp/counters.csv",
        "DCGM_EXPORTER_DEVICES_STR": "g:1",
        "DCGM_EXPORTER_NO_HOSTNAME": "true",
        "DCGM_EXPORTER_REPLACE_BLANKS_IN_MODEL_NAME": "1",
        "DCGM_EXPORTER_DEBUG": "yes",
        "DCGM_EXPORTER_LOG_FORMAT": "json",
    }

    config = parse_config([], env)
    assert config == Config(":9600", "/tmp/counters.csv", 5000, [1], True, True, True, "json")

    overridden = parse_config(["-a", ":9700", "-c", "1000"], env)
    assert overridden.address == ":9700"
    assert overridden.interval_ms == 1000


def test_devices_syntax() -> None:
    assert parse_config(["-d", "f"], {}).devices is None
    assert parse_config(["-d", "g:0,2"], {}).devices == [0, 2]
    assert parse_config(["-d", "g:0,0"], {}).devices == [0]
    with pytest.raises(SystemExit):
        parse_config(["-d", "i:0"], {})


def test_unsupported_dcgm_flag_rejected(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_config(["--kubernetes"], {})

    assert excinfo.value.code == 2
    assert "not supported by dcgm-exporter-lite" in capsys.readouterr().err


def test_run_end_to_end_with_fakes() -> None:
    port = _free_port()
    fake = FakeNvml(canonical_pair())
    config = Config(
        address=f"127.0.0.1:{port}",
        collectors=None,
        interval_ms=1000,
        devices=None,
        no_hostname=False,
        replace_blanks=False,
        debug=False,
        log_format="text",
    )
    stop = threading.Event()
    result: list[int] = []

    thread = threading.Thread(
        target=lambda: result.append(run(config, nvml_module=fake, stop_event=stop)),
        daemon=True,
    )
    thread.start()
    try:
        body = _wait_for_metrics(
            f"http://127.0.0.1:{port}/metrics", b"DCGM_FI_DEV_GPU_TEMP_CELSIUS"
        )
        assert b'gpu="0"' in body
    finally:
        stop.set()
        thread.join(5)

    assert result == [0]
    assert fake.shutdown_calls == 1


def test_run_nvml_init_failure_returns_1(capsys) -> None:
    fake = FakeNvml([], init_error=RuntimeError("no driver"))

    assert (
        run(Config("127.0.0.1:0", None, 1000, None, False, False, False, "text"), nvml_module=fake)
        == 1
    )
    assert "cannot initialize NVML" in capsys.readouterr().err


def test_run_old_driver_warns(caplog) -> None:
    fake = FakeNvml(canonical_pair(), driver="575.10")
    stop = threading.Event()
    stop.set()

    with caplog.at_level(logging.WARNING, logger="dcgm_exporter_lite"):
        assert (
            run(
                Config("127.0.0.1:0", None, 1000, None, False, False, False, "text"),
                nvml_module=fake,
                stop_event=stop,
            )
            == 0
        )

    assert "580" in caplog.text


def test_run_port_in_use_returns_1_no_traceback(capsys) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    try:
        result = run(
            Config(f"127.0.0.1:{port}", None, 1000, None, False, False, False, "text"),
            nvml_module=FakeNvml(canonical_pair()),
        )
    finally:
        listener.close()

    captured = capsys.readouterr()
    assert result == 1
    assert "cannot listen" in captured.err
    assert "Traceback" not in captured.err


def test_bad_interval_env_is_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_config([], {"DCGM_EXPORTER_INTERVAL": "abc"})

    assert exc_info.value.code == 2
    assert "invalid int value" in capsys.readouterr().err

"""Real-hardware smoketest. Run with ``pytest -m hardware`` on a GPU host."""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.hardware

EXPECTED = {
    "DCGM_FI_DEV_SM_CLOCK_HERTZ",
    "DCGM_FI_DEV_MEMORY_CLOCK_HERTZ",
    "DCGM_FI_DEV_GPU_TEMP_CELSIUS",
    "DCGM_FI_DEV_BOARD_POWER_WATTS",
    "DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL",
    "DCGM_FI_DEV_PCIE_REPLAY_TOTAL",
    "DCGM_FI_DEV_GPU_UTIL_RATIO",
    "DCGM_FI_DEV_MEMORY_UTIL_RATIO",
    "DCGM_FI_DEV_ENC_UTIL_RATIO",
    "DCGM_FI_DEV_DEC_UTIL_RATIO",
    "DCGM_FI_DEV_XID_ERROR",
    "DCGM_FI_DEV_FB_CAPACITY_BYTES",
    "DCGM_FI_DEV_FB_FREE_BYTES",
    "DCGM_FI_DEV_FB_USED_BYTES",
    "DCGM_FI_DEV_FB_RESERVED_BYTES",
    "DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO",
    "DCGM_FI_PROF_TENSOR_UTIL_RATIO",
    "DCGM_FI_PROF_DRAM_UTIL_RATIO",
    "DCGM_FI_PROF_PCIE_TX_BYTES",
    "DCGM_FI_PROF_PCIE_RX_BYTES",
}

DATACENTER_ONLY = {
    "DCGM_FI_DEV_MEMORY_TEMP_CELSIUS",
    "DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND",
    "DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL",
    "DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL",
    "DCGM_FI_DEV_ROW_REMAP_FAILED",
}

_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})? ([^\s]+)$")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)(?:="((?:[^"\\]|\\.)*)")?')
_ESCAPES = {"\\\\": "\\", '\\"': '"', "\\n": "\n"}


def _unescape(value: str) -> str:
    return re.sub(r"\\\\|\\\"|\\n", lambda match: _ESCAPES[match.group(0)], value)


def _samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        name, label_text, value_text = match.groups()
        labels = {key: _unescape(val) for key, val in _LABEL.findall(label_text or "")}
        samples.append((name, labels, float(value_text)))
    return samples


def _gpu0(text: str) -> dict[str, tuple[dict[str, str], float]]:
    result = {}
    for name, labels, value in _samples(text):
        if labels.get("gpu") == "0":
            result[name] = (labels, value)
    return result


def _nvidia_smi(query: str) -> str:
    return subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "--id=0"],
        text=True,
    ).strip()


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_healthy(port: int, process: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _, err = process.communicate()
            pytest.fail(f"exporter exited before becoming healthy:\n{err}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.1)
    process.terminate()
    _, err = process.communicate(timeout=5)
    pytest.fail(f"exporter did not become healthy within {timeout}s:\n{err}")


def _scrape(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        return response.read().decode("utf-8")


@pytest.fixture(scope="module")
def exporter():
    port = _free_port()
    executable = Path(sys.executable).with_name("dcgm-exporter-lite")
    if not executable.exists():
        pytest.fail(f"installed entry point not found at {executable}")

    process = subprocess.Popen(
        [str(executable), "-a", f"127.0.0.1:{port}", "-c", "1000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_healthy(port, process)
        time.sleep(2.5)
        text = _scrape(port)
    except Exception:
        if process.poll() is None:
            process.terminate()
        _, err = process.communicate(timeout=5)
        print(err, file=sys.stderr)
        raise

    yield process, text

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def test_expected_metrics_present(exporter) -> None:
    _, text = exporter
    names = set(_gpu0(text))

    assert names >= EXPECTED, f"missing: {sorted(EXPECTED - names)}"
    labels = _gpu0(text)["DCGM_FI_DEV_GPU_TEMP_CELSIUS"][0]
    assert labels["DCGM_FI_SYSTEM_DRIVER_VERSION"]


def test_plausible_values(exporter) -> None:
    _, text = exporter
    values = _gpu0(text)

    assert 1e8 < values["DCGM_FI_DEV_SM_CLOCK_HERTZ"][1] < 4e9
    for name, (_, value) in values.items():
        if name.endswith("_RATIO"):
            assert 0.0 <= value <= 1.0, f"{name}={value}"

    capacity = values["DCGM_FI_DEV_FB_CAPACITY_BYTES"][1]
    total_mib = float(_nvidia_smi("memory.total"))
    assert abs(capacity - total_mib * 2**20) <= total_mib * 2**20 * 0.02

    assert 10 < values["DCGM_FI_DEV_GPU_TEMP_CELSIUS"][1] < 100
    assert values["DCGM_FI_DEV_BOARD_POWER_WATTS"][1] > 0


def test_datacenter_only_absent_on_geforce(exporter) -> None:
    _, text = exporter
    values = _gpu0(text)
    model = next(iter(values.values()))[0].get("modelName", "")
    if "GeForce" not in model:
        pytest.skip(f"not a GeForce card: {model}")

    present = DATACENTER_ONLY & set(values)
    assert not present, f"datacenter-only metrics appeared on {model}: {sorted(present)}"


def test_labels(exporter) -> None:
    _, text = exporter
    labels = _gpu0(text)["DCGM_FI_DEV_GPU_TEMP_CELSIUS"][0]
    info = _nvidia_smi("uuid,driver_version,pci.bus_id")
    uuid, driver, pci = (part.strip() for part in info.split(","))

    assert labels["gpu"] == "0"
    assert labels["UUID"] == uuid
    assert labels["device"] == "nvidia0"
    assert labels["pci_bus_id"].upper() == pci.upper()
    assert labels["DCGM_FI_SYSTEM_DRIVER_VERSION"] == driver
    assert labels["hostname"] == socket.gethostname()


def test_runs_as_non_root() -> None:
    assert os.geteuid() != 0


def test_sigterm_exits_zero(exporter) -> None:
    process, _ = exporter

    process.send_signal(signal.SIGTERM)

    assert process.wait(timeout=10) == 0

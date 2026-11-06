from __future__ import annotations

import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from dcgm_exporter_lite.collector import Sample, Snapshot
from dcgm_exporter_lite.counters import CounterSpec
from dcgm_exporter_lite.exposition import make_registry, render
from dcgm_exporter_lite.server import make_server, parse_address

COUNTERS = [CounterSpec("DCGM_FI_DEV_GPU_TEMP_CELSIUS", "gauge", "GPU temperature (in C).")]


def _registry():
    snapshot = Snapshot(
        (Sample("DCGM_FI_DEV_GPU_TEMP_CELSIUS", (("gpu", "0"),), 41.0),),
        1,
        time.time(),
    )
    registry = make_registry(lambda: snapshot, COUNTERS)
    assert render(registry)
    return registry


@contextmanager
def _running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status, response.headers["Content-Type"], response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers["Content-Type"], exc.read()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (":9400", ("", 9400)),
        ("0.0.0.0:9400", ("0.0.0.0", 9400)),
        ("[::]:9400", ("::", 9400)),
        ("localhost:9400", ("localhost", 9400)),
    ],
)
def test_parse_address(text: str, expected: tuple[str, int]) -> None:
    assert parse_address(text) == expected


@pytest.mark.parametrize("text", ["9400", "host:", "host:99999", "host:abc", "::1:9400"])
def test_parse_address_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_address(text)


def test_ipv6_server_binds() -> None:
    try:
        server = make_server("[::1]:0", _registry(), lambda: True)
    except OSError:
        pytest.skip("IPv6 is unavailable")

    with _running(server) as port:
        status, _, body = _get(f"http://[::1]:{port}/health")
    assert status == 200
    assert body == b"OK"


def test_port_in_use_raises_oserror() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    try:
        with pytest.raises(OSError):
            make_server(f"127.0.0.1:{port}", _registry(), lambda: True)
    finally:
        listener.close()


def test_metrics_endpoint() -> None:
    registry = _registry()
    server = make_server("127.0.0.1:0", registry, lambda: True)

    with _running(server) as port:
        status, content_type, body = _get(f"http://127.0.0.1:{port}/metrics")

    assert status == 200
    assert content_type == CONTENT_TYPE_LATEST
    assert body == render(registry)


def test_health_before_first_collection() -> None:
    healthy = [False]
    server = make_server("127.0.0.1:0", _registry(), lambda: healthy[0])

    with _running(server) as port:
        status, _, body = _get(f"http://127.0.0.1:{port}/health")
        assert status == 503
        assert body == b"not ready\n"

        healthy[0] = True
        status, _, body = _get(f"http://127.0.0.1:{port}/health")
        assert status == 200
        assert body == b"OK"


def test_root_and_404() -> None:
    server = make_server("127.0.0.1:0", _registry(), lambda: True)

    with _running(server) as port:
        status, _, body = _get(f"http://127.0.0.1:{port}/")
        assert status == 200
        assert b"GPU Exporter" in body
        assert b"./metrics" in body

        status, _, _ = _get(f"http://127.0.0.1:{port}/nope")
        assert status == 404


def test_paths_ignore_query_string() -> None:
    server = make_server("127.0.0.1:0", _registry(), lambda: True)

    with _running(server) as port:
        status, _, _ = _get(f"http://127.0.0.1:{port}/metrics?collect[]=gpu")
        assert status == 200
        status, _, _ = _get(f"http://127.0.0.1:{port}/health?x=1")
        assert status == 200

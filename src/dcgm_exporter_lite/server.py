"""Minimal threaded HTTP server for /metrics, /health, and /."""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CollectorRegistry
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from .exposition import render

logger = logging.getLogger(__name__)

_INDEX_PAGE = b"""<html>
<head><title>GPU Exporter</title></head>
<body>
<h1>GPU Exporter</h1>
<p><a href="./metrics">Metrics</a></p>
<p><a href="./health">Health</a></p>
</body>
</html>
"""


def parse_address(addr: str) -> tuple[str, int]:
    """Parse ``host:port``, ``:port``, or ``[ipv6]:port``."""
    if not addr:
        raise ValueError("listen address must be HOST:PORT")

    if addr.startswith("["):
        end = addr.find("]")
        if end < 0:
            raise ValueError(f"invalid listen address {addr!r}: missing ']'")
        host = addr[1:end]
        rest = addr[end + 1 :]
        if not host:
            raise ValueError(f"invalid listen address {addr!r}: empty IPv6 host")
        if not rest.startswith(":"):
            raise ValueError(f"invalid listen address {addr!r}: expected [HOST]:PORT")
        port_text = rest[1:]
    else:
        if addr.count(":") != 1:
            if addr.count(":") > 1:
                raise ValueError(f"invalid listen address {addr!r}: IPv6 hosts must be bracketed")
            raise ValueError(f"invalid listen address {addr!r}: expected HOST:PORT")
        host, port_text = addr.rsplit(":", 1)

    if not port_text:
        raise ValueError(f"invalid listen address {addr!r}: missing port")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"invalid listen address {addr!r}: port is not a number") from exc
    if not 0 <= port <= 65535:
        raise ValueError(f"invalid listen address {addr!r}: port out of range")

    return host, port


def make_server(
    addr: str,
    registry: CollectorRegistry,
    healthy: Callable[[], bool],
) -> ThreadingHTTPServer:
    """Bind a threaded HTTP server; ``OSError`` means the bind failed."""
    host, port = parse_address(addr)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if self.path == "/metrics":
                body = render(registry)
                self._respond(200, CONTENT_TYPE_LATEST, body)
            elif self.path == "/health":
                if healthy():
                    self._respond(200, "text/plain; charset=utf-8", b"OK")
                else:
                    self._respond(503, "text/plain; charset=utf-8", b"not ready\n")
            elif self.path == "/":
                self._respond(200, "text/html; charset=utf-8", _INDEX_PAGE)
            else:
                self._respond(404, "text/plain; charset=utf-8", b"not found\n")

        def _respond(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            logger.debug("%s - %s", self.address_string(), format % args)

    family = socket.AF_INET6 if ":" in host else socket.AF_INET

    class Server(ThreadingHTTPServer):
        address_family = family
        daemon_threads = True
        allow_reuse_address = True

    return Server((host, port), Handler)

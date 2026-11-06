"""Command-line interface, logging, signals, and process wiring."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from . import nvml
from .collector import CollectionLoop, Collector
from .counters import CountersError, load_counters
from .exposition import make_registry
from .gpm import GpmSampler
from .server import make_server
from .xid import XidWatcher

logger = logging.getLogger(__name__)


def _package_version() -> str:
    try:
        return version("dcgm-exporter-lite")
    except PackageNotFoundError:
        return "0.1.0"


SupportedLogFormat = Literal["text", "json"]


@dataclass(frozen=True)
class Config:
    address: str
    collectors: str | None
    interval_ms: int
    devices: list[int] | None
    no_hostname: bool
    replace_blanks: bool
    debug: bool
    log_format: SupportedLogFormat


# 4.9.0's flag names that lite deliberately does not implement.
_UNSUPPORTED_FLAGS = frozenset(
    {
        "config-file",
        "watch-max-keep-age",
        "watch-max-keep-samples",
        "kubernetes",
        "kubernetes-enable-pod-labels",
        "kubernetes-enable-pod-uid",
        "kubernetes-gpu-id-type",
        "kubernetes-pod-label-allowlist-regex",
        "use-old-namespace",
        "remote-hostengine-info",
        "switch-devices",
        "cpu-devices",
        "fake-gpus",
        "configmap-data",
        "web-systemd-socket",
        "web-config-file",
        "web-read-timeout",
        "web-write-timeout",
        "max-concurrent-scrapes",
        "enable-exporter-metrics",
        "xid-count-window-size",
        "clock-events-count-window-size",
        "enable-dcgm-log",
        "dcgm-log-level",
        "dcgm-cleanup-timeout",
        "pod-resources-kubelet-socket",
        "hpc-job-mapping-dir",
        "hpc-job-mapping-max-file-bytes",
        "hpc-job-mapping-max-job-id-bytes",
        "hpc-job-mapping-max-jobs-per-file",
        "container-labels",
        "container-runtime-socket",
        "nvidia-resource-names",
        "kubernetes-virtual-gpus",
        "dump-enabled",
        "dump-directory",
        "dump-retention",
        "dump-compression",
        "dump-min-interval",
        "dump-max-file-bytes",
        "dump-max-total-bytes",
        "dump-max-files",
        "kubernetes-enable-dra",
        "disable-startup-validate",
        "enable-gpu-bind-unbind-watch",
        "gpu-bind-unbind-poll-interval",
        "enable-pprof",
    }
)

_UNSUPPORTED_SHORT_FLAGS = {
    "-k": "kubernetes",
    "-o": "use-old-namespace",
    "-p": "cpu-devices",
    "-r": "remote-hostengine-info",
    "-s": "switch-devices",
    "-m": "configmap-data",
    "-x": "xid-count-window-size",
}


def _reject_unsupported_flags(argv: Sequence[str]) -> None:
    for arg in argv:
        if arg == "--":
            break
        if arg.startswith("--"):
            name = arg[2:].split("=", 1)[0]
            if name in _UNSUPPORTED_FLAGS:
                _unsupported(name)
        elif arg in _UNSUPPORTED_SHORT_FLAGS:
            _unsupported(_UNSUPPORTED_SHORT_FLAGS[arg])


def _unsupported(name: str) -> None:
    print(
        f"--{name} is not supported by dcgm-exporter-lite "
        "(see README: Differences from DCGM Exporter)",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _env_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_devices(value: str | None) -> list[int] | None:
    if value is None or value == "f":
        return None
    if value.startswith("i:"):
        raise ValueError("MIG device selectors (i:) are not supported")
    if not value.startswith("g:"):
        raise ValueError("expected 'f' or 'g:<index>[,<index>...]'")
    items = value[2:].split(",")
    if not items or any(not item.strip() for item in items):
        raise ValueError("expected at least one GPU index after 'g:'")
    indices: list[int] = []
    for item in items:
        index = int(item)
        if index < 0:
            raise ValueError("GPU indices must be non-negative")
        if index not in indices:
            indices.append(index)
    return indices


def parse_config(argv: Sequence[str], env: Mapping[str, str]) -> Config:
    """Parse flags, then env vars, then defaults."""
    _reject_unsupported_flags(argv)

    parser = argparse.ArgumentParser(prog="dcgm-exporter-lite")
    parser.add_argument(
        "-a",
        "--address",
        default=env.get("DCGM_EXPORTER_LISTEN", ":9400"),
        help="listen address (default: :9400)",
    )
    parser.add_argument(
        "-f",
        "--collectors",
        default=env.get("DCGM_EXPORTER_COLLECTORS") or None,
        help="counters CSV path (default: packaged 4.9.0 counters)",
    )
    parser.add_argument(
        "-c",
        "--collect-interval",
        type=int,
        # A string default is converted by ``type``, so a bad env value is a
        # usage error rather than a traceback.
        default=env.get("DCGM_EXPORTER_INTERVAL", "30000"),
        help="collection interval in milliseconds (default: 30000)",
    )
    parser.add_argument(
        "-d",
        "--devices",
        default=env.get("DCGM_EXPORTER_DEVICES_STR") or None,
        help="GPU selector: 'f' or 'g:0,2'",
    )
    parser.add_argument(
        "-n",
        "--no-hostname",
        action="store_true",
        default=_env_bool(env.get("DCGM_EXPORTER_NO_HOSTNAME")),
        help="omit the hostname label",
    )
    parser.add_argument(
        "-rbmn",
        "--replace-blanks-in-model-name",
        action="store_true",
        default=_env_bool(env.get("DCGM_EXPORTER_REPLACE_BLANKS_IN_MODEL_NAME")),
        help="replace spaces in modelName with hyphens",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=_env_bool(env.get("DCGM_EXPORTER_DEBUG")),
        help="enable debug logging",
    )
    parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=env.get("DCGM_EXPORTER_LOG_FORMAT", "text"),
        help="log output format",
    )
    parser.add_argument("--version", action="version", version=_package_version())

    args = parser.parse_args(list(argv))

    if args.collect_interval <= 0:
        parser.error("--collect-interval must be greater than zero")
    if args.log_format not in ("text", "json"):
        parser.error("--log-format must be text or json")
    try:
        devices = _parse_devices(args.devices)
    except ValueError as exc:
        parser.error(f"--devices: {exc}")

    return Config(
        address=args.address,
        collectors=args.collectors,
        interval_ms=args.collect_interval,
        devices=devices,
        no_hostname=args.no_hostname,
        replace_blanks=args.replace_blanks_in_model_name,
        debug=args.debug,
        log_format=args.log_format,
    )


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging(config: Config) -> None:
    package_logger = logging.getLogger("dcgm_exporter_lite")
    package_logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    package_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    if config.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    package_logger.addHandler(handler)
    package_logger.propagate = True


def _install_stop_handlers(stop_event: threading.Event) -> dict[int, object]:
    if threading.current_thread() is not threading.main_thread():
        return {}

    previous: dict[int, object] = {}

    def request_stop(signum, frame) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        except (OSError, ValueError):
            continue
    return previous


def _restore_stop_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):
            continue


def run(
    config: Config,
    *,
    nvml_module=nvml,
    stop_event: threading.Event | None = None,
) -> int:
    """Wire the exporter together and serve until stopped."""
    _configure_logging(config)

    try:
        driver_version = nvml_module.init()
    except Exception as exc:
        print(f"error: cannot initialize NVML: {exc}", file=sys.stderr)
        return 1

    if nvml.driver_major(driver_version) < 580:
        logger.warning(
            "NVIDIA driver %s is older than 580; some metrics may be missing",
            driver_version,
        )

    try:
        devices = list(nvml_module.open_devices())
    except Exception as exc:
        print(f"error: cannot enumerate GPUs: {exc}", file=sys.stderr)
        nvml_module.shutdown()
        return 1

    if config.devices is not None:
        wanted = set(config.devices)
        devices = [device for device in devices if device.identity.index in wanted]

    if not devices:
        print("error: no GPUs match --devices", file=sys.stderr)
        nvml_module.shutdown()
        return 1

    try:
        counters = load_counters(config.collectors)
    except CountersError as exc:
        print(f"error: {exc}", file=sys.stderr)
        nvml_module.shutdown()
        return 1

    stop_event = stop_event or threading.Event()
    previous_handlers = _install_stop_handlers(stop_event)
    interval_s = config.interval_ms / 1000.0
    hostname = None if config.no_hostname else socket.gethostname()

    xid_source = None
    xid_watcher = None
    if hasattr(nvml_module, "NvmlXidEventSource"):
        try:
            xid_source = nvml_module.NvmlXidEventSource(devices)
            xid_watcher = XidWatcher(xid_source)
        except Exception as exc:
            logger.warning("cannot register XID events: %s", exc)

    collector = Collector(
        devices,
        counters,
        driver_version=driver_version,
        hostname=hostname,
        replace_blanks=config.replace_blanks,
        gpm=GpmSampler(devices),
        xid=xid_watcher,
        interval_s=interval_s,
    )

    try:
        collector.probe()
    except Exception as exc:
        print(f"error: GPU support probe failed: {exc}", file=sys.stderr)
        nvml_module.shutdown()
        _restore_stop_handlers(previous_handlers)
        return 1

    if xid_watcher is not None:
        xid_watcher.start()

    loop = CollectionLoop(collector, interval_s)
    loop.start()

    registry = make_registry(lambda: collector.latest, counters)

    # Unhealthy if no fresh snapshot within three intervals (at least 10 s).
    max_age_s = max(3 * interval_s, 10.0)

    def healthy() -> bool:
        return collector.healthy(max_age_s)

    try:
        server = make_server(config.address, registry, healthy)
    except OSError as exc:
        print(f"error: cannot listen on {config.address}: {exc}", file=sys.stderr)
        loop.stop()
        if xid_watcher is not None:
            xid_watcher.stop()
        nvml_module.shutdown()
        _restore_stop_handlers(previous_handlers)
        return 1

    serving = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.2},
        name="dcgm-http-server",
        daemon=True,
    )
    serving.start()
    logger.info("listening on %s", config.address)

    try:
        stop_event.wait()
    finally:
        server.shutdown()
        serving.join(5)
        loop.stop()
        if xid_watcher is not None:
            xid_watcher.stop()
        server.server_close()
        nvml_module.shutdown()
        _restore_stop_handlers(previous_handlers)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(sys.argv[1:] if argv is None else argv, os.environ)
    return run(config)

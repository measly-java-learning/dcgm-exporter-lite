"""dcgm-exporter-lite: unprivileged NVML metrics in DCGM Exporter 4.9.0 form."""

from __future__ import annotations


def main() -> int:
    from .cli import main as cli_main

    return cli_main()

"""Parse DCGM Exporter counter CSV files."""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CounterType = Literal["gauge", "counter", "label"]
_VALID_TYPES: frozenset[str] = frozenset(("gauge", "counter", "label"))
_DEFAULT_RESOURCE = ("data", "default-counters.csv")


class CountersError(ValueError):
    """A counter CSV file is malformed or unreadable."""


@dataclass(frozen=True)
class CounterSpec:
    """One enabled row of a DCGM counter CSV."""

    name: str
    type: CounterType
    help: str


def parse_counters(text: str, source: str) -> list[CounterSpec]:
    """Parse DCGM's ``name, type, help`` CSV format.

    Blank lines and lines whose first non-space character is ``#`` are ignored.
    A UTF-8 BOM and CRLF line endings are accepted.
    """
    counters: list[CounterSpec] = []
    seen: set[str] = set()

    for lineno, raw_line in enumerate(text.lstrip("\ufeff").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split(",", 2)
        if len(fields) != 3:
            raise CountersError(
                f"{source}:{lineno}: malformed row {line!r}: "
                f"expected 3 comma-separated fields, got {len(fields)}"
            )

        name, raw_type, help_text = (field.strip() for field in fields)
        if not name:
            raise CountersError(f"{source}:{lineno}: metric name is empty")
        if raw_type.lower() not in _VALID_TYPES:
            raise CountersError(
                f"{source}:{lineno}: unsupported metric type {raw_type!r} for {name!r}"
            )
        if name in seen:
            raise CountersError(f"{source}:{lineno}: duplicate metric name {name!r}")

        seen.add(name)
        counters.append(CounterSpec(name, raw_type.lower(), help_text))  # type: ignore[arg-type]

    return counters


def load_counters(path: str | None) -> list[CounterSpec]:
    """Load counters from ``path``, or the packaged 4.9.0 defaults when ``None``."""
    if path is None:
        resource = importlib.resources.files("dcgm_exporter_lite").joinpath(*_DEFAULT_RESOURCE)
        source = "default-counters.csv"
        try:
            text = resource.read_text(encoding="utf-8")
        except OSError as exc:
            raise CountersError(f"{source}: {exc}") from exc
    else:
        source = path
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise CountersError(f"{path}: {exc}") from exc

    return parse_counters(text, source)

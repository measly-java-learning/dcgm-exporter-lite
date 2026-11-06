from __future__ import annotations

import json
import re
from pathlib import Path

from dcgm_exporter_lite.counters import load_counters

DASHBOARD = Path(__file__).parent.parent / "grafana" / "dcgm-exporter-lite-dashboard.json"


def _dashboard() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _panels(dashboard: dict) -> list[dict]:
    result = []
    for panel in dashboard["panels"]:
        result.append(panel)
        result.extend(panel.get("panels", []))
    return result


def test_dashboard_is_valid_json_with_variables() -> None:
    dashboard = _dashboard()
    variables = {item["name"]: item for item in dashboard["templating"]["list"]}

    assert {"datasource", "instance", "gpu"} <= set(variables)
    assert variables["gpu"]["query"] == "label_values(DCGM_FI_DEV_GPU_TEMP_CELSIUS, gpu)"
    assert variables["instance"]["query"] == (
        "label_values(DCGM_FI_DEV_GPU_TEMP_CELSIUS, instance)"
    )


def test_only_dcgm_490_metrics() -> None:
    known = {counter.name for counter in load_counters(None)}
    used: set[str] = set()

    for panel in _panels(_dashboard()):
        for target in panel.get("targets", []):
            used.update(re.findall(r"\bDCGM_[A-Z0-9_]+\b", target["expr"]))

    assert used
    assert used <= known, f"unknown DCGM names: {sorted(used - known)}"


def test_every_panel_filters_instance_and_gpu() -> None:
    for panel in _panels(_dashboard()):
        if panel["type"] == "row":
            continue
        for target in panel["targets"]:
            assert 'instance=~"$instance"' in target["expr"], panel["title"]
            assert 'gpu=~"$gpu"' in target["expr"], panel["title"]


def test_rows_present() -> None:
    titles = [panel["title"] for panel in _panels(_dashboard()) if panel["type"] == "row"]

    assert titles == [
        "Fleet overview",
        "Utilization",
        "Memory",
        "Thermals & power",
        "Health",
        "I/O",
    ]


def test_required_operational_queries() -> None:
    exprs = {
        target["expr"] for panel in _panels(_dashboard()) for target in panel.get("targets", [])
    }

    assert any(
        "increase(DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL" in expr and "/ 3.6e6" in expr
        for expr in exprs
    )
    assert any(
        "max_over_time(DCGM_FI_DEV_GPU_UTIL_RATIO" in expr and "< 0.05" in expr for expr in exprs
    )
    assert any(
        "1 - DCGM_FI_DEV_FB_USED_BYTES" in expr and "DCGM_FI_DEV_FB_CAPACITY_BYTES" in expr
        for expr in exprs
    )
    assert any("DCGM_FI_DEV_XID_ERROR" in expr and "> 0" in expr for expr in exprs)


def test_idle_gpu_count_shows_zero_instead_of_no_data() -> None:
    exprs = [
        target.get("expr", "")
        for panel in _panels(_dashboard())
        for target in panel.get("targets", [])
    ]
    idle = [expr for expr in exprs if "max_over_time(DCGM_FI_DEV_GPU_UTIL_RATIO" in expr]

    assert idle
    assert all(expr.rstrip().endswith("or vector(0)") for expr in idle)

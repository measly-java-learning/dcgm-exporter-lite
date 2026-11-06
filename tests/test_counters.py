from __future__ import annotations

import pytest

from dcgm_exporter_lite.counters import CountersError, CounterSpec, load_counters, parse_counters


def test_default_has_27_rows() -> None:
    counters = load_counters(None)

    assert len(counters) == 27
    assert counters[0] == CounterSpec(
        "DCGM_FI_DEV_SM_CLOCK_HERTZ", "gauge", "SM clock frequency in hertz."
    )
    driver = next(c for c in counters if c.name == "DCGM_FI_SYSTEM_DRIVER_VERSION")
    assert driver.type == "label"


def test_comments_blank_lines_and_whitespace_ignored() -> None:
    text = "# x\n\n  A_B ,  gauge , help text \n"

    assert parse_counters(text, "test.csv") == [CounterSpec("A_B", "gauge", "help text")]


def test_help_may_contain_commas() -> None:
    text = "A_B,gauge,one, two, three\n"

    assert parse_counters(text, "test.csv") == [CounterSpec("A_B", "gauge", "one, two, three")]


def test_crlf_and_bom() -> None:
    text = "\ufeffA, gauge, h\r\nB, counter, h2\r\n"

    assert parse_counters(text, "test.csv") == [
        CounterSpec("A", "gauge", "h"),
        CounterSpec("B", "counter", "h2"),
    ]


@pytest.mark.parametrize("line", ["A, gauge", "A, histogram, h"])
def test_malformed_row_raises(line: str) -> None:
    with pytest.raises(CountersError) as excinfo:
        parse_counters(line + "\n", "test.csv")

    message = str(excinfo.value)
    assert "test.csv:1" in message
    assert "A" in message


def test_duplicate_name_raises() -> None:
    with pytest.raises(CountersError, match="duplicate"):
        parse_counters("A, gauge, h\nA, counter, h2\n", "test.csv")

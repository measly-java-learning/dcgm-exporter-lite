# dcgm-exporter-lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pip-installable, unprivileged Python exporter that serves NVML GPU metrics with DCGM
Exporter 4.9.0's exact metric names, units, and labels. It ships with a richer Grafana dashboard
and the docs needed to deploy it and later swap to the real DCGM Exporter.

**Architecture:**
- A thin NVML facade (`nvml.py`) converts `pynvml` errors into three exception types. All other
  code depends on a `GpuDevice` protocol, so tests use a pure-Python fake.
- A background collection loop probes which metrics each GPU supports, then builds immutable
  snapshots on the collect interval.
- A `prometheus_client` custom collector renders the latest snapshot. A standard-library threaded
  HTTP server serves `/metrics`, `/health`, and `/`.

**Tech Stack:**
- Python 3.12–3.14, `nvidia-ml-py==13.615.71`, `prometheus-client==0.26.0`
- Standard library: `argparse`, `http.server`, `threading`, `logging`, `importlib.resources`
- `pytest` (via `pytest-cov`), `ruff`, `uv`

**Spec:** `docs/superpowers/specs/2026-11-06-dcgm-exporter-lite-design.md`. Read it before any task.
Section references below (§n) point into it.

## Global Constraints

- Wire-compatible with **DCGM Exporter tag `4.9.0`**: metric names, types, units, label names,
  label order, and label value formats.
- **Never emit a value under a DCGM name that NVML cannot supply with the same meaning and unit.**
  Omit it instead. No zero/NaN placeholders.
- Runs as an unprivileged user. Needs no root, container, CDI, or DCGM.
- Runtime dependencies are only `nvidia-ml-py==13.615.71` and `prometheus-client==0.26.0`. Do not
  add others.
- `requires-python = ">=3.12"`. The build backend stays `uv_build`.
- Supported drivers: **580 and newer**. Older drivers log a WARNING, not a failure.
  `NVMLError_FunctionNotFound` is treated exactly like `NOT_SUPPORTED`.
- Defaults: address `:9400`, collect interval `30000` ms, collectors file = the packaged copy of
  4.9.0 `default-counters.csv`.
- Unsupported DCGM flags exit non-zero with a message. They are never silently ignored.
- Out of scope: MIG labels, Kubernetes/HPC labels, NVSwitch, DCGM health/policy, `DCGM_EXP_*`, YAML
  config, TLS, container image, PyPI publishing, 4.8.x names.
- Unit tests must pass on a machine with **no NVIDIA driver**. Hardware tests are marked `hardware`
  and run on `ryzen.local`.
- Lint: `uv run ruff check` and `uv run ruff format --check` are clean at every commit.

## Review Focus

1. **Scrape before the first collection finishes:** `/metrics` returns 200 with a valid, empty
   exposition, and `/health` returns 503. *(Task 8)*
2. **Mixed fleet:** GPU 0 supports a metric and GPU 1 returns not-supported. The family is emitted
   with only GPU 0's series, and GPU 1 gets no zero series. *(Task 6)*
3. **Counters CSV saved with CRLF line endings or a UTF-8 BOM** (edited on Windows) parses the same
   as a clean file. *(Task 1)*
4. **Listen address forms and a port already in use:**
   - `[::]:9400`, `0.0.0.0:9400`, `:9400`, and `localhost:9400` all parse.
   - A bound port exits non-zero with a one-line error, not a traceback. *(Task 8, Task 9)*
5. **A GPU is lost mid-run:** its series disappear, the others keep serving, and `/health` stays 200
   while at least one GPU succeeds. *(Task 6)*

---

## File Structure

```
src/dcgm_exporter_lite/
  __init__.py            main() -> delegates to cli.main
  __main__.py            python -m dcgm_exporter_lite
  data/default-counters.csv   vendored from 4.9.0
  counters.py            CSV parsing -> CounterSpec list
  nvml.py                pynvml facade, GpuDevice protocol, exceptions, event source
  metrics.py             MetricDef registry: DCGM name -> reader + unit conversion
  gpm.py                 GPM sampler (two-sample rates, per GPU)
  xid.py                 XID event watcher thread
  collector.py           support probing, Snapshot building, CollectionLoop thread
  exposition.py          prometheus_client Collector over a Snapshot
  server.py              address parsing, HTTP server (/metrics, /health, /)
  cli.py                 argparse + env vars, wiring, signals, logging
tests/
  conftest.py            fixtures: fake devices, fake event source
  fakes.py               FakeDevice, FakeEventSource, FakeGpm
  golden/default.prom    expected /metrics for the canonical fake GPU pair
  test_counters.py  test_nvml.py  test_metrics.py  test_gpm.py  test_xid.py
  test_collector.py test_exposition.py test_server.py test_cli.py
  hardware/test_smoke.py       @pytest.mark.hardware
scripts/smoketest-remote.sh    rsync + run hardware tests on a host
scripts/collect-datacenter-evidence.sh
contrib/systemd/dcgm-exporter-lite.service
grafana/dcgm-exporter-lite-dashboard.json
docs/notes/dcgm-4.9.0-contract.md
docs/testing/datacenter-verification.md
README.md
.github/workflows/ci.yml
```

---

### Task 1: Project scaffold, contract notes, and counters CSV

This task verifies the upstream contract first, because every later test depends on it.

**Files:**
- Create: `docs/notes/dcgm-4.9.0-contract.md`, `src/dcgm_exporter_lite/data/default-counters.csv`,
  `src/dcgm_exporter_lite/counters.py`, `src/dcgm_exporter_lite/__main__.py`, `tests/test_counters.py`,
  `.github/workflows/ci.yml`
- Modify: `src/dcgm_exporter_lite/__init__.py` (remove hello-world; `main()` calls `cli.main()` —
  add a temporary `cli.py` stub returning 0), `pyproject.toml` (description; `[tool.pytest.ini_options]`
  with `markers = ["hardware: needs a real NVIDIA GPU"]` and `addopts = "-m 'not hardware'"`;
  `[tool.ruff]` line-length 100, target py312, select `E,F,I,UP,B,SIM`)

**Interfaces:**
- Produces:
  - `CounterSpec(name: str, type: Literal["gauge","counter","label"], help: str)`, a frozen dataclass
  - `parse_counters(text: str, source: str) -> list[CounterSpec]`, which raises `CountersError(ValueError)`
  - `load_counters(path: str | None) -> list[CounterSpec]`. `None` loads the packaged default via `importlib.resources`.
  - `docs/notes/dcgm-4.9.0-contract.md`, which records the answers below. Later tasks cite it.

- [ ] **Step 1: Create the git baseline.** The default branch is already `main` (still with no commits).
  Commit the existing `uv init` files plus the spec and plan on `main`, then create the work branch
  `git checkout -b feat/exporter`.
- [ ] **Step 2: Verify the upstream contract.** Fetch tag 4.9.0 (`gh api repos/NVIDIA/dcgm-exporter/tarball/4.9.0`
  into a scratch dir). Read `tests/internal/metriccontract/`, `internal/pkg/server/`, `pkg/cmd/app.go`,
  `internal/pkg/appconfig/`, and the GPU label code. Record the following in `docs/notes/dcgm-4.9.0-contract.md`,
  each with a file:line citation:
  1. Exact label names and **order** on a GPU metric. Is `pci_bus_id` included? Is the hex upper- or lower-case?
  2. How the CSV `label` type (`DCGM_FI_SYSTEM_DRIVER_VERSION`) is rendered.
  3. `DCGM_FI_DEV_XID_ERROR`: extra labels (`err_code`/`err_msg`)? Value before any XID? Is it
     emitted at all before the first event?
  4. Counter rendering: `# HELP`/`# TYPE` names for `*_TOTAL` metrics, and whether `_created` appears.
  5. `/health` response code and body. `/` page content.
  6. Env-var names for `--collectors`, `--devices`, `--replace-blanks-in-model-name`, `--debug`,
     `--log-format`. The `--devices` syntax.
  7. Which DCGM field backs `BOARD_POWER_WATTS` (instant vs average) and `PROF_TENSOR_UTIL_RATIO`
     (any-tensor vs HMMA). Map these to `NVML_FI_DEV_POWER_INSTANT` vs `nvmlDeviceGetPowerUsage`,
     and to GPM `ANY_TENSOR_UTIL` vs `HMMA_TENSOR_UTIL`.
  8. GPM `PCIE_TX_PER_SEC`/`PCIE_RX_PER_SEC` unit, from the NVML header in the `nvidia-ml-py` wheel or the NVML docs.
  9. NVML `NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX/RX` semantics: cumulative counter (KiB?) or rate?
     Per link or total? This comes from the NVML docs. If they disagree, record it as "unverified"
     for the datacenter checklist (Task 11).
- [ ] **Step 3: Vendor the CSV.** Copy 4.9.0 `etc/default-counters.csv` byte-for-byte to
  `src/dcgm_exporter_lite/data/default-counters.csv`, keeping NVIDIA's header. Add one comment line
  at the top: `# Vendored from NVIDIA/dcgm-exporter tag 4.9.0`.
- [ ] **Step 4: Write failing tests** in `tests/test_counters.py`:
  - `test_default_has_27_rows`: `len(load_counters(None)) == 27`, the first row is
    `CounterSpec("DCGM_FI_DEV_SM_CLOCK_HERTZ","gauge","SM clock frequency in hertz.")`, and the
    `DCGM_FI_SYSTEM_DRIVER_VERSION` row has type `"label"`
  - `test_comments_blank_lines_and_whitespace_ignored`: `"# x\n\n  A_B ,  gauge , help text \n"` → `[CounterSpec("A_B","gauge","help text")]`
  - `test_help_may_contain_commas`: the third column is everything after the second comma, stripped
  - `test_crlf_and_bom` *(Review Focus 3)*: `"﻿A, gauge, h\r\nB, counter, h2\r\n"` parses to two specs
  - `test_malformed_row_raises`: `"A, gauge"` and `"A, histogram, h"` raise `CountersError`, and the message contains the source and line number
  - `test_duplicate_name_raises`
- [ ] **Step 5: Run** `uv run pytest tests/test_counters.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 6: Implement** `counters.py` per the Interfaces above. Use `str.split(",", 2)` and
  `text.lstrip("﻿").splitlines()`.
- [ ] **Step 7: Run** `uv run pytest -v && uv run ruff check && uv run ruff format --check`. Expected: all PASS.
- [ ] **Step 8: Add CI.** `.github/workflows/ci.yml` uses `astral-sh/setup-uv` and a matrix over
  Python 3.12/3.13/3.14 to run `uv run ruff check`, `uv run ruff format --check`, and `uv run pytest`.
- [ ] **Step 9: Commit** `feat: counters CSV parsing, vendored 4.9.0 defaults, contract notes`.

### Task 2: NVML facade and fake devices

**Files:**
- Create: `src/dcgm_exporter_lite/nvml.py`, `tests/fakes.py`, `tests/conftest.py`, `tests/test_nvml.py`

**Interfaces:**
- Produces (`nvml.py`):
  - Exceptions:
    - `Unsupported(Exception)`: `NVML_ERROR_NOT_SUPPORTED` or `NVMLError_FunctionNotFound`
    - `GpuLost(Exception)`: `NVML_ERROR_GPU_IS_LOST`
    - `ReadError(Exception)`: any other `NVMLError`
  - `_call(fn, *args)`: the single place where `pynvml.NVMLError` is mapped to those exceptions.
  - Frozen dataclasses:
    - `DeviceIdentity(index: int, uuid: str, pci_bus_id: str, minor: int, name: str)`
    - `MemoryInfo(total: int, free: int, used: int, reserved: int)` (bytes)
    - `RemappedRows(correctable: int, uncorrectable: int, pending: bool, failed: bool)`
  - `class GpuDevice(Protocol)` with attribute `identity: DeviceIdentity` and methods:
    - `mig_enabled() -> bool`
    - `clock_mhz(kind: Literal["sm","mem"]) -> int`
    - `gpu_temp_c() -> int`, `memory_temp_c() -> int`
    - `power_mw() -> int`, `energy_mj() -> int`, `pcie_replay() -> int`
    - `utilization_pct() -> tuple[int, int]` (gpu, memory)
    - `encoder_pct() -> int`, `decoder_pct() -> int`
    - `memory() -> MemoryInfo`, `remapped_rows() -> RemappedRows`
    - `nvlink_raw() -> tuple[int, int]` (tx, rx exactly as NVML returns them; semantics per contract notes item 9)
    - `gpm_supported() -> bool`
    - `gpm_sample() -> object`
    - `gpm_metrics(prev: object, cur: object, metric_ids: Sequence[int]) -> dict[int, float]` (per-id unsupported ids are absent)
  - `class NvmlDevice` implements `GpuDevice` over a pynvml handle.
  - `class XidEventSource(Protocol)`: `wait(timeout_ms: int) -> tuple[int, int] | None` returns `(gpu_index, xid_code)`. Also `close() -> None`.
  - `class NvmlXidEventSource(devices: Sequence[NvmlDevice])`: registers `nvmlEventTypeXidCriticalError`
    on every device that allows it. Devices that raise `Unsupported` are skipped and logged.
  - `init() -> str` initializes NVML and returns the driver version. `shutdown() -> None`. `open_devices() -> list[NvmlDevice]`.
  - `driver_major(version: str) -> int`
- Produces (`tests/fakes.py`):
  - `FakeDevice(identity, **values)`: each method returns `values[method_name]`. If the value is an
    exception instance or class, it raises it instead. `values` is mutable, so tests can change
    readings between cycles. The GPM methods take a scripted list of `dict[int, float]`, one per call.
  - `FakeEventSource(events: list[tuple[int,int]])` returns them in order, then `None`.
  - `canonical_pair() -> list[FakeDevice]` returns two datacenter-like GPUs with every value set,
    using the numbers in Task 7's golden file:
    - GPU 0: `GPU-00000000-0000-0000-0000-000000000000`, `00000000:17:00.0`, minor 0, `NVIDIA H100 80GB HBM3`
    - GPU 1: same, ending `…0001`, `00000000:2A:00.0`, minor 1

- [ ] **Step 1: Write failing tests** in `tests/test_nvml.py`, monkeypatching `pynvml` functions:
  - `test_not_supported_maps_to_unsupported`
  - `test_function_not_found_maps_to_unsupported`
  - `test_gpu_lost_maps`
  - `test_other_error_maps_to_read_error`
  - `test_identity_formats`: `pynvml.nvmlDeviceGetMinorNumber` → 3 gives `minor == 3`; `busId` is
    normalized to the case recorded in contract notes item 1; bytes and str names both decode
  - `test_driver_major`: `"580.65.06"` → 580, `"615.71.10"` → 615
  - `test_memory_uses_v2`: asserts `nvmlDeviceGetMemoryInfo` is called with `version=pynvml.nvmlMemory_v2`
- [ ] **Step 2: Run** `uv run pytest tests/test_nvml.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `nvml.py` and `tests/fakes.py`. NVML calls, from the spec §4.3 table:
  - `nvmlDeviceGetMigMode(h)[0] == NVML_DEVICE_MIG_ENABLE`
  - `nvmlDeviceGetFieldValues` for memory temp and the NVLink fields. Check each value's `nvmlReturn`
    and map a non-zero return through the same error mapping.
  - GPM through `nvmlGpmSampleAlloc`/`nvmlGpmSampleGet`/`nvmlGpmMetricsGet`, with `c_nvmlGpmMetricsGet_t`
    and `version = NVML_GPM_METRICS_GET_VERSION`, all ids in one call. Free samples with `nvmlGpmSampleFree` when replaced.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: NVML facade with error mapping and test fakes`.

### Task 3: Metric definitions and unit conversions

**Files:**
- Create: `src/dcgm_exporter_lite/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: the `GpuDevice` methods (Task 2).
- Produces:
  - `ReadContext(device: GpuDevice, gpm: Mapping[str, float] | None, xid_last: int | None, interval_s: float, state: dict)`.
    `gpm` is keyed by GPM metric name. `state` is per-(GPU, metric) scratch space, used for NVLink
    rate derivation if contract notes item 9 says the fields are counters.
  - `MetricDef(name: str, gpu_wide: bool, source: Literal["nvml","gpm","xid"], read: Callable[[ReadContext], float])`.
    `read` raises `Unsupported`/`ReadError`/`GpuLost`, or `NoValueYet` (a new exception for the
    first GPM or NVLink cycle).
  - `METRICS: dict[str, MetricDef]`: exactly the 25 non-label, non-vGPU rows of the spec §4.3 table.
    `DCGM_FI_DEV_VGPU_LICENSE_STATUS` is absent.
  - `GPM_IDS: dict[str, int]`: the GPM metric names used, mapped to `pynvml.NVML_GPM_METRIC_*` ids.
  - `gpu_wide` is True for clocks, temps, power, energy, PCIe replay, FB*, row remap, NVLink, and
    XID. It is False for the util ratios and `PROF_*` (§5.3 MIG rule).

- [ ] **Step 1: Write failing tests** in `tests/test_metrics.py`. Use one `FakeDevice` and assert exact values:
  - `test_clock_hz`: `clock_mhz("sm")=1980` → `DCGM_FI_DEV_SM_CLOCK_HERTZ == 1_980_000_000.0`. Same check for the memory clock.
  - `test_power_watts`: `power_mw=350_123` → `350.123` (or the field-value path, if contract notes item 7 says instant)
  - `test_energy_joules`: `energy_mj=1_032_632` → `1032.632`
  - `test_util_ratios`: `utilization_pct=(87, 42)` → `0.87`, `0.42`. `encoder_pct=5` → `0.05`.
  - `test_fb_bytes`: `MemoryInfo(17103323136, 16461987840, 250347520, 390987776)` maps to CAPACITY/FREE/USED/RESERVED unchanged
  - `test_row_remap`: `RemappedRows(3, 1, False, True)` → CORRECTABLE 3, UNCORRECTABLE 1, FAILED 1.0
  - `test_prof_from_gpm`: `gpm={"GRAPHICS_UTIL": 55.0, <tensor id per notes>: 12.5, "DRAM_BW_UTIL": 30.0}` → ratios `0.55`, `0.125`, `0.30`
  - `test_prof_pcie_bytes_per_second`: converts using the unit from contract notes item 8 (e.g. if MiB/s: `1.125` → `1_179_648.0`)
  - `test_prof_absent_when_no_gpm`: `gpm=None` raises `NoValueYet`
  - `test_xid`: `xid_last=79` → 79.0. `xid_last=None` behaves per contract notes item 3.
  - `test_nvlink`: per contract notes item 9. If the fields are counters: the first cycle raises
    `NoValueYet`, and the second cycle gives `(Δtx+Δrx)*unit/interval_s`.
  - `test_registry_matches_default_csv`: `set(METRICS) == {c.name for c in load_counters(None) if c.type != "label"} - {"DCGM_FI_DEV_VGPU_LICENSE_STATUS"}`
- [ ] **Step 2: Run** `uv run pytest tests/test_metrics.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `metrics.py`, one small reader function per metric. No class hierarchy.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: DCGM 4.9.0 metric definitions with NVML unit conversions`.

### Task 4: GPM sampler

**Files:**
- Create: `src/dcgm_exporter_lite/gpm.py`, `tests/test_gpm.py`

**Interfaces:**
- Consumes: `GpuDevice.gpm_supported/gpm_sample/gpm_metrics`, `GPM_IDS`.
- Produces: `GpmSampler(devices: Sequence[GpuDevice])` with method
  `step(index: int) -> Mapping[str, float] | None`. It takes a new sample, computes metrics against
  the previous sample, and returns `None` on the first call or when GPM is unsupported (checked once
  per device and cached). An `Unsupported` result for a single metric id just leaves that key out.

- [ ] **Step 1: Write failing tests:**
  - `test_first_step_returns_none`
  - `test_second_step_returns_named_values`
  - `test_unsupported_device_always_none_and_never_samples`: assert that `gpm_sample` is never called
  - `test_missing_metric_id_omitted`
  - `test_sample_error_resets_baseline`: after a `ReadError` the next step returns `None`, and the step after that returns values
- [ ] **Step 2: Run** `uv run pytest tests/test_gpm.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `gpm.py`.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: GPM two-sample sampler`.

### Task 5: XID watcher

**Files:**
- Create: `src/dcgm_exporter_lite/xid.py`, `tests/test_xid.py`

**Interfaces:**
- Consumes: `XidEventSource` (Task 2).
- Produces:
  - `XidWatcher(source: XidEventSource)`, a daemon `threading.Thread`
  - `start()`
  - `stop(timeout: float = 2.0)`: sets an Event, then joins. `wait` is called with a 500 ms timeout, so stop is prompt.
  - `last(index: int) -> int | None`
  - `available: bool` (False if the source registered no devices)
  - The last code is stored per GPU under a lock.

- [ ] **Step 1: Write failing tests:**
  - `test_records_last_code_per_gpu`: events `[(0, 13), (1, 79), (0, 31)]` → `last(0)==31`, `last(1)==79`
  - `test_none_before_event`
  - `test_stop_returns_promptly`: `stop()` finishes in under 1.5 s
  - `test_source_error_logged_and_thread_survives`: the source raises `ReadError` once, then yields an event
- [ ] **Step 2: Run** `uv run pytest tests/test_xid.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `xid.py`.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: XID event watcher`.

### Task 6: Collector, support probing, snapshots, collection loop

**Files:**
- Create: `src/dcgm_exporter_lite/collector.py`, `tests/test_collector.py`

**Interfaces:**
- Consumes: `CounterSpec`, `METRICS`, `ReadContext`, `NoValueYet`, `GpmSampler.step`,
  `XidWatcher.last/available`, `GpuDevice`, and the exceptions.
- Produces:
  - `Sample(name: str, labels: tuple[tuple[str, str], ...], value: float)`, frozen
  - `Snapshot(samples: tuple[Sample, ...], ok_gpus: int, taken_at: float)`, frozen
  - `Collector(devices, counters, *, driver_version: str, hostname: str | None, replace_blanks: bool, gpm: GpmSampler, xid: XidWatcher | None, interval_s: float)` with:
    - `probe() -> None`: computes the per-GPU enabled set from the requested counters.
      - Counter names missing from `METRICS` → one WARNING each.
      - `Unsupported` on the first read → one INFO per (metric, GPU), and the metric is disabled for that GPU.
      - MIG GPU → disable non-`gpu_wide` metrics, with one INFO.
      - GPM-sourced metrics are probed with `gpm_supported()`. XID is probed with `xid.available`.
    - `collect_once() -> Snapshot`
    - `latest: Snapshot | None`
    - `labels_for(device) -> tuple[tuple[str,str], ...]`: label order per contract notes item 1.
      `Hostname` is omitted when `hostname is None`. `modelName` has spaces replaced with `_` when
      `replace_blanks`. Each `label`-type counter (e.g. `DCGM_FI_SYSTEM_DRIVER_VERSION`) adds a label.
  - `CollectionLoop(collector, interval_s)`, a daemon thread: `start()`, `stop()`. It calls
    `collect_once()` immediately, then every interval. An unexpected exception is logged, and the
    loop continues.
- Collection rules:
  - `ReadError` → skip that sample this cycle. WARNING, rate-limited to once per (metric, GPU) per 5 minutes.
  - `NoValueYet` → skip silently.
  - `GpuLost` → ERROR once. The GPU is excluded from all later cycles.
  - `Unsupported` at collect time → disable the metric for that GPU (INFO).
  - A GPU counts toward `ok_gpus` if at least one sample succeeded this cycle.

- [ ] **Step 1: Write failing tests** using `canonical_pair()`:
  - `test_labels_order_and_values`: exact tuple for GPU 0 per contract notes item 1
  - `test_no_hostname_and_replace_blanks`
  - `test_mixed_fleet_unsupported_omitted` *(Review Focus 2)*: GPU 1 `memory_temp_c=Unsupported`
    → the snapshot has a `DCGM_FI_DEV_MEMORY_TEMP_CELSIUS` sample for GPU 0 only
  - `test_function_not_found_treated_as_unsupported`
  - `test_unknown_counter_warns_once` (use `caplog`)
  - `test_mig_gpu_only_gpu_wide`: GPU 0 `mig_enabled=True` → no `*_UTIL_RATIO` or `PROF_*` for GPU 0. Temperature and power are present.
  - `test_read_error_skips_sample_then_recovers`
  - `test_gpu_lost_mid_run` *(Review Focus 5)*: cycle 1 has both GPUs. GPU 1 then raises `GpuLost`,
    so cycle 2 has only GPU 0 samples and `ok_gpus == 1`. Cycle 3 makes no calls to GPU 1.
  - `test_first_cycle_has_no_prof_second_has`
  - `test_loop_runs_and_stops`: interval 0.05 s, `latest` is set within 1 s, and `stop()` joins
- [ ] **Step 2: Run** `uv run pytest tests/test_collector.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `collector.py`.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: collector with support probing, MIG handling, and collection loop`.

### Task 7: Prometheus exposition and golden test

**Files:**
- Create: `src/dcgm_exporter_lite/exposition.py`, `tests/test_exposition.py`, `tests/golden/default.prom`

**Interfaces:**
- Consumes: `Snapshot`, `CounterSpec`.
- Produces:
  - `SnapshotCollector(get_snapshot: Callable[[], Snapshot | None], counters: Sequence[CounterSpec])`,
    implementing `prometheus_client.registry.Collector.collect()`. Families are yielded in CSV order,
    and a family with no samples is skipped.
  - `make_registry(get_snapshot, counters) -> CollectorRegistry`. It is a fresh
    `CollectorRegistry(auto_describe=False)`, so the default process/platform/gc collectors are not
    included.
  - `render(registry) -> bytes`, which wraps `generate_latest`.

- [ ] **Step 1: Write the golden file** `tests/golden/default.prom`. Hand-write the expected
  exposition for the canonical pair's second cycle (so GPM and XID values are present). Follow the
  formats in contract notes items 1–4 exactly: HELP from the CSV, TYPE, label order, counter naming,
  and float formatting as `prometheus_client` renders it.
- [ ] **Step 2: Write failing tests:**
  - `test_golden_default`: `render(...)` equals the golden file byte-for-byte
  - `test_counter_names_match_dcgm`: `DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL` appears exactly as in
    DCGM: no `_total_total`, and no `_created` unless contract notes item 4 says DCGM emits it
  - `test_no_process_or_python_metrics`
  - `test_empty_snapshot_renders_valid_empty` *(Review Focus 1)*: `get_snapshot` returns `None` → `render(...) == b""`
  - `test_label_value_escaping`: a model name with `"` and `\` is escaped
- [ ] **Step 3: Run** `uv run pytest tests/test_exposition.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement** `exposition.py`.
  - Use `GaugeMetricFamily` for gauges.
  - For counters, use `CounterMetricFamily` with the name minus `_TOTAL`, if that renders exactly as
    DCGM does. Otherwise use a `Metric` with `type="counter"` and sample names set explicitly.
  - Let the golden test decide.
- [ ] **Step 5: Run** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 6: Commit** `feat: Prometheus exposition matching DCGM 4.9.0 golden output`.

### Task 8: HTTP server

**Files:**
- Create: `src/dcgm_exporter_lite/server.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: `render`, `CollectorRegistry`.
- Produces:
  - `parse_address(addr: str) -> tuple[str, int]`. Raises `ValueError` with a readable message on bad input.
  - `make_server(addr: str, registry, healthy: Callable[[], bool]) -> ThreadingHTTPServer`. Raises
    `OSError` if the bind fails. Routes:
    - `GET /metrics` → 200, `CONTENT_TYPE_LATEST`
    - `GET /health` → 200/503 with the body per contract notes item 5
    - `GET /` → a small HTML page linking to `/metrics`
    - any other path → 404
  - Request logging goes to `logging` at DEBUG, not stderr.

- [ ] **Step 1: Write failing tests** *(Review Focus 4)*:
  - `test_parse_address`:
    - `":9400"` → `("", 9400)`
    - `"0.0.0.0:9400"` → `("0.0.0.0", 9400)`
    - `"[::]:9400"` → `("::", 9400)`
    - `"localhost:9400"` → `("localhost", 9400)`
    - `"9400"`, `"host:"`, and `"host:99999"` raise `ValueError`
  - `test_ipv6_server_binds`: `[::1]:0`. Skip if IPv6 is unavailable.
  - `test_port_in_use_raises_oserror`
  - `test_metrics_endpoint`: start on `127.0.0.1:0` in a thread; GET `/metrics` → 200, correct content type, and body equals `render(registry)`
  - `test_health_before_first_collection` *(Review Focus 1)*: `healthy=lambda: False` → 503. With True → 200.
  - `test_root_and_404`
- [ ] **Step 2: Run** `uv run pytest tests/test_server.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `server.py`. Set `ThreadingHTTPServer.address_family` to `AF_INET6`
  when the host contains `:`. Set `daemon_threads = True`.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: HTTP server for /metrics, /health, /`.

### Task 9: CLI, wiring, signals

**Files:**
- Create/replace: `src/dcgm_exporter_lite/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `Config(address: str, collectors: str | None, interval_ms: int, devices: list[int] | None, no_hostname: bool, replace_blanks: bool, debug: bool, log_format: Literal["text","json"])`, frozen
  - `parse_config(argv: Sequence[str], env: Mapping[str, str]) -> Config`. Raises `SystemExit(2)` with
    a message for unsupported flags. Precedence: flags, then env vars, then defaults. Flag and env
    names follow spec §5.1 and contract notes item 6.
  - `main(argv: Sequence[str] | None = None) -> int`
  - `run(config: Config, *, nvml_module=nvml, stop_event: threading.Event | None = None) -> int`.
    This is the testable core; `main` builds the config and calls `run`.

- Behavior of `run`:
  1. Set up logging.
  2. `nvml.init()`. On failure, print `error: cannot initialize NVML: <reason>` and return 1.
  3. Driver major < 580 → WARNING.
  4. Open devices and filter by `--devices`. If no devices remain → return 1 with a message.
  5. Load counters. On `CountersError` → return 1.
  6. Build the collector, `probe()`, and start the XID watcher and collection loop.
  7. `make_server`. On `OSError` → `error: cannot listen on <addr>: <reason>` and return 1 (no traceback).
  8. Serve until SIGTERM/SIGINT or `stop_event`. Then stop the threads, call `nvml.shutdown()`, and return 0.
  - `healthy()` is `latest is not None and latest.ok_gpus > 0`.
- Unsupported flags are rejected by name. Write the list in `cli.py` from 4.9.0's `flag_names.go`:
  every flag not in spec §5.1, e.g. `--kubernetes`, `--config-file`, `--remote-hostengine-info`,
  `--web-config-file`, `--fake-gpus`, `--switch-devices`, `--cpu-devices`, `--use-old-namespace`, ….
  - The message is `--<flag> is not supported by dcgm-exporter-lite (see README: Differences from DCGM Exporter)`.
  - Unknown flags that are not DCGM flags get argparse's normal error.

- [ ] **Step 1: Write failing tests:**
  - `test_defaults`: `Config(":9400", None, 30000, None, False, False, False, "text")`
  - `test_short_aliases`: `-a :9500 -f x.csv -c 1000`
  - `test_env_vars`: e.g. `DCGM_EXPORTER_LISTEN=":9600"`, `DCGM_EXPORTER_INTERVAL="5000"`. A flag beats the env var.
  - `test_devices_syntax`: `f` → None, `g:0,2` → `[0, 2]`. MIG selectors (`i:…`) are rejected.
  - `test_unsupported_dcgm_flag_rejected`: `--kubernetes` → `SystemExit`, and stderr contains `not supported by dcgm-exporter-lite`
  - `test_run_end_to_end_with_fakes`: a fake `nvml_module` returns `canonical_pair()`. Start `run`
    in a thread on `127.0.0.1:0`, then GET `/metrics` until it contains `DCGM_FI_DEV_GPU_TEMP_CELSIUS`
    (timeout 3 s). Set `stop_event`, and `run` returns 0.
  - `test_run_nvml_init_failure_returns_1`
  - `test_run_old_driver_warns`: `"575.10"` → WARNING containing `580`
  - `test_run_port_in_use_returns_1_no_traceback` *(Review Focus 4)*
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `cli.py`. JSON log format means one JSON object per line, built with the
  standard `logging.Formatter`.
- [ ] **Step 4: Run** `uv run pytest -v && uv run ruff check`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: CLI with DCGM-compatible flags and process wiring`.

### Task 10: Hardware smoketest on ryzen.local

**Files:**
- Create: `tests/hardware/test_smoke.py`, `scripts/smoketest-remote.sh`

**Interfaces:**
- Consumes: the installed `dcgm-exporter-lite` command.
- Produces: `scripts/smoketest-remote.sh HOST`. It rsyncs the repo to `~/dcgm-exporter-lite-smoke`
  on HOST (excluding `.venv`), creates a venv at `~/dcgm-exporter-lite-smoke/.venv` with `python3 -m venv`, runs `pip install -e . pytest`,
  and runs `pytest -m hardware -v`. It exits with pytest's status.

- [ ] **Step 1: Write** `tests/hardware/test_smoke.py` (`pytestmark = pytest.mark.hardware`). The
  test starts `dcgm-exporter-lite -a 127.0.0.1:<free port> -c 1000` as a subprocess and waits for
  `/health` to return 200 (timeout 10 s). After 2.5 s it scrapes `/metrics` and checks:
  - `test_expected_metrics_present`: every ✅ metric in spec §4.3 is present for gpu `"0"`
  - `test_plausible_values`:
    - `1e8 < SM_CLOCK_HERTZ < 4e9`
    - every `*_RATIO` is in `[0, 1]`
    - `FB_CAPACITY_BYTES` is within 2% of `16311 * 2**20` (or of `nvidia-smi --query-gpu=memory.total` × 2²⁰)
    - `10 < GPU_TEMP_CELSIUS < 100`
    - `BOARD_POWER_WATTS > 0`
  - `test_datacenter_only_absent_on_geforce`: `MEMORY_TEMP_CELSIUS`, `NVLINK_THROUGHPUT_BYTES_PER_SECOND`,
    and `ROW_REMAP_*` are absent. Skip if the model name does not contain `GeForce`.
  - `test_labels`: `UUID` matches `nvidia-smi --query-gpu=uuid`, `device="nvidia0"`, and
    `DCGM_FI_SYSTEM_DRIVER_VERSION` matches `nvidia-smi --query-gpu=driver_version`
  - `test_runs_as_non_root`: `os.geteuid() != 0`
  - `test_sigterm_exits_zero`
- [ ] **Step 2: Run** `scripts/smoketest-remote.sh ryzen.local`. Expected: all PASS. If a value
  check fails, fix the conversion in `metrics.py` (with a unit test) rather than loosening the bound.
- [ ] **Step 3: Record** the scraped `/metrics` output in the PR description or commit message body as evidence.
- [ ] **Step 4: Commit** `test: hardware smoketest for real NVML on ryzen.local`.

### Task 11: Grafana dashboard

**Files:**
- Create: `grafana/dcgm-exporter-lite-dashboard.json`, `tests/test_dashboard.py`

**Interfaces:**
- Consumes: the metric names in `load_counters(None)`.

- [ ] **Step 1: Write failing tests** in `tests/test_dashboard.py`:
  - `test_dashboard_is_valid_json_with_variables`: `templating.list` names include `datasource`,
    `instance`, and `gpu`, and `gpu`'s query is `label_values(DCGM_FI_DEV_GPU_TEMP_CELSIUS, gpu)`
  - `test_only_dcgm_490_metrics`: every `DCGM_[A-Z_]+` token in any panel `expr` is a name from the
    default CSV (so the dashboard works against real DCGM Exporter 4.9.0)
  - `test_every_panel_filters_instance_and_gpu`: each `expr` contains `instance=~"$instance"` and `gpu=~"$gpu"`
  - `test_rows_present`: row titles are exactly `Fleet overview`, `Utilization`, `Memory`, `Thermals & power`, `Health`, `I/O`
- [ ] **Step 2: Run** `uv run pytest tests/test_dashboard.py -v`. Expected: FAIL.
- [ ] **Step 3: Build** the dashboard (Grafana schema for current Grafana, `${datasource}` on every
  panel), using the panel suggestions in spec §6. Datacenter-only panels have `noValue: "Not reported by this GPU"`.
  Required queries:
  - Energy in range: `sum(increase(DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL{…}[$__range])) / 3.6e6` (kWh)
  - Idle GPUs: `count(max_over_time(DCGM_FI_DEV_GPU_UTIL_RATIO{…}[1h]) < 0.05)`
  - FB headroom: `1 - DCGM_FI_DEV_FB_USED_BYTES / DCGM_FI_DEV_FB_CAPACITY_BYTES`
  - Recent XIDs: a table of `DCGM_FI_DEV_XID_ERROR{…} > 0`
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5: Manual check across the network split.** The exporter runs on `ryzen.local`;
  Prometheus and Grafana run on the development workstation (which has `docker`, not podman).
  - On `ryzen.local`, from the Task 10 smoketest venv (`~/dcgm-exporter-lite-smoke`), start the
    exporter listening on all interfaces: `.venv/bin/dcgm-exporter-lite -a 0.0.0.0:9400 -c 5000`.
    The user has pre-approved binding `0.0.0.0` on `ryzen.local` for this check.
  - On the workstation, from a scratch dir, write `prometheus.yml` with a 15s scrape job
    `dcgm-exporter-lite` targeting `ryzen.local:9400`, then:
    ```
    docker network create dcgm-lite-check
    docker run -d --name prom --network dcgm-lite-check -p 9090:9090 \
      -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml:ro" prom/prometheus
    docker run -d --name grafana --network dcgm-lite-check -p 3000:3000 grafana/grafana
    ```
  - Confirm `http://localhost:9090/targets` shows the ryzen target UP. Add a Prometheus datasource
    in Grafana (`http://prom:9090`) and import the dashboard JSON (Grafana HTTP API with the default
    admin credentials is fine for this throwaway instance).
  - Let it scrape for at least 10 minutes, with some GPU load on ryzen if available, then confirm
    every non-datacenter panel shows data. Report and fix any panel that shows an error (as opposed
    to "No data").
  - Tear down: stop the exporter on ryzen (`Ctrl-C`/kill); `docker rm -f prom grafana && docker network rm dcgm-lite-check`.
- [ ] **Step 6: Commit** `feat: operational-value Grafana dashboard`.

### Task 12: Documentation, systemd unit, datacenter verification kit

**Files:**
- Create: `README.md` (replace), `contrib/systemd/dcgm-exporter-lite.service`,
  `docs/testing/datacenter-verification.md`, `scripts/collect-datacenter-evidence.sh`
- Modify: `pyproject.toml` (`description`, `classifiers` add 3.13/3.14, `[project.urls]`)

- [ ] **Step 1: README.** Cover every item in spec §7:
  - What it is, and the bridge story
  - Install, quick start, flags table
  - Metric coverage table:
    - Columns: ✅ verified on hardware / 🖥 datacenter-only, **unverified on hardware** / ❌ omitted, with the reason
    - Values taken from Task 10 results
  - GPM note: Hopper+ only; differs from DCGM DCP sampling
  - MIG behavior
  - Driver floor 580+
  - Prometheus scrape example
  - Dashboard import, with a link to NVIDIA's 4.9.0 dashboard
  - **"Swapping to DCGM Exporter"** checklist
  - "Differences from DCGM Exporter"
- [ ] **Step 2: systemd user unit.** `ExecStart=%h/.local/bin/dcgm-exporter-lite`,
  `Restart=on-failure`, `WantedBy=default.target`. Its install instructions in the README are
  `systemctl --user daemon-reload && systemctl --user enable --now dcgm-exporter-lite` and
  `loginctl enable-linger $USER`.
- [ ] **Step 3: Datacenter verification checklist** (spec §8). It must be runnable by someone who
  has not seen this repo, and contain:
  - prerequisites
  - install commands
  - for each 🖥 metric and each GPM metric: expected presence, and the `nvidia-smi -q` section to cross-check against
  - load generation steps, using a short PyTorch or `cuda-samples` matmul and an NVLink peer copy
    for multi-GPU, with commands
  - MIG checks: enable MIG on one GPU, then confirm that only GPU-wide metrics remain
  - the NVLink counter-vs-rate question, with explicit pass/fail criteria
  - exactly what to send back
- [ ] **Step 4: Evidence script.** `scripts/collect-datacenter-evidence.sh [ADDR]` writes
  `dcgm-lite-evidence-<host>-<date>.tar.gz` containing:
  - `nvidia-smi -q`
  - `nvidia-smi topo -m`
  - three `/metrics` scrapes 30 s apart
  - exporter version, driver version, and `uname -a`
  
  Run `shellcheck` on it if available.
- [ ] **Step 5: Verify.**
  - `uv build` produces a wheel.
  - `unzip -l dist/*.whl | grep default-counters.csv` shows the data file.
  - A fresh venv install (`python -m venv /tmp/x && /tmp/x/bin/pip install dist/*.whl && /tmp/x/bin/dcgm-exporter-lite --help`) works.
  - `uv run pytest && uv run ruff check` passes.
- [ ] **Step 6: Commit** `docs: README, systemd user unit, datacenter verification kit`.

# dcgm-exporter-lite — Design

Date: 2026-11-06
Status: Draft for review (builder check done; driver floor, MIG, datacenter testing settled)

## 1. Intent

NVIDIA's [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) is the standard way to get GPU
metrics into Prometheus, but it is hard to adopt in some environments:

- Its container needs a CDI hook that breaks in some podman setups.
- DCGM needs root to install and to run.
- It is a Go program. Some teams do not have approval or infrastructure for Go.

**dcgm-exporter-lite** is a Python package that can be `pip install`ed into a virtual environment
and run with ordinary user credentials. It reads GPU data from the NVIDIA driver through NVML
(`nvidia-ml-py`) and serves it in Prometheus format (`prometheus-client`).

### Why it exists: a bridge, not a replacement

The main purpose is to **show the operational value of GPU metrics now**, with no special
privileges. That evidence can justify the exceptions needed to run the real DCGM Exporter. When
that happens, operators replace this exporter with DCGM Exporter and **change no monitoring
configuration**. Prometheus scrape configs, Grafana dashboards, and alert rules must keep working
unchanged.

Every compatibility decision below follows from that swap requirement.

### Users

Operators of mostly **datacenter-class NVIDIA GPUs** (A100/H100/B200 class) who want DCGM-style
monitoring but cannot meet DCGM Exporter's privilege, container, or language requirements.

## 2. Goals, non-goals, anti-goals

### Goals

1. **Wire compatibility with DCGM Exporter 4.9.0** (released 2026-11-06). For every metric we emit,
   the metric name, type, unit, label names, and label value formats match what 4.9.0 emits for
   the same GPU.
2. Runs as an unprivileged user from a venv. Needs no root, container, CDI, or DCGM.
3. Matches DCGM Exporter's operational surface: default listen address `:9400`, `/metrics`,
   `/health`, the same names for the flags we support, and the same counters-CSV format.
4. Ships a **richer Grafana dashboard** that shows operational value. It uses only DCGM 4.9.0 metric
   names, so it also works unchanged against the real DCGM Exporter.
5. Documentation: install, run as a systemd **user** service, configure Prometheus, import the
   dashboard, and swap to DCGM Exporter.

### Non-goals (this version)

- The 4.8.x (pre-rename) metric names and units, and the 1.x `dcgm_*` names.
- MIG instance metrics and labels (`GPU_I_ID`, `GPU_I_PROFILE`, …).
- Kubernetes, container, DRA, or HPC-job attribution labels (`pod`, `namespace`, `container`, `hpc_job`, …).
- NVSwitch, CPU, and NVLink-per-link metrics.
- DCGM health, policy (`DCGM_EXP_FIELD_POLICY_*`), clock-event, and `DCGM_EXP_*` exporter metrics.
- YAML config files (v1/v2), ConfigMap sources, TLS/web-config, systemd socket activation.
- A container image.
- Podman device pass-through docs. **Stretch goal for later.**

### Anti-goals (a failure even if it technically works)

- **A wrong value under a DCGM name.** A metric that NVML cannot supply with the same meaning and
  unit is **omitted**. It is never approximated under the DCGM name. A gap in a panel is acceptable.
  A misleading value is not.
- **Requiring root**, setuid helpers, or extra system packages beyond the NVIDIA driver.
- **Any monitoring-config change at swap time.** Changing a scrape job, dashboard, or alert
  because we named or labeled something differently is a failure.
- Inventing new metric names in the `DCGM_FI_*` namespace.

## 3. Constraints

- Python **>= 3.12** (from `pyproject.toml`). The test host has 3.14, so support 3.12–3.14.
- Runtime dependencies (already pinned): `nvidia-ml-py==13.615.71`, `prometheus-client==0.26.0`.
  Add no other runtime dependencies without a strong reason.
- Build backend `uv_build`. Dev tools: `pytest-cov`, `ruff`. The entry point `dcgm-exporter-lite`
  maps to `dcgm_exporter_lite:main`. The package is currently a `uv init` stub.
- **Supported NVIDIA drivers: 580 and newer.** 580 is the last driver branch that supports Volta,
  so Volta (V100) is the oldest architecture. Rules for older or limited drivers:
  - NVML calls absent from a supported driver (`NVMLError_FunctionNotFound`) or from the GPU's
    architecture are treated exactly like `NOT_SUPPORTED`: the metric is omitted for that GPU.
  - At startup, a driver older than 580 logs a WARNING (not a hard failure).
  - GPM, and so all `PROF_*` metrics, exists only on Hopper and newer. On Volta and Ampere those
    metrics are omitted.
  - The README states the 580+ floor.
- The development machine has **no NVIDIA driver**. Unit tests must run against a fake NVML.
- Real-hardware smoketest host: `ryzen.local` (password-free SSH). It has an NVIDIA GeForce RTX
  5060 Ti (Blackwell, arch 10), driver 615.71.10, CUDA 13.4, Python 3.14, and podman.
  `/dev/nvidia*` is world-readable/writable. This is a consumer card, so the datacenter-only paths
  (row remap, HBM temperature, NVLink) cannot be checked on real hardware here.

## 4. Metric contract

### 4.1 Source of truth

DCGM Exporter tag **`4.9.0`**:

- `etc/default-counters.csv`: the default metric list (27 rows).
- `tests/internal/metriccontract/`: canonical exposition examples, including label order.
- `grafana/dcgm-exporter-dashboard.json`: NVIDIA's dashboard, for reference and comparison.

Vendor a copy of 4.9.0's `default-counters.csv` into the package as the default collectors file.
Keep NVIDIA's header and license comment and record the upstream tag.

### 4.2 Labels

Every GPU metric carries these labels, in this order (as in 4.9.0):

| Label | Value | NVML source |
|---|---|---|
| `gpu` | NVML device index, e.g. `"0"` | enumeration index |
| `UUID` | e.g. `"GPU-7e1510cf-2c7e-57e9-daf0-42f851c85119"` | `nvmlDeviceGetUUID` |
| `pci_bus_id` | e.g. `"00000000:2B:00.0"` | `nvmlDeviceGetPciInfo_v3().busId` |
| `device` | `"nvidia<minor>"`, e.g. `"nvidia0"` | `nvmlDeviceGetMinorNumber` |
| `modelName` | e.g. `"NVIDIA GeForce RTX 5060 Ti"` | `nvmlDeviceGetName` |
| `Hostname` | host name (omitted with `--no-hostname`) | `socket.gethostname()` |
| `DCGM_FI_SYSTEM_DRIVER_VERSION` | e.g. `"615.71.10"` | `nvmlSystemGetDriverVersion` |

`DCGM_FI_SYSTEM_DRIVER_VERSION` appears as a label because the CSV row type is `label`. Handle the
CSV `label` type generically: the field becomes a label on every metric, not a metric.

**Builder must verify** the exact label set, order, and value formats against the
`metriccontract` tests and the 4.9.0 source. In particular, check whether `pci_bus_id` is upper-
or lower-case hex, and whether `XID_ERROR` carries extra labels (`err_code`/`err_msg`) in 4.9.0.
Record the verified contract in a single test fixture (golden exposition file) that the whole test
suite uses.

### 4.3 Coverage map (the 27 default counters)

Status uses the results of an NVML probe run as an unprivileged user on `ryzen.local` (RTX 5060 Ti):

- ✅ = worked on the test card
- 🖥 = datacenter-only, "Not Supported" on GeForce
- ❌ = not implemented

| DCGM 4.9.0 metric | Type | NVML source → conversion | Status |
|---|---|---|---|
| `DCGM_FI_DEV_SM_CLOCK_HERTZ` | gauge | `nvmlDeviceGetClockInfo(SM)` MHz × 1e6 | ✅ |
| `DCGM_FI_DEV_MEMORY_CLOCK_HERTZ` | gauge | `nvmlDeviceGetClockInfo(MEM)` MHz × 1e6 | ✅ |
| `DCGM_FI_DEV_MEMORY_TEMP_CELSIUS` | gauge | field `NVML_FI_DEV_MEMORY_TEMP` | 🖥 |
| `DCGM_FI_DEV_GPU_TEMP_CELSIUS` | gauge | `nvmlDeviceGetTemperature(GPU)` | ✅ |
| `DCGM_FI_DEV_BOARD_POWER_WATTS` | gauge | `nvmlDeviceGetPowerUsage` mW ÷ 1000 (verify vs `NVML_FI_DEV_POWER_INSTANT`, which DCGM 4.7 may use) | ✅ |
| `DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL` | counter | `nvmlDeviceGetTotalEnergyConsumption` mJ ÷ 1000 | ✅ |
| `DCGM_FI_DEV_PCIE_REPLAY_TOTAL` | counter | `nvmlDeviceGetPcieReplayCounter` | ✅ |
| `DCGM_FI_DEV_GPU_UTIL_RATIO` | gauge | `nvmlDeviceGetUtilizationRates().gpu` ÷ 100 | ✅ |
| `DCGM_FI_DEV_MEMORY_UTIL_RATIO` | gauge | `nvmlDeviceGetUtilizationRates().memory` ÷ 100 | ✅ |
| `DCGM_FI_DEV_ENC_UTIL_RATIO` | gauge | `nvmlDeviceGetEncoderUtilization()[0]` ÷ 100 | ✅ |
| `DCGM_FI_DEV_DEC_UTIL_RATIO` | gauge | `nvmlDeviceGetDecoderUtilization()[0]` ÷ 100 | ✅ |
| `DCGM_FI_DEV_XID_ERROR` | gauge | NVML event set, `nvmlEventTypeXidCriticalError`; value = last XID code | ✅ (registers as user) |
| `DCGM_FI_DEV_FB_CAPACITY_BYTES` | gauge | `nvmlDeviceGetMemoryInfo(v2).total` | ✅ |
| `DCGM_FI_DEV_FB_FREE_BYTES` | gauge | `…v2.free` | ✅ |
| `DCGM_FI_DEV_FB_USED_BYTES` | gauge | `…v2.used` | ✅ |
| `DCGM_FI_DEV_FB_RESERVED_BYTES` | gauge | `…v2.reserved` | ✅ |
| `DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND` | gauge | fields `NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX/RX` (total across links; derive a rate if they are counters) | 🖥 |
| `DCGM_FI_DEV_VGPU_LICENSE_STATUS` | gauge | — | ❌ omit (NVML only reports "not vGPU" on bare metal; semantics unclear) |
| `DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL` | counter | `nvmlDeviceGetRemappedRows` | 🖥 |
| `DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL` | counter | `nvmlDeviceGetRemappedRows` | 🖥 |
| `DCGM_FI_DEV_ROW_REMAP_FAILED` | gauge | `nvmlDeviceGetRemappedRows` (isPending/failureOccurred) | 🖥 |
| `DCGM_FI_SYSTEM_DRIVER_VERSION` | label | `nvmlSystemGetDriverVersion` | ✅ |
| `DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO` | gauge | GPM `GRAPHICS_UTIL` (percent) ÷ 100 | ✅ |
| `DCGM_FI_PROF_TENSOR_UTIL_RATIO` | gauge | GPM `ANY_TENSOR_UTIL` ÷ 100 (DCGM's text says HMMA pipe; see note) | ✅ |
| `DCGM_FI_PROF_DRAM_UTIL_RATIO` | gauge | GPM `DRAM_BW_UTIL` ÷ 100 | ✅ |
| `DCGM_FI_PROF_PCIE_TX_BYTES` | gauge (B/s) | GPM `PCIE_TX_PER_SEC` → bytes/s (unit needs verifying) | ✅ |
| `DCGM_FI_PROF_PCIE_RX_BYTES` | gauge (B/s) | GPM `PCIE_RX_PER_SEC` → bytes/s (unit needs verifying) | ✅ |

**Rules:**

- **Units must match DCGM exactly.** Each conversion above has a unit test. Clocks are Hz, energy J,
  utilization a 0–1 ratio, memory bytes.
- At startup, probe each requested metric on each GPU. If NVML returns `NVML_ERROR_NOT_SUPPORTED`,
  log one INFO line per (metric, GPU) and never emit it for that GPU. Do not emit zero or NaN in
  its place.
- A metric that fails transiently at collection time (any error other than not-supported) is
  skipped for that cycle and logged at WARNING, rate-limited.
- **The GPM-backed `PROF_*` metrics are a deliberate decision.** GPM is NVIDIA's own NVML interface
  to the same hardware activity counters. It needs Hopper or newer (the probe confirmed it on
  Blackwell GeForce). It differs from DCGM's DCP sampling: GPM computes over the interval between
  two samples. The README documents this difference. On pre-Hopper GPUs these metrics are omitted.
  For tensor, choose between `ANY_TENSOR_UTIL` and `HMMA_TENSOR_UTIL` by checking which one matches
  DCGM 4.7's `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` definition. The probe found both supported.
- The GPM PCIe per-second units were not clear in the probe (`PCIE_TX_PER_SEC` read 1.125 at idle).
  The builder must confirm the unit (likely MiB/s) from the NVML headers or docs, or by
  cross-checking against `nvmlDeviceGetPcieThroughput` (KB/s) under load. If neither GPM nor that
  function can produce correct bytes/s, omit the metric.
- **XID:** a daemon thread waits on an NVML event set for all GPUs. It records the last XID code
  per GPU. `DCGM_FI_DEV_XID_ERROR` reports the last XID seen (0 before any event), matching DCGM's
  "value of the last XID error encountered". The builder must check 4.9.0's behavior for value
  before any XID, and for any extra labels.

### 4.4 Metric types and HELP text

- Type comes from the CSV (`gauge`/`counter`/`label`).
- `prometheus-client` appends `_total` to counters automatically. The 4.9.0 counter names already
  end in `_total`, and the output must match DCGM's exposition exactly. The builder must check how
  the client library renders these names (a custom collector yielding `CounterMetricFamily`
  normally strips and re-adds `_total` and also emits `_created`) and make the output identical to
  DCGM's. That includes no extra `_created` series, unless DCGM emits them.
- HELP text comes from the CSV third column, as in DCGM.
- Do not register the default `process_*` / `python_*` collectors on `/metrics`. DCGM does not
  expose Go runtime metrics by default (`--enable-exporter-metrics` is off). Supporting that flag to
  expose the Python equivalents is optional.

### 4.5 Custom counters files

`-f/--collectors PATH` accepts a CSV in DCGM's format: `name, type, help`, with `#` comments and
blank lines ignored and whitespace trimmed. Handling of each row:

- **Known and implemented:** collected.
- **Known DCGM name that we do not implement** (e.g. one of the many optional fields in the full
  DCGM field list): log a WARNING at startup and ignore the row.
- **Malformed row:** exit non-zero with a clear error, as DCGM does.

The set of implemented metrics may grow beyond the 27 defaults when NVML supports a DCGM field
cleanly. Examples: ECC counters, throttle reasons, PCIe link gen/width, fan speed, power limit,
and `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` equivalents. Use 4.9.0 names and units, and add each one
only after checking its exact name, unit, and meaning in 4.9.0. **Which extras to add is the
builder's call**, guided by what the dashboard needs (§6).

## 5. Runtime behavior

### 5.1 Command line

Use DCGM Exporter 4.9.0 flag names, aliases, and environment variables wherever the behavior is
the same:

| Flag | Alias | Env | Default | Notes |
|---|---|---|---|---|
| `--address` | `-a` | `DCGM_EXPORTER_LISTEN` | `:9400` | `host:port`, `[::]:port` for IPv6 |
| `--collectors` | `-f` | `DCGM_EXPORTER_COLLECTORS` (verify) | packaged `default-counters.csv` | |
| `--collect-interval` | `-c` | `DCGM_EXPORTER_INTERVAL` | `30000` (ms) | |
| `--devices` | `-d` | verify | all | DCGM's GPU selection syntax, minimal subset: `f` (all) or `g:0,1` (indices) |
| `--no-hostname` | | `DCGM_EXPORTER_NO_HOSTNAME` | off | |
| `--replace-blanks-in-model-name` | | verify | off | spaces in `modelName` → `_` |
| `--debug` | | verify | off | DEBUG logging |
| `--log-format` | | verify | text | `text`/`json` if cheap; otherwise text only |
| `--version` | | | | |

- Any other DCGM flag (Kubernetes, hostengine, config-file, web-config, …) **exits non-zero with a
  message** saying it is not supported in lite mode. It is never silently ignored.
- Argument parsing uses `argparse` from the standard library.

### 5.2 HTTP

- `GET /metrics`: Prometheus text exposition. Implemented with `prometheus-client`.
- `GET /health`: 200 when NVML is initialized and the last collection cycle succeeded for at least
  one GPU, 503 otherwise. Match the response body/format of 4.9.0's handler if it is simple.
- `GET /`: a small HTML page linking to `/metrics`, as DCGM does.
- Standard-library `http.server` (threaded) or `prometheus_client`'s WSGI app under `wsgiref`. No
  new dependencies.

### 5.3 Collection model

- A collection thread reads NVML every `collect-interval` and builds an immutable snapshot of all
  samples. `/metrics` serializes the latest snapshot, so a scrape never calls NVML. This matches
  DCGM's watch-and-cache model.
- GPM: keep the previous `nvmlGpmSample` per GPU. Each cycle, take a new sample and compute metrics
  over (previous, current). The first cycle emits no `PROF_*` values.
- XID: a separate event thread (§4.3).
- Startup order: initialize NVML, enumerate devices, probe support, start the threads, then start
  HTTP.
- **MIG mode** (decided by design author, consistent with 4.9.0's behavior): for a GPU with MIG
  enabled (`nvmlDeviceGetMigMode` current = enabled), emit only GPU-wide metrics, once per physical
  GPU, with no MIG labels. GPU-wide metrics are clocks, temperatures, power, energy, PCIe replay,
  framebuffer, row remap, and XID. **Omit** GPU/memory/encoder/decoder utilization and all `PROF_*`
  metrics for that GPU, and log one INFO line explaining why. `--devices` MIG selectors are
  rejected as unsupported.
- Errors:
  - NVML init failure at startup (no driver or no devices) → exit non-zero with a clear message.
  - A GPU that disappears or returns `GPU_IS_LOST` mid-run → drop that GPU's series, log an
    ERROR, keep serving the others.
- Shutdown: on SIGTERM/SIGINT, stop the threads, call `nvmlShutdown`, and exit 0.
- Logging: standard `logging` to stderr (journald picks it up under systemd).

## 6. Grafana dashboard

A **richer dashboard** whose job is to make the operational case. It answers: "Are our GPUs busy?
Are we wasting capacity? Are they healthy? What do they cost to run?"

Requirements:

- Uses **only DCGM Exporter 4.9.0 metric names and labels**, so it renders against the real
  exporter after the swap.
- Variables: `datasource`, `instance`, and `gpu`, filled the same way as NVIDIA's dashboard
  (`label_values(DCGM_FI_DEV_GPU_TEMP_CELSIUS, …)`), plus `Hostname`/`modelName` where useful.
- Panels that depend on datacenter-only metrics show "No data" gracefully on cards without them.
  No panel is broken.
- Exported JSON for current Grafana, at `grafana/dcgm-exporter-lite-dashboard.json`.
- The README links to NVIDIA's dashboard for comparison.

Suggested content (layout and panel choice are the builder's call):

- **Fleet overview row:** GPU count; average and max GPU utilization; GPUs idle (< 5% for 1h); total
  power now; energy over the selected range (kWh, from `increase()` of the energy counter); total
  and used framebuffer.
- **Utilization:** GPU util, tensor-core activity, graphics-engine activity, and DRAM bandwidth per
  GPU; a utilization heatmap across GPUs over time.
- **Memory:** FB used vs capacity per GPU, and headroom %.
- **Thermals & power:** GPU temp, memory temp (datacenter), power per GPU, SM/memory clocks.
- **Health:** last XID by GPU with a table of recent non-zero XIDs, PCIe replay rate, row-remap
  counters (datacenter).
- **I/O:** PCIe TX/RX, NVLink throughput (datacenter), encoder/decoder utilization.

## 7. Packaging & docs

- PyPI-ready package (`uv build`). Publishing to PyPI is **out of scope** for this version (left to
  the user).
- The packaged data file `default-counters.csv` is loaded via `importlib.resources`.
- README:
  - what it is and why (the bridge story)
  - install (pip/uv)
  - run
  - flags
  - the metric coverage table with ✅/🖥/❌
  - the GPM note
  - Prometheus scrape example
  - dashboard import
  - **"Swapping to DCGM Exporter"** checklist (same port, same names; remove the lite service)
  - known differences
- `contrib/systemd/dcgm-exporter-lite.service`: a systemd **user** unit, with instructions for
  `systemctl --user enable --now` and `loginctl enable-linger`.

## 8. Testing

- **Fake NVML:** a test double that replaces the `pynvml` functions used. It is configured per test
  with devices, values, and per-call not-supported or error responses. All unit tests run on a
  machine with no driver.
- **Golden exposition test:** a fixed fake GPU (or two) renders `/metrics` and compares it to a
  checked-in expected text taken from the 4.9.0 metric contract (names, label order, formats,
  units, HELP/TYPE lines).
- **Unit tests:** every unit conversion; CSV parsing (comments, whitespace, malformed rows, label
  type); not-supported omission; GPM first-cycle behavior; XID event recording; GPU-lost handling;
  CLI flag parsing and rejection of unsupported flags; `/health` states.
- **Hardware smoketest:** a script or `pytest` marker (`-m hardware`) that runs on `ryzen.local`.
  It starts the exporter, scrapes `/metrics`, and checks:
  - the ✅ metrics are present with plausible values (e.g. clocks > 1e8 Hz, 0 ≤ ratios ≤ 1,
    capacity ≈ 16 GiB)
  - the 🖥 metrics are absent
  
  It runs as the ordinary user.
- **Datacenter verification (manual, run by the user and reported back).** The datacenter-only
  metrics (🖥: memory temp, NVLink throughput, row remap), the MIG-mode behavior, and the older
  driver range (580+) are implemented and tested against the fake NVML only. These are **shipped,
  but marked "unverified on hardware"** in the README coverage table until the user reports
  results.
  - The builder writes `docs/testing/datacenter-verification.md`. It is a self-contained checklist
    someone else can run on an A100/H100/B200 host.
  - The checklist covers: setup commands; which metrics to expect; how to cross-check each value
    against `nvidia-smi -q` (or real DCGM Exporter, if available); how to create load (PCIe,
    NVLink, tensor) to sanity-check rates; MIG-mode checks; and exactly what output to capture and
    send back (e.g. a `/metrics` dump plus `nvidia-smi -q` output).
  - Where possible, include a small helper script that collects the evidence into one archive.
  - **NVLink in particular:** the checklist must settle whether the NVML throughput fields are
    cumulative counters or rates. Until that is confirmed, the implementation follows the NVML
    header documentation.
- **Optional compatibility check:** if someone has a host where real DCGM Exporter 4.9.0 runs,
  diff the metric and label sets. Out of scope to automate now.
- CI is the builder's call (ruff + pytest on 3.12–3.14 is the obvious default).

## 9. Left to the builder

- Internal module structure.
- Choice of HTTP server among standard-library options.
- Which optional DCGM fields beyond the 27 defaults to implement (§4.5).
- Dashboard layout and exact panel queries (§6).
- Whether to support `--log-format json` and `--enable-exporter-metrics`.
- CI setup.

## 10. References

- DCGM Exporter 4.9.0: https://github.com/NVIDIA/dcgm-exporter/tree/4.9.0
- 4.9.0 release notes (rename and unit changes): https://github.com/NVIDIA/dcgm-exporter/releases/tag/4.9.0
- DCGM Exporter metrics docs: https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html
- NVIDIA dashboard: https://github.com/NVIDIA/dcgm-exporter/tree/4.9.0/grafana
- NVML API reference (GPM, field values, events): https://docs.nvidia.com/deploy/nvml-api/

# DCGM Exporter 4.9.0 wire contract

Verified against `NVIDIA/dcgm-exporter` tag `4.9.0` (tarball commit `4e663a5`), plus the
`nvidia-ml-py==13.615.71` constants and the NVML API reference. Every later task cites this file.

Line references below are paths inside the 4.9.0 source tree.

## 1. GPU label names and order

`internal/pkg/rendermetrics/render_metrics.go:349-389` builds GPU labels in this fixed order:

1. `gpu` — device index as a decimal string, e.g. `"0"`
2. `UUID` — `nvmlDeviceGetUUID`
3. `pci_bus_id` — `nvmlDeviceGetPciInfo_v3().busId`; **included**
4. `device` — `"nvidia" + minor`, e.g. `"nvidia0"`
5. `modelName` — `nvmlDeviceGetName`
6. `hostname` — **lower-case**; omitted when empty (`--no-hostname`)

Then CSV `label` rows and per-metric attributes are appended in sorted key order
(`render_metrics.go:504-510`, `labelPairs.addMap` at `render_metrics.go:556-570`).

Consequences for lite:

- The spec table's `Hostname` capitalization is wrong for actual 4.9.0 output. Emit `hostname`.
- `pci_bus_id` is always present. On `ryzen.local`, `nvidia-smi --query-gpu=pci.bus_id` reports
  `00000000:2B:00.0`; NVML's bus ID is normalized to upper-case hex with the same shape.
- With `--replace-blanks-in-model-name`, 4.9.0 collapses whitespace and replaces each space with a
  **hyphen**, not an underscore (`gpu_collector.go:1191-1198`). Lite follows 4.9.0.
- The `label`-type counter's label name is its CSV field name
  (`gpu_collector.go:1204-1209`), so the packaged default emits
  `DCGM_FI_SYSTEM_DRIVER_VERSION="<driver>"`.

## 2. CSV `label` type rendering

A `label` row is never a metric family. `labelsFromValues` (`gpu_collector.go:1172-1179`) collects
the field value into a map; `addMetricLabel` (`gpu_collector.go:1204-1210`) stores it under the CSV
field name. The renderer appends it to every GPU metric. `internal/pkg/collector/base_collector.go:143`
does the same for the event collectors.

## 3. `DCGM_FI_DEV_XID_ERROR`

4.9.0 adds two attributes to the XID sample (`internal/pkg/collector/gpu_collector.go:1128-1134`):

- `err_code` — the decimal XID code
- `err_msg` — text from `internal/pkg/collector/xid_errors.go:20`; code `0` is `"No Error"`, unknown
  codes are `"unknown error"` (`gpu_collector.go:1160-1167`)

The metric value is the last XID code from the DCGM field. Before any XID the field value is `0`, so
4.9.0 emits a sample with `err_code="0"`, `err_msg="No Error"` when the field is available. The
4.9.0 field-support probe for `ryzen.local` records this field as N/A on that GeForce card
(`tests/internal/metriccontract/testdata/field-support-document.json`), which is why lite gates the
whole family on the NVML event source being available and otherwise omits it.

Lite: when the XID watcher is available and no event has arrived, emit `0.0` with `err_code="0"`,
`err_msg="No Error"`. When the watcher is unavailable, probe disables the metric for every GPU.

## 4. Counter rendering

`tests/internal/metriccontract/metrics_test.go:71-73` and
`tests/internal/exportercontract/contracts_test.go:44-46` show the exact shape:

```
# HELP DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL Total energy.
# TYPE DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL counter
DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL{gpu="0",...} 196764
```

The family name already contains `_TOTAL`. No `_total_total` suffix and no `_created` series.
`prometheus_client.CounterMetricFamily` appends `_total` again, so lite must build counters with
`prometheus_client.core.Metric(..., typ="counter")` and explicit sample names.

## 5. `/health` and `/`

- 4.9.0 `Health` (`internal/pkg/server/server.go:616-642`) writes body `OK` and returns **200**
  whenever the registry is non-nil, including before the first scrape; with a nil registry it still
  returns 200 and writes `OK - reload in progress`.
- Lite intentionally differs per spec §5.2 and Review Focus 1: 200 `OK` only after a snapshot with
  at least one healthy GPU; before that, 503 with body `not ready\n`.
- `/` (`server.go:126-147`) is a small HTML page titled `GPU Exporter` with links to `./metrics` and
  `./health`.

## 6. Flags, env vars, device syntax

From `pkg/cmd/app.go`:

| Flag | Env | Line |
|---|---|---|
| `--collectors` / `-f` | `DCGM_EXPORTER_COLLECTORS` | 149 |
| `--address` / `-a` | `DCGM_EXPORTER_LISTEN` | 166 |
| `--collect-interval` / `-c` | `DCGM_EXPORTER_INTERVAL` | 208 |
| `--devices` / `-d` | `DCGM_EXPORTER_DEVICES_STR` | 320 |
| `--no-hostname` / `-n` | `DCGM_EXPORTER_NO_HOSTNAME` | 329 |
| `--replace-blanks-in-model-name` / `-rbmn` | `DCGM_EXPORTER_REPLACE_BLANKS_IN_MODEL_NAME` | 374 |
| `--debug` | `DCGM_EXPORTER_DEBUG` | 383 |
| `--log-format` | `DCGM_EXPORTER_LOG_FORMAT` | 418 |

`--devices` selector syntax: `f` selects every GPU (the default, `Flex: true` at
`internal/pkg/appconfig/defaults.go:40`; `f` is also the canonical form at
`internal/pkg/devicewatcher/device_watcher.go:844`). `g:0,2` selects GPU indices 0 and 2. `i:` selects
MIG instances and is rejected by lite. The rest of 4.9.0's selector grammar is out of scope.

## 7. Power and tensor sources

- `DCGM_FI_DEV_BOARD_POWER_WATTS` is the 4.9.0 rename of `DCGM_FI_DEV_POWER_USAGE` field 155
  (`internal/pkg/counters/default_counter_sources_test.go:145`). DCGM's power-usage field is backed
  by `nvmlDeviceGetPowerUsage` (mW), not `NVML_FI_DEV_POWER_INSTANT`. Lite uses `power_mw() / 1000`.
- `DCGM_FI_PROF_TENSOR_UTIL_RATIO` is the rename of `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, whose help is
  explicitly "tensor (HMMA) pipe" (`default_counter_sources_test.go:177`). Lite uses GPM
  `HMMA_TENSOR_UTIL` (id 7), not `ANY_TENSOR_UTIL`. GPM utilization values are percentages
  (`0.0-100.0`), so divide by 100.

## 8. GPM PCIe units

NVML documents `NVML_GPM_METRIC_PCIE_TX_PER_SEC` and `..._RX_PER_SEC` as **MiB/sec**
(`nvidia-ml-py` `pynvml.py:6629-6630`; NVML API "GPM Enums"). Lite multiplies by `1024 * 1024`
to emit bytes/sec for `DCGM_FI_PROF_PCIE_TX_BYTES` / `..._RX_BYTES`.

4.9.0 passes DCGM's value through with no scaling: `internal/pkg/signalcatalog/catalog.go:322-341`
binds both fields with `ConversionIdentity`. Any TX correction would have to live in the DCGM host
engine's profiling module, not in the exporter. The open `NVIDIA/DCGM` repo has only field definitions for
`DCGM_FI_PROF_PCIE_*` and a skeleton `modules/profiling`, so the source does not settle it. 4.9.0 also rejects the older
`DCGM_FI_DEV_PCIE_TX/RX_THROUGHPUT` fields and names the `PROF_` fields as their replacement
(`internal/pkg/counters/counter_config.go:406-416`). The CSV help says the rate includes "both
protocol headers and data payloads".

## 9. NVML NVLink throughput fields

`pynvml.py:688-691` names fields 138/139 "NVLink Throughput Counters" with unit **KiB**, and
`NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX/RX`. They are cumulative counters, not rates. DCGM converts
them to an instantaneous MiB/s rate by differencing (`NVIDIA/dcgm-exporter` PR #658, describing
`ReadAndCacheNvLinkBandwidth`).

Lite: first cycle raises `NoValueYet`; later cycles emit
`(Δtx + Δrx) * 1024 / elapsed_s` bytes/sec (measured monotonic time between readings) for `DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND`.
The fields aggregate all links, matching the family's "across all lanes" meaning. Marked
**unverified on hardware** until the datacenter checklist confirms both the counter semantics and
the aggregate scope.

## 10. Metric coverage note

The packaged default CSV has 27 enabled rows. Lite implements 25 non-label, non-vGPU metrics;
`DCGM_FI_DEV_VGPU_LICENSE_STATUS` is deliberately omitted per spec §4.3.

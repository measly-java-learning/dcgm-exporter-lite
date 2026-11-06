# dcgm-exporter-lite

A pip-installable, unprivileged Python exporter that serves NVIDIA GPU metrics with **DCGM
Exporter 4.9.0** metric names, types, units, and labels.

It reads NVML through `nvidia-ml-py` and serves Prometheus text through `prometheus-client`. It
needs no root, no DCGM install, no container, and no CDI hook. The intended use is a bridge: show
the operational value of GPU metrics now, then swap in the real DCGM Exporter without changing
scrape configs, dashboards, or alerts.

## Requirements

- NVIDIA driver **580 or newer** (older drivers run with a warning; some metrics are missing).
- Python **3.12–3.14**.
- Linux with NVML available to the running user. `/dev/nvidia*` is normally world-readable.

## Install

The package is not published to PyPI. Install a wheel or the source tree:

```bash
# from a built wheel
python3 -m venv .venv
.venv/bin/pip install dist/dcgm_exporter_lite-0.1.0-py3-none-any.whl

# or from a checkout
python3 -m venv .venv
.venv/bin/pip install .
```

With `uv`:

```bash
uv venv
uv pip install dist/dcgm_exporter_lite-0.1.0-py3-none-any.whl
```

## Quick start

```bash
.venv/bin/dcgm-exporter-lite -a :9400 -c 30000
curl -s http://localhost:9400/health          # OK once the first collection succeeds
curl -s http://localhost:9400/metrics
```

`/metrics` serializes the latest immutable snapshot; a scrape never calls NVML directly.
`/health` returns 503 before the first successful collection, 200 afterwards, and 503 again if no
fresh collection has completed within three collect intervals (minimum 10 s), e.g. when an NVML call hangs.

### Flags

| Flag | Alias | Environment | Default |
|---|---|---|---|
| `--address` | `-a` | `DCGM_EXPORTER_LISTEN` | `:9400` |
| `--collectors` | `-f` | `DCGM_EXPORTER_COLLECTORS` | packaged 4.9.0 `default-counters.csv` |
| `--collect-interval` | `-c` | `DCGM_EXPORTER_INTERVAL` | `30000` ms |
| `--devices` | `-d` | `DCGM_EXPORTER_DEVICES_STR` | all GPUs |
| `--no-hostname` | `-n` | `DCGM_EXPORTER_NO_HOSTNAME` | off |
| `--replace-blanks-in-model-name` | `-rbmn` | `DCGM_EXPORTER_REPLACE_BLANKS_IN_MODEL_NAME` | off |
| `--debug` | | `DCGM_EXPORTER_DEBUG` | off |
| `--log-format` | | `DCGM_EXPORTER_LOG_FORMAT` | `text` (`text`/`json`) |
| `--version` | | | |

`--devices` accepts `f` (all GPUs) or `g:0,2`. MIG selectors (`i:`) are rejected. Flags beat
environment variables; environment variables beat defaults. Other DCGM Exporter flags
(`--kubernetes`, `--config-file`, `--web-config-file`, `--remote-hostengine-info`, …) exit with a
message instead of being silently ignored.

## Metric coverage

Verified on an NVIDIA GeForce RTX 5060 Ti (driver 615.71.10, non-root). Datacenter-only metrics are
implemented and unit-tested against the fake NVML; they are **unverified on hardware** until the
checklist in [`docs/testing/datacenter-verification.md`](docs/testing/datacenter-verification.md)
is run on an A100/H100/B200 host.

- ✅ verified on hardware
- ◐ present on hardware with in-range idle values; not yet observed under a real event or load
- 🖥 datacenter-only, **unverified on hardware**
- ❌ omitted

| DCGM 4.9.0 metric | Type | Status | Source / reason |
|---|---|---|---|
| `DCGM_FI_DEV_SM_CLOCK_HERTZ` | gauge | ✅ | `nvmlDeviceGetClockInfo(SM)` × 1e6 |
| `DCGM_FI_DEV_MEMORY_CLOCK_HERTZ` | gauge | ✅ | `nvmlDeviceGetClockInfo(MEM)` × 1e6 |
| `DCGM_FI_DEV_MEMORY_TEMP_CELSIUS` | gauge | 🖥 | NVML field `NVML_FI_DEV_MEMORY_TEMP` |
| `DCGM_FI_DEV_GPU_TEMP_CELSIUS` | gauge | ✅ | `nvmlDeviceGetTemperature(GPU)` |
| `DCGM_FI_DEV_BOARD_POWER_WATTS` | gauge | ✅ | `nvmlDeviceGetPowerUsage` mW ÷ 1000 |
| `DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL` | counter | ✅ | `nvmlDeviceGetTotalEnergyConsumption` mJ ÷ 1000 |
| `DCGM_FI_DEV_PCIE_REPLAY_TOTAL` | counter | ✅ | `nvmlDeviceGetPcieReplayCounter` |
| `DCGM_FI_DEV_GPU_UTIL_RATIO` | gauge | ✅ | utilization `.gpu` ÷ 100 |
| `DCGM_FI_DEV_MEMORY_UTIL_RATIO` | gauge | ✅ | utilization `.memory` ÷ 100 |
| `DCGM_FI_DEV_ENC_UTIL_RATIO` | gauge | ✅ | encoder utilization ÷ 100 |
| `DCGM_FI_DEV_DEC_UTIL_RATIO` | gauge | ✅ | decoder utilization ÷ 100 |
| `DCGM_FI_DEV_XID_ERROR` | gauge | ◐ | XID event set (registration verified as non-root; no XID observed yet); last code per GPU, plus `err_code`/`err_msg` labels |
| `DCGM_FI_DEV_FB_CAPACITY_BYTES` | gauge | ✅ | `nvmlDeviceGetMemoryInfo(v2).total` |
| `DCGM_FI_DEV_FB_FREE_BYTES` | gauge | ✅ | `.free` |
| `DCGM_FI_DEV_FB_USED_BYTES` | gauge | ✅ | `.used` |
| `DCGM_FI_DEV_FB_RESERVED_BYTES` | gauge | ✅ | `.reserved` |
| `DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND` | gauge | 🖥 | cumulative KiB fields → rate across all links |
| `DCGM_FI_DEV_VGPU_LICENSE_STATUS` | gauge | ❌ | NVML only reports “not vGPU” on bare metal; meaning is unclear |
| `DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL` | counter | 🖥 | `nvmlDeviceGetRemappedRows` |
| `DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL` | counter | 🖥 | `nvmlDeviceGetRemappedRows` |
| `DCGM_FI_DEV_ROW_REMAP_FAILED` | gauge | 🖥 | `nvmlDeviceGetRemappedRows` failure flag |
| `DCGM_FI_SYSTEM_DRIVER_VERSION` | label | ✅ | driver version as a label on every metric |
| `DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO` | gauge | ◐ | GPM `GRAPHICS_UTIL` ÷ 100 |
| `DCGM_FI_PROF_TENSOR_UTIL_RATIO` | gauge | ◐ | GPM `HMMA_TENSOR_UTIL` ÷ 100 |
| `DCGM_FI_PROF_DRAM_UTIL_RATIO` | gauge | ◐ | GPM `DRAM_BW_UTIL` ÷ 100 |
| `DCGM_FI_PROF_PCIE_TX_BYTES` | gauge | ◐ | GPM `PCIE_TX_PER_SEC` MiB/s × 2²⁰ |
| `DCGM_FI_PROF_PCIE_RX_BYTES` | gauge | ◐ | GPM `PCIE_RX_PER_SEC` MiB/s × 2²⁰ |

The packaged default is the 4.9.0 `default-counters.csv` (27 rows). A custom `-f` file may add
other rows; names this exporter does not implement are logged once and ignored.

### GPM and the `PROF_*` metrics

GPM is NVIDIA’s NVML interface to the same hardware activity counters. It exists on Hopper and
newer (including Blackwell GeForce). The exporter takes two samples and computes metrics over the
interval, so the first collection cycle has no `PROF_*` values. This differs from DCGM’s DCP
sampling, and is the one place where values can differ from the real exporter. On Volta and Ampere
these metrics are absent.

### Labels

Every GPU metric carries, in order:

```
gpu, UUID, pci_bus_id, device, modelName, hostname, DCGM_FI_SYSTEM_DRIVER_VERSION
```

XID samples append `err_code` and `err_msg`. `--no-hostname` omits `hostname`.
`--replace-blanks-in-model-name` collapses whitespace and replaces spaces with `-`, matching 4.9.0.

### MIG

For a GPU with MIG enabled, only GPU-wide metrics are exported once per physical GPU, with no MIG
labels: clocks, temperatures, power, energy, PCIe replay, framebuffer, row remap, and XID.
Utilization ratios and all `PROF_*` metrics are omitted for that GPU, with one INFO log line.
MIG instance labels and `i:` device selectors are out of scope.

## Prometheus

```yaml
scrape_configs:
  - job_name: dcgm-exporter-lite
    scrape_interval: 15s
    static_configs:
      - targets: ["gpu-host:9400"]
```

Metric and label names match DCGM Exporter 4.9.0. `/metrics` does not expose Python or process
metrics.

## Grafana

Import [`grafana/dcgm-exporter-lite-dashboard.json`](grafana/dcgm-exporter-lite-dashboard.json) in
Grafana (Dashboards → New → Import), then pick your Prometheus data source. The dashboard uses
only 4.9.0 metric names and the `instance`/`gpu` labels, so it also works against the real DCGM
Exporter. Datacenter-only panels show “Not reported by this GPU” on cards that lack the source.

For comparison, NVIDIA’s 4.9.0 dashboard is at
<https://github.com/NVIDIA/dcgm-exporter/tree/4.9.0/grafana>.

## systemd user service

```bash
# The unit expects the entry point at ~/.local/bin. If you installed into a
# venv, link it there first:
mkdir -p ~/.local/bin
ln -sf /path/to/.venv/bin/dcgm-exporter-lite ~/.local/bin/dcgm-exporter-lite

mkdir -p ~/.config/systemd/user
cp contrib/systemd/dcgm-exporter-lite.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now dcgm-exporter-lite
loginctl enable-linger "$USER"
```

The unit runs `%h/.local/bin/dcgm-exporter-lite` and restarts on failure. Logs go to the user
journal (`journalctl --user -u dcgm-exporter-lite`).

## Swapping to DCGM Exporter

1. Keep the same listen port and `/metrics` path.
2. Stop the lite service:
   `systemctl --user disable --now dcgm-exporter-lite`
3. Start DCGM Exporter 4.9.0 on the same address (container or package, with its usual
   privileges).
4. Confirm Prometheus reports the target UP and the job still has its historical series.
5. Confirm the dashboard and alert rules need no edits. They should not: the metric names,
   types, units, and labels are the same.
6. Verify the driver-version label, `pci_bus_id`, and the XID `err_code`/`err_msg` labels. The
   lite exporter emits them in 4.9.0 form.
7. Datacenter-only series (memory temperature, NVLink, row remap) start appearing after the swap;
   the dashboard panels already render them.
8. If you use MIG, expect DCGM to add `GPU_I_ID`/`GPU_I_PROFILE` labels. The lite dashboard does
   not select on them, so it keeps working.

## Differences from DCGM Exporter

- **No MIG instance metrics or labels.** MIG-enabled GPUs export only GPU-wide metrics.
- **No Kubernetes, HPC-job, NVSwitch, CPU, or per-link NVLink metrics/labels.**
- **`PROF_*` uses GPM, not DCP.** Hopper+ only; two-sample rates; first cycle has no values.
- **No DCGM health, policy, clock-event, or `DCGM_EXP_*` metrics.**
- **No YAML config, ConfigMap source, TLS/web config, systemd socket activation, or container
  image.**
- **Minimal `--devices` grammar** (`f` or `g:0,2`); MIG selectors are rejected.
- **Unsupported DCGM flags exit non-zero** with a README pointer instead of being ignored.
- **`/health` returns 503 before the first successful collection, and when collection goes stale.**
  4.9.0 returns 200 earlier.
- **Logging content/format differs**, though `--log-format json` produces one JSON object per line.

## Development

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest
```

Hardware tests are marked `hardware` and excluded by default:

```bash
scripts/smoketest-remote.sh ryzen.local
```

Datacenter-only metrics need the manual checklist in
[`docs/testing/datacenter-verification.md`](docs/testing/datacenter-verification.md). The verified
4.9.0 wire contract is recorded in
[`docs/notes/dcgm-4.9.0-contract.md`](docs/notes/dcgm-4.9.0-contract.md).

## License

MIT for this project. The vendored `default-counters.csv` retains NVIDIA’s header and is from
DCGM Exporter 4.9.0.

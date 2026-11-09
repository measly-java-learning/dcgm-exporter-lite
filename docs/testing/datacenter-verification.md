# Datacenter verification checklist

This checklist verifies the metrics that cannot be exercised on a GeForce card: memory
temperature, NVLink throughput, row remapping, and the GPM-backed `PROF_*` families. It also
verifies MIG handling and the NVLink counter-vs-rate assumption.

Run it on an A100/H100/B200-class host with NVIDIA driver 580 or newer, Python 3.12+, and an
ordinary non-root account. `nvidia-smi` and `curl` must be installed. Do not run the load
generators on a production GPU without approval.

## 1. Install and start

```bash
python3 -m venv ~/dcgm-lite-verify
~/dcgm-lite-verify/bin/pip install dcgm-exporter-lite   # or a local wheel under test

~/dcgm-lite-verify/bin/dcgm-exporter-lite -a 127.0.0.1:9400 -c 5000
```

In another shell:

```bash
curl -fsS http://127.0.0.1:9400/health          # expect 200 OK
curl -fsS http://127.0.0.1:9400/metrics > /tmp/baseline.prom
```

Record the following once, before load:

```bash
nvidia-smi -q > /tmp/nvidia-smi-q-before.txt
nvidia-smi topo -m > /tmp/nvidia-smi-topo.txt
```

## 2. Metric presence and cross-checks

Use the GPU UUID from `nvidia-smi --query-gpu=uuid --format=csv,noheader` to pick the right
samples. Compare each value with the command in the third column.

| Metric | Expected | Cross-check | Pass/fail |
|---|---|---|---|
| `DCGM_FI_DEV_MEMORY_TEMP_CELSIUS` | present on datacenter GPUs | `nvidia-smi -q -d TEMPERATURE` → `Memory Current Temp` | Pass if within 5 °C. Fail if absent, or if it matches GPU temp instead. |
| `DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND` | present on NVLink systems | `nvidia-smi -q -d NVLINK` (or `nvidia-smi nvlink -s`) | See §5. Pass only if the value is a rate and covers all links. |
| `DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL` | present | `nvidia-smi -q -d ROW_REMAPPER` → correctable rows | Exact match. |
| `DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL` | present | same section → uncorrectable rows | Exact match. |
| `DCGM_FI_DEV_ROW_REMAP_FAILED` | present | same section → remapping failure flag | `1` iff the failure flag is set, else `0`. |
| `DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO` | present on Hopper+ | `nvidia-smi dmon -s u` GPU utilization, or `dcgmi dmon` if available | `0..1`; rises under a graphics/compute load. |
| `DCGM_FI_PROF_TENSOR_UTIL_RATIO` | present on Hopper+ | compare with `nvidia-smi dmon` / `dcgmi dmon` tensor counters if available | `0..1`; clearly rises during the FP16/BF16 matmul load in §4. |
| `DCGM_FI_PROF_DRAM_UTIL_RATIO` | present on Hopper+ | `nvidia-smi dmon -s m` memory utilization; `dcgmi dmon` DRAM counters | `0..1`; rises during a memory-bound copy. |
| `DCGM_FI_PROF_PCIE_TX_BYTES` / `..._RX_BYTES` | present on Hopper+ | `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` for link state; compare with a host↔device copy rate | Rises during §4 PCIe load; not negative. |
| `DCGM_FI_DEV_XID_ERROR` | present with `err_code`/`err_msg` | no safe way to inject on production | Pass if the series exists with code `0` and `err_msg="No Error"`; do not trigger XIDs deliberately. |

The driver-version label `DCGM_FI_SYSTEM_DRIVER_VERSION` must equal
`nvidia-smi --query-gpu=driver_version --format=csv,noheader`.

## 3. Load generation

Use one or more of these while scraping. The goal is to make the relevant ratios and rates
leave zero.

### Tensor / graphics load (PyTorch)

```bash
python3 - <<'PY'
import time, torch
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
end = time.time() + 90
while time.time() < end:
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
PY
```

This exercises HMMA tensor pipes. Observe `DCGM_FI_PROF_TENSOR_UTIL_RATIO` and
`DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO` during the run.

### PCIe load

```bash
python3 - <<'PY'
import time, torch
host = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
dev = torch.empty_like(host, device="cuda")
end = time.time() + 60
while time.time() < end:
    dev.copy_(host)   # host -> device
    host.copy_(dev)   # device -> host
    torch.cuda.synchronize()
PY
```

`DCGM_FI_PROF_PCIE_TX_BYTES` and `..._RX_BYTES` should both be non-zero.

### NVLink peer copy (multi-GPU)

```bash
python3 - <<'PY'
import time, torch
assert torch.cuda.device_count() >= 2
src = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
dst = torch.empty_like(src, device="cuda:1")
end = time.time() + 60
while time.time() < end:
    dst.copy_(src)
    torch.cuda.synchronize()
PY
```

If peer access is unavailable, use `cuda-samples`' `p2pBandwidthLatencyTest` instead.

### cuda-samples alternative

```bash
git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/bandwidthTest && make && ./bandwidthTest
cd ../../5_Domain_Specific/p2pBandwidthLatencyTest && make && ./p2pBandwidthLatencyTest
```

## 4. MIG behavior

MIG setup needs root. The exporter itself must still run as your ordinary user.

```bash
# Choose a profile that exists for the GPU; list them with:
nvidia-smi mig -lgip

sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi -i 0 mig -cgi 1g.10gb -C    # example; use a profile printed above
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader
```

Restart the exporter and scrape it. For GPU 0, expect **only GPU-wide** metrics:

- clocks, GPU/memory temperature, power, energy, PCIe replay
- framebuffer capacity/free/used/reserved
- row remap
- XID

Expect **no** `*_UTIL_RATIO` and no `DCGM_FI_PROF_*` samples for that GPU. Other GPUs must keep
their full set. The exporter is run as a normal user, so `--devices` still uses physical GPU
indices.

Restore the host:

```bash
sudo nvidia-smi -i 0 -mig 0
```

## 5. NVLink: counter or rate?

NVML fields 138/139 (`NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX/RX`) are documented as cumulative
KiB counters. dcgm-exporter-lite samples them and emits
`(Δtx + Δrx) * 1024 / elapsed_s` bytes/sec, where `elapsed_s` is the measured time between readings. Verify that assumption.

1. At idle, capture two `nvidia-smi -q -d NVLINK` outputs 15 s apart. The relevant counters must
   not increase without traffic.
2. Start the NVLink peer copy from §3 and capture the exporter's
   `DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND` plus two `nvidia-smi -q -d NVLINK` outputs
   15 s apart.
3. **Pass** if:
   - the exporter emits no value on its first collection cycle, then a positive rate;
   - the rate is within 20% of `(Δtx + Δrx) * 1024 / 15` using NVML's `DATA_TX`/`DATA_RX` values
     (or the corresponding `nvidia-smi nvlink` totals);
   - the rate falls back toward zero within one collection interval after the copy stops;
   - the value covers all links, matching the family's "across all lanes" help text.
4. **Fail** if the exporter reports a raw cumulative counter, a per-link value, a negative value,
   or a value that does not change with load. Capture the two `nvidia-smi` snapshots and the three
   scrapes; the fix is in `metrics.py`'s NVLink conversion.

## 6. What to send back

Run the evidence collector while the exporter is serving:

```bash
./scripts/collect-datacenter-evidence.sh http://127.0.0.1:9400
```

It writes `dcgm-lite-evidence-<host>-<date>.tar.gz` containing:

- `nvidia-smi-q.txt`
- `nvidia-smi-topo.txt`
- `driver-version.txt`, `uname.txt`, `exporter-version.txt`
- `metrics/metrics-1.prom`, `metrics-2.prom`, `metrics-3.prom` (30 s apart)

Send that archive plus a short note covering:

- GPU model and count, driver version, MIG mode
- which load generators ran and for how long
- pass/fail for each row in §2 and for §5
- any metric that was absent, had the wrong unit, or disagreed with `nvidia-smi`
- the exact `nvidia-smi` section used for each datacenter-only comparison

## Verification results

### 2026-11-09: A100 and H100 PCIe (v1.0.0)

Run by the maintainer on one A100 host and one H100 PCIe host. The H100 is a PCIe-only part with
no NVLink bridges, so §5 could not be run.

| Check | Result |
|---|---|
| `DCGM_FI_DEV_MEMORY_TEMP_CELSIUS` | Pass: agrees with `nvidia-smi` |
| `DCGM_FI_DEV_ROW_REMAP_*` | Pass: matches `nvidia-smi -q -d ROW_REMAPPER` |
| `DCGM_FI_DEV_XID_ERROR` | Pass: `0` with `err_msg="No Error"`; no XID was triggered |
| `DCGM_FI_SYSTEM_DRIVER_VERSION` label | Pass: matches `nvidia-smi` |
| `DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL` (A100) | Pass: `rate()` matches the `BOARD_POWER_WATTS` gauge |
| `DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO`, `..._TENSOR_UTIL_RATIO`, `..._DRAM_UTIL_RATIO` | Pass: respond to load |
| `DCGM_FI_PROF_PCIE_TX_BYTES` / `..._RX_BYTES` | Pass: respond to load; within 2% of `nvidia-smi dmon` over a sustained 60 s copy |

The `PROF_*` rows apply to the H100 only. They are GPM-backed, and the A100 (Ampere) has no GPM, so
it emits no `PROF_*` samples by design.

| Not covered | |
|---|---|
| §5 NVLink counter-vs-rate | Not run: no NVLink on either host |
| §4 MIG behavior | Not reported |

Still open: `DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND` (§5) needs an NVLink-connected
SXM/HGX host. MIG handling (§4) can be checked on either of these hosts, but needs root to
enable MIG. A real XID event has not been observed.

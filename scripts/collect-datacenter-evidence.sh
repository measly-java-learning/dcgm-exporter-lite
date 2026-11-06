#!/usr/bin/env bash
# Collect the evidence needed to verify datacenter-only metrics.
set -euo pipefail

ADDR="${1:-http://127.0.0.1:9400}"
HOST="$(hostname -s)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="dcgm-lite-evidence-${HOST}-${STAMP}.tar.gz"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

mkdir -p "${WORK}/metrics"

nvidia-smi -q > "${WORK}/nvidia-smi-q.txt"
nvidia-smi topo -m > "${WORK}/nvidia-smi-topo.txt"
nvidia-smi --query-gpu=driver_version --format=csv,noheader > "${WORK}/driver-version.txt"
uname -a > "${WORK}/uname.txt"
{
  if command -v dcgm-exporter-lite >/dev/null 2>&1; then
    dcgm-exporter-lite --version
  elif [ -x ./.venv/bin/dcgm-exporter-lite ]; then
    ./.venv/bin/dcgm-exporter-lite --version
  elif python3 -c 'import dcgm_exporter_lite' >/dev/null 2>&1; then
    python3 -c 'from importlib.metadata import version; print(version("dcgm-exporter-lite"))'
  else
    echo "unknown"
  fi
} > "${WORK}/exporter-version.txt" 2>&1

for i in 1 2 3; do
  echo "scrape ${i} of 3 from ${ADDR}" >&2
  curl -fsS "${ADDR}/metrics" > "${WORK}/metrics/metrics-${i}.prom"
  if [ "${i}" -lt 3 ]; then
    sleep 30
  fi
done

tar -czf "${OUT}" -C "${WORK}" .
echo "${PWD}/${OUT}"

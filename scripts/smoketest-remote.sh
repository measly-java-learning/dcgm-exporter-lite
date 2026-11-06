#!/usr/bin/env bash
# Run the hardware smoketest on a remote GPU host.
set -euo pipefail

HOST="${1:?usage: $0 HOST}"
DEST="dcgm-exporter-lite-smoke"

rsync -az --delete \
  --exclude '.venv' \
  --exclude '.git' \
  --exclude 'dist' \
  --exclude '__pycache__' \
  ./ "${HOST}:${DEST}/"

ssh "${HOST}" bash -s -- "${DEST}" <<'REMOTE'
set -euo pipefail
cd "$HOME/$1"
python3 -m venv .venv
.venv/bin/pip install --disable-pip-version-check -q -e . pytest
.venv/bin/pytest -o addopts='' -m hardware -v
REMOTE

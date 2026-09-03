#!/bin/sh
set -eu

if [ ! -f "${DB_PATH:-/data/data.db}" ]; then
  echo "Database not found. Provision the private evaluation database on the mounted volume." >&2
  exit 1
fi
mkdir -p "${AUDIO_DIR:-/data/audio_fma}"
exec gunicorn -b 0.0.0.0:5000 eval_server:app

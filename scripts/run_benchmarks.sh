#!/usr/bin/env bash
# Uso: bash scripts/run_benchmarks.sh [corridas]
set -euo pipefail
RUNS=${1:-30}

mkdir -p results output
make hough_global

{
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | tail -n 1
  date
} > results/system_info.txt

rm -f results/bitacora_global.csv
./hough_global runway.pgm --runs "$RUNS" --csv results/bitacora_global.csv

#!/usr/bin/env bash
# Uso: bash scripts/run_benchmarks.sh [rondas] [corridas]
set -euo pipefail
ROUNDS=${1:-3}
RUNS=${2:-10}

mkdir -p results output
make all variants optimizaciones

{
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | tail -n 1
  grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs
  date
} > results/system_info.txt

# Intercaladas: nadie sale favorecido
rm -f results/bitacora_{global,const,shared}.csv
for r in $(seq "$ROUNDS"); do
  for v in global const shared; do
    ./hough_$v runway.pgm --runs "$RUNS" --csv results/bitacora_$v.csv > results/log_$v.txt
  done
done
grep -h "CPU:" results/log_*.txt | head -n 1 >> results/system_info.txt

./variants runway.pgm --runs 30 | tee results/variantes.txt
./optimizaciones runway.pgm --runs 30 | tee results/optimizaciones.txt
# Sin python: correr en host
if command -v python3 >/dev/null; then
  python3 scripts/compare_outputs.py | tee results/compare_outputs.txt
fi

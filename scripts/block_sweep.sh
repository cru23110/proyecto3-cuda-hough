#!/usr/bin/env bash
# Uso: bash scripts/block_sweep.sh [corridas]
set -euo pipefail
RUNS=${1:-30}
CSV=results/bitacora_block_sweep.csv

mkdir -p results output/sweep
make all
rm -f "$CSV" results/ocupancia.txt

for b in 128 256 512 1024; do
  for v in global const shared; do
    occ=$(./hough_$v runway.pgm --runs "$RUNS" --block "$b" --csv "$CSV" --out output/sweep/runway \
          | grep "Ocupancia" | cut -d: -f2)
    echo "$v,$b,$occ" | tee -a results/ocupancia.txt
  done
done
rm -rf output/sweep

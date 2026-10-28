#!/usr/bin/env bash
# Full scene aggregation with automatic resume (each finished scene is a parquet file; a crash
# - typically a transient EE error or the machine running out of memory - is simply re-run).
# Usage: bash revision/02_edge_aggregation/run_full.sh 2024 [1985 2002]
cd "$(dirname "$0")/../.."
export GDAL_CACHEMAX=64
for year in "$@"; do
  until [ -f "revision/data/geometry/$year/qc.json" ]; do sleep 60; done
  for attempt in $(seq 1 40); do
    python revision/02_edge_aggregation/aggregate_scenes.py --year "$year" --all --workers 4 \
      >> "revision/data/full_run_$year.log" 2>&1 && break
    echo "attempt $attempt failed $(date)" >> "revision/data/full_run_$year.log"
    sleep 60
  done
  echo "YEAR $year FINISHED $(date)" >> "revision/data/full_run_$year.log"
done

#!/usr/bin/env bash
# Rebuild one year from MapBiomas Collection 10.1 (decision D-11) and re-run the downstream steps.
# Products built from the Collection 9 raster are kept (renamed *_c9), never deleted.
# Usage: bash revision/02_edge_aggregation/rebuild_year_c10.sh 2002
set -u
cd "$(dirname "$0")/../.."
export GDAL_CACHEMAX=64
Y=$1
D=revision/data
log() { echo "=== $* $(date)"; }

log "fetch + label $Y"
python revision/01_data_preparation/rebuild_class_raster.py --year "$Y" || exit 1

log "preserve Collection 9 products"
for p in "$D/geometry/$Y" "$D/scenes/$Y" "$D/tables/full_polygon_ring_$Y.parquet" \
         "$D/covariates/dem_zones_$Y.parquet" "$D/covariates/class_stability_$Y.parquet"; do
  [ -e "$p" ] && [ ! -e "${p%.parquet}_c9${p##*${p%.parquet}}" ] && mv "$p" "${p%.parquet}_c9${p##*${p%.parquet}}"
done

log "edge geometry $Y"
python revision/01_data_preparation/build_edge_geometry.py --year "$Y" || exit 1

log "scene aggregation $Y"
for attempt in $(seq 1 20); do
  python revision/02_edge_aggregation/aggregate_scenes.py --year "$Y" --all --workers 4 && break
  echo "attempt $attempt failed $(date)"; sleep 60
done

log "covariates $Y"
python revision/05_covariates/build_covariates.py demzones --year "$Y"
python revision/01_data_preparation/class_stability.py --year "$Y"
log "REBUILD $Y DONE"

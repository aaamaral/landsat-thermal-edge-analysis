#!/usr/bin/env bash
# Re-run every downstream estimate on the consistent (all Collection 10.1) database, in order,
# one heavy job at a time. Usage: bash revision/08_statistics/rerun_all.sh
set -u
cd "$(dirname "$0")/../.."
export GDAL_CACHEMAX=64
Y="1985 2002 2024"
S=revision/08_statistics
run() { echo "=== $* $(date)"; python "$@" 2>&1 | grep -v -i "warn\|▕"; echo "--- exit ${PIPESTATUS[0]} $(date)"; }

run revision/04_landscape/landscape_metrics_revision.py --year 2002
run $S/core_estimates.py --source full --year $Y --B 1000
run revision/03_database_audit/audit_database.py --year $Y
run $S/orientation.py --year $Y --B 1000
run $S/covariate_models.py --year $Y
run $S/spatial_autocorrelation.py --year $Y
run $S/spatial_block_ci.py --year $Y --B 1000
run $S/landscape_models.py --year $Y
run $S/map_date_sensitivity.py --year $Y --B 1000
run revision/09_figures/fig_results.py
echo "RERUN ALL DONE $(date)"

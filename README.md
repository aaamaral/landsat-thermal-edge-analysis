# Edge-associated thermal patterns in Landsat surface temperature

## Overview

This repository contains the code used to produce the results of the article listed below. The
pipeline describes how mapped land-cover boundaries (MapBiomas Collection 10.1) appear in Landsat
Collection 2 Level 2 surface temperature (TST) in the State of Paraná, Brazil, in three
year–sensor conditions: 1985 (Landsat 5 TM), 2002 (Landsat 7 ETM+) and 2024 (Landsat 8/9 TIRS).
It computes three descriptors of the observed edge pattern:

- **DET**, edge thermal difference: the TST of the 0–30 m ring inside a polygon minus the polygon's
  own interior reference band in the same scene;
- **DEP**, edge effect depth: the transition breakpoint of the TST-contrast profile with distance
  from the boundary (linear–plateau fit), i.e. the end of the steep near-edge regime, not the
  distance at which the contrast vanishes;
- **DTEC**, directional thermal edge contrast: the edge contrast of a class computed separately for
  each neighbouring class.

All three are contrasts in the TST product. None of them is a measurement of heat transfer.
Uncertainty comes from a crossed polygon × scene bootstrap, a spatial block bootstrap and two-way
cluster-robust regression.

The repository holds code, documentation and the Landsat scene list only. No raster, no derived
table and no aggregated database is distributed (see [Data availability](#data-availability)).

## Article

*Edge-associated thermal patterns in Landsat surface temperature: magnitude, transition depth and
directional contrasts across land-cover classes.* Adriano Araújo do Amaral. Manuscript under review.

## Requirements

- Python 3.12 (tested with Python 3.12.1).
- The packages in `requirements.txt`, with the versions used in the study: NumPy, pandas, SciPy,
  PyArrow, DuckDB, statsmodels, patsy, rasterio, pyproj, Shapely, GeoPandas, pyogrio, joblib,
  Matplotlib, Cartopy and pyshp (map figures) and the Earth Engine Python API (`earthengine-api`).
- GDAL, PROJ and GEOS are used through rasterio, pyogrio, pyproj and Shapely. The binary wheels
  installed by `pip` include them, so no separate system installation is needed on Windows, macOS
  or Linux x86-64. Conda-forge builds also work.
- Bash, for the three `.sh` wrappers (Git Bash or WSL on Windows). Each wrapper only calls the
  Python scripts listed in the workflow, which can also be run directly.
- A Google Earth Engine account and a Google Cloud project registered for Earth Engine, for the
  steps marked **EE** in the workflow.
- Disk and memory for a full run: the scene × zone database has about 17, 26 and 76 million rows
  for 1985, 2002 and 2024 (Parquet, tens of GB). Rasters are read by windows and blocks so that the
  scripts run on a workstation with limited RAM.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_synthetic.py       # synthetic checks, no Earth Engine needed (a few seconds)
```

## Repository structure

```
LICENSE                        MIT License
.gitattributes, .gitignore     line endings; files kept out of version control
README.md                      this file
requirements.txt               Python dependencies (versions used in the study)
data/README.md                 external data sources and the files the user must provide
outputs/scene_manifests/       Landsat scene list of the study (922 candidates; 672 used)
scripts/build_scene_manifest.py    rebuilds the scene list from Earth Engine (EE)
scripts/prepare_reference_grid.py  builds the 30 m reference grid of Paraná
scripts/gee_python/            Earth Engine settings: collections, QA_PIXEL masking, scaling;
                               polygon identifiers
scripts/landscape/             landscape-metric functions (imported by step 6; run standalone
                               with python -m scripts.landscape.<module>)
revision/src/tedge/            core library
    paths.py                   project paths, years, sensors, input locations
    fetch.py                   Earth Engine: per-scene TST/NDVI/NDMI image, grid-aligned tiles
    geometry.py                distance rings, neighbouring class, edge orientation, contacts
    aggregate.py               pixel -> zone aggregation, thermal-support variants
    engine.py                  DuckDB tables: polygon × ring × scene, contrasts, interfaces
    profiles.py                profiles, DET, DEP (linear–plateau breakpoint), DTEC, crossed bootstrap
revision/01_data_preparation/  MapBiomas class raster, polygon labels, edge geometry, class stability
revision/02_edge_aggregation/  scene × polygon × neighbouring class × ring aggregation (EE)
revision/03_database_audit/    integrity checks, support and retention tables
revision/04_landscape/         landscape metrics of the polygons
revision/05_covariates/        SRTM topography and ERA5-Land meteorology (EE)
revision/08_statistics/        estimates, sensitivity analyses, inference and models
revision/09_figures/           figures
    cartography.py             map projection, boundaries, graticule, north arrow, scale bar
    fig_study_area.py          Figure 2, location of the study area
    fig_results.py             Figures 3-7 and S2 (Figure 3: spatial coverage map)
    fig_conceptual.py          Figure 1;  fig_spatial_examples.py: Figure S3
revision/10_tables/            supplementary tables
tests/test_synthetic.py        minimal tests on synthetic data
```

Run every command from the repository root. Scripts locate the root from their own position, so
the relative layout must be kept. Intermediate data are written to `revision/data/` and results to
`revision/outputs/`; both folders are created by the scripts and are git-ignored.

## External data

| Dataset | Access | Used for |
|---|---|---|
| Landsat Collection 2 Level 2 (Tier 1): Landsat 5 TM `LANDSAT/LT05/C02/T1_L2`, Landsat 7 ETM+ `LANDSAT/LE07/C02/T1_L2`, Landsat 8/9 `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` | Google Earth Engine | Surface temperature (ST band), NDVI, NDMI, QA_PIXEL |
| MapBiomas Brazil Collection 10.1 land cover, `projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_coverage_v1` | Google Earth Engine (public asset) | Annual class maps of 1985, 2002 and 2024 and of the adjacent years |
| ERA5-Land hourly, `ECMWF/ERA5_LAND/HOURLY` | Google Earth Engine | Scene-level meteorological covariates |
| SRTM 1 arc-second, `USGS/SRTMGL1_003` | Google Earth Engine | Elevation, slope and aspect |
| State boundary of Paraná | Supplied by the user (for example, the official IBGE state boundary) | Extent of the scene search and clip of the reference grid |
| IBGE 2022 state boundaries, `BR_UF_2022` (SIRGAS 2000) | Downloaded by the user from geoftp.ibge.gov.br (see [Map figures](#map-figures-figures-2-and-3)) | Figures 2 and 3 |
| Natural Earth 1:10m Admin 0 countries (public domain) | Downloaded automatically by Cartopy on first use | International boundaries in Figure 2 |

These datasets are not redistributed here. They remain subject to the licences and terms of use of
their providers (USGS/NASA for Landsat and SRTM, the MapBiomas Project, ECMWF/Copernicus for
ERA5-Land, IBGE or the provider of the boundary file; Natural Earth is in the public domain) and of
Google Earth Engine. More details are in `data/README.md`.

## Environment variables

| Variable | Meaning | Needed by |
|---|---|---|
| `EE_PROJECT` | Google Cloud project registered for Earth Engine | every step marked **EE** |
| `PARANA_GPKG` | path to the Paraná boundary (GeoPackage or another vector format readable by GeoPandas). Default: `data/external/PARANA.gpkg` | `prepare_reference_grid.py`, `build_scene_manifest.py` |
| `IBGE_UF` | path to the IBGE `BR_UF_2022.zip` file (optional). Default: `data/external/ibge/BR_UF_2022.zip` | `fig_study_area.py`, `fig_results.py` (Figure 3) |

```bash
earthengine authenticate                     # once per machine; stores credentials outside the repository
export EE_PROJECT=your-cloud-project         # PowerShell: $env:EE_PROJECT="your-cloud-project"
export PARANA_GPKG=/path/to/PARANA.gpkg      # PowerShell: $env:PARANA_GPKG="C:\path\to\PARANA.gpkg"
```

When `EE_PROJECT` is not set, the Earth Engine steps stop with a message that explains how to set
it. When the boundary or a required input file is missing, the scripts stop with a message naming
the file and the command that creates it. No credentials, tokens or project identifiers are stored
in this repository.

## Reproducing the reference grid

All rasters of the analysis share one 30 m grid. The original study used a class raster of Paraná
as template; that raster is not distributed. `scripts/prepare_reference_grid.py` rebuilds a grid
with the same definition:

| Property | Value |
|---|---|
| CRS | Albers Equal Area with the IBGE parameters for Brazil: `+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs` (the full WKT is in the script) |
| Resolution | 30 m × 30 m, north-up |
| Alignment | pixel corners on x = 4,936,261.972889816 + 30 i and y = 8,824,729.933096489 − 30 j (m); this point is the upper-left corner of the study grid |
| Extent | default (`--extent study`): 22,656 columns × 15,576 rows from that corner, as in the study. `--extent boundary` instead fits the grid to the bounding box of the boundary, on the same lattice (`--pad` adds pixels on each side) |
| Clip | pixels whose centre lies inside the boundary are 1, all others 0 |

Steps:

1. Obtain a polygon of the State of Paraná in any CRS (it is reprojected by the script).
2. Set `PARANA_GPKG` to its path, or pass `--boundary path`.
3. Run `python scripts/prepare_reference_grid.py`. The grid is written to
   `revision/data/inputs/reference_grid_30m.tif` (int16, tiled and compressed, about 350 million
   pixels; under a minute on a workstation).
4. Step 1 of the workflow uses this file automatically as the template on which MapBiomas
   Collection 10.1 is sampled.

With the default options, the grid, CRS and extent match those of the study. The clip follows the
boundary file supplied, so the set of in-state pixels along the border can differ slightly from the
original raster; with the boundary used in the study, 99.996 % of the original in-state pixels were
reproduced, with differences confined to border pixels. The grid is therefore spatially and
analytically equivalent to the original, not a byte-for-byte copy, and a different boundary file
can change a small number of border polygons.

## Workflow

Steps marked **EE** use Google Earth Engine; **heavy** marks steps that take many hours or days, or
tens of GB of disk.

| Step | Command | Output | Notes |
|---|---|---|---|
| 0a. Scene list (optional) | `python scripts/build_scene_manifest.py` | a rebuilt copy of the list | **EE**, minutes. The study list is already included |
| 0b. Reference grid | `python scripts/prepare_reference_grid.py` | `revision/data/inputs/reference_grid_30m.tif` | light (under a minute) |
| 1. Class raster and polygons | `python revision/01_data_preparation/rebuild_class_raster.py --year 1985` (repeat for 2002 and 2024) | `revision/data/inputs/parana_albers_{year}.tif`, `polygon_labels_{year}.tif`, polygon table | **EE**, **heavy** (≈370 tiles of 1024 × 1024 px per year, then labelling) |
| 2. Edge geometry | `python revision/01_data_preparation/build_edge_geometry.py --year 1985 2002 2024` | `revision/data/geometry/{year}/` | **heavy** (local) |
| 3. Temporal class stability | `python revision/01_data_preparation/class_stability.py --year 1985 2002 2024` | `revision/data/covariates/class_stability_{year}.parquet` | **EE** |
| 4. Scene aggregation | `bash revision/02_edge_aggregation/run_full.sh 1985 2002 2024` (wraps `aggregate_scenes.py --all`, resumes after interruptions) | `revision/data/scenes/{year}/*.parquet` | **EE**, **heavy**: 672 scenes, days on a workstation |
| 5. Covariates | `python revision/05_covariates/build_covariates.py dem`, then `demzones`, then `era5` | topographic and ERA5-Land tables | **EE** |
| 6. Landscape metrics | `python revision/04_landscape/landscape_metrics_revision.py --year {year}` for each year | `revision/data/inputs/landscape_{year}.csv` | **heavy** (local) |
| 7. Estimates, sensitivity analyses, inference and models | `bash revision/08_statistics/rerun_all.sh` (runs the scripts below in order) | `revision/outputs/full/*.csv` | **heavy** (local; reads the whole database, 1,000 bootstrap replicates) |
| 8. Figures | `python revision/09_figures/fig_conceptual.py`, `fig_study_area.py`, `fig_results.py`, `fig_spatial_examples.py` | `revision/outputs/figures/` | `fig_spatial_examples.py` uses **EE** for one scene; the map figures need the IBGE file (see [Map figures](#map-figures-figures-2-and-3)) |
| 9. Supplementary tables | `python revision/10_tables/supplementary_tables.py` | `revision/outputs/tables/` | light |

`revision/02_edge_aggregation/rebuild_year_c10.sh <year>` runs steps 1–4 and the covariates for one
year. `rerun_all.sh` runs `landscape_metrics_revision.py` for 2002 only, because that was the year
rebuilt during the revision; on a fresh run, execute step 6 for all three years first.

What each stage does:

1. **Scene selection.** The scene list contains every Collection 2 Tier 1 Level 2 scene
   intersecting Paraná on WRS-2 paths 220–224 and rows 76–78 in 1985, 2002 and 2024. Scenes with
   `cloud_cover` ≤ 50 % are processed (109, 162 and 401; 672 in total), selected in
   `aggregate_scenes.py`.
2. **Quality screening.** QA_PIXEL bits 0–5 (fill, dilated cloud, cirrus, cloud, cloud shadow,
   snow) mask pixels one by one; a masked pixel only reduces the valid count of its zone
   (`scripts/gee_python/edge_config.py`, `revision/src/tedge/fetch.py`). TST is
   ST × 0.00341802 + 149.0 − 273.15 (°C).
3. **Grid.** `prepare_reference_grid.py` (above).
4. **Land cover.** MapBiomas Collection 10.1 is sampled on the grid for each year
   (`rebuild_class_raster.py`).
5. **Polygons.** 8-connected components of each class, labelled in a fixed order
   (`rebuild_class_raster.py`, identifiers from `scripts/gee_python/polygon_ids.py`).
6. **Distance-to-edge aggregation.** Distance rings of 30 m from each polygon's own boundary
   (ring 0 = boundary pixel layer, rings to 300 m, then a pooled ring), the neighbouring class
   across the nearest boundary and the edge orientation (`geometry.py`, `build_edge_geometry.py`).
   Each scene is fetched on the grid and aggregated by polygon × neighbouring class × ring
   (`aggregate.py`, `aggregate_scenes.py`).
7. **DET.** Ring 0 minus the polygon's interior reference band in the same scene (`profiles.py`,
   `engine.py`, `core_estimates.py`).
8. **DEP.** Linear–plateau breakpoint of the contrast profile, with right-censoring and an
   exponential competitor (`profiles.py`, `core_estimates.py`).
9. **DTEC.** Ring 0 of each (polygon, neighbouring class) interface minus the polygon's reference
   band, and the asymmetry of each class pair (`engine.py`, `core_estimates.py`).
10. **Sensitivity analyses.** Thermal support, edge exclusion, reference window, valid fraction,
    map–scene date and season, edge orientation (scripts listed below).
11. **Inference.** Crossed polygon × scene bootstrap, spatial block bootstrap, two-way
    cluster-robust regression, REML variance components.
12. **Tables** and 13. **Figures** from the result tables of step 7.

Table S13 (MapBiomas class accuracy) is not computed by this code. It reports the statistics
published by the MapBiomas accuracy application (Collection 10, Atlantic Forest biome), as
described in the Supplementary Material.

## Main analyses

| Analysis | Script(s) |
|---|---|
| Edge aggregation (scene × polygon × neighbouring class × ring) | `revision/src/tedge/geometry.py`, `aggregate.py`, `fetch.py`; `revision/01_data_preparation/build_edge_geometry.py`; `revision/02_edge_aggregation/aggregate_scenes.py` |
| DET | `revision/src/tedge/engine.py` (`contrasts`), `profiles.py`; `revision/08_statistics/core_estimates.py` (`det_sweep.csv`) |
| DEP | `revision/src/tedge/profiles.py` (`broken_stick`, `dep_from_replicates`, `threshold_depth`); `core_estimates.py` (`profiles.csv`, `dep.csv`) |
| DTEC | `revision/src/tedge/engine.py` (`interface_contrasts`); `core_estimates.py` (`dtec_pairs.csv`) |
| Thermal-support sensitivity (30 m, 60/90/120 m moving windows, Gaussian kernel with FWHM equal to the native thermal resolution) | `revision/src/tedge/aggregate.py` (`support_variants`), `aggregate_scenes.py`; `core_estimates.py` |
| Edge exclusion (first 1–3 rings removed) | `core_estimates.py` (`excl`) |
| Crossed bootstrap (polygon × scene, Poisson weights) | `revision/src/tedge/profiles.py` (`pigeonhole_means`); `core_estimates.py`, `orientation.py`, `map_date_sensitivity.py` |
| Spatial block bootstrap (25 km and 50 km blocks) and Moran correlograms | `revision/08_statistics/spatial_block_ci.py`, `spatial_autocorrelation.py` |
| Covariate models (two-way cluster-robust SE) and REML variance decomposition | `revision/08_statistics/covariate_models.py`; covariates from `revision/05_covariates/build_covariates.py` |
| Landscape metrics | `scripts/landscape/*`, `revision/04_landscape/landscape_metrics_revision.py`, `revision/08_statistics/landscape_models.py` |
| Map–scene date, season, class stability and edge orientation | `revision/01_data_preparation/class_stability.py`, `revision/08_statistics/map_date_sensitivity.py`, `orientation.py` |
| Database audit, support and retention | `revision/03_database_audit/audit_database.py` |
| Tables | `audit_database.py`, `core_estimates.py`, `covariate_models.py`, `revision/10_tables/supplementary_tables.py` |
| Figures | Fig. 1 `revision/09_figures/fig_conceptual.py`; Fig. 2 `fig_study_area.py`; Figs. 3–7 and S2 `fig_results.py`; Fig. S3 `fig_spatial_examples.py`; Fig. S1 `supplementary_tables.py` |

## Map figures (Figures 2 and 3)

Both maps use the Albers equal-area projection of the analysis grid (IBGE parameters for Brazil,
GRS 1980; see [Reproducing the reference grid](#reproducing-the-reference-grid)), a graticule in
degrees, a north arrow and a scale bar whose length is calibrated with geodesic distances.

**Boundary data.** Brazilian state boundaries come from the official IBGE 2022 territorial grid,
dataset `BR_UF_2022` (SIRGAS 2000), published at geoftp.ibge.gov.br. Download the zip once and keep
it compressed:

```bash
mkdir -p data/external/ibge
curl -o data/external/ibge/BR_UF_2022.zip \
  https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/Brasil/BR/BR_UF_2022.zip
```

The scripts read `data/external/ibge/BR_UF_2022.zip`, or the path in the environment variable
`IBGE_UF`. The file is git-ignored and not redistributed here. International boundaries come from
Natural Earth 1:10m Admin 0 countries (public domain), which Cartopy downloads and caches
automatically on first use (internet access needed once).

**Commands.**

| Figure | Command | Inputs |
|---|---|---|
| Figure 2, location of the study area | `python revision/09_figures/fig_study_area.py` | IBGE `BR_UF_2022`, Natural Earth |
| Figure 3, spatial coverage | `python revision/09_figures/fig_results.py` (also draws Figures 4–7 and S2) | `revision/outputs/audit/coverage_grid.csv` from `audit_database.py` (step 5), IBGE `BR_UF_2022` |

Outputs are written to `revision/outputs/figures/` (`fig02_study_area`, `fig03_coverage`, PNG and
PDF). In Figure 3 each square is one of the real 10 km cells of the coverage grid of the analysis
(cell index × 10 km in the analysis CRS), and its colour is the number of usable polygons read from
the pipeline output; no value is interpolated or recomputed for display.

## Tests

`python tests/test_synthetic.py` (or `python -m pytest tests` if pytest is installed) runs in a few
seconds without Earth Engine. It builds a synthetic landscape with a known edge signal and checks
that the pipeline (edge geometry → zone aggregation with QA-invalid pixels → DuckDB tables →
contrasts) recovers DET, a DEP breakpoint at 90 m and the two directional contrasts, and that the
crossed bootstrap gives reproducible intervals. It also checks the reference-grid alignment, the
scene list, the error messages for missing settings, and the absence of local paths and secrets.

## Limitations of reproducibility

- **Earth Engine catalogue.** Results depend on the Earth Engine assets at the time of processing.
  USGS, ECMWF and MapBiomas may reprocess or replace assets. Transient Earth Engine errors are
  handled by re-running (`run_full.sh` resumes).
- **Reference grid.** The grid is rebuilt with the study definition; border pixels depend on the
  boundary file (see above).
- **Bootstrap.** Estimates use 1,000 replicates with fixed random seeds. With the same database and
  package versions the intervals are reproducible; other NumPy versions may change the last digit
  of interval limits.
- **Computation.** The full aggregation of the 672 scenes takes days on a workstation and tens of
  GB of disk.

## Data availability

All input data come from public sources: Landsat Collection 2 Level 2 (obtained through Google
Earth Engine), MapBiomas Collection 10.1, ERA5-Land and SRTM, plus a state boundary of Paraná
supplied by the user. The derived data used in the analyses (class rasters on the reference grid,
the scene × polygon × neighbouring class × ring database, intermediate tables and result tables)
are not distributed. They can be rebuilt from the public sources with the code in this repository,
following the workflow above.

## Citation

If you use this code, please cite the article above.

Repository URL: https://github.com/aaamaral/landsat-thermal-edge-analysis

## License

The code is released under the MIT License (see `LICENSE`). The external data used by the code are
not covered by this licence; they remain subject to the licences and terms of their providers.

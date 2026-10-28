# External data

No raster, vector or derived tabular data are stored in this repository. The pipeline reads public
sources through Google Earth Engine and one boundary file supplied by the user. The external data
remain subject to the licences and terms of use of their providers and of Google Earth Engine.

## Accessed through Google Earth Engine (no download needed)

| Dataset | Earth Engine ID | Used for |
|---|---|---|
| Landsat 5 TM Collection 2 Level 2 | `LANDSAT/LT05/C02/T1_L2` | Surface temperature (ST band), NDVI, NDMI, QA_PIXEL (1985) |
| Landsat 7 ETM+ Collection 2 Level 2 | `LANDSAT/LE07/C02/T1_L2` | Same (2002) |
| Landsat 8/9 OLI/TIRS Collection 2 Level 2 | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` | Same (2024) |
| MapBiomas Brazil Collection 10.1 land cover | `projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_coverage_v1` | Annual class maps (1985, 2002, 2024 and adjacent years) |
| ERA5-Land hourly | `ECMWF/ERA5_LAND/HOURLY` | Scene-level meteorological covariates |
| SRTM 1 arc-second | `USGS/SRTMGL1_003` | Elevation, slope and aspect |

## Supplied by the user

| File | Location | Notes |
|---|---|---|
| State boundary of Paraná | path in the environment variable `PARANA_GPKG`, or `data/external/PARANA.gpkg` | Any polygon of the state in a vector format readable by GeoPandas, with a defined CRS (for example, the official IBGE state boundary). Used by `scripts/prepare_reference_grid.py` and `scripts/build_scene_manifest.py`. |
| IBGE 2022 state boundaries, `BR_UF_2022.zip` (SIRGAS 2000) | `data/external/ibge/BR_UF_2022.zip`, or the path in `IBGE_UF` | Official IBGE territorial grid, from https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/Brasil/BR/BR_UF_2022.zip. Keep it zipped. Used only by the map figures (Figures 2 and 3). |

## Downloaded automatically

| Dataset | Notes |
|---|---|
| Natural Earth 1:10m Admin 0 countries | Public domain. Downloaded and cached by Cartopy on first use; international boundaries in Figure 2. |

## Built by the code

| File | Command | Notes |
|---|---|---|
| `revision/data/inputs/reference_grid_30m.tif` | `python scripts/prepare_reference_grid.py` | 30 m grid of the study (CRS, alignment and extent in the main README), 1 inside Paraná, 0 outside. Template of every raster of the analysis. |

## Included

- `outputs/scene_manifests/landsat_scene_manifest_y1985-2002-2024_p220-224_r76-78.csv`: all Tier 1
  Collection 2 Level 2 scenes intersecting Paraná on WRS-2 paths 220–224 and rows 76–78 for the
  three years (922 scenes). The pipeline uses the scenes with `cloud_cover` ≤ 50 % (109, 162 and
  401 scenes; 672 in total). `scripts/build_scene_manifest.py` rebuilds the list with the same
  filters.

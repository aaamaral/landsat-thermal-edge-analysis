# Aggregated derived data: edge-associated thermal patterns in Landsat surface temperature (Paraná, Brazil)

Aggregated derived dataset accompanying the article *Edge-associated thermal patterns in Landsat
surface temperature: magnitude, transition depth and directional contrasts across land-cover classes
in Paraná, southern Brazil* (A. A. do Amaral, Remote Sensing Applications: Society and Environment).

Code: https://github.com/aaamaral/landsat-thermal-edge-analysis (archived release: see the Zenodo
record of the software). This dataset lets users recompute the published point estimates and the
crossed polygon × scene bootstrap intervals **without** re-running the Earth Engine extraction and
raster processing. The scene × polygon × distance-ring table (`polygon_ring/`) also allows other
windows, edge exclusion and the thermal-support experiments to be recomputed. The full scene ×
polygon × neighbouring class × distance-ring database is not included. It can be rebuilt from the
public inputs with the code.

## Conditions and units

- Year–sensor conditions: 1985 (Landsat 5 TM), 2002 (Landsat 7 ETM+), 2024 (Landsat 8/9 TIRS);
  Landsat Collection 2 Level 2 surface temperature (TST, °C), 30 m analysis grid; MapBiomas
  Collection 10.1 land cover for the same year.
- `focal_class`, `nb_class`: MapBiomas Collection 10.1 legend codes (e.g. 3 forest formation,
  9 silviculture, 15 pasture, 21 mosaic of uses, 24 urban area, 33 river, lake and ocean,
  39 soybean, 41 other temporary crops).
- `label`: polygon identifier (connected component of one class) **within one condition**. Labels are
  not comparable between years.
- `scene_id`: Landsat Collection 2 product identifier (full list in `results/S_scenes.csv`).
- Rings: ring *k* (k = 0…9) holds the pixels at distance [30k, 30k + 30) m inside the polygon from
  its mapped boundary (centre 15, 45, …, 285 m), and ring 10 holds every pixel ≥ 300 m.
- Reference band of window W: rings W/30 and W/30 + 1 (distances [W, W + 60) m); for W = 300 m,
  ring 10. Ring means require ≥ 3 valid pixels (no valid-fraction threshold).
- Coordinates: Albers equal-area, IBGE parameters for Brazil
  (`+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +units=m`).

## Files

| Path | One row per | Columns |
|---|---|---|
| `contrasts/contrasts_W150_{year}.parquet` | scene × polygon (polygons eligible for W = 150 m) | `scene_id`, `label`, `focal_class`, `c0`…`c4` (TST of ring k minus the reference band, °C; `c0` = DET), `ref` (reference-band TST, °C) |
| `contrasts/contrasts_W300_{year}.parquet` | scene × polygon (eligible for W = 300 m) | as above with `c0`…`c9` (profile to 300 m, used for DEP) |
| `dtec/dtec_W150_ring0_{year}.parquet` | scene × polygon × neighbouring class | `dtec` (TST of the edge ring adjacent to `nb_class` minus the polygon reference band, °C), `nv_edge`/`nt_edge` (valid/total pixels of that edge zone), `contact_faces` (shared 4-neighbour pixel faces), `normal_e`, `normal_n` (summed outward normal of the contact) |
| `polygons/polygons_{year}.parquet` | polygon | `label`, `focal_class`, `n_px` (area in 30 m pixels), `max_ring`, `centroid_x`, `centroid_y`, `perimeter_faces`, `limit_faces` (faces on the study-area limit) |
| `results/*.csv` | — | Class-level results behind the figures and tables (estimates and 95 % intervals): `S_main_results.csv` (DET, DEP; Table 2), `profiles.csv` (Fig. 4), `det_sweep.csv` (Table 3, Fig. 5), `dtec_pairs.csv` (Figs. 6–7), `variance_components.csv` (Table 4), the other supplementary tables, and `S_scenes.csv` (scene list) |
| `polygon_ring/polygon_ring_{year}.parquet` | scene × polygon × ring (neighbours pooled) | `n_total`, `n_valid`, ring-mean TST for the thermal supports `t30`, `box60`, `box90`, `box120`, `psf`, and `ndvi`, `ndmi`: needed for other windows, edge exclusion and thermal-support experiments |
| `MANIFEST.csv` | file | size, number of rows, SHA-256 |
| `verify_deposit.py` | — | recomputes DET, the W = 300 m profiles and DTEC from the tables above and compares them with `results/` |

## Reproducing the estimates

Point estimates are means over the observed (scene, polygon) cells, or (scene, interface) cells for
DTEC; e.g. forest DET in 1985 = mean of `c0` over the rows of `contrasts_W150_1985.parquet` with
`focal_class == 3`. The intervals in the article come from a crossed bootstrap that resamples polygons
(or interfaces) and scenes with independent Poisson(1) weights (function `pigeonhole_means` in
`revision/src/tedge/profiles.py` of the code repository). Run

```
python verify_deposit.py .
```

to check every DET, W = 300 m profile and DTEC point estimate against `results/` (tolerance 1e-4 °C).

## Sources and licence

Derived from Landsat Collection 2 Level 2 products (U.S. Geological Survey, public domain) and
MapBiomas Collection 10.1 (CC BY 4.0). Dataset licence: CC BY 4.0.

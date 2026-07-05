# UrbanCool Melbourne

**ML-powered urban heat vulnerability mapping for Greater Melbourne.**

UrbanCool Melbourne fuses Victorian Government urban heat island and vegetation cover polygons, City of Melbourne tree/canopy data, OpenStreetMap urban morphology, Bureau of Meteorology weather records, and ABS Census demographics to train a gradient-boosted model that identifies which SA2 areas (suburbs) are most vulnerable during heatwaves. The final deliverable is a Streamlit dashboard with an interactive heat map, SHAP-based explanations, and a "what-if" green-infrastructure simulator.

See [UrbanCool_Melbourne_Project_Guide.md](UrbanCool_Melbourne_Project_Guide.md) for the full project plan, data source details, and 7-day implementation schedule.

## Project Status

Day 1 complete — environment set up, all data sources downloaded and validated (including the manually-ordered urban heat/vegetation shapefiles).

## Data Sources

| # | Source | Role | Access |
|---|--------|------|--------|
| 1 | [Vic Planning — vegetation, heat & land use data](https://www.planning.vic.gov.au/guides-and-resources/Data-spatial-and-insights/melbournes-vegetation-heat-and-land-use-data) | Target variable (urban heat island / heat vulnerability index polygons) | Manual order via DataShare Vic cart checkout (no public API) — see `download_urban_heat_data()` for direct dataset links |
| 2 | [City of Melbourne — tree inventory](https://data.melbourne.vic.gov.au/explore/dataset/trees-with-species-and-dimensions-urban-forest/) | Tree density, species, health features | Automated (Open Data API) |
| 3 | [City of Melbourne — tree canopy polygons](https://data.melbourne.vic.gov.au/explore/dataset/tree-canopies-2021-urban-forest/) | Canopy coverage ratio | Automated (Open Data API) |
| 4 | OpenStreetMap via `osmnx` | Building density, roads, parks, water | Automated |
| 5 | [BOM climate data](http://www.bom.gov.au/climate/data/) | Contextual weather features | Automated (recent obs) / manual (full historical) |
| 6 | [ABS — SA2 boundaries](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3) | Spatial unit boundaries | Automated |

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\Activate.ps1 for PowerShell
pip install -r requirements.txt
cp .env.example .env            # fill in any optional API keys
```

## Downloading Raw Data

```bash
python -m src.data.download
```

This fetches everything that has a public API (trees, tree canopies, OSM features, BOM recent observations, ABS SA2 boundaries) into `data/raw/`. The Victorian Government urban heat data has no public API and must be downloaded manually — the script prints instructions when run.

Validate what landed in `data/raw/`:

```bash
python -m src.data.validation
```

## Configuration

Study area, CRS (`EPSG:28355` — GDA94 / MGA Zone 55), spatial unit (SA2), and data source parameters live in [config/settings.yaml](config/settings.yaml).

## Project Structure

```
urbancool-melbourne/
├── config/settings.yaml   # Paths, CRS, spatial boundaries, model params
├── data/{raw,interim,processed}
├── notebooks/              # Exploration, feature engineering, training, SHAP
├── src/
│   ├── data/                # download.py, loaders.py, validation.py
│   ├── features/             # spatial.py, urban_morphology.py, vegetation.py
│   ├── models/                # train.py, evaluate.py, predict.py
│   └── visualization/         # maps.py
├── app/streamlit_app.py     # Dashboard
├── tests/
├── models/                  # Trained model artifacts (gitignored)
└── outputs/                 # Generated maps, reports
```

## Known Limitations

- The urban heat/vegetation data (2018, 2014) predates the current tree inventory — a temporal mismatch acknowledged as a known limitation (see project guide).
- The urban heat/HVI/vegetation datasets are vector polygons (ESRI Shapefile, delivered in geographic GDA94/EPSG:4283), not raster GeoTIFFs as the original project guide assumed. There's no `load_heat_raster()`/`rasterstats.zonal_stats()` step for this source — join the polygons to SA2 boundaries with a regular geopandas spatial join/overlay instead, reprojecting to EPSG:28355 first. Target variable is `UHI18_M` (mean Urban Heat Island value) in `HEAT_URBAN_HEAT_2018.shp`, joined via `SA2_MAIN16`; see `config/settings.yaml`'s `data_sources.urban_heat` for all six dataset UUIDs. Note the 2014 vs 2018 heat vulnerability index shapefiles use different column names for the same concept (`HVI` vs `HVI_INDEX`) — align these during Day 2 cleaning.
- BOM's Climate Data Online historical exports are gated behind a session token generated in-browser; this repo automates recent (72h) observations only. Historical daily records require a manual CDO export — see `src/data/download.py::download_bom_weather` docstring.
- OSM building footprints for the full Greater Melbourne bbox (~660k features with hundreds of sparse tag columns) can exhaust RAM if fetched in one shot; `download_osm_features` fetches buildings and water tile-by-tile via `_features_from_bbox_tiled` to bound memory use and improve resilience to Overpass throttling (Overpass briefly throttled this connection during the Day 1 run after heavy building/road/park queries — a retry once the throttle cleared succeeded).

---
title: UrbanCool Melbourne
emoji: 🌡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# UrbanCool Melbourne

**ML-powered urban heat vulnerability mapping for Greater Melbourne.**

UrbanCool Melbourne fuses Victorian Government urban heat island and vegetation cover polygons, City of Melbourne tree/canopy data, OpenStreetMap urban morphology, Bureau of Meteorology weather records, and ABS Census demographics to train a Random Forest model that identifies which SA2 areas (suburbs) are most vulnerable during heatwaves. The final deliverable is a FastAPI + static-console dashboard (see DAY_6.md) with an interactive heat map, SHAP-based explanations, and a "what-if" green-infrastructure simulator.

> The YAML block above is [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-config-reference) metadata; GitHub ignores it and renders this file normally. It only matters if this repo (or its content) is pushed to a Space using the Docker SDK; see "Deploying to Hugging Face Spaces" below.

See [UrbanCool_Melbourne_Project_Guide.md](UrbanCool_Melbourne_Project_Guide.md) for the full project plan, data source details, and 7-day implementation schedule.

## Project Status

Day 6 complete. See [DAY_6.md](DAY_6.md) for the finished dashboard: a working What-If Simulator (`app/components/what_if.py`), a Feature Explorer with a live per-suburb SHAP waterfall (`app/components/feature_explorer.py`), a persistent model-accuracy KPI row, a visual redesign (custom Streamlit theme, hero header, styled metric cards; see DAY_6.md Part B3), and a `tests/` suite (10/10 passing). Day 5: dashboard Part 1 (heat map), see [DAY_5.md](DAY_5.md). Day 4: model training & evaluation, see [DAY_4.md](DAY_4.md); spatially-grouped CV across SA2/SA1 resolutions found SA1 generalizes better (held-out test R²=0.570, RMSE=1.14°C); tuned Random Forest (capped at 100 trees to fit the deployed dashboard's memory budget), SHAP explainability, saved to `models/best_model.joblib`. Day 3: feature engineering, see [DAY_3.md](DAY_3.md) (`data/processed/feature_matrix.csv`, 361 SA2 rows × 26 columns). Day 2: exploration & cleaning, see [DAY_2.md](DAY_2.md). Day 1: environment set up, all data sources downloaded and validated.

## Data Sources

| # | Source | Role | Access |
|---|--------|------|--------|
| 1 | [Vic Planning: vegetation, heat & land use data](https://www.planning.vic.gov.au/guides-and-resources/Data-spatial-and-insights/melbournes-vegetation-heat-and-land-use-data) | Target variable (urban heat island / heat vulnerability index polygons) | Manual order via DataShare Vic cart checkout (no public API); see `download_urban_heat_data()` for direct dataset links |
| 2 | [City of Melbourne: tree inventory](https://data.melbourne.vic.gov.au/explore/dataset/trees-with-species-and-dimensions-urban-forest/) | Tree density, species, health features | Automated (Open Data API) |
| 3 | [City of Melbourne: tree canopy polygons](https://data.melbourne.vic.gov.au/explore/dataset/tree-canopies-2021-urban-forest/) | Canopy coverage ratio | Automated (Open Data API) |
| 4 | OpenStreetMap via `osmnx` | Building density, roads, parks, water | Automated |
| 5 | [BOM climate data](http://www.bom.gov.au/climate/data/) | Contextual weather features | Automated (recent obs) / manual (full historical) |
| 6 | [ABS: SA2 boundaries](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3) | Spatial unit boundaries | Automated |

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

This fetches everything that has a public API (trees, tree canopies, OSM features, BOM recent observations, ABS SA2 boundaries) into `data/raw/`. The Victorian Government urban heat data has no public API and must be downloaded manually. The script prints instructions when run.

Validate what landed in `data/raw/`:

```bash
python -m src.data.validation
```

## Running the Pipeline

Once raw data is downloaded and validated:

```bash
python -m src.data.loaders         # clean raw data -> data/interim/*.parquet
python -m src.features.spatial     # build feature_matrix.csv + feature_matrix_sa1.csv
python -m src.models.train         # train, tune, and save models/best_model.joblib
python -m src.models.predict       # predictions -> data/processed/predictions_sa1.csv / predictions_sa2.csv
python -m src.visualization.maps   # optional standalone Folium map -> outputs/heat_vulnerability_map.html
python -m app.build_static         # bake the dashboard's map/KPIs/charts -> app/static/index.html
uvicorn app.main:app --reload      # launch the dashboard at localhost:8000
pytest tests/ -v                   # run the test suite
```

## Deploying to Hugging Face Spaces

The dashboard (`app/`) is a plain FastAPI app with a static HTML/CSS/JS frontend (no
Streamlit), so it deploys as a [Docker SDK Space](https://huggingface.co/docs/hub/spaces-sdks-docker) rather than a Python-runtime Space. This
repo's root `Dockerfile` and the YAML block at the top of this README (Spaces reads
`README.md`'s frontmatter for the Space's title/SDK config) are both already set up for
this.

**Before deploying**: `models/`, `data/interim/sa2_boundaries.parquet`, and
`data/processed/predictions_sa2.csv` are gitignored from *this* GitHub repo (they're
large, regenerable build artifacts, see `.gitignore`), but a Space is its own separate
git repo, and the Dockerfile needs those specific files present in *its* build context.
Run the "Running the Pipeline" commands above locally first, then either:

- create a Space on huggingface.co (Docker SDK), add it as a second git remote, and
  `git push` this repo's content to it, including the normally-gitignored files above
  (`git add -f <path>` for just those, on the Space remote's branch); or
- use the Spaces web UI's file upload for `models/`, `data/interim/sa2_boundaries.parquet`,
  and `data/processed/predictions_sa2.csv` after pushing the rest of the code.

See `DAY_6.md` for the fuller reasoning behind this architecture.

## Configuration

Study area, CRS (`EPSG:28355`, GDA94 / MGA Zone 55), spatial unit (SA2), and data source parameters live in [config/settings.yaml](config/settings.yaml).

## Project Structure

```
urbancool-melbourne/
├── config/settings.yaml   # Paths, CRS, spatial boundaries, model params
├── data/{raw,interim,processed}
├── notebooks/              # Exploration, feature engineering, training, SHAP
├── src/
│   ├── data/                # download.py, loaders.py, validation.py
│   ├── features/             # spatial.py, urban_morphology.py, vegetation.py
│   ├── models/                # train.py, predict.py
│   └── visualization/         # maps.py
├── app/
│   ├── main.py                # FastAPI app + API endpoints
│   ├── core.py                 # Shared model/data/SHAP/What-If logic
│   ├── build_static.py          # Bakes app/static/index.html from real data
│   ├── templates/index_template.html
│   └── static/                  # styles.css, script.js, generated index.html + PNGs
├── Dockerfile, .dockerignore  # Hugging Face Spaces (Docker SDK) deployment
├── tests/                    # conftest.py, test_features.py, test_pipeline.py, test_api.py
├── models/                  # Trained model artifacts (gitignored)
└── outputs/                 # Generated maps, reports
```

## Known Limitations

- The urban heat/vegetation data (2018, 2014) predates the current tree inventory, a temporal mismatch acknowledged as a known limitation (see project guide).
- The urban heat/HVI/vegetation datasets are vector polygons (ESRI Shapefile, delivered in geographic GDA94/EPSG:4283), not raster GeoTIFFs as the original project guide assumed. There's no `load_heat_raster()`/`rasterstats.zonal_stats()` step for this source; join the polygons to SA2 boundaries with a regular geopandas spatial join/overlay instead, reprojecting to EPSG:28355 first. Target variable is `UHI18_M` (mean Urban Heat Island value) in `HEAT_URBAN_HEAT_2018.shp`, joined via `SA2_MAIN16`; see `config/settings.yaml`'s `data_sources.urban_heat` for all six dataset UUIDs. Note the 2014 vs 2018 heat vulnerability index shapefiles use different column names for the same concept (`HVI` vs `HVI_INDEX`); align these during Day 2 cleaning.
- BOM's Climate Data Online historical exports are gated behind a session token generated in-browser; this repo automates recent (72h) observations only. Historical daily records require a manual CDO export; see `src/data/download.py::download_bom_weather` docstring.
- OSM building footprints for the full Greater Melbourne bbox (~660k features with hundreds of sparse tag columns) can exhaust RAM if fetched in one shot; `download_osm_features` fetches buildings and water tile-by-tile via `_features_from_bbox_tiled` to bound memory use and improve resilience to Overpass throttling (Overpass briefly throttled this connection during the Day 1 run after heavy building/road/park queries; a retry once the throttle cleared succeeded).

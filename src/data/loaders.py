"""Dataset-specific loading and cleaning functions for UrbanCool Melbourne.

Each ``load_and_clean_*`` function reads one raw source from ``data/raw/``,
applies the cleaning rules found during Day 2 exploration (see
``notebooks/01_data_exploration.ipynb`` and ``DAY_2.md``), reprojects to the
project's working CRS, and returns a ready-to-use GeoDataFrame. Nothing here
writes to disk — see ``save_interim`` / ``src/data/download.py``'s
counterpart for that.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Real-world street tree DBH tops out well under this; larger values in the
# raw data are measurement/entry errors (max observed was 26,027cm).
MAX_PLAUSIBLE_DBH_CM = 300


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _raw_dir(config: dict) -> Path:
    return PROJECT_ROOT / config["paths"]["data_raw"]


def _study_area_polygon(config: dict):
    from shapely.geometry import box

    min_lon, min_lat, max_lon, max_lat = config["study_area"]["bbox"]
    return box(min_lon, min_lat, max_lon, max_lat)


def load_and_clean_trees(config: dict) -> gpd.GeoDataFrame:
    """Load the Melbourne tree inventory and clean it into a GeoDataFrame.

    - Drops rows missing coordinates.
    - Converts diameter_breast_height to float, nulling out physically
      implausible values (>300cm) rather than dropping the row, since the
      other tree attributes are still usable.
    - Standardises species names (strips whitespace).
    - useful_life_expectency has no literal "Not Assessed" string in this
      dataset (unlike the project guide's assumption) — missing values are
      already NaN, so they're left as NaN and labelled explicitly.
    """
    path = _raw_dir(config) / "melbourne_trees.csv"
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.lstrip("﻿") for c in df.columns]

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    df["diameter_breast_height"] = pd.to_numeric(df["diameter_breast_height"], errors="coerce")
    implausible = df["diameter_breast_height"] > MAX_PLAUSIBLE_DBH_CM
    if implausible.any():
        logger.warning("Nulling %d trees with implausible DBH > %dcm", implausible.sum(), MAX_PLAUSIBLE_DBH_CM)
        df.loc[implausible, "diameter_breast_height"] = pd.NA

    for col in ("common_name", "scientific_name", "genus", "family"):
        df[col] = df[col].astype("string").str.strip()

    df["useful_life_expectency"] = df["useful_life_expectency"].fillna("Not Assessed")

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )
    return gdf.to_crs(config["crs"]["working"])


def load_and_clean_canopy(config: dict) -> gpd.GeoDataFrame:
    """Load tree canopy polygons, fix invalid geometries, and clip to the study area."""
    path = _raw_dir(config) / "tree_canopies.geojson"
    gdf = gpd.read_file(path)

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        logger.warning("Fixing %d invalid canopy geometries with buffer(0)", invalid.sum())
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    study_area = gpd.GeoSeries([_study_area_polygon(config)], crs="EPSG:4326").to_crs(gdf.crs)
    gdf = gpd.clip(gdf, study_area)
    return gdf.to_crs(config["crs"]["working"])


# Columns worth keeping per OSM layer — parks in particular arrives with
# ~140 mostly-empty free-form tag columns (buildings/water were already
# trimmed at download time; parks wasn't).
_OSM_KEEP_COLUMNS = {
    "buildings": ["building", "name"],
    "roads": ["osmid", "highway", "name", "oneway", "length"],
    "parks": ["leisure", "landuse", "name"],
    "water": ["natural", "waterway", "name"],
}


def load_and_clean_osm(config: dict) -> dict[str, gpd.GeoDataFrame]:
    """Load OSM buildings, roads, parks and water, trimming sparse tag columns and adding building area."""
    osm_dir = _raw_dir(config) / "osm"
    working_crs = config["crs"]["working"]
    layers: dict[str, gpd.GeoDataFrame] = {}

    for name, keep_cols in _OSM_KEEP_COLUMNS.items():
        gdf = gpd.read_file(osm_dir / f"{name}.geojson")
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            logger.warning("Fixing %d invalid %s geometries with buffer(0)", invalid.sum(), name)
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
        cols = [c for c in keep_cols if c in gdf.columns] + ["geometry"]
        gdf = gdf[cols].to_crs(working_crs)
        if name == "roads":
            # osmnx edge attributes (osmid, highway, name, oneway) can be a
            # scalar or a list per edge when multiple OSM ways were merged
            # into one graph edge — mixed list/scalar columns break Arrow.
            for col in ("osmid", "highway", "name", "oneway"):
                if col in gdf.columns:
                    gdf[col] = gdf[col].apply(
                        lambda v: ";".join(map(str, v)) if isinstance(v, list) else v
                    ).astype("string")
        layers[name] = gdf

    layers["buildings"]["building_area_sqm"] = layers["buildings"].geometry.area
    return layers


def load_and_clean_heat(config: dict) -> dict[str, gpd.GeoDataFrame]:
    """Load the manually-ordered urban heat / vegetation / HVI shapefiles.

    These are vector polygons (ESRI Shapefile, GDA94 geographic), not a
    raster — the project guide's ``load_heat_raster()`` doesn't apply here.
    Each dataset is reprojected to the working CRS and clipped to the study
    area. Column names are NOT harmonised across years here (e.g. HVI vs
    HVI_INDEX between the 2014 and 2018 heat vulnerability index shapefiles)
    — that's a Day 3 feature-engineering concern, not a Day 2 loading one.
    """
    heat_dir = _raw_dir(config) / "urban_heat"
    working_crs = config["crs"]["working"]
    study_area = gpd.GeoSeries([_study_area_polygon(config)], crs="EPSG:4326")

    datasets: dict[str, gpd.GeoDataFrame] = {}
    for dataset_dir in sorted(p for p in heat_dir.iterdir() if p.is_dir()):
        shapefiles = list(dataset_dir.glob("**/*.shp"))
        if not shapefiles:
            continue
        gdf = gpd.read_file(shapefiles[0])
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            logger.warning("Fixing %d invalid geometries in %s with buffer(0)", invalid.sum(), dataset_dir.name)
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
        clip_area = study_area.to_crs(gdf.crs)
        gdf = gpd.clip(gdf, clip_area)
        datasets[dataset_dir.name] = gdf.to_crs(working_crs)
    return datasets


def load_and_clean_abs_boundaries(config: dict) -> gpd.GeoDataFrame:
    """Load ABS SA2 boundaries and filter to Greater Melbourne (GCC_NAME21)."""
    shapefiles = list((_raw_dir(config) / "abs_boundaries").glob("*.shp"))
    gdf = gpd.read_file(shapefiles[0])
    melb = gdf[gdf["GCC_NAME21"] == "Greater Melbourne"].copy()
    return melb.to_crs(config["crs"]["working"])


def save_interim(gdf: gpd.GeoDataFrame, config: dict, name: str) -> Path:
    """Save a cleaned GeoDataFrame to data/interim/<name>.parquet."""
    dest_dir = PROJECT_ROOT / config["paths"]["data_interim"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{name}.parquet"
    gdf.to_parquet(dest)
    logger.info("Saved %d rows to %s", len(gdf), dest)
    return dest


def main() -> None:
    config = load_config()

    save_interim(load_and_clean_trees(config), config, "trees")
    save_interim(load_and_clean_canopy(config), config, "tree_canopy")
    save_interim(load_and_clean_abs_boundaries(config), config, "sa2_boundaries")

    for name, gdf in load_and_clean_osm(config).items():
        save_interim(gdf, config, f"osm_{name}")

    for name, gdf in load_and_clean_heat(config).items():
        save_interim(gdf, config, name.lower())


if __name__ == "__main__":
    main()

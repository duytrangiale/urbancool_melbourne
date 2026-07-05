"""Basic validation checks for raw data landed by ``src/data/download.py``.

Run as a script to print a report covering: which expected files are
present, row/feature counts, CRS, and missing-value summaries.

    python -m src.data.validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _raw_dir(config: dict) -> Path:
    return PROJECT_ROOT / config["paths"]["data_raw"]


def validate_melbourne_trees(config: dict) -> dict:
    path = _raw_dir(config) / "melbourne_trees.csv"
    if not path.exists():
        return {"present": False}

    df = pd.read_csv(path, sep=None, engine="python")
    return {
        "present": True,
        "rows": len(df),
        "columns": list(df.columns),
        "missing_counts": df.isna().sum().to_dict(),
    }


def validate_tree_canopies(config: dict) -> dict:
    path = _raw_dir(config) / "tree_canopies.geojson"
    if not path.exists():
        return {"present": False}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])
    return {
        "present": True,
        "feature_count": len(features),
        "geometry_types": sorted({f["geometry"]["type"] for f in features if f.get("geometry")}),
    }


def validate_osm_features(config: dict) -> dict:
    osm_dir = _raw_dir(config) / "osm"
    expected = ["buildings.geojson", "roads.geojson", "parks.geojson", "water.geojson"]
    report = {}
    for name in expected:
        path = osm_dir / name
        if not path.exists():
            report[name] = {"present": False}
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        report[name] = {"present": True, "feature_count": len(data.get("features", []))}
    return report


def validate_bom_weather(config: dict) -> dict:
    path = _raw_dir(config) / "bom_weather_recent.json"
    if not path.exists():
        return {"present": False}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    obs = data.get("observations", {}).get("data", [])
    return {"present": True, "observation_count": len(obs)}


def validate_abs_boundaries(config: dict) -> dict:
    abs_dir = _raw_dir(config) / "abs_boundaries"
    if not abs_dir.exists():
        return {"present": False}

    shapefiles = list(abs_dir.glob("*.shp"))
    if not shapefiles:
        return {"present": False}

    import geopandas as gpd

    gdf = gpd.read_file(shapefiles[0])
    return {
        "present": True,
        "rows": len(gdf),
        "crs": str(gdf.crs),
        "columns": list(gdf.columns),
    }


def validate_urban_heat(config: dict) -> dict:
    """Load each manually-ordered urban heat/vegetation shapefile and report shape/CRS/validity.

    Datasets are nested under data/raw/urban_heat/<DATASET_NAME>/.../*.shp
    (DataShare Vic's export layout), so each dataset's own .shp is located
    independently rather than assuming a flat directory.
    """
    heat_dir = _raw_dir(config) / "urban_heat"
    if not heat_dir.exists():
        return {"present": False}

    import geopandas as gpd

    dataset_dirs = [p for p in heat_dir.iterdir() if p.is_dir()]
    if not dataset_dirs:
        return {"present": False}

    report: dict = {}
    for dataset_dir in sorted(dataset_dirs):
        shapefiles = list(dataset_dir.glob("**/*.shp"))
        if not shapefiles:
            report[dataset_dir.name] = {"present": False}
            continue
        gdf = gpd.read_file(shapefiles[0])
        report[dataset_dir.name] = {
            "present": True,
            "rows": len(gdf),
            "crs": str(gdf.crs),
            "geom_types": sorted(gdf.geom_type.unique().tolist()),
            "valid_geometries": f"{gdf.geometry.is_valid.sum()}/{len(gdf)}",
            "columns": list(gdf.columns),
        }
    return report


def main() -> None:
    config = load_config()

    checks = {
        "melbourne_trees": validate_melbourne_trees(config),
        "tree_canopies": validate_tree_canopies(config),
        "osm_features": validate_osm_features(config),
        "bom_weather": validate_bom_weather(config),
        "abs_boundaries": validate_abs_boundaries(config),
        "urban_heat": validate_urban_heat(config),
    }

    logger.info("Validation report:")
    for name, report in checks.items():
        logger.info("  %s: %s", name, report)


if __name__ == "__main__":
    main()

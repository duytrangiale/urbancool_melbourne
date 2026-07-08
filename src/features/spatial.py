"""Combines vegetation, urban morphology, and target-variable features into the final feature matrix.

Run as a script to produce ``data/processed/feature_matrix.csv``:

    python -m src.features.spatial
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from src.data.loaders import PROJECT_ROOT, load_config
from src.features._utils import area_weighted_mean
from src.features.urban_morphology import compute_urban_features
from src.features.vegetation import (
    city_of_melbourne_coverage_mask,
    compute_canopy_features,
    compute_state_vegetation_features,
    compute_tree_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CODE_COL = "SA2_CODE21"


def compute_heat_target(heat_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame, code_col: str = CODE_COL) -> pd.DataFrame:
    """Area-weighted mean/max UHI (2018) per SA2.

    Uses the same area-weighted overlay technique as the state vegetation features
    (see src/features/_utils.py and DAY_3.md) — NOT a plain groupby on SA2_MAIN16,
    because that 2016-vintage code doesn't reliably match the 2021 SA2 boundaries used
    everywhere else in this pipeline (see DAY_2.md, section B7).
    """
    agg = area_weighted_mean(heat_gdf, boundaries_gdf, ["UHI18_M"], code_col)
    result = agg.reindex(boundaries_gdf[code_col].values)
    result.index.name = code_col
    result = result.reset_index().rename(
        columns={"UHI18_M": "mean_uhi_2018", "UHI18_M_max": "max_uhi_2018", "mesh_block_count": "heat_mesh_block_count"}
    )
    return result[[code_col, "mean_uhi_2018", "max_uhi_2018", "heat_mesh_block_count"]]


def build_feature_matrix(config: dict | None = None) -> pd.DataFrame:
    """Load every cleaned interim dataset, compute all features, and merge into one table."""
    config = config or load_config()
    interim = PROJECT_ROOT / config["paths"]["data_interim"]

    sa2 = gpd.read_parquet(interim / "sa2_boundaries.parquet")
    trees = gpd.read_parquet(interim / "trees.parquet")
    canopy = gpd.read_parquet(interim / "tree_canopy.parquet")
    veg2018 = gpd.read_parquet(interim / "vegetation_cover_2018.parquet")
    buildings = gpd.read_parquet(interim / "osm_buildings.parquet")
    roads = gpd.read_parquet(interim / "osm_roads.parquet")
    parks = gpd.read_parquet(interim / "osm_parks.parquet")
    water = gpd.read_parquet(interim / "osm_water.parquet")
    heat = gpd.read_parquet(interim / "heat_urban_heat_2018.parquet")

    logger.info("Computing City of Melbourne coverage mask...")
    coverage_mask = city_of_melbourne_coverage_mask(canopy)
    logger.info("Computing tree features (City of Melbourne LGA only)...")
    tree_feats = compute_tree_features(trees, canopy, sa2, CODE_COL, coverage_mask=coverage_mask)
    logger.info("Computing canopy features (City of Melbourne LGA only)...")
    canopy_feats = compute_canopy_features(canopy, sa2, CODE_COL, coverage_mask=coverage_mask)
    logger.info("Computing state vegetation features (full coverage)...")
    veg_feats = compute_state_vegetation_features(veg2018, sa2, CODE_COL)
    logger.info("Computing urban morphology features...")
    urban_feats = compute_urban_features(buildings, roads, parks, water, sa2, CODE_COL)
    logger.info("Computing heat target...")
    heat_target = compute_heat_target(heat, sa2, CODE_COL)

    matrix = sa2[[CODE_COL, "SA2_NAME21"]].copy()
    matrix["area_sqkm"] = sa2.geometry.area / 1_000_000

    for feats in (tree_feats, canopy_feats, veg_feats, urban_feats, heat_target):
        matrix = matrix.merge(feats, on=CODE_COL, how="left")

    return matrix


def save_feature_matrix(matrix: pd.DataFrame, config: dict | None = None):
    config = config or load_config()
    dest_dir = PROJECT_ROOT / config["paths"]["data_processed"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "feature_matrix.csv"
    matrix.to_csv(dest, index=False)
    logger.info("Saved %d rows x %d cols to %s", matrix.shape[0], matrix.shape[1], dest)
    return dest


def main() -> None:
    config = load_config()
    matrix = build_feature_matrix(config)
    save_feature_matrix(matrix, config)


if __name__ == "__main__":
    main()

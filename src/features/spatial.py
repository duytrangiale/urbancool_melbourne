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
    """Area-weighted mean/max UHI (2018) per boundary.

    Uses the same area-weighted overlay technique as the state vegetation features
    (see src/features/_utils.py and DAY_3.md) — NOT a plain groupby on SA2_MAIN16,
    because that 2016-vintage code doesn't reliably match the 2021 boundaries used
    everywhere else in this pipeline (see DAY_2.md, section B7).
    """
    agg = area_weighted_mean(heat_gdf, boundaries_gdf, ["UHI18_M"], code_col)
    result = agg.reindex(boundaries_gdf[code_col].values)
    result.index.name = code_col
    result = result.reset_index().rename(
        columns={"UHI18_M": "mean_uhi_2018", "UHI18_M_max": "max_uhi_2018", "mesh_block_count": "heat_mesh_block_count"}
    )
    return result[[code_col, "mean_uhi_2018", "max_uhi_2018", "heat_mesh_block_count"]]


def compute_demographic_features(hvi_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame, code_col: str = CODE_COL) -> pd.DataFrame:
    """Area-weighted population density and care-need share, from the Vic Government's
    2018 Heat Vulnerability Index dataset (2016-Census-derived, published on 2016 SA1
    boundaries — same area-weighted-overlay technique as compute_heat_target handles the
    boundary-vintage mismatch).

    Deliberately excludes the dataset's own ``HVI_INDEX`` column: per its metadata PDF,
    HVI_INDEX is itself computed *from* Landsat-derived Land Surface Temperature (the
    same underlying heat signal as this project's ``mean_uhi_2018`` target) combined with
    these demographic indicators — including it as a model feature would be target
    leakage, the same reasoning that excludes max_uhi_2018/heat_mesh_block_count (see
    DAY_4.md, B1). POPU_DENS and PER_CARE are pure 2016-Census demographic inputs, not
    derived from heat data, so they're safe to use as predictors.

    Note: the exact units of POPU_DENS/PER_CARE are not independently confirmed here (the
    Vic Government data dictionary PDF that defines them returned HTTP 403 to an
    automated fetch) — used as relative numeric features, which doesn't require knowing
    the exact unit, but worth checking before quoting an absolute value publicly.
    """
    agg = area_weighted_mean(hvi_gdf, boundaries_gdf, ["POPU_DENS", "PER_CARE"], code_col)
    result = agg.reindex(boundaries_gdf[code_col].values)
    result.index.name = code_col
    result = result.reset_index().rename(columns={"POPU_DENS": "population_density", "PER_CARE": "pct_population_needing_care"})
    return result[[code_col, "population_density", "pct_population_needing_care"]]


def build_feature_matrix(
    boundaries_file: str = "sa2_boundaries.parquet",
    code_col: str = CODE_COL,
    name_col: str = "SA2_NAME21",
    config: dict | None = None,
) -> pd.DataFrame:
    """Load every cleaned interim dataset, compute all features, and merge into one table
    at the given boundary resolution (SA2 by default; pass the SA1 boundaries/code column
    for the higher-resolution matrix — see DAY_4.md's Part G for why).
    """
    config = config or load_config()
    interim = PROJECT_ROOT / config["paths"]["data_interim"]

    boundaries = gpd.read_parquet(interim / boundaries_file)
    trees = gpd.read_parquet(interim / "trees.parquet")
    canopy = gpd.read_parquet(interim / "tree_canopy.parquet")
    veg2018 = gpd.read_parquet(interim / "vegetation_cover_2018.parquet")
    buildings = gpd.read_parquet(interim / "osm_buildings.parquet")
    roads = gpd.read_parquet(interim / "osm_roads.parquet")
    parks = gpd.read_parquet(interim / "osm_parks.parquet")
    water = gpd.read_parquet(interim / "osm_water.parquet")
    heat = gpd.read_parquet(interim / "heat_urban_heat_2018.parquet")
    hvi = gpd.read_parquet(interim / "heat_vulnerability_index_2018.parquet")

    logger.info("Computing City of Melbourne coverage mask...")
    coverage_mask = city_of_melbourne_coverage_mask(canopy)
    logger.info("Computing tree features (City of Melbourne LGA only)...")
    tree_feats = compute_tree_features(trees, canopy, boundaries, code_col, coverage_mask=coverage_mask)
    logger.info("Computing canopy features (City of Melbourne LGA only)...")
    canopy_feats = compute_canopy_features(canopy, boundaries, code_col, coverage_mask=coverage_mask)
    logger.info("Computing state vegetation features (full coverage)...")
    veg_feats = compute_state_vegetation_features(veg2018, boundaries, code_col)
    logger.info("Computing urban morphology features...")
    urban_feats = compute_urban_features(buildings, roads, parks, water, boundaries, code_col)
    logger.info("Computing heat target...")
    heat_target = compute_heat_target(heat, boundaries, code_col)
    logger.info("Computing demographic features...")
    demo_feats = compute_demographic_features(hvi, boundaries, code_col)

    matrix = boundaries[[code_col, name_col]].copy()
    matrix["area_sqkm"] = boundaries.geometry.area / 1_000_000

    for feats in (tree_feats, canopy_feats, veg_feats, urban_feats, heat_target, demo_feats):
        matrix = matrix.merge(feats, on=code_col, how="left")

    return matrix


def save_feature_matrix(matrix: pd.DataFrame, filename: str = "feature_matrix.csv", config: dict | None = None):
    config = config or load_config()
    dest_dir = PROJECT_ROOT / config["paths"]["data_processed"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    matrix.to_csv(dest, index=False)
    logger.info("Saved %d rows x %d cols to %s", matrix.shape[0], matrix.shape[1], dest)
    return dest


def main() -> None:
    config = load_config()

    sa2_matrix = build_feature_matrix("sa2_boundaries.parquet", "SA2_CODE21", "SA2_NAME21", config)
    save_feature_matrix(sa2_matrix, "feature_matrix.csv", config)

    sa1_matrix = build_feature_matrix("sa1_boundaries.parquet", "SA1_CODE21", "SA2_NAME21", config)
    save_feature_matrix(sa1_matrix, "feature_matrix_sa1.csv", config)


if __name__ == "__main__":
    main()

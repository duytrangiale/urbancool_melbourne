"""Tree and canopy feature engineering — aggregates point/polygon vegetation data up to SA2 level.

Every function here takes a cleaned GeoDataFrame (from ``src/data/loaders.py``) plus the
SA2 boundaries GeoDataFrame, and returns a plain (non-spatial) DataFrame indexed by the
SA2 code, ready to be merged into the final feature matrix in ``src/features/spatial.py``.

Two vegetation data sources are combined here, and they do NOT cover the same area:

- The City of Melbourne tree inventory and canopy polygons only cover the City of
  Melbourne LGA (23 of 361 Greater Melbourne SA2 areas) — confirmed during Day 3
  exploration (see DAY_3.md). Features from this source are set to NaN, not 0, outside
  that coverage area, since 0 would wrongly claim "no trees" for suburbs this dataset
  simply never surveyed.
- The Victorian Government vegetation-cover dataset (Mesh Block level, part of the same
  collection as the heat data) covers 96% of SA2 areas, so it's used as the primary,
  full-coverage vegetation feature source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from src.features._utils import area_weighted_mean

# Ordinal mid-point (in years) used to turn the categorical useful_life_expectency
# field into a numeric feature. "> 41 years" is open-ended, so 45 is a representative
# (not exact) mid-point for that bucket.
_USEFUL_LIFE_MIDPOINT_YEARS = {
    "< 10 years": 5.0,
    "11 - 20 years": 15.5,
    "21 - 30 years": 25.5,
    "31 - 40 years": 35.5,
    "> 41 years": 45.0,
    "Not Assessed": np.nan,
}


def _shannon_diversity(species: pd.Series) -> float:
    """Shannon diversity index H = -sum(p_i * ln(p_i)) over species proportions."""
    counts = species.value_counts()
    if counts.empty:
        return np.nan
    proportions = counts / counts.sum()
    return float(-(proportions * np.log(proportions)).sum())


def city_of_melbourne_coverage_mask(canopy_gdf: gpd.GeoDataFrame):
    """Convex hull of the canopy polygons, as a proxy for "area this data source actually surveyed".

    Canopy is derived from a near-complete aerial-imagery sweep of the LGA (unlike
    individual street trees, which have real gaps even within a covered area), so its
    convex hull is a good approximation of the LGA boundary without needing to download
    a separate council-boundary file.

    Uses each polygon's *centroid*, not the full polygon, before unioning: a convex hull
    only cares about the outermost points, and unioning ~58K simple points is orders of
    magnitude faster than unioning ~58K complex polygons (a full-polygon union computes
    exact boundary intersections; a point union just collects coordinates) — the
    resulting hull is visually identical for this coverage-classification purpose. This
    one call is also expensive enough that callers should compute it once and pass it to
    both ``compute_tree_features`` and ``compute_canopy_features`` rather than letting
    each recompute it (see DAY_3.md for the ~15-minute performance bug this caused).
    """
    return canopy_gdf.geometry.centroid.union_all().convex_hull


def compute_tree_features(
    trees_gdf: gpd.GeoDataFrame,
    canopy_gdf: gpd.GeoDataFrame,
    boundaries_gdf: gpd.GeoDataFrame,
    code_col: str = "SA2_CODE21",
    coverage_mask=None,
) -> pd.DataFrame:
    """Spatially join trees to SA2 areas and aggregate to per-area tree statistics.

    Returns a DataFrame indexed by ``code_col`` with: tree_count, tree_density_per_ha,
    mean_diameter_cm, species_diversity_shannon, pct_long_life_expectancy,
    mean_useful_life_years — all NaN outside the City of Melbourne LGA coverage area.

    ``coverage_mask``: pass a precomputed ``city_of_melbourne_coverage_mask(canopy_gdf)``
    if also calling ``compute_canopy_features`` on the same data, to avoid computing it
    twice (see that function's docstring).
    """
    joined = gpd.sjoin(trees_gdf, boundaries_gdf[[code_col, "geometry"]], predicate="within", how="inner")

    area_ha = (boundaries_gdf.set_index(code_col).geometry.area / 10_000).rename("area_ha")

    grouped = joined.groupby(code_col)
    tree_count = grouped.size().rename("tree_count")
    mean_diameter = grouped["diameter_breast_height"].mean().rename("mean_diameter_cm")
    species_diversity = grouped["scientific_name"].apply(_shannon_diversity).rename("species_diversity_shannon")

    assessed = joined[joined["useful_life_expectency"] != "Not Assessed"]
    assessed_grouped = assessed.groupby(code_col)
    pct_long_life = (
        assessed_grouped["useful_life_expectency"]
        .apply(lambda s: (s == "> 41 years").mean())
        .rename("pct_long_life_expectancy")
    )

    life_years = joined["useful_life_expectency"].map(_USEFUL_LIFE_MIDPOINT_YEARS)
    mean_useful_life = life_years.groupby(joined[code_col]).mean().rename("mean_useful_life_years")

    result = pd.concat([tree_count, mean_diameter, species_diversity, pct_long_life, mean_useful_life], axis=1)
    result = result.reindex(boundaries_gdf[code_col])
    result.index.name = code_col

    if coverage_mask is None:
        coverage_mask = city_of_melbourne_coverage_mask(canopy_gdf)
    covered = boundaries_gdf.set_index(code_col).geometry.intersects(coverage_mask)

    covered = covered.reindex(result.index).fillna(False)
    # Within the coverage area, "no matching row" genuinely means zero trees.
    result.loc[covered, "tree_count"] = result.loc[covered, "tree_count"].fillna(0)
    # Outside it, every column is unknown/not-surveyed — leave as NaN, not zero.
    result.loc[~covered, :] = np.nan
    result["tree_count"] = result["tree_count"].astype("Int64")

    result["tree_density_per_ha"] = result["tree_count"] / result.index.map(area_ha)

    result = result.reset_index()
    return result[
        [code_col, "tree_count", "tree_density_per_ha", "mean_diameter_cm",
         "species_diversity_shannon", "pct_long_life_expectancy", "mean_useful_life_years"]
    ]


def compute_canopy_features(
    canopy_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame, code_col: str = "SA2_CODE21", coverage_mask=None
) -> pd.DataFrame:
    """Overlay City of Melbourne tree canopy polygons with SA2 boundaries.

    Returns ``canopy_coverage_ratio_city`` — NaN outside the LGA coverage area (see
    module docstring). This is a supplementary, high-resolution feature; the
    full-coverage vegetation feature is ``compute_state_vegetation_features`` below.

    ``coverage_mask``: see ``compute_tree_features`` — pass the same precomputed mask if
    calling both functions to avoid recomputing it.
    """
    overlay = gpd.overlay(
        canopy_gdf[["geometry"]], boundaries_gdf[[code_col, "geometry"]], how="intersection"
    )
    overlay["canopy_area_sqm"] = overlay.geometry.area

    canopy_area = overlay.groupby(code_col)["canopy_area_sqm"].sum()
    sa2_area = boundaries_gdf.set_index(code_col).geometry.area

    result = pd.DataFrame({"canopy_area_sqm": canopy_area}).reindex(boundaries_gdf[code_col])
    result.index.name = code_col

    if coverage_mask is None:
        coverage_mask = city_of_melbourne_coverage_mask(canopy_gdf)
    covered = boundaries_gdf.set_index(code_col).geometry.intersects(coverage_mask)
    covered = covered.reindex(result.index).fillna(False)
    result.loc[covered, "canopy_area_sqm"] = result.loc[covered, "canopy_area_sqm"].fillna(0.0)
    result.loc[~covered, :] = np.nan

    result["canopy_coverage_ratio_city"] = result["canopy_area_sqm"] / result.index.map(sa2_area)
    result = result.reset_index()
    return result[[code_col, "canopy_coverage_ratio_city"]]


def compute_state_vegetation_features(veg_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame, code_col: str = "SA2_CODE21") -> pd.DataFrame:
    """Area-weighted mean vegetation-cover percentages from the Victorian Government dataset.

    Full-coverage (96% of SA2 areas — see module docstring) primary vegetation feature.
    ``PERANYVEG``/``PERANYTREE`` are already 0-100 mesh-block-level percentages; the
    weighted mean here is the standard way to combine values from polygons of a finer
    geography (Mesh Block) into a coarser one (SA2) — see DAY_3.md.
    """
    agg = area_weighted_mean(veg_gdf, boundaries_gdf, ["PERANYVEG", "PERANYTREE"], code_col)
    result = agg.reindex(boundaries_gdf[code_col].values)
    result.index.name = code_col
    result = result.reset_index().rename(
        columns={"PERANYVEG": "vegetation_cover_pct_state", "PERANYTREE": "tree_cover_pct_state"}
    )
    return result[[code_col, "vegetation_cover_pct_state", "tree_cover_pct_state"]]

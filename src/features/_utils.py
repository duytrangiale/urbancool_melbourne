"""Shared geospatial aggregation helpers used by more than one feature module."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd


def area_weighted_mean(
    source_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame, value_cols: list[str], code_col: str
) -> pd.DataFrame:
    """Aggregate polygon attribute columns up to a coarser boundary via area-weighted mean.

    ``source_gdf`` (e.g. Mesh Block-level polygons) is overlaid with ``boundaries_gdf``
    (e.g. SA2 polygons), splitting any source polygon that straddles a boundary into the
    pieces that fall in each area. Each piece's contribution to the mean is weighted by
    its post-split area — this is the correct way to combine polygons of two different,
    only-partially-aligned geographies (see DAY_3.md for why a plain groupby-by-code
    would be wrong here).

    Returns a DataFrame indexed by ``code_col`` with one weighted-mean column per
    ``value_cols`` entry, plus ``<col>_max`` for reference.
    """
    overlay = gpd.overlay(
        source_gdf[value_cols + ["geometry"]], boundaries_gdf[[code_col, "geometry"]], how="intersection"
    )
    overlay["_piece_area"] = overlay.geometry.area

    grouped = overlay.groupby(code_col)
    total_area = grouped["_piece_area"].sum()

    result = pd.DataFrame(index=total_area.index)
    for col in value_cols:
        weighted_sum = (overlay[col] * overlay["_piece_area"]).groupby(overlay[code_col]).sum()
        result[col] = weighted_sum / total_area
        result[f"{col}_max"] = grouped[col].max()

    result["mesh_block_count"] = grouped.size()
    return result

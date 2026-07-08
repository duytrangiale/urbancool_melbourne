"""Urban morphology feature engineering — buildings, roads, parks, water, aggregated to SA2 level."""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

# Typical carriageway width in metres per OSM `highway` type, used only to turn road
# *length* (all we actually have) into an approximate road *area* for the
# impervious-surface estimate. These are reasonable Australian-road-width defaults, not
# measured values — see DAY_3.md for why an approximation is unavoidable here.
_ROAD_WIDTH_M = {
    "motorway": 14.0, "motorway_link": 10.0,
    "trunk": 12.0, "trunk_link": 9.0,
    "primary": 10.0, "primary_link": 8.0,
    "secondary": 9.0, "secondary_link": 7.0,
    "tertiary": 8.0, "tertiary_link": 7.0,
    "unclassified": 6.0, "residential": 6.0,
    "living_street": 5.0, "service": 4.0,
}
_DEFAULT_ROAD_WIDTH_M = 6.0


def compute_building_features(buildings_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame, code_col: str = "SA2_CODE21") -> pd.DataFrame:
    """Assign each building to the SA2 whose polygon contains its centroid, then aggregate.

    Uses a centroid-based join rather than a full polygon overlay (gpd.overlay) for
    performance: with 660K buildings, splitting every polygon that happens to straddle
    an SA2 boundary is expensive for a negligible accuracy gain, since the number of
    buildings actually crossing a boundary line is a tiny fraction of the total (see
    DAY_3.md for the reasoning and the measured trade-off).
    """
    centroids = buildings_gdf.copy()
    centroids["geometry"] = buildings_gdf.geometry.centroid

    joined = gpd.sjoin(centroids, boundaries_gdf[[code_col, "geometry"]], predicate="within", how="inner")

    area_ha = boundaries_gdf.set_index(code_col).geometry.area / 10_000
    sa2_area_sqm = boundaries_gdf.set_index(code_col).geometry.area

    grouped = joined.groupby(code_col)
    building_count = grouped.size().rename("building_count")
    total_building_area = grouped["building_area_sqm"].sum().rename("total_building_area_sqm")
    mean_building_area = grouped["building_area_sqm"].mean().rename("mean_building_area_sqm")

    result = pd.concat([building_count, total_building_area, mean_building_area], axis=1)
    result = result.reindex(boundaries_gdf[code_col]).fillna({"building_count": 0, "total_building_area_sqm": 0.0})
    result.index.name = code_col
    result["building_count"] = result["building_count"].astype(int)

    result["building_density_per_ha"] = result["building_count"] / result.index.map(area_ha)
    result["building_coverage_ratio"] = result["total_building_area_sqm"] / result.index.map(sa2_area_sqm)

    result = result.reset_index()
    return result[[code_col, "building_count", "building_density_per_ha", "building_coverage_ratio", "mean_building_area_sqm", "total_building_area_sqm"]]


def compute_road_features(roads_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame, code_col: str = "SA2_CODE21") -> pd.DataFrame:
    """Clip roads to SA2 boundaries (one vectorised overlay, not a per-area loop) and aggregate.

    Also estimates road surface area from length x an assumed width-per-road-type, for
    the impervious-surface estimate — roads only have length in this dataset, not a
    mapped carriageway polygon, so this is a deliberate approximation (see module
    docstring and DAY_3.md).
    """
    overlay = gpd.overlay(roads_gdf[["highway", "geometry"]], boundaries_gdf[[code_col, "geometry"]], how="intersection")
    overlay["clipped_length_m"] = overlay.geometry.length

    primary_type = overlay["highway"].fillna("").apply(lambda s: s.split(";")[0] if s else "")
    overlay["width_m"] = primary_type.map(_ROAD_WIDTH_M).fillna(_DEFAULT_ROAD_WIDTH_M)
    overlay["estimated_area_sqm"] = overlay["clipped_length_m"] * overlay["width_m"]

    grouped = overlay.groupby(code_col)
    road_length_m = grouped["clipped_length_m"].sum().rename("road_length_m")
    estimated_road_area = grouped["estimated_area_sqm"].sum().rename("estimated_road_area_sqm")

    sa2_area_sqkm = boundaries_gdf.set_index(code_col).geometry.area / 1_000_000

    result = pd.concat([road_length_m, estimated_road_area], axis=1)
    result = result.reindex(boundaries_gdf[code_col]).fillna(0.0)
    result.index.name = code_col

    result["road_length_km"] = result["road_length_m"] / 1_000
    result["road_density_km_per_sqkm"] = result["road_length_km"] / result.index.map(sa2_area_sqkm)

    result = result.reset_index()
    return result[[code_col, "road_length_km", "road_density_km_per_sqkm", "estimated_road_area_sqm"]]


def compute_park_water_features(parks_gdf: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame, code_col: str = "SA2_CODE21") -> pd.DataFrame:
    """Park and water-body coverage ratios via polygon overlay.

    Both ``parks_gdf`` (a park can be tagged on a single OSM node, giving a handful of
    Point features) and ``water_gdf`` (polygons for lakes/bays, lines for OSM
    'waterway' rivers, plus a few points) mix geometry types. ``gpd.overlay`` requires a
    single geometry type per input, and only polygons have a defined area anyway, so
    both are filtered to their Polygon/MultiPolygon subset here. The excluded
    line/point features are still used for the nearest-water distance feature (see
    compute_distance_features), where geometry type doesn't matter.
    """
    sa2_area_sqm = boundaries_gdf.set_index(code_col).geometry.area

    park_polygons = parks_gdf[parks_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    park_overlay = gpd.overlay(park_polygons[["geometry"]], boundaries_gdf[[code_col, "geometry"]], how="intersection")
    park_overlay["area_sqm"] = park_overlay.geometry.area
    park_area = park_overlay.groupby(code_col)["area_sqm"].sum()

    water_polygons = water_gdf[water_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    water_overlay = gpd.overlay(water_polygons[["geometry"]], boundaries_gdf[[code_col, "geometry"]], how="intersection")
    water_overlay["area_sqm"] = water_overlay.geometry.area
    water_area = water_overlay.groupby(code_col)["area_sqm"].sum()

    result = pd.DataFrame({"park_area_sqm": park_area, "water_area_sqm": water_area})
    result = result.reindex(boundaries_gdf[code_col]).fillna(0.0)
    result.index.name = code_col

    result["park_coverage_ratio"] = result["park_area_sqm"] / result.index.map(sa2_area_sqm)
    result["water_coverage_ratio"] = result["water_area_sqm"] / result.index.map(sa2_area_sqm)

    result = result.reset_index()
    return result[[code_col, "park_coverage_ratio", "water_coverage_ratio"]]


def compute_distance_features(boundaries_gdf: gpd.GeoDataFrame, parks_gdf: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame, code_col: str = "SA2_CODE21") -> pd.DataFrame:
    """Distance from each SA2's centroid to the nearest park and nearest water feature (metres)."""
    centroids = gpd.GeoDataFrame(
        boundaries_gdf[[code_col]], geometry=boundaries_gdf.geometry.centroid, crs=boundaries_gdf.crs
    )

    nearest_park = gpd.sjoin_nearest(centroids, parks_gdf[["geometry"]], distance_col="dist_to_nearest_park_m")
    nearest_park = nearest_park.drop_duplicates(subset=code_col)[[code_col, "dist_to_nearest_park_m"]]

    nearest_water = gpd.sjoin_nearest(centroids, water_gdf[["geometry"]], distance_col="dist_to_nearest_water_m")
    nearest_water = nearest_water.drop_duplicates(subset=code_col)[[code_col, "dist_to_nearest_water_m"]]

    result = nearest_park.merge(nearest_water, on=code_col, how="outer")
    return result


def compute_urban_features(
    buildings_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    parks_gdf: gpd.GeoDataFrame,
    water_gdf: gpd.GeoDataFrame,
    boundaries_gdf: gpd.GeoDataFrame,
    code_col: str = "SA2_CODE21",
) -> pd.DataFrame:
    """Combine all urban morphology features, plus an approximate impervious_ratio."""
    buildings = compute_building_features(buildings_gdf, boundaries_gdf, code_col)
    roads = compute_road_features(roads_gdf, boundaries_gdf, code_col)
    parks_water = compute_park_water_features(parks_gdf, water_gdf, boundaries_gdf, code_col)
    distances = compute_distance_features(boundaries_gdf, parks_gdf, water_gdf, code_col)

    result = buildings.merge(roads, on=code_col).merge(parks_water, on=code_col).merge(distances, on=code_col)

    sa2_area_sqm = boundaries_gdf.set_index(code_col).geometry.area
    impervious_area = result["total_building_area_sqm"] + result["estimated_road_area_sqm"]
    result["impervious_ratio"] = (impervious_area / result[code_col].map(sa2_area_sqm)).clip(upper=1.0)

    return result.drop(columns=["total_building_area_sqm", "estimated_road_area_sqm"])

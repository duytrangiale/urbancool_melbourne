"""Interactive Folium choropleth map of predicted heat vulnerability by SA2 (suburb).

Run as a script to produce ``outputs/heat_vulnerability_map.html``:

    python -m src.visualization.maps
"""

from __future__ import annotations

import logging

import folium
import geopandas as gpd
import pandas as pd

from src.data.loaders import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MELBOURNE_CENTER = [-37.8136, 144.9631]

# Tooltip fields shown on hover, in order — a mix of the prediction itself and the
# features most responsible for it (per DAY_4.md's SHAP analysis: tree cover and
# vegetation cover dominate, followed by population density and imperviousness).
TOOLTIP_FIELDS = [
    "SA2_NAME21", "predicted_mean_uhi_2018", "tree_cover_pct_state",
    "vegetation_cover_pct_state", "impervious_ratio", "population_density",
]
TOOLTIP_ALIASES = [
    "Suburb", "Predicted heat (°C above non-urban baseline)", "Tree cover (%)",
    "Vegetation cover (%)", "Impervious ratio", "Population density (per km²)",
]


def load_predictions_gdf(config: dict | None = None) -> gpd.GeoDataFrame:
    """SA2 boundaries joined to their aggregated predictions, reprojected to WGS84
    (EPSG:4326) — Folium/Leaflet requires geographic lat/lon, not this project's working
    projected CRS (EPSG:28355), so this reprojection is not optional."""
    config = config or load_config()
    interim = PROJECT_ROOT / config["paths"]["data_interim"]
    processed = PROJECT_ROOT / config["paths"]["data_processed"]

    sa2 = gpd.read_parquet(interim / "sa2_boundaries.parquet")
    predictions = pd.read_csv(processed / "predictions_sa2.csv", dtype={"SA2_CODE21": str})

    gdf = sa2[["SA2_CODE21", "geometry"]].merge(predictions, on="SA2_CODE21", how="left")

    # Simplify while still in the projected (metres) CRS, before reprojecting: Folium
    # embeds the full geometry as GeoJSON *twice* (once for the choropleth fill, once for
    # the tooltip layer), and full-resolution ABS coastline detail made the unsimplified
    # map ~19MB. A 30m tolerance is imperceptible at any zoom level this suburb-overview
    # map is meant to be viewed at, and cuts vertex count by roughly 85%.
    gdf["geometry"] = gdf.geometry.simplify(30, preserve_topology=True)
    gdf = gdf.to_crs("EPSG:4326")

    # Round for readable tooltips/legend; fill display-only NaNs (a handful of SA2s lack
    # some context features — see DAY_3/4.md — predictions themselves have full coverage).
    for col in ["predicted_mean_uhi_2018", "tree_cover_pct_state", "vegetation_cover_pct_state", "impervious_ratio", "population_density"]:
        gdf[col] = gdf[col].round(2)
    gdf[TOOLTIP_FIELDS[2:]] = gdf[TOOLTIP_FIELDS[2:]].fillna("no data")

    return gdf


def create_heat_vulnerability_map(predictions_gdf: gpd.GeoDataFrame, feature_name: str = "predicted_mean_uhi_2018") -> folium.Map:
    """Interactive choropleth of predicted heat by SA2, with a hover tooltip showing the
    suburb name, predicted value, and the features most responsible for it."""
    m = folium.Map(location=MELBOURNE_CENTER, zoom_start=10, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=predictions_gdf.__geo_interface__,
        data=predictions_gdf,
        columns=["SA2_CODE21", feature_name],
        key_on="feature.properties.SA2_CODE21",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.3,
        legend_name="Predicted heat (°C above non-urban baseline)",
        nan_fill_color="lightgray",
    ).add_to(m)

    folium.GeoJson(
        predictions_gdf,
        style_function=lambda _: {"fillOpacity": 0, "weight": 0},
        tooltip=folium.GeoJsonTooltip(fields=TOOLTIP_FIELDS, aliases=TOOLTIP_ALIASES, sticky=True),
    ).add_to(m)

    return m


def save_map(m: folium.Map, filename: str = "heat_vulnerability_map.html", config: dict | None = None):
    config = config or load_config()
    dest_dir = PROJECT_ROOT / config["paths"]["outputs"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    m.save(str(dest))
    logger.info("Saved map to %s", dest)
    return dest


def main() -> None:
    config = load_config()
    gdf = load_predictions_gdf(config)
    m = create_heat_vulnerability_map(gdf)
    save_map(m, config=config)


if __name__ == "__main__":
    main()

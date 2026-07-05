"""Automated data fetching for UrbanCool Melbourne.

Each ``download_*`` function fetches one raw data source into ``data/raw/``.
Run as a script to fetch everything that has a public API:

    python -m src.data.download

The Victorian Government urban heat data has no public API and must be
downloaded by hand; ``download_urban_heat_data`` prints instructions and
checks whether the expected file is already present.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
TIMEOUT = 60
# BOM blocks non-browser User-Agent strings on its public JSON feeds; a
# browser-like UA is required there and works fine for the other sources too.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _raw_dir(config: dict) -> Path:
    raw_dir = PROJECT_ROOT / config["paths"]["data_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def download_melbourne_trees(config: dict) -> Path:
    """Fetch the City of Melbourne tree inventory (CSV) via the Open Data API."""
    src = config["data_sources"]["melbourne_trees"]
    url = f"{src['base_url']}/{src['dataset_id']}/exports/{src['format']}"
    dest = _raw_dir(config) / "melbourne_trees.csv"

    logger.info("Downloading Melbourne tree inventory from %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def download_tree_canopies(config: dict) -> Path:
    """Fetch the City of Melbourne tree canopy polygons (GeoJSON) via the Open Data API."""
    src = config["data_sources"]["tree_canopies"]
    url = f"{src['base_url']}/{src['dataset_id']}/exports/{src['format']}"
    dest = _raw_dir(config) / "tree_canopies.geojson"

    logger.info("Downloading tree canopy polygons from %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def download_urban_heat_data(config: dict) -> Path | None:
    """Victorian Planning urban heat/vegetation data has no public API.

    The data is distributed as vector polygons (ESRI Shapefile / MapInfo /
    KML — NOT a raster GeoTIFF, despite the DELWP report title) via
    DataShare Vic, a cart-and-checkout portal in front of DEECA's GeoNetwork
    catalogue. There is no direct download URL to script against; each
    dataset has to be added to an order and checked out by hand, choosing
    "ESRI Shapefile" as the format so it loads with geopandas/fiona.

    Prints jump-to links for the six vegetation/heat/vulnerability datasets
    (see ``config['data_sources']['urban_heat']['datasets']``) and checks
    whether files have already been placed in ``data/raw/urban_heat/`` by
    hand.
    """
    src = config["data_sources"]["urban_heat"]
    dest_dir = _raw_dir(config) / "urban_heat"
    expected = list(dest_dir.glob("*")) if dest_dir.exists() else []

    if expected:
        logger.info("Found %d existing file(s) in %s — skipping manual step.", len(expected), dest_dir)
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset_links = "\n".join(
        f"       - {name}: https://datashare.maps.vic.gov.au/search?q=uuid={uuid}"
        for name, uuid in src.get("datasets", {}).items()
    )
    logger.warning(
        "Urban heat data must be ordered manually (vector polygons, not raster):\n"
        "  1. Visit the dataset pages below and click 'Add to order' on each:\n"
        "%s\n"
        "  2. Open 'My Order list' (cart icon) and check out at /order/configuration/,\n"
        "     choosing format = ESRI Shapefile (SHP).\n"
        "  3. Unzip and place the downloaded .shp/.dbf/.shx/.prj files in %s\n"
        "  Background: %s\n",
        dataset_links,
        dest_dir,
        src["portal_url"],
    )
    return None


def _tile_bbox(ox_bbox: tuple[float, float, float, float], n_lon: int, n_lat: int):
    """Split a (left, bottom, right, top) bbox into an n_lon x n_lat grid of sub-bboxes."""
    import numpy as np

    left, bottom, right, top = ox_bbox
    lons = np.linspace(left, right, n_lon + 1)
    lats = np.linspace(bottom, top, n_lat + 1)
    for i in range(n_lon):
        for j in range(n_lat):
            yield (lons[i], lats[j], lons[i + 1], lats[j + 1])


def _features_from_bbox_tiled(
    ox, ox_bbox, tags: dict, keep_columns: list[str], n_lon: int = 4, n_lat: int = 4, retries: int = 3
):
    """Fetch OSM features tile-by-tile, keeping only ``keep_columns`` + geometry.

    OSM elements can carry hundreds of distinct free-form tag keys; osmnx
    turns each into a column, and building that wide, mostly-sparse
    GeoDataFrame for a whole-metro bbox in one shot can exhaust RAM (a
    Greater-Melbourne buildings pull needed 4.5+ GB for a single dtype
    block). Tiling keeps each intermediate GeoDataFrame small, and columns
    are trimmed before concatenation so the final result stays lean.

    Smaller per-tile requests are also cheaper for the Overpass server,
    which helps when a single whole-metro request gets soft-throttled
    (observed as a connect timeout) after a burst of heavy queries — each
    tile gets its own retries instead of failing the entire fetch.
    """
    import time

    import geopandas as gpd
    import pandas as pd

    frames = []
    for tile in _tile_bbox(ox_bbox, n_lon, n_lat):
        tile_gdf = None
        for attempt in range(retries):
            try:
                tile_gdf = ox.features_from_bbox(bbox=tile, tags=tags)
                break
            except ox._errors.InsufficientResponseError:
                tile_gdf = None
                break
            except requests.exceptions.RequestException as exc:
                logger.warning("Tile %s attempt %d/%d failed: %s", tile, attempt + 1, retries, exc)
                time.sleep(5 * (attempt + 1))
        if tile_gdf is None or tile_gdf.empty:
            continue
        cols = [c for c in keep_columns if c in tile_gdf.columns]
        frames.append(tile_gdf[cols + ["geometry"]])

    if not frames:
        return gpd.GeoDataFrame(columns=keep_columns + ["geometry"], geometry="geometry")

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)


def download_osm_features(config: dict) -> dict[str, Path]:
    """Fetch building footprints, roads, parks and water bodies from OpenStreetMap via osmnx."""
    import osmnx as ox

    bbox = config["study_area"]["bbox"]  # [min_lon, min_lat, max_lon, max_lat]
    # osmnx expects (left, bottom, right, top)
    ox_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
    dest_dir = _raw_dir(config) / "osm"
    dest_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    logger.info("Downloading OSM building footprints for bbox %s (tiled to bound memory use)", ox_bbox)
    buildings = _features_from_bbox_tiled(ox, ox_bbox, tags={"building": True}, keep_columns=["building", "name"])
    dest = dest_dir / "buildings.geojson"
    buildings.to_file(dest, driver="GeoJSON")
    outputs["buildings"] = dest
    logger.info("Saved %d building features to %s", len(buildings), dest)

    logger.info("Downloading OSM road network for bbox %s", ox_bbox)
    road_graph = ox.graph_from_bbox(bbox=ox_bbox, network_type="drive")
    _, roads = ox.graph_to_gdfs(road_graph)
    dest = dest_dir / "roads.geojson"
    roads.to_file(dest, driver="GeoJSON")
    outputs["roads"] = dest
    logger.info("Saved %d road segments to %s", len(roads), dest)

    logger.info("Downloading OSM parks/green space for bbox %s", ox_bbox)
    parks = ox.features_from_bbox(
        bbox=ox_bbox, tags={"leisure": ["park", "garden"], "landuse": ["recreation_ground", "grass"]}
    )
    dest = dest_dir / "parks.geojson"
    parks.to_file(dest, driver="GeoJSON")
    outputs["parks"] = dest
    logger.info("Saved %d park features to %s", len(parks), dest)

    logger.info("Downloading OSM water bodies for bbox %s (tiled for resilience)", ox_bbox)
    water = _features_from_bbox_tiled(
        ox, ox_bbox, tags={"natural": "water", "waterway": True}, keep_columns=["natural", "waterway", "name"]
    )
    dest = dest_dir / "water.geojson"
    water.to_file(dest, driver="GeoJSON")
    outputs["water"] = dest
    logger.info("Saved %d water features to %s", len(water), dest)

    return outputs


def download_bom_weather(config: dict) -> Path:
    """Fetch recent (~72h) half-hourly observations for the configured BOM station.

    BOM's Climate Data Online historical exports are gated behind a session
    token generated in-browser (a ``p_c`` parameter tied to a "load a data
    file" click), so full historical daily records can't be scripted
    reliably. This function pulls the public JSON feed of recent
    observations instead — sufficient for contextual/validation weather
    features. For full historical daily max/min/rainfall, request a data
    file manually at http://www.bom.gov.au/climate/data/ for the station in
    ``config['data_sources']['bom_weather']['station_id']``.
    """
    wmo_id = config["data_sources"]["bom_weather"]["wmo_id"]
    # IDV60801 = Victorian half-hourly observations product, keyed by WMO number.
    url = f"https://www.bom.gov.au/fwo/IDV60801/IDV60801.{wmo_id}.json"
    dest = _raw_dir(config) / "bom_weather_recent.json"

    logger.info("Downloading recent BOM observations from %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("Saved %s (%.1f KB)", dest, dest.stat().st_size / 1e3)
    return dest


def download_abs_boundaries(config: dict) -> Path:
    """Fetch ABS SA2 digital boundary shapefiles (ASGS Edition 3)."""
    url = (
        "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/"
        "jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip"
    )
    dest_dir = _raw_dir(config) / "abs_boundaries"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ABS SA2 boundaries from %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest_dir)
    logger.info("Extracted ABS SA2 boundaries to %s", dest_dir)
    return dest_dir


def main() -> None:
    config = load_config()

    results = {
        "melbourne_trees": download_melbourne_trees(config),
        "tree_canopies": download_tree_canopies(config),
        "abs_boundaries": download_abs_boundaries(config),
        "bom_weather": download_bom_weather(config),
        "osm_features": download_osm_features(config),
        "urban_heat": download_urban_heat_data(config),
    }

    logger.info("Download summary:")
    for name, result in results.items():
        logger.info("  %-16s -> %s", name, result)


if __name__ == "__main__":
    main()

"""Bakes the parts of the dashboard that don't change per-request (the map, KPI values,
hottest/coolest lists, feature-importance bars, suburb dropdown options, and the two
dark-themed global charts) into app/static/index.html and app/static/*.png.

Run whenever the model/predictions are regenerated:

    python -m app.build_static
"""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from app.core import ACCENT, DARK_RCPARAMS, DARK_SURFACE, INK, _brighten_dark_text, complete_suburbs, green_ramp, heat_ramp, load_model, load_sa2_df, load_shap_explainer
from src.data.loaders import PROJECT_ROOT, load_config
from src.models.train import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

APP_DIR = PROJECT_ROOT / "app"
STATIC_DIR = APP_DIR / "static"
SVG_W, SVG_H = 900, 760


def _poly_to_path(geom, project) -> str:
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    parts = []
    for poly in polys:
        coords = list(poly.exterior.coords)
        d = "M " + " L ".join(f"{project(x, y)[0]:.1f},{project(x, y)[1]:.1f}" for x, y in coords) + " Z"
        parts.append(d)
    return " ".join(parts)


def _load_sa2_geometry() -> "gpd.GeoDataFrame":
    config = load_config()
    interim = PROJECT_ROOT / config["paths"]["data_interim"]
    sa2 = gpd.read_parquet(interim / "sa2_boundaries.parquet")
    predictions = load_sa2_df()
    cols = ["SA2_CODE21", "SA2_NAME21", "predicted_mean_uhi_2018", "tree_cover_pct_state", "vegetation_cover_pct_state", "impervious_ratio"]
    gdf = sa2[["SA2_CODE21", "geometry"]].merge(predictions[cols], on="SA2_CODE21")
    # Simplified for a lightweight embedded SVG — imperceptible at this zoom, ~90% fewer
    # vertices than the full ABS boundary detail (see the Day 5 map's identical reasoning).
    gdf["geometry"] = gdf.geometry.simplify(80, preserve_topology=True)
    return gdf.to_crs("EPSG:4326")


def _build_sa2_paths() -> tuple[str, dict]:
    """Main map: one <path> per SA2, pre-coloured for all three layers (heat/tree/veg) so
    the frontend can switch layers by swapping a data attribute into `fill` — no ramp
    logic duplicated in JavaScript, Python stays the single source of truth for colour."""
    gdf = _load_sa2_geometry()
    minx, miny, maxx, maxy = gdf.total_bounds

    def project(x, y):
        px = (x - minx) / (maxx - minx) * SVG_W
        py = SVG_H - (y - miny) / (maxy - miny) * SVG_H
        return px, py

    ranges = {}
    for col, ramp in [("predicted_mean_uhi_2018", heat_ramp), ("tree_cover_pct_state", green_ramp), ("vegetation_cover_pct_state", green_ramp)]:
        vmin, vmax = gdf[col].min(), gdf[col].max()
        ranges[col] = (float(vmin), float(vmax))

    path_tags = []
    for _, row in gdf.iterrows():
        name = str(row["SA2_NAME21"]).replace("&", "&amp;").replace('"', "&quot;")
        heat_t = (row["predicted_mean_uhi_2018"] - ranges["predicted_mean_uhi_2018"][0]) / (ranges["predicted_mean_uhi_2018"][1] - ranges["predicted_mean_uhi_2018"][0])
        fill_heat = heat_ramp(heat_t)
        attrs = [f'data-name="{name}"', f'data-val-heat="{row["predicted_mean_uhi_2018"]:.2f}"', f'data-fill-heat="{fill_heat}"']

        if pd.notna(row["tree_cover_pct_state"]):
            tree_t = (row["tree_cover_pct_state"] - ranges["tree_cover_pct_state"][0]) / (ranges["tree_cover_pct_state"][1] - ranges["tree_cover_pct_state"][0])
            attrs += [f'data-val-tree="{row["tree_cover_pct_state"]:.1f}"', f'data-fill-tree="{green_ramp(tree_t)}"']
        if pd.notna(row["vegetation_cover_pct_state"]):
            veg_t = (row["vegetation_cover_pct_state"] - ranges["vegetation_cover_pct_state"][0]) / (ranges["vegetation_cover_pct_state"][1] - ranges["vegetation_cover_pct_state"][0])
            attrs += [f'data-val-veg="{row["vegetation_cover_pct_state"]:.1f}"', f'data-fill-veg="{green_ramp(veg_t)}"']

        path_tags.append(f'<path d="{_poly_to_path(row.geometry, project)}" fill="{fill_heat}" class="sa2-poly" {" ".join(attrs)}></path>')

    ranges_out = {
        "heat": ranges["predicted_mean_uhi_2018"],
        "tree": ranges["tree_cover_pct_state"],
        "veg": ranges["vegetation_cover_pct_state"],
    }
    return "\n".join(path_tags), ranges_out


def _build_suburb_lookup() -> dict:
    """One entry per suburb: a small self-contained path (its own local 0-220 viewBox,
    not the main map's coordinate space — otherwise a small suburb would render as a
    barely-visible dot) for the What-If Simulator's mini-map, plus the raw feature values
    needed to draw the green/grey proportion bar client-side without another API call."""
    gdf = _load_sa2_geometry()
    lookup = {}
    pad = 14
    inner = 220 - 2 * pad
    for _, row in gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        span = max(maxx - minx, maxy - miny) or 1.0
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

        def project(x, y, cx=cx, cy=cy, span=span):
            px = pad + inner / 2 + (x - cx) / span * inner
            py = pad + inner / 2 - (y - cy) / span * inner
            return px, py

        lookup[row["SA2_NAME21"]] = {
            "d": _poly_to_path(row.geometry, project),
            "tree": None if pd.isna(row["tree_cover_pct_state"]) else round(float(row["tree_cover_pct_state"]), 1),
            "veg": None if pd.isna(row["vegetation_cover_pct_state"]) else round(float(row["vegetation_cover_pct_state"]), 1),
            "impervious": round(float(row["impervious_ratio"]), 3),
        }
    return lookup


def _rank_rows(df: pd.DataFrame, ascending: bool, cls: str) -> str:
    ranked = df.dropna(subset=["predicted_mean_uhi_2018"]).sort_values("predicted_mean_uhi_2018", ascending=ascending).head(6)
    rows = []
    for _, row in ranked.iterrows():
        name = row["SA2_NAME21"]
        rows.append(
            f'<li data-suburb="{name}" tabindex="0" role="button">'
            f'<span class="rank-name">{name}</span>'
            f'<span class="rank-val {cls}">{row["predicted_mean_uhi_2018"]:.2f}&deg;C</span></li>'
        )
    return "\n".join(rows)


def _bar_rows() -> str:
    fi = pd.read_csv(PROJECT_ROOT / "models" / "feature_importance.csv")
    top = fi.head(6)
    maxv = top["importance"].max()
    labels = {
        "tree_cover_pct_state": "Tree cover", "vegetation_cover_pct_state": "Vegetation cover",
        "population_density": "Population density", "area_sqkm": "Suburb area",
        "pct_population_needing_care": "Care-need share", "dist_to_nearest_water_m": "Dist. to water",
        "impervious_ratio": "Impervious ratio", "building_density_per_ha": "Building density",
        "road_density_km_per_sqkm": "Road density", "park_coverage_ratio": "Park coverage",
        "water_coverage_ratio": "Water coverage", "mean_building_area_sqm": "Mean building area",
        "building_coverage_ratio": "Building coverage", "dist_to_nearest_park_m": "Dist. to park",
    }
    rows = []
    for _, row in top.iterrows():
        pct_of_max = row["importance"] / maxv * 100
        pct_label = row["importance"] * 100
        label = labels.get(row["feature"], row["feature"])
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{label}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct_of_max:.1f}%"></div></div>'
            f'<span class="bar-val">{pct_label:.1f}%</span></div>'
        )
    return "\n".join(rows)


def _suburb_options(names: list[str]) -> str:
    return "\n".join(f'<option value="{n}">{n}</option>' for n in names)


def _render_dark_shap_summary():
    df = load_sa2_df()
    model = load_model()
    explainer = load_shap_explainer()

    sample = df.sample(n=min(300, len(df)), random_state=42)
    X = sample[FEATURE_COLS]
    X_imputed = pd.DataFrame(model.named_steps["impute"].transform(X), columns=FEATURE_COLS, index=X.index)
    explanation = explainer(X_imputed)

    with plt.rc_context(DARK_RCPARAMS):
        shap.summary_plot(explanation, X_imputed, show=False)
        fig = plt.gcf()
        fig.suptitle("Which features push predicted heat up or down? (300-suburb sample)", color=INK, y=1.02, fontsize=11)
        _brighten_dark_text(fig)
        fig.set_size_inches(8, 6.4)
        fig.tight_layout()
        fig.savefig(STATIC_DIR / "shap_summary_dark.png", dpi=140, bbox_inches="tight", facecolor=DARK_SURFACE)
        plt.close(fig)
    logger.info("Wrote shap_summary_dark.png (%d-suburb sample)", len(sample))


def _render_dark_scatter():
    df = load_sa2_df()
    scatter_cols = ["tree_cover_pct_state", "vegetation_cover_pct_state", "impervious_ratio"]
    titles = {"tree_cover_pct_state": "Tree cover (%)", "vegetation_cover_pct_state": "Vegetation cover (%)", "impervious_ratio": "Impervious ratio"}
    with plt.rc_context(DARK_RCPARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
        for ax, col in zip(axes, scatter_cols):
            sub = df[[col, "predicted_mean_uhi_2018"]].dropna()
            ax.scatter(sub[col], sub["predicted_mean_uhi_2018"], s=14, color=ACCENT, alpha=0.6, edgecolor="none")
            z = np.polyfit(sub[col], sub["predicted_mean_uhi_2018"], 1)
            xs = np.linspace(sub[col].min(), sub[col].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color="#eef5f7", linewidth=1.3)
            ax.set_xlabel(titles[col])
            if col == scatter_cols[0]:
                ax.set_ylabel("predicted heat (°C)")
        fig.suptitle("Predicted heat vs. each feature, across all 361 SA2 areas", color=INK, y=1.03, fontsize=11)
        _brighten_dark_text(fig)
        fig.tight_layout()
        fig.savefig(STATIC_DIR / "scatter_dark.png", dpi=140, bbox_inches="tight", facecolor=DARK_SURFACE)
        plt.close(fig)
    logger.info("Wrote scatter_dark.png")


def main() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_sa2_df()

    paths_svg, ranges = _build_sa2_paths()
    suburb_lookup = _build_suburb_lookup()
    hottest_html = _rank_rows(df, ascending=False, cls="v-hot")
    coolest_html = _rank_rows(df, ascending=True, cls="v-cool")
    bars_html = _bar_rows()
    all_names = sorted(df["SA2_NAME21"].dropna().unique().tolist())
    complete_names = complete_suburbs()
    vmin, vmax = ranges["heat"]

    template = (APP_DIR / "templates" / "index_template.html").read_text(encoding="utf-8")
    out = template.replace("__SA2_PATHS__", paths_svg)
    out = out.replace("__V_MIN__", f"{vmin:.1f}")
    out = out.replace("__V_MID__", f"{(vmin + vmax) / 2:.1f}")
    out = out.replace("__V_MAX__", f"{vmax:.1f}")
    out = out.replace("__HOTTEST_ROWS__", hottest_html)
    out = out.replace("__COOLEST_ROWS__", coolest_html)
    out = out.replace("__BAR_ROWS__", bars_html)
    out = out.replace("__MEAN_HEAT__", f"{df['predicted_mean_uhi_2018'].mean():.2f}")
    out = out.replace("__SUBURB_OPTIONS__", _suburb_options(all_names))
    out = out.replace("__SUBURB_OPTIONS_COMPLETE__", _suburb_options(complete_names))
    out = out.replace("__LAYER_RANGES_JSON__", json.dumps(ranges))
    out = out.replace("__SUBURB_LOOKUP_JSON__", json.dumps(suburb_lookup))

    (STATIC_DIR / "index.html").write_text(out, encoding="utf-8")
    logger.info("Wrote %s (%d SA2 polygons, %d suburbs, %d complete for What-If)", STATIC_DIR / "index.html", len(df), len(all_names), len(complete_names))

    _render_dark_shap_summary()
    _render_dark_scatter()


if __name__ == "__main__":
    main()

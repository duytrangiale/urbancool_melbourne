"""Shared logic between the static-site build step (app/build_static.py) and the live
API (app/main.py) — model/data loading (cached as module-level singletons, since both
the build script and the API server are short-lived-per-process and only need to load
the model once), the heat colour ramp, and the What-If / SHAP computations themselves.
Keeping this in one place means the build script and the API can never quietly diverge
on how a prediction or a SHAP plot is computed.
"""

from __future__ import annotations

import io
import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")

from src.data.loaders import PROJECT_ROOT, load_config
from src.models.feature_columns import FEATURE_COLS

SLIDER_COLS = ["tree_cover_pct_state", "vegetation_cover_pct_state", "impervious_ratio", "park_coverage_ratio"]

# Dark rcParams matching app/static/styles.css's --surface/--border/--text-* tokens, so a
# server-rendered chart (SHAP waterfall, scatter plots) sits in its <img> wrapper without
# a jarring light rectangle — see DAY_6.md for why this differs from the (light-themed)
# notebook charts.
DARK_SURFACE = "#0f1e29"
DARK_RCPARAMS = {
    "figure.facecolor": DARK_SURFACE,
    "axes.facecolor": DARK_SURFACE,
    "savefig.facecolor": DARK_SURFACE,
    "axes.edgecolor": "#2c4a60",
    "axes.labelcolor": "#eef5f7",
    "axes.grid": True,
    "grid.color": "#2c4a60",
    "grid.linewidth": 0.7,
    "text.color": "#eef5f7",
    # Brightened from #8fa8b3 — axis tick numbers were reported hard to read; matches the
    # same contrast-driven fix applied to app/static/styles.css's --text-muted.
    "xtick.color": "#c3d3db",
    "ytick.color": "#c3d3db",
    "xtick.labelcolor": "#c3d3db",
    "ytick.labelcolor": "#c3d3db",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
ACCENT = "#35d0c4"
INK = "#eef5f7"


def _brighten_dark_text(fig, replacement: str = "#eef5f7") -> None:
    """SHAP's plotting functions (summary_plot, waterfall) hardcode some text colours
    internally rather than respecting matplotlib rcParams — walk every text object
    actually rendered and brighten anything too dim to read comfortably on this dark
    theme. Confirmed by direct inspection (not assumed) that SHAP renders some labels
    *twice*, once as a real tick label (which does pick up rcParams) and once as a
    separately-drawn annotation hardcoded to grey (``#999999``) sitting on top of it —
    that grey technically clears a bare 4.5:1 contrast minimum against this dark surface,
    but reads as muddy/dim next to the surrounding near-white text, which is why a
    saturation check (not a contrast-ratio pass/fail) decides what counts as "a real
    colour to leave alone" here: SHAP's vivid red/blue value labels are strongly
    saturated (their max and min RGB channels are far apart) and are left untouched;
    low-saturation greys are brightened regardless of whether they'd technically pass.
    """
    import matplotlib.colors
    import matplotlib.text

    for text_obj in fig.findobj(matplotlib.text.Text):
        color = text_obj.get_color()
        try:
            r, g, b = matplotlib.colors.to_rgb(color)
        except (ValueError, TypeError):
            continue
        saturation = max(r, g, b) - min(r, g, b)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if saturation < 0.15 and luminance < 0.75:
            text_obj.set_color(replacement)

_model = None
_explainer = None
_sa2_df = None
_model_info = None


def load_model():
    global _model
    if _model is None:
        config = load_config()
        _model = joblib.load(PROJECT_ROOT / config["paths"]["models"] / "best_model.joblib")
    return _model


def load_shap_explainer():
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(load_model().named_steps["model"])
    return _explainer


def load_sa2_df() -> pd.DataFrame:
    global _sa2_df
    if _sa2_df is None:
        config = load_config()
        processed = PROJECT_ROOT / config["paths"]["data_processed"]
        _sa2_df = pd.read_csv(processed / "predictions_sa2.csv", dtype={"SA2_CODE21": str})
    return _sa2_df


def load_model_info() -> dict:
    global _model_info
    if _model_info is None:
        config = load_config()
        models_dir = PROJECT_ROOT / config["paths"]["models"]
        info = json.loads((models_dir / "model_info.json").read_text())
        info["test_metrics"] = json.loads((models_dir / "test_metrics.json").read_text())
        _model_info = info
    return _model_info


def _interpolate_ramp(t: float, stops: list[tuple[float, tuple[int, int, int]]], fallback: str) -> str:
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            r = round(c0[0] + (c1[0] - c0[0]) * f)
            g = round(c0[1] + (c1[1] - c0[1]) * f)
            b = round(c0[2] + (c1[2] - c0[2]) * f)
            return f"#{r:02x}{g:02x}{b:02x}"
    return fallback


def heat_ramp(t: float) -> str:
    """Amber -> orange -> red, matching styles.css's --heat-cool/--heat-mid/--heat-hot."""
    stops = [(0.0, (255, 210, 138)), (0.5, (255, 138, 76)), (1.0, (232, 72, 58))]
    return _interpolate_ramp(t, stops, "#e8483a")


def green_ramp(t: float) -> str:
    """Light -> dark green, for the Tree Cover / Vegetation Cover map layers — a
    one-hue sequential ramp (never a rainbow), distinct from the heat ramp's warm hues
    so the two layers are never visually confusable."""
    stops = [(0.0, (18, 38, 30)), (0.5, (45, 122, 90)), (1.0, (110, 220, 170))]
    return _interpolate_ramp(t, stops, "#6edcaa")


def all_suburbs() -> list[str]:
    return sorted(load_sa2_df()["SA2_NAME21"].dropna().unique().tolist())


def complete_suburbs() -> list[str]:
    """Suburbs with complete vegetation/impervious data — see DAY_3.md section C1 for why
    8 of 361 are excluded from anything that perturbs those specific columns."""
    df = load_sa2_df()
    complete = df.dropna(subset=SLIDER_COLS)
    return sorted(complete["SA2_NAME21"].dropna().unique().tolist())


def get_snapshot(suburb: str) -> dict:
    """The suburb's feature values, plus a predicted-heat figure computed the *same way*
    the What-If Simulator and SHAP waterfall compute theirs: one live model.predict()
    call on this suburb's aggregated feature row (not the map's precomputed value, which
    is built differently — see DAY_6.md Part G2). Both live in the same view (the What-If
    Simulator's own snapshot panel, and Feature Explorer's SHAP panel), so they need to
    agree with each other exactly; a small gap against the Heat Map tab's number is
    expected and explained there, not hidden."""
    df = load_sa2_df()
    matches = df.loc[df["SA2_NAME21"] == suburb]
    if matches.empty:
        raise KeyError(suburb)
    row = matches.iloc[0]
    model = load_model()
    predicted = float(model.predict(pd.DataFrame([row[FEATURE_COLS]]))[0])
    return {
        "tree_cover_pct_state": float(row["tree_cover_pct_state"]),
        "vegetation_cover_pct_state": float(row["vegetation_cover_pct_state"]),
        "impervious_ratio": float(row["impervious_ratio"]),
        "population_density": float(row["population_density"]),
        "predicted_mean_uhi_2018": predicted,
    }


def compute_whatif(suburb: str, extra_tree: float, extra_veg: float, green_conversion: float) -> dict:
    df = load_sa2_df()
    matches = df.loc[df["SA2_NAME21"] == suburb]
    if matches.empty:
        raise KeyError(suburb)
    row = matches.iloc[0]
    model = load_model()

    base = row[FEATURE_COLS].copy()
    modified = base.copy()
    modified["tree_cover_pct_state"] = np.clip(modified["tree_cover_pct_state"] + extra_tree, 0, 100)
    modified["vegetation_cover_pct_state"] = np.clip(modified["vegetation_cover_pct_state"] + extra_veg, 0, 100)
    conversion_fraction = green_conversion / 100.0
    modified["impervious_ratio"] = np.clip(modified["impervious_ratio"] - conversion_fraction, 0, 1)
    modified["park_coverage_ratio"] = np.clip(modified["park_coverage_ratio"] + conversion_fraction, 0, 1)

    current_pred = float(model.predict(pd.DataFrame([base]))[0])
    modified_pred = float(model.predict(pd.DataFrame([modified]))[0])
    return {"current": current_pred, "modified": modified_pred, "delta": modified_pred - current_pred}


def render_shap_waterfall_png(suburb: str) -> bytes:
    df = load_sa2_df()
    matches = df.loc[df["SA2_NAME21"] == suburb]
    if matches.empty:
        raise KeyError(suburb)
    row = matches.iloc[0]
    model = load_model()
    explainer = load_shap_explainer()

    X = pd.DataFrame([row[FEATURE_COLS]])
    X_imputed = pd.DataFrame(model.named_steps["impute"].transform(X), columns=FEATURE_COLS)
    explanation = explainer(X_imputed)

    with plt.rc_context(DARK_RCPARAMS):
        shap.plots.waterfall(explanation[0], show=False)
        fig = plt.gcf()
        fig.suptitle(f"SHAP waterfall — {suburb}", color=INK, y=1.03, fontsize=11)
        _brighten_dark_text(fig)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=DARK_SURFACE)
        plt.close(fig)
    buf.seek(0)
    return buf.read()

"""Generates heat predictions for every SA1 in Greater Melbourne using the saved model,
then aggregates them up to SA2 (suburb) level for the dashboard.

The model is trained at SA1 resolution (see DAY_4.md's Part G), but SA1 units are small,
unnamed sub-areas — not what a "select a suburb" dashboard wants. Every SA1 nests inside
exactly one SA2 (the ABS hierarchy is strictly nested — see DAY_3.md), so aggregating
predictions up to SA2 is a plain area-weighted groupby, not a spatial overlay.

Run as a script to produce ``data/processed/predictions_sa1.csv`` and
``data/processed/predictions_sa2.csv``:

    python -m src.models.predict
"""

from __future__ import annotations

import logging

import geopandas as gpd
import joblib
import pandas as pd

from src.data.loaders import PROJECT_ROOT, load_config
from src.models.train import FEATURE_COLS, RESOLUTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PREDICTION_COL = "predicted_mean_uhi_2018"


def generate_sa1_predictions(config: dict | None = None) -> pd.DataFrame:
    """Predict heat for every SA1 in the feature matrix, not just the ones with a known
    target — a prediction only needs the feature columns, which the model's own imputer
    already handles gaps in (see DAY_4.md for which columns have gaps and why)."""
    config = config or load_config()
    processed = PROJECT_ROOT / config["paths"]["data_processed"]
    models_dir = PROJECT_ROOT / config["paths"]["models"]

    model = joblib.load(models_dir / "best_model.joblib")
    sa1 = RESOLUTIONS["SA1"]
    features = pd.read_csv(processed / sa1["feature_matrix"], dtype={sa1["code_col"]: str})

    features[PREDICTION_COL] = model.predict(features[FEATURE_COLS])
    logger.info("Predicted %s for all %d SA1 rows (%d have a known actual value for comparison)", PREDICTION_COL, len(features), features["mean_uhi_2018"].notna().sum())
    return features[[sa1["code_col"], "SA2_NAME21", PREDICTION_COL, "mean_uhi_2018"] + FEATURE_COLS]


def aggregate_predictions_to_sa2(sa1_predictions: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Area-weighted mean of SA1 predictions per parent SA2 — a plain groupby (not an
    overlay) since SA1 boundaries nest exactly inside SA2 boundaries. Merges the result
    onto the existing SA2 feature matrix so the dashboard has both the aggregated
    prediction and the SA2-level context features (which come from the original
    area-weighted overlay in src/features/spatial.py, not re-derived here)."""
    config = config or load_config()
    interim = PROJECT_ROOT / config["paths"]["data_interim"]
    processed = PROJECT_ROOT / config["paths"]["data_processed"]

    sa1_boundaries = gpd.read_parquet(interim / RESOLUTIONS["SA1"]["boundaries"])
    weights = sa1_boundaries[["SA1_CODE21", "SA2_CODE21"]].copy()
    weights["_area"] = sa1_boundaries.geometry.area

    weighted = sa1_predictions.merge(weights, on="SA1_CODE21", how="left")
    weighted["_weighted_pred"] = weighted[PREDICTION_COL] * weighted["_area"]

    grouped = weighted.groupby("SA2_CODE21")
    sa2_pred = (grouped["_weighted_pred"].sum() / grouped["_area"].sum()).rename(PREDICTION_COL).reset_index()

    sa2_features = pd.read_csv(processed / RESOLUTIONS["SA2"]["feature_matrix"], dtype={"SA2_CODE21": str})
    result = sa2_features.merge(sa2_pred, on="SA2_CODE21", how="left")
    logger.info("Aggregated to %d SA2 rows (%d with a predicted value)", len(result), result[PREDICTION_COL].notna().sum())
    return result


def save_predictions(sa1_predictions: pd.DataFrame, sa2_predictions: pd.DataFrame, config: dict | None = None):
    config = config or load_config()
    dest_dir = PROJECT_ROOT / config["paths"]["data_processed"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    sa1_path = dest_dir / "predictions_sa1.csv"
    sa2_path = dest_dir / "predictions_sa2.csv"
    sa1_predictions.to_csv(sa1_path, index=False)
    sa2_predictions.to_csv(sa2_path, index=False)
    logger.info("Saved %d SA1 rows to %s, %d SA2 rows to %s", len(sa1_predictions), sa1_path, len(sa2_predictions), sa2_path)
    return sa1_path, sa2_path


def main() -> None:
    config = load_config()
    sa1_predictions = generate_sa1_predictions(config)
    sa2_predictions = aggregate_predictions_to_sa2(sa1_predictions, config)
    save_predictions(sa1_predictions, sa2_predictions, config)


if __name__ == "__main__":
    main()

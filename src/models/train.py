"""Baseline model comparison, spatially-aware cross-validation, hyperparameter tuning,
and artifact saving for the SA2-level urban heat regression task.

Run as a script to reproduce the full Day 4 pipeline end-to-end:

    python -m src.models.train
"""

from __future__ import annotations

import json
import logging
import warnings

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data.loaders import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Cosmetic-only: inside a Pipeline + cross_validate, LGBMRegressor's sklearn wrapper warns
# that it was "fitted with feature names" when scored against the imputer's numpy output.
# Predictions are unaffected (verified by comparing scored metrics against the other
# models' plausible ranges) — this is a known upstream quirk, not a real data issue.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

TARGET_COL = "mean_uhi_2018"
GROUP_COL = "SA3_CODE21"

# The City of Melbourne tree/canopy columns (tree_count, canopy_coverage_ratio_city, etc.)
# are ~89% missing outside the inner LGA (see DAY_3.md section C1) and already have a
# full-coverage substitute (vegetation_cover_pct_state / tree_cover_pct_state), so they're
# excluded here rather than imputed — imputing a value for 89% of rows would mostly be
# guessing, not signal. max_uhi_2018 and heat_mesh_block_count are dropped as target
# leakage: both come from the same heat-overlay computation as mean_uhi_2018 itself
# (see DAY_4.md for the reasoning).
FEATURE_COLS = [
    "area_sqkm",
    "vegetation_cover_pct_state",
    "tree_cover_pct_state",
    "building_density_per_ha",
    "building_coverage_ratio",
    "mean_building_area_sqm",
    "road_density_km_per_sqkm",
    "park_coverage_ratio",
    "water_coverage_ratio",
    "dist_to_nearest_park_m",
    "dist_to_nearest_water_m",
    "impervious_ratio",
]

_BOOSTING_PARAM_DIST = {
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [3, 5, 7, 10],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5],
}
_FOREST_PARAM_DIST = {
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [3, 5, 7, 10, None],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", 0.5, 1.0],
}
_RIDGE_PARAM_DIST = {
    "model__alpha": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
}
PARAM_DIST_BY_MODEL = {
    "XGBoost": _BOOSTING_PARAM_DIST,
    "LightGBM": _BOOSTING_PARAM_DIST,
    "Random Forest": _FOREST_PARAM_DIST,
    "Ridge": _RIDGE_PARAM_DIST,
}


def load_model_data(config: dict | None = None) -> pd.DataFrame:
    """Load the feature matrix, attach an SA3 grouping column, and drop rows with no target."""
    config = config or load_config()
    processed = PROJECT_ROOT / config["paths"]["data_processed"]
    interim = PROJECT_ROOT / config["paths"]["data_interim"]

    features = pd.read_csv(processed / "feature_matrix.csv", dtype={"SA2_CODE21": str})
    sa2 = gpd.read_parquet(interim / "sa2_boundaries.parquet")

    data = features.merge(sa2[["SA2_CODE21", "SA3_CODE21", "SA3_NAME21"]], on="SA2_CODE21", how="left")
    n_before = len(data)
    data = data.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    logger.info("Loaded %d SA2 rows, dropped %d with no heat target -> %d modeling rows", n_before, n_before - len(data), len(data))
    return data


def spatial_train_test_split(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """80/20 split grouped by SA3 (~40 sub-regions of a few adjacent SA2s each), so whole
    neighbourhoods land on one side of the split. A plain random split would put spatially
    adjacent, autocorrelated SA2s on both sides and overstate test performance (see
    DAY_4.md for why this matters for this dataset specifically).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(data, groups=data[GROUP_COL]))
    return data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx].reset_index(drop=True)


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """One sklearn Pipeline per candidate model, each with its own imputer (and scaler for
    Ridge) fit inside the pipeline so cross-validation never leaks fold statistics."""
    return {
        "Mean baseline": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=200, random_state=random_state)),
        ]),
        "XGBoost": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=random_state, verbosity=0)),
        ]),
        "LightGBM": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=random_state, verbose=-1)),
        ]),
    }


def evaluate_with_cv(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Spatially-grouped 5-fold CV (GroupKFold by SA3) for each model. Returns RMSE / MAE /
    R^2 mean +/- std across folds, sorted best (lowest RMSE) first."""
    cv = GroupKFold(n_splits=n_splits)
    scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}
    rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X, y, groups=groups, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "RMSE_mean": -scores["test_RMSE"].mean(), "RMSE_std": scores["test_RMSE"].std(),
            "MAE_mean": -scores["test_MAE"].mean(), "MAE_std": scores["test_MAE"].std(),
            "R2_mean": scores["test_R2"].mean(), "R2_std": scores["test_R2"].std(),
        })
    return pd.DataFrame(rows).sort_values("RMSE_mean").reset_index(drop=True)


def tune_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series, random_state: int = 42, n_iter: int = 30) -> RandomizedSearchCV:
    """RandomizedSearchCV over a model-appropriate hyperparameter space, using the same
    spatially-grouped CV splitter as the baseline comparison."""
    cv = GroupKFold(n_splits=5)
    pipe = build_models(random_state)[model_name]
    param_distributions = PARAM_DIST_BY_MODEL[model_name]
    search = RandomizedSearchCV(
        pipe, param_distributions, n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=random_state, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search


def evaluate_on_test(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
    }


def extract_feature_importance(pipeline: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """Works for both tree ensembles (feature_importances_) and linear models (|coef_|).
    The two are not on the same scale and shouldn't be compared across model types."""
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_)
    else:
        raise ValueError(f"{type(estimator).__name__} has neither feature_importances_ nor coef_")
    return pd.DataFrame({"feature": feature_cols, "importance": values}).sort_values("importance", ascending=False).reset_index(drop=True)


def save_artifacts(model: Pipeline, feature_cols: list[str], cv_summary: pd.DataFrame, best_params: dict, test_metrics: dict, config: dict | None = None):
    config = config or load_config()
    models_dir = PROJECT_ROOT / config["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "best_model.joblib")
    extract_feature_importance(model, feature_cols).to_csv(models_dir / "feature_importance.csv", index=False)
    cv_summary.to_csv(models_dir / "cv_summary.csv", index=False)
    with open(models_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    with open(models_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Saved model + artifacts to %s", models_dir)
    return models_dir


def main() -> dict:
    config = load_config()
    model_cfg = config["model"]
    data = load_model_data(config)

    train, test = spatial_train_test_split(data, test_size=model_cfg["test_size"], random_state=model_cfg["random_state"])
    logger.info(
        "Train: %d rows / %d SA3 groups | Test: %d rows / %d SA3 groups",
        len(train), train[GROUP_COL].nunique(), len(test), test[GROUP_COL].nunique(),
    )

    X_train, y_train, groups_train = train[FEATURE_COLS], train[TARGET_COL], train[GROUP_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    models = build_models(model_cfg["random_state"])
    cv_summary = evaluate_with_cv(models, X_train, y_train, groups_train)
    logger.info("Baseline CV comparison (grouped 5-fold, by SA3):\n%s", cv_summary.to_string(index=False))

    real_models = cv_summary[cv_summary["model"] != "Mean baseline"]
    best_name = real_models.iloc[0]["model"]
    logger.info("Best baseline by CV RMSE: %s", best_name)

    search = tune_model(best_name, X_train, y_train, groups_train, model_cfg["random_state"])
    logger.info("Tuned %s best CV RMSE: %.4f | best params: %s", best_name, -search.best_score_, search.best_params_)

    test_metrics = evaluate_on_test(search.best_estimator_, X_test, y_test)
    logger.info("Held-out test metrics for tuned %s: %s", best_name, test_metrics)

    # Report generalization using the untouched test set above, then refit the tuned
    # hyperparameters on ALL rows (train + test) for the artifact that ships in models/ —
    # standard practice once the held-out estimate has already been recorded.
    final_pipe = build_models(model_cfg["random_state"])[best_name]
    final_pipe.set_params(**search.best_params_)
    final_pipe.fit(data[FEATURE_COLS], data[TARGET_COL])

    save_artifacts(final_pipe, FEATURE_COLS, cv_summary, search.best_params_, test_metrics, config)

    return {
        "cv_summary": cv_summary,
        "best_name": best_name,
        "best_params": search.best_params_,
        "test_metrics": test_metrics,
        "final_model": final_pipe,
    }


if __name__ == "__main__":
    main()

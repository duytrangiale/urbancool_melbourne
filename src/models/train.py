"""Baseline model comparison, spatially-aware cross-validation, hyperparameter tuning,
and artifact saving for the urban heat regression task — at both SA2 and SA1 resolution
(see DAY_4.md's Part G for why SA1 was added: ~30x more training rows using
data already on disk, no new downloads beyond ABS SA1 boundaries).

Run as a script to build both resolutions, compare them on a held-out spatial test set,
and save whichever generalises better as ``models/best_model.joblib``:

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
from src.models.feature_columns import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Cosmetic-only: inside a Pipeline + cross_validate, LGBMRegressor's sklearn wrapper warns
# that it was "fitted with feature names" when scored against the imputer's numpy output.
# Predictions are unaffected (verified by comparing scored metrics against the other
# models' plausible ranges) — this is a known upstream quirk, not a real data issue.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

TARGET_COL = "mean_uhi_2018"

# Each resolution's feature matrix, boundary file, own-level code column, and the
# one-level-up geography to group spatial CV/splits by (SA3 for SA2, SA2 for SA1) — see
# DAY_4.md A1 for why grouping one level up matters.
RESOLUTIONS = {
    "SA2": {
        "feature_matrix": "feature_matrix.csv",
        "boundaries": "sa2_boundaries.parquet",
        "code_col": "SA2_CODE21",
        "group_col": "SA3_CODE21",
    },
    "SA1": {
        "feature_matrix": "feature_matrix_sa1.csv",
        "boundaries": "sa1_boundaries.parquet",
        "code_col": "SA1_CODE21",
        "group_col": "SA2_CODE21",
    },
}

# The City of Melbourne tree/canopy columns (tree_count, canopy_coverage_ratio_city, etc.)
# are ~89% missing outside the inner LGA (see DAY_3.md section C1) and already have a
# full-coverage substitute (vegetation_cover_pct_state / tree_cover_pct_state), so they're
# excluded here rather than imputed — imputing a value for 89% of rows would mostly be
# guessing, not signal. max_uhi_2018 and heat_mesh_block_count are dropped as target
# leakage: both come from the same heat-overlay computation as mean_uhi_2018 itself.
_BOOSTING_PARAM_DIST = {
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [3, 5, 7, 10],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5],
}
# n_estimators capped at 150 (not the 500 an unconstrained search would happily pick):
# the deployed dashboard runs this model inside a 512MB container (see DAY_6.md's
# deployment section), and a 500-tree forest plus its SHAP explainer measured ~730MB
# resident, well past that. 100-150 trees loses ~0.002 RMSE / ~0.002 R2 on the held-out
# spatial test set versus 500 (noise-level) while fitting comfortably in memory.
_FOREST_PARAM_DIST = {
    "model__n_estimators": [50, 100, 150],
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


def load_model_data(resolution: str = "SA2", config: dict | None = None) -> pd.DataFrame:
    """Load a resolution's feature matrix, attach its one-level-up grouping column, and
    drop rows with no target."""
    config = config or load_config()
    res = RESOLUTIONS[resolution]
    processed = PROJECT_ROOT / config["paths"]["data_processed"]
    interim = PROJECT_ROOT / config["paths"]["data_interim"]

    features = pd.read_csv(processed / res["feature_matrix"], dtype={res["code_col"]: str})
    boundaries = gpd.read_parquet(interim / res["boundaries"])

    data = features.merge(boundaries[[res["code_col"], res["group_col"]]].drop_duplicates(), on=res["code_col"], how="left")
    n_before = len(data)
    data = data.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    logger.info(
        "[%s] Loaded %d rows, dropped %d with no heat target -> %d modeling rows",
        resolution, n_before, n_before - len(data), len(data),
    )
    return data


def spatial_train_test_split(data: pd.DataFrame, group_col: str, test_size: float = 0.2, random_state: int = 42):
    """80/20 split grouped by ``group_col`` (the geography one level up from the modeling
    unit), so whole neighbourhoods land on one side of the split. A plain random split
    would put spatially adjacent, autocorrelated rows on both sides and overstate test
    performance (see DAY_4.md A1 for why this matters for this dataset specifically).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(data, groups=data[group_col]))
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
    """Spatially-grouped 5-fold CV for each model. Returns RMSE / MAE / R^2 mean +/- std
    across folds, sorted best (lowest RMSE) first."""
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


def run_pipeline(resolution: str, config: dict | None = None) -> dict:
    """Full train/tune/evaluate pipeline for one resolution (SA2 or SA1). Returns
    everything needed to compare resolutions and, if selected, save artifacts."""
    config = config or load_config()
    model_cfg = config["model"]
    res = RESOLUTIONS[resolution]

    data = load_model_data(resolution, config)
    train, test = spatial_train_test_split(data, res["group_col"], test_size=model_cfg["test_size"], random_state=model_cfg["random_state"])
    logger.info(
        "[%s] Train: %d rows / %d %s groups | Test: %d rows / %d %s groups",
        resolution, len(train), train[res["group_col"]].nunique(), res["group_col"],
        len(test), test[res["group_col"]].nunique(), res["group_col"],
    )

    X_train, y_train, groups_train = train[FEATURE_COLS], train[TARGET_COL], train[res["group_col"]]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    models = build_models(model_cfg["random_state"])
    cv_summary = evaluate_with_cv(models, X_train, y_train, groups_train)
    logger.info("[%s] Baseline CV comparison (grouped 5-fold, by %s):\n%s", resolution, res["group_col"], cv_summary.to_string(index=False))

    real_models = cv_summary[cv_summary["model"] != "Mean baseline"]
    best_name = real_models.iloc[0]["model"]
    logger.info("[%s] Best baseline by CV RMSE: %s", resolution, best_name)

    search = tune_model(best_name, X_train, y_train, groups_train, model_cfg["random_state"])
    logger.info("[%s] Tuned %s best CV RMSE: %.4f | best params: %s", resolution, best_name, -search.best_score_, search.best_params_)

    test_metrics = evaluate_on_test(search.best_estimator_, X_test, y_test)
    logger.info("[%s] Held-out test metrics for tuned %s: %s", resolution, best_name, test_metrics)

    # Report generalization using the untouched test set above, then refit the tuned
    # hyperparameters on ALL rows (train + test) for the artifact that could ship in
    # models/ — standard practice once the held-out estimate has already been recorded.
    final_pipe = build_models(model_cfg["random_state"])[best_name]
    final_pipe.set_params(**search.best_params_)
    final_pipe.fit(data[FEATURE_COLS], data[TARGET_COL])

    return {
        "resolution": resolution,
        "n_rows": len(data),
        "n_groups": data[res["group_col"]].nunique(),
        "cv_summary": cv_summary,
        "best_name": best_name,
        "best_params": search.best_params_,
        "test_metrics": test_metrics,
        "test_model": search.best_estimator_,  # train-only fit, for plotting held-out predictions
        "final_model": final_pipe,  # refit on all rows (train+test), the artifact that gets saved
        "X_test": X_test,
        "y_test": y_test,
    }


def save_artifacts(result: dict, config: dict | None = None):
    config = config or load_config()
    models_dir = PROJECT_ROOT / config["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result["final_model"], models_dir / "best_model.joblib")
    extract_feature_importance(result["final_model"], FEATURE_COLS).to_csv(models_dir / "feature_importance.csv", index=False)
    result["cv_summary"].to_csv(models_dir / "cv_summary.csv", index=False)
    with open(models_dir / "best_params.json", "w") as f:
        json.dump(result["best_params"], f, indent=2)
    with open(models_dir / "test_metrics.json", "w") as f:
        json.dump(result["test_metrics"], f, indent=2)
    with open(models_dir / "model_info.json", "w") as f:
        json.dump({"resolution": result["resolution"], "model": result["best_name"], "n_rows": result["n_rows"], "n_groups": result["n_groups"]}, f, indent=2)

    logger.info("Saved model + artifacts to %s (%s / %s)", models_dir, result["resolution"], result["best_name"])
    return models_dir


def main() -> dict:
    config = load_config()

    results = {resolution: run_pipeline(resolution, config) for resolution in RESOLUTIONS}

    comparison = pd.DataFrame([
        {
            "resolution": r["resolution"], "model": r["best_name"], "n_rows": r["n_rows"], "n_groups": r["n_groups"],
            "test_RMSE": r["test_metrics"]["RMSE"], "test_MAE": r["test_metrics"]["MAE"], "test_R2": r["test_metrics"]["R2"],
        }
        for r in results.values()
    ]).sort_values("test_RMSE")
    logger.info("Resolution comparison (held-out spatial test set):\n%s", comparison.to_string(index=False))

    winner = comparison.iloc[0]["resolution"]
    logger.info("Winner: %s (lower held-out test RMSE)", winner)

    models_dir = PROJECT_ROOT / config["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(models_dir / "resolution_comparison.csv", index=False)

    save_artifacts(results[winner], config)

    return {"results": results, "comparison": comparison, "winner": winner}


if __name__ == "__main__":
    main()

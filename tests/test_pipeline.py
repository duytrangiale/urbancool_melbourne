"""End-to-end pipeline tests against small synthetic sample data — fast, deterministic,
and independent of the real (large, slow-to-build) geospatial datasets. Verifies the
modeling and prediction-aggregation *plumbing* (splits, training, aggregation math), not
the real model's accuracy (that's DAY_4.md's job, against real data).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.models.predict import aggregate_predictions_to_sa2, PREDICTION_COL
from src.models.train import (
    FEATURE_COLS,
    TARGET_COL,
    build_models,
    evaluate_on_test,
    evaluate_with_cv,
    spatial_train_test_split,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def synthetic_modeling_data() -> pd.DataFrame:
    """80 synthetic rows across 8 groups (10 rows each) with a feature that actually
    predicts the target, so a trained model has something real to find — this is a
    plumbing test, not an accuracy test, but a model that can't beat a constant on
    obviously-informative synthetic data would indicate something is really broken.
    """
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame({col: rng.uniform(0, 100, n) for col in FEATURE_COLS})
    df["group"] = np.repeat(np.arange(8), 10)
    # Target driven mostly by one feature (like tree_cover_pct_state in the real model)
    # plus a bit of noise from the rest.
    df[TARGET_COL] = 20 - 0.15 * df[FEATURE_COLS[0]] + rng.normal(0, 1.0, n)
    return df


def test_spatial_train_test_split_has_no_group_overlap(synthetic_modeling_data):
    train, test = spatial_train_test_split(synthetic_modeling_data, group_col="group", test_size=0.25, random_state=0)

    assert set(train["group"]).isdisjoint(set(test["group"])), "train/test must not share groups (spatial leakage)"
    assert len(train) + len(test) == len(synthetic_modeling_data)


def test_models_train_and_predict_without_error(synthetic_modeling_data):
    train, test = spatial_train_test_split(synthetic_modeling_data, group_col="group", test_size=0.25, random_state=0)
    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    models = build_models(random_state=0)
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test), f"{name} produced the wrong number of predictions"
        assert np.isfinite(preds).all(), f"{name} produced non-finite predictions"


def test_real_models_beat_mean_baseline_on_informative_synthetic_data(synthetic_modeling_data):
    # Not a claim about real-world accuracy (see DAY_4.md for that) — just a sanity check
    # that the training/evaluation plumbing (build_models, evaluate_with_cv) correctly
    # lets a real signal through, using GroupKFold(3) since this fixture only has 8 groups.
    train, _ = spatial_train_test_split(synthetic_modeling_data, group_col="group", test_size=0.25, random_state=0)
    models = build_models(random_state=0)
    cv_summary = evaluate_with_cv(models, train[FEATURE_COLS], train[TARGET_COL], train["group"], n_splits=3)

    baseline_rmse = cv_summary.set_index("model").loc["Mean baseline", "RMSE_mean"]
    best_real_rmse = cv_summary[cv_summary["model"] != "Mean baseline"]["RMSE_mean"].min()
    assert best_real_rmse < baseline_rmse


def test_aggregate_predictions_to_sa2_area_weights_correctly(tmp_path, monkeypatch):
    # Two SA1s under one SA2: a small one (area=10, prediction=30) and a large one
    # (area=90, prediction=10) -> the SA2-level prediction must be pulled toward the
    # large one (12), not a plain unweighted average of the two (20) — same
    # area-weighting principle as test_area_weighted_mean_weights_by_area_not_count,
    # applied to predictions instead of raw source data (see DAY_5.md section A4).
    sa1_boundaries = gpd.GeoDataFrame(
        {"SA1_CODE21": ["1", "2"], "SA2_CODE21": ["A", "A"]},
        geometry=[box(0, 0, 10, 1), box(0, 1, 10, 10)],  # areas: 10 and 90
        crs="EPSG:28355",
    )
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    sa1_boundaries.to_parquet(interim_dir / "sa1_boundaries.parquet")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    sa2_features = pd.DataFrame({"SA2_CODE21": ["A"], "SA2_NAME21": ["Test Suburb"], "some_feature": [1.0]})
    sa2_features.to_csv(processed_dir / "feature_matrix.csv", index=False)

    config = {"paths": {"data_interim": "interim", "data_processed": "processed"}}
    monkeypatch.setattr("src.models.predict.PROJECT_ROOT", tmp_path)

    sa1_predictions = pd.DataFrame({
        "SA1_CODE21": ["1", "2"],
        PREDICTION_COL: [30.0, 10.0],
    })

    result = aggregate_predictions_to_sa2(sa1_predictions, config)

    assert result.loc[result["SA2_CODE21"] == "A", PREDICTION_COL].iloc[0] == pytest.approx(12.0)

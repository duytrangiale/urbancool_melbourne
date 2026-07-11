"""Feature engineering tests — synthetic geometries with known, hand-computable answers
for the aggregation math, plus schema checks against the real generated feature matrix
when it's present (skipped otherwise, e.g. in a fresh clone before the pipeline has run).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.features._utils import area_weighted_mean
from src.features.vegetation import _shannon_diversity
from src.models.train import FEATURE_COLS, TARGET_COL

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_area_weighted_mean_splits_and_weights_correctly():
    # One 10x10 boundary square, tiled exactly by a left half (value=10) and a right half
    # (value=20), each 50% of the area -> the weighted mean must be the plain average,
    # 15, not skewed toward either half.
    boundary = gpd.GeoDataFrame({"CODE": ["A"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:28355")
    source = gpd.GeoDataFrame(
        {"value": [10.0, 20.0]},
        geometry=[box(0, 0, 5, 10), box(5, 0, 10, 10)],
        crs="EPSG:28355",
    )

    result = area_weighted_mean(source, boundary, ["value"], "CODE")

    assert result.loc["A", "value"] == pytest.approx(15.0)
    assert result.loc["A", "value_max"] == pytest.approx(20.0)
    assert result.loc["A", "mesh_block_count"] == 2


def test_area_weighted_mean_weights_by_area_not_count():
    # A small sliver (10% of area, value=100) and a large piece (90% of area, value=0)
    # -> the weighted mean must be dominated by the large piece (10), not a plain
    # unweighted average of the two values (50) — this is the whole point of the
    # area-weighted technique over a naive groupby mean (see DAY_3.md section A2).
    boundary = gpd.GeoDataFrame({"CODE": ["A"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:28355")
    source = gpd.GeoDataFrame(
        {"value": [100.0, 0.0]},
        geometry=[box(0, 0, 1, 10), box(1, 0, 10, 10)],
        crs="EPSG:28355",
    )

    result = area_weighted_mean(source, boundary, ["value"], "CODE")

    assert result.loc["A", "value"] == pytest.approx(10.0)


def test_shannon_diversity_uniform_is_zero():
    species = pd.Series(["oak"] * 5)
    assert _shannon_diversity(species) == pytest.approx(0.0)


def test_shannon_diversity_two_even_species_is_ln2():
    species = pd.Series(["oak", "oak", "elm", "elm"])
    assert _shannon_diversity(species) == pytest.approx(np.log(2))


def test_shannon_diversity_empty_is_nan():
    assert np.isnan(_shannon_diversity(pd.Series([], dtype=object)))


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "processed" / "feature_matrix.csv").exists(),
    reason="feature_matrix.csv not built yet — run `python -m src.features.spatial` first",
)
def test_real_feature_matrix_has_expected_schema_and_ranges():
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "feature_matrix.csv")

    for col in FEATURE_COLS + [TARGET_COL]:
        assert col in df.columns, f"expected column {col!r} missing from feature_matrix.csv"

    # Percentages and ratios must stay in their physically valid ranges even with NaNs.
    for col in ["vegetation_cover_pct_state", "tree_cover_pct_state"]:
        assert df[col].dropna().between(0, 100).all(), f"{col} has values outside [0, 100]"
    for col in ["building_coverage_ratio", "park_coverage_ratio", "water_coverage_ratio", "impervious_ratio"]:
        assert df[col].dropna().between(0, 1).all(), f"{col} has values outside [0, 1]"
    assert (df["area_sqkm"] > 0).all(), "SA2 areas must be positive"

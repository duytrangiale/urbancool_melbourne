"""The trained model's feature columns, kept separate from src/models/train.py so
inference-time code (app/core.py) doesn't have to import lightgbm/xgboost and the rest
of the training machinery just to know which columns to feed the model — that import
alone costs ~150MB of resident memory, a real constraint on the dashboard's deploy target.
"""

from __future__ import annotations

# population_density / pct_population_needing_care (from the Vic Government Heat
# Vulnerability Index dataset's 2016-Census-derived indicators, NOT its HVI_INDEX column,
# which is itself heat-derived and would be leakage) are new as of the SA1 round — see
# DAY_4.md's Part G and src/features/spatial.py::compute_demographic_features.
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
    "population_density",
    "pct_population_needing_care",
]

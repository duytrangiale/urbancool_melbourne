# UrbanCool Melbourne — dashboard container.
#
# IMPORTANT: this image does NOT run the data/training pipeline (that needs the full
# raw data sources and takes minutes, not a Docker build's job). It expects
# data/interim/sa2_boundaries.parquet, data/processed/predictions_sa2.csv, and models/*
# (except best_model.joblib, fetched separately below — see that step) to already
# exist in the build context — i.e. you've already run, locally:
#
#   python -m src.data.loaders
#   python -m src.features.spatial
#   python -m src.models.train
#   python -m src.models.predict
#
# See DAY_6.md's deployment section for why (and for the Hugging Face Spaces caveat:
# those files are gitignored from the main GitHub repo for size/reproducibility reasons,
# so a Spaces deployment needs them committed to the Space's own repo instead).
FROM python:3.10-slim

WORKDIR /code

# geopandas/shapely/fiona/pyproj ship self-contained wheels on PyPI for this Python/OS
# combination, so no system GDAL/GEOS install is needed here — kept minimal deliberately.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY config/ config/
COPY src/ src/
COPY app/ app/

# Only the specific data/model files the dashboard actually reads at runtime or build
# time (see app/core.py, app/build_static.py) — not the full data/ tree.
COPY data/interim/sa2_boundaries.parquet data/interim/sa2_boundaries.parquet
COPY data/processed/predictions_sa2.csv data/processed/predictions_sa2.csv
COPY models/model_info.json models/model_info.json
COPY models/test_metrics.json models/test_metrics.json
COPY models/feature_importance.csv models/feature_importance.csv

# best_model.joblib is 208MB, over GitHub's 100MB push limit, so it can't be COPY'd
# from the build context like the files above. Hosted as a plain public file on
# Hugging Face Hub instead and fetched here with `requests` (already installed above).
RUN mkdir -p models && python -c "import requests; open('models/best_model.joblib', 'wb').write(requests.get('https://huggingface.co/duytrangiale/urbancool-melbourne-model/resolve/main/best_model.joblib', timeout=300).content)"

# Bakes app/static/index.html + the dark-themed SHAP/scatter PNGs from the data above —
# see app/build_static.py. Takes a few minutes (the global SHAP sample is the slow part).
RUN python -m app.build_static

# Hugging Face Spaces' Docker SDK expects the app on 7860 (and sets no PORT env var,
# so the ${PORT:-7860} fallback covers it); Render sets PORT itself (default 10000) and
# auto-detects whatever the container actually binds to. This one CMD works on both.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

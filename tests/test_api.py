"""FastAPI endpoint tests — run against the real trained model/predictions (skipped if
they haven't been built yet, same convention as test_features.py's real-data check).
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "best_model.joblib").exists()
    or not (PROJECT_ROOT / "data" / "processed" / "predictions_sa2.csv").exists(),
    reason="model/predictions not built yet — run `python -m src.models.train && python -m src.models.predict` first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_suburbs_endpoint_returns_known_counts(client):
    resp = client.get("/api/suburbs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["suburbs"]) == 361
    # 8 suburbs lack complete vegetation/impervious data — DAY_3.md section C1.
    assert len(data["complete_suburbs"]) == 353


def test_snapshot_endpoint_known_suburb(client):
    resp = client.get("/api/snapshot/Abbotsford")
    assert resp.status_code == 200
    data = resp.json()
    for key in ["tree_cover_pct_state", "vegetation_cover_pct_state", "impervious_ratio", "population_density", "predicted_mean_uhi_2018"]:
        assert key in data


def test_snapshot_endpoint_unknown_suburb_is_404(client):
    resp = client.get("/api/snapshot/Not A Real Suburb")
    assert resp.status_code == 404


def test_whatif_no_change_matches_current_prediction(client):
    resp = client.post("/api/whatif", json={"suburb": "Abbotsford", "extra_tree": 0, "extra_veg": 0, "green_conversion": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["current"] == pytest.approx(data["modified"])
    assert data["delta"] == pytest.approx(0.0)


def test_whatif_more_tree_cover_cools(client):
    # See DAY_6.md D1 — tree cover is the model's strongest single cooling driver.
    resp = client.post("/api/whatif", json={"suburb": "Abbotsford", "extra_tree": 30, "extra_veg": 0, "green_conversion": 0})
    data = resp.json()
    assert data["delta"] < 0, "more tree cover should reduce predicted heat"


def test_whatif_unknown_suburb_is_404(client):
    resp = client.post("/api/whatif", json={"suburb": "Not A Real Suburb"})
    assert resp.status_code == 404


def test_shap_endpoint_returns_png(client):
    resp = client.get("/api/shap/Abbotsford")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_shap_endpoint_unknown_suburb_is_404(client):
    resp = client.get("/api/shap/Not A Real Suburb")
    assert resp.status_code == 404


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "UrbanCool Melbourne" in resp.text


def test_static_files_are_served(client):
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/script.js").status_code == 200

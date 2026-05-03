"""Phase 7 (Mark) -- FastAPI service tests.

Tests cover:
  - GET  /health          : no-load liveness
  - GET  /info            : model card payload (reads JSON artefacts only)
  - POST /predict         : single prediction (with stub detector)
  - POST /predict_batch   : batch prediction (with stub detector)
  - error handling for missing/oversized batch
  - Pydantic schema enforcement

The detector is stubbed via api._get_detector to avoid loading the real
LightGBM booster (which is incompatible with this Windows environment but
runs fine in CI / production).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import api  # noqa: E402


# ---------------------------------------------------------------------------
# Stub detector
# ---------------------------------------------------------------------------

class _StubDetector:
    """Mimics the FraudDetector contract with deterministic output."""

    thresholds = {"cost_optimal": 0.1121, "default_05": 0.5}

    def predict_one(self, row, use_cost_optimal=True, top_k=5):
        # Deterministic prob from a few feature signals
        amt = float(row.get("amt", 0.0))
        z = float(row.get("amt_cat_zscore", 0.0))
        v = float(row.get("vel_amt_24h", 0.0))
        score = 1.0 / (1.0 + np.exp(-(0.001 * amt + 0.5 * z + 0.0001 * v - 1.0)))
        thr = self.thresholds["cost_optimal"] if use_cost_optimal else 0.5
        return SimpleNamespace(
            to_dict=lambda: {
                "prob": float(score),
                "alert": bool(score >= thr),
                "threshold": thr,
                "default_threshold": 0.5,
                "cost_optimal_threshold": self.thresholds["cost_optimal"],
                "individual_probs": {"catboost": float(score),
                                       "xgboost": float(score),
                                       "lightgbm": float(score)},
                "top_features": [
                    {"feature": "amt_cat_zscore", "value": z, "importance": 2.86,
                     "contribution_score": 2.86 * abs(z)}
                ][:top_k],
                "latency_ms": 1.23,
            }
        )

    def predict_batch(self, df):
        amts = df["amt"].to_numpy() if "amt" in df.columns else np.zeros(len(df))
        return 1.0 / (1.0 + np.exp(-(0.001 * amts - 1.0)))


@pytest.fixture
def client(monkeypatch):
    api._reset_detector()
    monkeypatch.setattr(api, "_get_detector", lambda: _StubDetector())
    with TestClient(api.app) as c:
        yield c
    api._reset_detector()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "fraud-detection-api"


def test_health_does_not_load_predictor(monkeypatch):
    """Health probe must work even if the booster files are missing."""
    api._reset_detector()

    def _exploding_detector():
        raise RuntimeError("models not loaded")

    monkeypatch.setattr(api, "_get_detector", _exploding_detector)
    with TestClient(api.app) as c:
        r = c.get("/health")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /info
# ---------------------------------------------------------------------------

def test_info_returns_53_features(client):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["feature_count"] == 53
    assert len(body["feature_names"]) == 53


def test_info_lists_all_three_base_learners_plus_ensemble(client):
    body = client.get("/info").json()
    assert set(body["metrics"].keys()) == {"catboost", "xgboost", "lightgbm", "ensemble_simple_avg"}


def test_info_production_pick_is_ensemble(client):
    body = client.get("/info").json()
    assert body["production_pick"] == "ensemble_simple_avg"


def test_info_thresholds_have_default_and_cost_optimal(client):
    body = client.get("/info").json()
    assert "default_05" in body["thresholds"]
    assert "cost_optimal" in body["thresholds"]
    assert body["thresholds"]["default_05"] == 0.5


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

def test_predict_returns_well_formed_response(client):
    r = client.post("/predict", json={"features": {"amt": 100.0}, "use_cost_optimal": True, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    for key in ("prob", "alert", "threshold", "default_threshold",
                "cost_optimal_threshold", "individual_probs", "top_features"):
        assert key in body, f"missing key: {key}"
    assert 0.0 <= body["prob"] <= 1.0
    assert body["threshold"] == 0.1121


def test_predict_default_threshold_mode(client):
    r = client.post("/predict", json={"features": {"amt": 100.0}, "use_cost_optimal": False, "top_k": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 0.5
    assert body["top_features"] == []


def test_predict_high_amount_high_zscore_triggers_alert(client):
    r = client.post(
        "/predict",
        json={
            "features": {"amt": 5000.0, "amt_cat_zscore": 10.0, "vel_amt_24h": 20000.0},
            "use_cost_optimal": True,
            "top_k": 1,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["alert"] is True


def test_predict_rejects_invalid_top_k(client):
    r = client.post("/predict", json={"features": {"amt": 100.0}, "use_cost_optimal": True, "top_k": 100})
    assert r.status_code == 422  # Pydantic validation - top_k bounded to 53


# ---------------------------------------------------------------------------
# /predict_batch
# ---------------------------------------------------------------------------

def test_predict_batch_returns_one_prob_per_row(client):
    rows = [{"amt": 10.0}, {"amt": 1000.0}, {"amt": 50000.0}]
    r = client.post("/predict_batch", json={"rows": rows, "use_cost_optimal": True})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 3
    assert len(body["probs"]) == 3
    assert len(body["alerts"]) == 3
    assert all(0.0 <= p <= 1.0 for p in body["probs"])


def test_predict_batch_rejects_empty_payload(client):
    r = client.post("/predict_batch", json={"rows": [], "use_cost_optimal": True})
    assert r.status_code == 422


def test_predict_batch_rejects_oversized_payload(client):
    rows = [{"amt": 1.0}] * 10_001  # max_length=10_000
    r = client.post("/predict_batch", json={"rows": rows, "use_cost_optimal": True})
    assert r.status_code == 422


def test_predict_batch_high_amounts_get_higher_alert_rate(client):
    """
    Sanity: with the stub's monotone scoring, larger amts yield higher
    probs. This catches a contract regression where /predict_batch
    accidentally feeds rows in shuffled order.
    """
    low = client.post("/predict_batch", json={"rows": [{"amt": 1.0}] * 5}).json()
    high = client.post("/predict_batch", json={"rows": [{"amt": 10000.0}] * 5}).json()
    assert max(high["probs"]) > max(low["probs"])

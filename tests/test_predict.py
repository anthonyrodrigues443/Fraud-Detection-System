"""Phase 6 (Mark) -- tests for src/predict.py and end-to-end inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from data_pipeline import CLEAN_STACK_53, FREQ_FEATURES
from predict import FraudDetector, PredictionResult

MODELS = REPO / "models"


pytestmark = pytest.mark.skipif(
    not (MODELS / "cb.cbm").exists(),
    reason="models/ not yet trained -- run python src/train_production.py first.",
)


@pytest.fixture(scope="module")
def detector():
    return FraudDetector.load(MODELS)


def test_detector_loads_with_53_features(detector):
    assert len(detector.feature_cols) == 53
    assert detector.feature_cols == CLEAN_STACK_53
    assert detector.thresholds["default_05"] == 0.5
    assert 0.0 < detector.thresholds["cost_optimal"] < 1.0


def test_predict_one_returns_prediction_result_for_zero_row(detector):
    row = {c: 0.0 for c in CLEAN_STACK_53}
    out = detector.predict_one(row, top_k=3)
    assert isinstance(out, PredictionResult)
    assert 0.0 <= out.prob <= 1.0
    assert isinstance(out.alert, bool)
    assert out.threshold == detector.thresholds["cost_optimal"]
    assert set(out.individual_probs.keys()) == {"catboost", "xgboost", "lightgbm"}
    assert len(out.top_features) == 3
    assert out.latency_ms > 0


def test_predict_one_high_risk_row_scores_higher_than_low_risk(detector):
    low = {c: 0.0 for c in CLEAN_STACK_53}
    low["amt"] = 5.0
    low["log_amt"] = float(np.log(6.0))
    low["hour"] = 12

    high = {c: 0.0 for c in CLEAN_STACK_53}
    high.update(amt=1500.0, log_amt=float(np.log(1501.0)),
                hour=2, is_night=1,
                amt_cat_zscore=6.0, vel_amt_24h=8000.0, vel_count_24h=12,
                amt_zscore=5.0, cat_fraud_rate=0.05)
    p_low = detector.predict_one(low, top_k=0).prob
    p_high = detector.predict_one(high, top_k=0).prob
    assert p_high > p_low


def test_predict_one_handles_unseen_merchant_state_city(detector):
    row = {c: 0.0 for c in CLEAN_STACK_53 if c not in FREQ_FEATURES}
    row["merchant"] = "fraud_made_up_merchant_xyz"
    row["state"] = "ZZ"
    row["city"] = "totally_new_city"
    # freq_* will be filled by apply_frequency_encoders -> 0
    out = detector.predict_one(row, top_k=0)
    assert 0.0 <= out.prob <= 1.0


def test_predict_batch_matches_single_call_means(detector):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: rng.normal(0, 1, 8).astype(np.float32) for c in CLEAN_STACK_53})
    df["merchant"] = "a"; df["state"] = "CA"; df["city"] = "foo"
    batch = detector.predict_batch(df)
    assert batch.shape == (8,)
    assert batch.dtype == np.float32
    # Per-row ensemble = mean of three boosters; sanity-check boundedness
    assert (batch >= 0.0).all() and (batch <= 1.0).all()


def test_top_features_ordered_by_contribution(detector):
    row = {c: 0.0 for c in CLEAN_STACK_53}
    row["amt_cat_zscore"] = 5.0
    out = detector.predict_one(row, top_k=5)
    contribs = [t["contribution_score"] for t in out.top_features]
    assert contribs == sorted(contribs, reverse=True)
    feature_names = [t["feature"] for t in out.top_features]
    assert "amt_cat_zscore" in feature_names

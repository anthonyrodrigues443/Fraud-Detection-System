"""Phase 7 (Anthony) -- end-to-end inference tests.

Goes beyond test_predict.py by testing:
  - Round-trip consistency: same input always produces same output
  - Batch vs single-call agreement
  - Threshold-based alert logic
  - PredictionResult serialization
  - Edge cases: extreme values, all-zero, all-max
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
sys.path.insert(0, str(REPO / "src"))

from data_pipeline import CLEAN_STACK_53, FREQ_FEATURES
from predict import FraudDetector, PredictionResult


pytestmark = pytest.mark.skipif(
    not (MODELS / "cb.cbm").exists(),
    reason="models/ not yet trained -- run python src/train_production.py first.",
)


@pytest.fixture(scope="module")
def detector():
    return FraudDetector.load(MODELS)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_input_produces_same_output(detector):
    row = {c: 0.5 for c in CLEAN_STACK_53}
    r1 = detector.predict_one(row, top_k=3)
    r2 = detector.predict_one(row, top_k=3)
    assert r1.prob == r2.prob
    assert r1.alert == r2.alert
    assert r1.individual_probs == r2.individual_probs


# ---------------------------------------------------------------------------
# Batch vs single agreement
# ---------------------------------------------------------------------------

def test_batch_probs_match_single_probs(detector):
    rng = np.random.default_rng(42)
    n = 5
    df = pd.DataFrame({c: rng.normal(0, 1, n).astype(np.float32) for c in CLEAN_STACK_53})
    df["merchant"] = "test_merchant"
    df["state"] = "CA"
    df["city"] = "test_city"

    batch_probs = detector.predict_batch(df)

    for i in range(n):
        row = df.iloc[i].to_dict()
        single = detector.predict_one(row, top_k=0)
        assert abs(single.prob - float(batch_probs[i])) < 1e-4, (
            f"Row {i}: single={single.prob:.6f} vs batch={batch_probs[i]:.6f}"
        )


# ---------------------------------------------------------------------------
# Alert logic
# ---------------------------------------------------------------------------

def test_alert_flag_respects_cost_optimal_threshold(detector):
    thr = detector.thresholds["cost_optimal"]

    low_row = {c: 0.0 for c in CLEAN_STACK_53}
    high_row = {c: 0.0 for c in CLEAN_STACK_53}
    high_row.update(amt=5000.0, log_amt=float(np.log(5001.0)),
                    hour=3, is_night=1,
                    amt_cat_zscore=10.0, vel_amt_24h=20000.0,
                    vel_count_24h=50, amt_zscore=8.0,
                    cat_fraud_rate=0.1)

    r_low = detector.predict_one(low_row, use_cost_optimal=True, top_k=0)
    r_high = detector.predict_one(high_row, use_cost_optimal=True, top_k=0)

    if r_low.prob < thr:
        assert r_low.alert is False
    if r_high.prob >= thr:
        assert r_high.alert is True


def test_default_threshold_mode(detector):
    row = {c: 0.0 for c in CLEAN_STACK_53}
    r = detector.predict_one(row, use_cost_optimal=False, top_k=0)
    assert r.threshold == 0.5
    assert r.default_threshold == 0.5


# ---------------------------------------------------------------------------
# PredictionResult serialization
# ---------------------------------------------------------------------------

def test_prediction_result_to_dict_roundtrip(detector):
    row = {c: 1.0 for c in CLEAN_STACK_53}
    result = detector.predict_one(row, top_k=3)
    d = result.to_dict()

    assert isinstance(d, dict)
    assert set(d.keys()) == {
        "prob", "alert", "threshold", "default_threshold",
        "cost_optimal_threshold", "individual_probs",
        "top_features", "latency_ms",
    }
    assert isinstance(json.dumps(d), str)


def test_prediction_result_probs_bounded(detector):
    rng = np.random.default_rng(99)
    for _ in range(10):
        row = {c: float(rng.normal(0, 3)) for c in CLEAN_STACK_53}
        r = detector.predict_one(row, top_k=0)
        assert 0.0 <= r.prob <= 1.0
        for name, p in r.individual_probs.items():
            assert 0.0 <= p <= 1.0, f"{name} prob {p} out of [0,1]"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_extreme_amount_does_not_crash(detector):
    row = {c: 0.0 for c in CLEAN_STACK_53}
    row["amt"] = 1_000_000.0
    row["log_amt"] = float(np.log(1_000_001.0))
    row["amt_cat_zscore"] = 100.0
    r = detector.predict_one(row, top_k=0)
    assert 0.0 <= r.prob <= 1.0


def test_negative_features_do_not_crash(detector):
    row = {c: -10.0 for c in CLEAN_STACK_53}
    r = detector.predict_one(row, top_k=0)
    assert 0.0 <= r.prob <= 1.0


def test_top_features_count_matches_request(detector):
    row = {c: 1.0 for c in CLEAN_STACK_53}
    for k in (0, 1, 5, 10, 53):
        r = detector.predict_one(row, top_k=k)
        assert len(r.top_features) == min(k, 53)


def test_individual_probs_has_all_three_models(detector):
    row = {c: 0.0 for c in CLEAN_STACK_53}
    r = detector.predict_one(row, top_k=0)
    assert set(r.individual_probs.keys()) == {"catboost", "xgboost", "lightgbm"}


def test_ensemble_prob_is_mean_of_individuals(detector):
    row = {c: 0.5 for c in CLEAN_STACK_53}
    r = detector.predict_one(row, top_k=0)
    expected = np.mean(list(r.individual_probs.values()))
    assert abs(r.prob - expected) < 1e-6

"""Phase 7 (Anthony) -- structural tests for the training pipeline and model artifacts.

These tests verify that:
  - All required model artifacts exist and are loadable
  - Production metrics meet minimum quality floors
  - Threshold artifacts are internally consistent
  - Feature column ordering matches the canonical CLEAN_STACK_53
  - The model card exists and covers required sections
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "src"))

from data_pipeline import CLEAN_STACK_53


# ---------------------------------------------------------------------------
# Artifact existence
# ---------------------------------------------------------------------------

REQUIRED_MODEL_FILES = [
    "cb.cbm", "xgb.json", "lgb.txt",
    "freq_encoders.json", "feature_cols.json",
    "threshold.json", "production_metrics.json",
]


@pytest.mark.parametrize("filename", REQUIRED_MODEL_FILES)
def test_model_artifact_exists(filename):
    assert (MODELS / filename).exists(), f"Missing artifact: models/{filename}"


def test_model_card_exists():
    card = MODELS / "model_card.md"
    assert card.exists()
    text = card.read_text()
    for section in ("Model overview", "Intended use", "Training data",
                    "Performance", "Limitations", "Ethical considerations"):
        assert section.lower() in text.lower(), f"Model card missing section: {section}"


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

def test_feature_cols_match_canonical_order():
    cols = json.loads((MODELS / "feature_cols.json").read_text())
    assert cols == CLEAN_STACK_53, "feature_cols.json diverges from CLEAN_STACK_53"


def test_feature_cols_has_53_entries():
    cols = json.loads((MODELS / "feature_cols.json").read_text())
    assert len(cols) == 53


# ---------------------------------------------------------------------------
# Threshold artifacts
# ---------------------------------------------------------------------------

def test_threshold_json_has_required_keys():
    thr = json.loads((MODELS / "threshold.json").read_text())
    assert "default_05" in thr
    assert "cost_optimal" in thr
    assert thr["default_05"] == 0.5


def test_cost_optimal_threshold_in_valid_range():
    thr = json.loads((MODELS / "threshold.json").read_text())
    assert 0.001 < thr["cost_optimal"] < 0.5, (
        f"Cost-optimal threshold {thr['cost_optimal']} outside expected range (0.001, 0.5)"
    )


def test_cost_optimal_saves_money_vs_default():
    thr = json.loads((MODELS / "threshold.json").read_text())
    assert thr["cost_at_optimal_dollars"] < thr["cost_at_05_dollars"], (
        "Cost-optimal threshold should have lower expected cost than default 0.5"
    )


# ---------------------------------------------------------------------------
# Production metrics quality floors
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prod_metrics():
    return json.loads((MODELS / "production_metrics.json").read_text())


def test_ensemble_auprc_above_floor(prod_metrics):
    auprc = prod_metrics["models"]["ensemble_simple_avg"]["auprc"]
    assert auprc >= 0.97, f"Ensemble AUPRC {auprc:.4f} below floor 0.97"


def test_ensemble_auroc_above_floor(prod_metrics):
    auroc = prod_metrics["models"]["ensemble_simple_avg"]["auroc"]
    assert auroc >= 0.99, f"Ensemble AUROC {auroc:.4f} below floor 0.99"


def test_ensemble_f1_at_05_above_floor(prod_metrics):
    f1 = prod_metrics["models"]["ensemble_simple_avg"]["f1_at_05"]
    assert f1 >= 0.90, f"Ensemble F1@0.5 {f1:.4f} below floor 0.90"


def test_all_base_learners_above_auprc_floor(prod_metrics):
    for name in ("catboost", "xgboost", "lightgbm"):
        auprc = prod_metrics["models"][name]["auprc"]
        assert auprc >= 0.95, f"{name} AUPRC {auprc:.4f} below floor 0.95"


def test_production_pick_is_ensemble(prod_metrics):
    assert prod_metrics["production_pick"] == "ensemble_simple_avg"


def test_train_test_sizes_reasonable(prod_metrics):
    assert prod_metrics["n_train"] > 800_000
    assert prod_metrics["n_test"] > 200_000
    assert prod_metrics["fraud_rate_train"] < 0.01
    assert prod_metrics["fraud_rate_test"] < 0.01


def test_ensemble_beats_every_individual_on_cost(prod_metrics):
    ens_cost = prod_metrics["models"]["ensemble_simple_avg"]["cost_at_optimal_threshold"]
    for name in ("catboost", "xgboost", "lightgbm"):
        ind_cost = prod_metrics["models"][name]["cost_at_optimal_threshold"]
        assert ens_cost <= ind_cost, (
            f"Ensemble cost ${ens_cost:.0f} should be <= {name} cost ${ind_cost:.0f}"
        )


# ---------------------------------------------------------------------------
# Frequency encoders
# ---------------------------------------------------------------------------

def test_freq_encoders_cover_all_three_columns():
    enc = json.loads((MODELS / "freq_encoders.json").read_text())
    assert set(enc.keys()) == {"merchant", "state", "city"}
    for col, mapping in enc.items():
        assert len(mapping) > 0, f"Empty encoder for {col}"
        assert all(isinstance(v, int) for v in mapping.values())

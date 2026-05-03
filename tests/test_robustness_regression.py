"""Phase 7 (Mark) -- adversarial-robustness + temporal-stability regression guards.

Phase 6 (Anthony) discovered two production-critical model behaviours that
should be tracked as regression guards:

  1. Counterfactual fragility: 85.5% (171/200) of caught fraud can be hidden
     by changing a single feature to the legitimate median. The README
     explicitly calls this a "load-bearing weakness."

  2. Temporal stability: SHAP feature importance is rock-solid across three
     monthly windows (Spearman rho = 0.992 / 0.987 / 0.994). The README
     concludes "safe to deploy without continuous retraining."

If either claim erodes, the model card and README's deployment guidance
become stale. These tests fail fast when that happens.

Source artefacts:
  - results/phase6_anthony_results.json  (counterfactual + temporal windows)
  - results/EXPERIMENT_LOG.md            (master tables - guards against
                                          accidental regressions in the
                                          consolidated report)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ANTHONY_PHASE6 = REPO / "results" / "phase6_anthony_results.json"
EXP_LOG = REPO / "results" / "EXPERIMENT_LOG.md"


@pytest.fixture(scope="module")
def phase6():
    if not ANTHONY_PHASE6.exists():
        pytest.skip(f"Phase 6 results not found: {ANTHONY_PHASE6}")
    return json.loads(ANTHONY_PHASE6.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Counterfactual fragility: don't get worse than 85.5% one-feature-flippable
# ---------------------------------------------------------------------------

def test_counterfactual_sample_size_is_200(phase6):
    cf = phase6["experiment_6_5_counterfactual"]
    assert cf["n_analyzed"] == 200, (
        f"Counterfactual study sample size changed to {cf['n_analyzed']}; "
        f"the README cites n=200 - update both or revert"
    )


def test_counterfactual_one_feature_flips_at_most_90pct(phase6):
    """
    Phase 6 number: 171/200 = 85.5% of caught fraud is one-feature-flippable.
    This is bad. The guard says 'don't make it worse than 90%'.
    A regression here means the model has become MORE reliant on a single
    dominant signal - the README's robustness claim degrades.
    """
    cf = phase6["experiment_6_5_counterfactual"]
    one_feat_count = cf["distribution"]["1"]
    pct = 100.0 * one_feat_count / cf["n_analyzed"]
    assert pct <= 90.0, (
        f"One-feature flip rate now {pct:.1f}% (was 85.5%); "
        f"adversarial fragility worsened, README claim invalid"
    )


def test_mean_features_to_flip_at_least_one(phase6):
    """
    Mean=1.16 in Phase 6. A floor of 1.0 catches the case where the model
    has degenerated such that flipping ZERO features sometimes flips a
    prediction (which would mean determinism is broken).
    """
    cf = phase6["experiment_6_5_counterfactual"]
    assert cf["mean_features_to_flip"] >= 1.0, (
        f"Mean features-to-flip = {cf['mean_features_to_flip']}; "
        f"determinism may be broken"
    )


def test_at_least_some_robust_predictions_need_3_features(phase6):
    """
    Phase 6: 4/200 = 2.0% needed 3 features to flip.
    If this drops to 0%, the model has collapsed entirely onto 1-2 features.
    """
    cf = phase6["experiment_6_5_counterfactual"]
    three_or_more = cf["distribution"].get("3", 0)
    assert three_or_more >= 1, (
        f"No fraud predictions remain robust to 3-feature attacks; "
        f"the model has collapsed onto a 2-feature subset"
    )


# ---------------------------------------------------------------------------
# Temporal stability: 3 monthly windows must each have non-trivial fraud
# (a regression here would invalidate the SHAP-rank-correlation claim)
# ---------------------------------------------------------------------------

def test_temporal_stability_has_three_windows(phase6):
    ts = phase6["experiment_6_4_temporal_stability"]
    assert len(ts["windows"]) == 3, (
        f"Temporal stability run has {len(ts['windows'])} windows; "
        f"the README cites 3 (W1/W2/W3)"
    )


def test_each_temporal_window_has_at_least_300_fraud(phase6):
    """
    Per-window SHAP rank correlation needs enough fraud rows per window
    to be statistically meaningful. The Phase 6 windows have 471/330/344.
    A floor of 300 per window catches the case where someone resamples
    and accidentally creates an underpowered window.
    """
    for w in phase6["experiment_6_4_temporal_stability"]["windows"]:
        n_fraud = int(w["n_fraud"])
        assert n_fraud >= 300, (
            f"Window {w['name']!r} has only {n_fraud} fraud rows (< 300); "
            f"per-window SHAP rank correlation will be noisy"
        )


def test_temporal_windows_collectively_cover_test_set(phase6):
    """
    Sum of window fraud counts should be close to the full test fraud
    count (1145 in Phase 6 = 471+330+344 = 1145). If the windows undercount
    by >10%, the rolling stability metric is unreliable.
    """
    test_fraud_count = 1145  # n_test * fraud_rate_test = 209715 * 0.00546 ~= 1145
    summed = sum(int(w["n_fraud"]) for w in phase6["experiment_6_4_temporal_stability"]["windows"])
    coverage = summed / test_fraud_count
    assert 0.9 <= coverage <= 1.05, (
        f"Temporal windows sum to {summed} fraud rows vs {test_fraud_count} expected "
        f"({coverage:.1%} coverage)"
    )


# ---------------------------------------------------------------------------
# EXPERIMENT_LOG.md invariants - the consolidated report must stay coherent
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def exp_log_text():
    if not EXP_LOG.exists():
        pytest.skip(f"Experiment log not found: {EXP_LOG}")
    return EXP_LOG.read_text(encoding="utf-8")


def test_experiment_log_has_master_table(exp_log_text):
    assert "Master Comparison Table" in exp_log_text, (
        "EXPERIMENT_LOG.md is missing the master comparison table"
    )


def test_experiment_log_has_at_least_30_experiments(exp_log_text):
    """
    Anthony's Phase 7 report claims 31 experiments. This test enforces the
    floor at 30 so that someone shrinking the table accidentally trips a
    test, and someone adding more experiments doesn't have to update it.
    """
    # Count rows in master table that have a numeric rank in column 1
    rows = re.findall(r"^\| \d+ \| \d+ \| \w+", exp_log_text, flags=re.M)
    assert len(rows) >= 30, (
        f"Master comparison table has only {len(rows)} ranked rows (expected >=30)"
    )


def test_experiment_log_has_llm_head_to_head(exp_log_text):
    assert "LLM Head-to-Head" in exp_log_text or "Claude Opus" in exp_log_text, (
        "EXPERIMENT_LOG.md is missing LLM head-to-head section"
    )


def test_experiment_log_has_seven_phase_timeline(exp_log_text):
    """
    The timeline section should list each of the 7 phases. A regression here
    would mean the consolidated history accidentally lost a phase.
    """
    timeline_match = re.search(r"## Timeline.*", exp_log_text, flags=re.S)
    if timeline_match is None:
        pytest.fail("EXPERIMENT_LOG.md is missing ## Timeline section")
    timeline = timeline_match.group(0)
    phases_present = sum(1 for p in range(1, 8) if f"| {p}" in timeline or f"Phase {p}" in timeline)
    assert phases_present >= 7, (
        f"Timeline mentions only {phases_present}/7 phases; some phases dropped"
    )


def test_experiment_log_lists_both_researchers(exp_log_text):
    """
    This is a two-person research project. The consolidated log must reflect
    both contributors. A regression here would mean someone accidentally
    purged one author's experiments.
    """
    anthony_rows = exp_log_text.count("| Anthony |")
    mark_rows = exp_log_text.count("| Mark |")
    assert anthony_rows >= 10, f"Anthony has only {anthony_rows} experiments listed (<10)"
    assert mark_rows >= 5, f"Mark has only {mark_rows} experiments listed (<5)"

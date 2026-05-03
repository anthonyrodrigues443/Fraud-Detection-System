"""Phase 7 (Mark) -- Streamlit `app.py` static smoke tests.

Anthony's tests cover src/* but NOT app.py. The Streamlit demo is part of
the production surface (it's what reviewers click through), so it deserves
its own regression coverage.

These tests are *static* - they do NOT actually start a Streamlit server.
Instead they:
  - parse app.py with `ast`
  - assert the module imports the expected predictor + data-pipeline names
  - assert that st.cache_resource / st.cache_data are used (no full reload
    on every interaction)
  - assert that demo-transaction sampling uses 10 fraud + 10 legit (the
    number cited in the Phase 6 report)
  - assert the file references the production model card and metrics

This means they're ~10 ms each and run in CI without GPU/booster setup.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
APP_PY = REPO / "app.py"


@pytest.fixture(scope="module")
def app_source():
    if not APP_PY.exists():
        pytest.skip(f"app.py not found at {APP_PY}")
    return APP_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_tree(app_source):
    return ast.parse(app_source)


# ---------------------------------------------------------------------------
# Static structure
# ---------------------------------------------------------------------------

def test_app_imports_fraud_detector(app_tree):
    names = set()
    for node in ast.walk(app_tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    assert "FraudDetector" in names, "app.py must import FraudDetector from predict"


def test_app_imports_data_pipeline_helpers(app_tree):
    names = set()
    for node in ast.walk(app_tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    expected = {"sample_test_transactions", "fit_frequency_encoders", "apply_frequency_encoders"}
    missing = expected - names
    assert not missing, f"app.py missing data_pipeline imports: {missing}"


def test_app_uses_streamlit_caching(app_source):
    """
    Without caching, every UI interaction would reload the booster + parquet
    (>200 MB, ~40s startup). The two cache decorators are load-bearing.
    """
    assert "@st.cache_resource" in app_source, "app.py must use st.cache_resource for the predictor"
    assert "@st.cache_data" in app_source, "app.py must use st.cache_data for sampled txns / metrics"


def test_app_samples_10_fraud_and_10_legit(app_source):
    """
    Phase 6 report: 'Streamlit demo sample: 10 fraud + 10 legit, stratified
    (random_state=7)'. If someone changes the n, the screenshot in the
    README and the model card become stale.
    """
    match = re.search(r"sample_test_transactions\([^)]*\)", app_source, flags=re.S)
    assert match is not None, "app.py does not call sample_test_transactions"
    call = match.group(0)
    assert "n_fraud=10" in call, f"Demo sample should request 10 fraud rows: {call}"
    assert "n_legit=10" in call, f"Demo sample should request 10 legit rows: {call}"
    assert "random_state=7" in call, f"Demo sample should pin random_state=7 for reproducibility: {call}"


def test_app_references_production_model_artifacts(app_source):
    assert "production_metrics.json" in app_source, (
        "app.py must read production_metrics.json for the sidebar metrics"
    )
    assert "FraudDetector.load(MODELS)" in app_source or "FraudDetector.load()" in app_source, (
        "app.py must call FraudDetector.load() to instantiate the predictor"
    )


def test_app_shows_both_thresholds(app_source):
    """
    The Phase-6 demo's headline insight is 'cost-optimal vs default-0.5'.
    Both must be in the UI.
    """
    assert "cost_optimal" in app_source, "app.py must surface cost_optimal threshold"
    assert "default_threshold" in app_source or "0.5" in app_source, (
        "app.py must surface default-0.5 threshold for comparison"
    )


def test_app_displays_per_base_learner_probs(app_source):
    """
    A core selling point of the Phase-6 demo is showing the ensemble vs
    each base learner. Removing the bar chart would cripple the explanation.
    """
    assert "individual_probs" in app_source, (
        "app.py must show individual_probs (per-base-learner output)"
    )


def test_app_displays_top_features(app_source):
    assert "top_features" in app_source, "app.py must surface top_features attribution"


def test_app_uses_53_feature_pipeline(app_source):
    """
    A regression here would mean someone wired up a different feature stack
    (e.g., the deprecated 39-feature stack) for the demo, breaking parity
    with the production predictor.
    """
    assert "CLEAN_STACK_53" in app_source, "app.py must import the canonical 53-feature stack"


# ---------------------------------------------------------------------------
# README parity - the README quotes specific numbers from app.py
# ---------------------------------------------------------------------------

def test_app_page_title_matches_phase(app_source):
    """
    page_title is what shows up on the browser tab when the app runs.
    Anthony's README shows it as 'Fraud Detection ...'. A change here
    would break the screenshot in the README.
    """
    match = re.search(r"page_title=([\"'])([^\"']+)\1", app_source)
    assert match is not None, "app.py must set st.set_page_config(page_title=...)"
    title = match.group(2)
    assert "Fraud Detection" in title, f"page_title should mention 'Fraud Detection', got {title!r}"

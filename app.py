"""
Phase 6 (Mark) -- Streamlit production demo.

Loads the simple-average ensemble (CB + XGB + LGB) and exposes a live demo
that lets the user:
    - pick a real test transaction (10 fraud + 10 legit, stratified)
    - tweak the headline fields (amt, hour, category, ...) and re-predict
    - see the ensemble's probability vs each base learner's vote
    - compare alerts under the default 0.5 threshold vs Phase-4 cost-optimal
    - see top-5 contributing features (CatBoost importance x z-scored value)
    - view the production model card in the sidebar

This is the production demo that ships from Phase 6. Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from data_pipeline import (
    load_full_dataset, fit_frequency_encoders, apply_frequency_encoders,
    sample_test_transactions, CLEAN_STACK_53,
)
from predict import FraudDetector


PARQUET = REPO / "data" / "processed" / "mark_phase3_full.parquet"
MODELS = REPO / "models"


# ---------------------------------------------------------------------------- #
# Caching
# ---------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading FraudDetector ensemble (CB + XGB + LGB) ...")
def load_detector():
    return FraudDetector.load(MODELS)


@st.cache_data(show_spinner="Sampling test transactions ...")
def load_demo_transactions():
    train_df, test_df = load_full_dataset(PARQUET)
    encoders = fit_frequency_encoders(train_df)
    test_df = apply_frequency_encoders(test_df, encoders)
    return sample_test_transactions(test_df, n_fraud=10, n_legit=10, random_state=7)


@st.cache_data
def load_metrics():
    return json.loads((MODELS / "production_metrics.json").read_text())


# ---------------------------------------------------------------------------- #
# Page
# ---------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Fraud Detection — Phase 6 (Mark)",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Fraud Detection — production ensemble")
st.caption(
    "Phase 5 production winner: simple-average of CatBoost + XGBoost + LightGBM. "
    "Phase 4 cost-optimal threshold ≈ 0.11. Trained on 838,860 transactions "
    "(53 features, temporal split). This demo runs the full inference path live."
)

detector = load_detector()
demo_txns = load_demo_transactions()
metrics = load_metrics()

# ---------------------------------------------------------------------------- #
# Sidebar -- model card + headline metrics
# ---------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Production model")
    ens = metrics["models"]["ensemble_simple_avg"]
    st.metric("Test AUPRC", f"{ens['auprc']:.4f}")
    st.metric("Test AUROC", f"{ens['auroc']:.4f}")
    st.metric("F1 @ thr=0.5", f"{ens['f1_at_05']:.4f}")
    st.metric("Cost-optimal threshold", f"{ens['cost_optimal_threshold']:.4f}")
    st.metric("Min expected cost (test)", f"${ens['cost_at_optimal_threshold']:,.0f}")

    st.divider()
    st.subheader("vs Phase-5 LLM frontier")
    st.caption(
        "From Phase-5 head-to-head on a 50-row stratified sample:\n"
        "- **CatBoost**: F1 = 1.000, latency 0.1 ms/row\n"
        "- **Claude Opus 4.6**: F1 = 0.864, latency 24,225 ms/row\n"
        "- **Claude Haiku 4.5**: F1 = 0.485, latency 12,906 ms/row\n\n"
        "Specialist beats frontier on every measurable axis. "
        "(GPT-5.4 codex run hit usage limit — re-run scheduled May 6.)"
    )

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "- 53 engineered features (Anthony 39 + Mark 14):\n"
        "  baseline (17), velocity (8), amount-deviation (5), temporal (3),\n"
        "  geographic (2), category (4), merchant-side (4), interactions (4),\n"
        "  frequency-encoded (3) and Mark statistical (6).\n"
        "- Three boosters trained independently, probabilities **averaged**.\n"
        "  Phase-5 result: a uniform mean beat every single learner AND a\n"
        "  trainable LogReg meta (which overfit despite 125k cal samples).\n"
        "- Cost-sensitive threshold: FN cost = transaction $amount,\n"
        "  FP cost = \\$1.50 (analyst review time).\n"
        "  Cost-optimal threshold beats threshold 0.5 by "
        f"\\${ens['cost_at_05'] - ens['cost_at_optimal_threshold']:,.0f} on test."
    )

# ---------------------------------------------------------------------------- #
# Main -- pick a transaction
# ---------------------------------------------------------------------------- #
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("1. Pick a test transaction")
    options = []
    for t in demo_txns:
        m = t["meta"]
        tag = "🚨 FRAUD" if t["label"] == 1 else "✅ legit"
        amt = m.get("amt", 0.0)
        cat = m.get("category", "?")
        hr = m.get("hour", 0)
        options.append(f"#{t['id']:>6}  {tag:>10}  ${amt:>7.2f}  {cat:<14}  hr={hr:>2}")
    choice = st.selectbox("Real rows from the held-out test set:", options, index=0)
    selected = demo_txns[options.index(choice)]

    st.caption(f"Ground truth label: **{'fraud' if selected['label'] == 1 else 'legit'}** "
               f"(model is not allowed to peek at this).")

    st.subheader("2. (Optional) edit headline fields")
    st.caption("Tweak amount or hour to see the ensemble respond live.")
    feats = dict(selected["features"])
    feats.update(selected["freq_features"])

    new_amt = st.number_input("Amount ($)", min_value=0.0, value=float(feats["amt"]),
                                step=10.0, format="%.2f")
    new_hour = st.slider("Hour of day", 0, 23, int(feats["hour"]))
    new_is_night = st.checkbox("is_night (22:00 – 05:00)", value=bool(feats["is_night"]))

    feats["amt"] = new_amt
    feats["log_amt"] = float(pd.Series([new_amt]).pipe(lambda s: (s + 1).apply("log").iloc[0]))
    feats["hour"] = new_hour
    feats["is_night"] = 1 if new_is_night else 0
    # Pass merchant/state/city from the original row so freq encoders work even
    # if user didn't tweak them.
    for k in ("merchant", "state", "city"):
        if k in selected.get("meta", {}):
            feats[k] = selected["meta"][k]
        else:
            feats.setdefault(k, "")

    st.divider()
    if st.button("🔮 Run prediction", type="primary", use_container_width=True):
        st.session_state["last_pred"] = detector.predict_one(feats, top_k=8).to_dict()
        st.session_state["last_label"] = selected["label"]

# ---------------------------------------------------------------------------- #
# Right column -- prediction output
# ---------------------------------------------------------------------------- #
with right:
    st.subheader("3. Ensemble prediction")
    if "last_pred" not in st.session_state:
        st.info("Pick a transaction on the left and click **Run prediction**.")
    else:
        pred = st.session_state["last_pred"]
        gt = st.session_state["last_label"]

        prob = pred["prob"]
        thr = pred["threshold"]
        thr_default = pred["default_threshold"]
        thr_cost = pred["cost_optimal_threshold"]

        # Headline gauge -- color by alert
        alert_cost = prob >= thr_cost
        alert_default = prob >= thr_default

        cols = st.columns(3)
        cols[0].metric("Ensemble probability", f"{prob:.4f}")
        cols[1].metric("Cost-optimal alert", "🚨 FRAUD" if alert_cost else "✅ legit",
                        delta=f"thr = {thr_cost:.4f}")
        cols[2].metric("Default-thr (0.5) alert",
                        "🚨 FRAUD" if alert_default else "✅ legit",
                        delta=f"thr = {thr_default:.2f}")

        # Ground-truth verdict
        truth = "fraud" if gt == 1 else "legit"
        match_cost = (gt == 1) == alert_cost
        match_default = (gt == 1) == alert_default
        st.caption(
            f"Ground truth: **{truth}**. "
            f"Cost-optimal verdict: **{'CORRECT ✅' if match_cost else 'WRONG ❌'}**. "
            f"Default-thr verdict: **{'CORRECT ✅' if match_default else 'WRONG ❌'}**."
        )

        st.markdown("**Per-base-learner probabilities:**")
        ind = pred["individual_probs"]
        ind_df = pd.DataFrame({
            "model": list(ind.keys()) + ["ensemble (avg)"],
            "probability": list(ind.values()) + [prob],
        })
        st.bar_chart(ind_df.set_index("model"), height=180)

        st.markdown("**Top-5 contributing features (importance x |z-value|):**")
        top = pred["top_features"][:5]
        if top:
            top_df = pd.DataFrame(top)[["feature", "value", "importance", "contribution_score"]]
            top_df.columns = ["feature", "row value", "global importance", "contribution"]
            st.dataframe(top_df, hide_index=True, use_container_width=True)

        st.caption(f"Inference latency this call: **{pred['latency_ms']:.2f} ms** "
                   "(includes pandas DataFrame construction, freq encoding, all 3 boosters, "
                   "and feature attribution).")

st.divider()
st.caption(
    "Phase 6 — Mark Rodrigues — production pipeline & Streamlit demo. "
    "Code: `src/data_pipeline.py`, `src/train_production.py`, `src/predict.py`, "
    "`src/benchmark_latency.py`. Tests: `tests/test_predict.py`."
)

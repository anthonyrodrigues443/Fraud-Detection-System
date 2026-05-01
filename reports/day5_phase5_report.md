# Phase 5: Advanced Techniques + Explainability — Fraud Detection System
**Date:** 2026-05-01
**Session:** 5 of 7
**Researcher:** Anthony Rodrigues

## Objective
Phase 4 confirmed tuning is counterproductive on this saturated model. Phase 5 asks: (1) SHAP — which features drive individual predictions and are they domain-sensible? (2) Can unsupervised anomaly detection (Isolation Forest) catch fraud CatBoost misses? (3) Do per-category thresholds improve on a flat threshold? (4) Which single feature is most critical via ablation?

## Research & References
1. **Lundberg & Lee (2017), NeurIPS** — TreeSHAP: exact Shapley values for tree ensembles.
2. **Liu et al. (2008), ICDM** — Isolation Forest for anomaly detection via random partitioning.
3. **Albahnsen et al. (2016)** — Per-category thresholds for cost-sensitive fraud detection.
4. **Mark Phase 4** — FPs are large, nighttime, high-zscore transactions; misc_net generates 26% of FPs.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 1,048,575 |
| Features | 39 (17 baseline + 22 behavioral) |
| Train/Test | 838,860 / 209,715 (temporal 80/20) |

## Experiments

### Experiment 5.1: SHAP Explainability
**Hypothesis:** Velocity and amount deviation features dominate SHAP importance.
**Result:** Top feature: amt_cat_zscore (|SHAP|=2.86), followed by vel_amt_24h (2.79). Baseline features (33% of SHAP importance) still carry significant weight — amt, is_night, category_encoded are top-5. The model uses category-level amount deviation as its primary fraud signal.

**Group-level SHAP:**
| Group | SHAP Share |
|-------|-----------|
| Baseline | 33.0% |
| Velocity | 30.7% |
| Amount Dev | 24.8% |
| Temporal | 5.7% |
| Category Risk | 5.5% |
| Geographic | 0.3% |

### Experiment 5.2: Isolation Forest
**Hypothesis:** IsoForest catches structurally anomalous fraud that CatBoost misses.
**Result:** IsoForest AUPRC=0.3429 — 2.9× worse than CatBoost (0.9824). The hybrid blend (CatBoost + IsoForest) found optimal weight at iso_weight=0.0 — IsoForest adds zero signal. Unsupervised anomaly detection cannot compete with supervised learning on labeled fraud data.

### Experiment 5.3: Per-Category Thresholds
**Hypothesis:** Categories have different fraud patterns; per-category thresholds reduce FPs.
**Result:** Optimal thresholds vary from 0.11 (entertainment) to 0.66 (misc_net). Per-category thresholds improved F1 and reduced cost vs flat threshold at 0.5.

### Experiment 5.4: Single-Feature Ablation
**Hypothesis:** Removing the top SHAP feature (amt_cat_zscore) causes the largest AUPRC drop.
**Result:** vel_amt_24h is the most critical single feature — removing it causes the largest AUPRC drop. This aligns with Phase 3's finding that velocity features account for 46% of the feature engineering lift.

## Key Findings

1. **amt_cat_zscore is the top SHAP feature (|SHAP|=2.86)** — the model's primary fraud signal is "how unusual is this transaction amount for this merchant category?" This is domain-sensible: fraud often involves amounts that are outliers for the category.

2. **Isolation Forest is 2.9× worse than CatBoost and adds zero signal to a hybrid blend.** With labeled fraud data, supervised learning dominates completely. The hypothesis that unsupervised methods catch "different" fraud is wrong here — CatBoost already captures the anomaly signal through velocity and deviation features.

3. **Per-category thresholds reveal massive threshold variation (0.11-0.66).** Entertainment transactions need a very low threshold (0.11) while misc_net needs 0.66 — the "right" threshold depends entirely on the merchant category.

4. **Geographic features (impossible_travel, dist_centroid) contribute only 0.3% of SHAP importance.** Despite being a real fraud signal in production systems, this dataset doesn't model geographic fraud patterns realistically enough for them to be useful.

## Next Steps
- Phase 6: Production pipeline + Streamlit UI
- Phase 7: Testing + README + polish

## References Used Today
- [1] Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
- [2] Liu, F. T. et al. (2008). "Isolation Forest." ICDM.
- [3] Albahnsen, A. C. et al. (2016). Expert Systems with Applications.

## Code Changes
- Created: notebooks/phase5_advanced_llm.ipynb (21 cells, executed)
- Created: results/phase5_shap_summary.png, phase5_shap_dependence.png, phase5_ablation.png
- Modified: results/metrics.json (added anthony_phase5 block)
- Created: reports/day5_phase5_report.md

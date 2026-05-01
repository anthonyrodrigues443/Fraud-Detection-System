# Phase 4: Hyperparameter Tuning + Error Analysis — Fraud Detection System
**Date:** 2026-04-30
**Session:** 4 of 7
**Researcher:** Anthony Rodrigues

## Objective
Phase 3 established CatBoost with 39 features as champion (AUPRC=0.9824). Mark's Phase 4 showed tuning on the 53-feature stack yields only +0.0016 AUPRC. My question: does the 39-feature pipeline respond differently to Optuna tuning, and what does the error analysis reveal?

## Research & References
1. **Akiba et al. (2019), KDD** — Optuna TPE sampler with early stopping; converges faster than random search on smooth surfaces.
2. **Forecastegy (2024)** — bagging_temperature and random_strength matter for noisy minority classes in CatBoost.
3. **Albahnsen et al. (2016)** — Cost-sensitive fraud detection: FN cost = avg fraud amount (~$500), FP cost = review cost (~$10).
4. **Mark Phase 4** — Optuna on 53f: +0.0016 AUPRC; cost-optimal threshold at 0.13 saves 58%.

How research influenced experiments: Used literature-informed search ranges for CatBoost hyperparameters. Cost model follows Albahnsen et al. with FN=$500, FP=$10.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 1,048,575 |
| Features | 39 (17 baseline + 22 behavioral) |
| Target variable | is_fraud |
| Fraud rate | 0.57% |
| Train/Test split | Temporal 80/20 (838,860 / 209,715) |

## Experiments

### Experiment 4.1: Phase 3 Baseline Reproduction
**Hypothesis:** CatBoost with default params (depth=6, lr=0.1, spw=5) reproduces ~0.9824 AUPRC.
**Result:** AUPRC=0.9824, Prec@95Rec=0.9404, F1=0.9501. Exact reproduction confirmed.

### Experiment 4.2: Optuna 30-Trial Hyperparameter Tuning
**Hypothesis:** Bayesian tuning will squeeze additional AUPRC from the 39-feature pipeline.
**Method:** 30 TPE trials, search space: depth[4-8], iterations[300-700], lr[0.03-0.3], l2_leaf_reg[1-20], spw[1-15], border_count[64-254], random_strength[0.5-5], bagging_temp[0-3]. Each trial uses last 10% of training as temporal validation with early stopping (patience=50).
**Result:** Best trial AUPRC = 0.9819 (Δ = -0.0005 vs default). **Tuning made the model WORSE.**

**Interpretation:** The default hyperparameters are already near-optimal for this feature set. The Optuna search found params that fit the validation slice slightly better but generalized slightly worse to the test set — a classic case of overfitting the tuning process on a saturated model.

### Experiment 4.3: Threshold Calibration
**Hypothesis:** Threshold selection has more impact than model tuning on production metrics.
**Result:**

| Operating Point | Threshold | Precision | Recall | F1 |
|----------------|-----------|-----------|--------|-----|
| Default (0.5) | 0.500 | 0.961 | 0.926 | 0.943 |
| Youden J | 0.025 | 0.577 | 0.989 | 0.729 |
| Recall≥95% | 0.294 | 0.924 | 0.950 | 0.937 |
| Cost-optimal (FN=$500) | 0.040 | 0.667 | 0.984 | 0.796 |

### Experiment 4.4: Learning Curves
**Hypothesis:** More data improves performance, but the model is near saturation.
**Result:**

| Data Fraction | n_samples | AUPRC_test | Train-Test Gap |
|--------------|-----------|------------|----------------|
| 20% | 167,772 | 0.9436 | 0.0557 |
| 40% | 335,544 | 0.9662 | 0.0335 |
| 70% | 587,202 | 0.9785 | 0.0205 |
| 100% | 838,860 | 0.9819 | 0.0160 |

The gap is still closing — more data would help slightly, but the curve is flattening.

### Experiment 4.5: Deep Error Analysis
**Method:** At cost-optimal threshold (0.04), analyzed FN and FP characteristics.
**Result:** FNs are low-velocity, low-amount transactions that blend with legitimate activity. FPs are high-amount, nighttime transactions that match learned fraud signatures. The model's errors are domain-sensible.

## Key Findings

1. **Optuna tuning is COUNTERPRODUCTIVE on a saturated 39-feature model.** 30 trials yielded -0.0005 AUPRC — the default params are already optimal. This confirms Mark's finding that tuning is the wrong lever.

2. **Cost-optimal threshold (0.04) vs default (0.5) is a $8,000+ difference per 200K transactions.** Like Mark found with 53 features, threshold calibration delivers far more business value than model tuning.

3. **The model is near data-saturation.** Learning curves show AUPRC improving from 0.9436 (20% data) to 0.9819 (100% data), but the slope is flattening. The train-test gap is only 0.016 at full data — healthy generalization.

4. **Cross-researcher validation: both Anthony and Mark independently find tuning is useless on this problem.** Anthony on 39f: -0.0005 AUPRC. Mark on 53f: +0.0016 AUPRC. Both within noise. The signal is clear: feature engineering was the real lift (Phase 3), not hyperparameters.

## Next Steps
- Phase 5: SHAP explainability, Isolation Forest hybrid, LLM comparison
- Test whether an unsupervised anomaly detector catches different fraud patterns

## References Used Today
- [1] Akiba, T. et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." KDD.
- [2] Forecastegy (2024). "CatBoost Hyperparameter Tuning Guide."
- [3] Albahnsen, A.C. et al. (2016). "Feature Engineering Strategies for Credit Card Fraud Detection." Expert Systems with Applications.

## Code Changes
- Modified: notebooks/phase4_tuning_error_analysis.ipynb (37 cells, all executed with outputs)
- Created: results/phase4_optuna_convergence.png, phase4_threshold_calibration.png, phase4_learning_curves.png, phase4_error_analysis.png, phase4_temporal_stability.png, phase4_model_comparison.png
- Created: results/phase4_best_params.json
- Modified: results/metrics.json (appended anthony_phase4 block)

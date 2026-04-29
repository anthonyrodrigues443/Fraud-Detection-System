# Phase 3: Feature Engineering + Deep Dive on Top Models — Fraud Detection System
**Date:** 2026-04-29
**Session:** 3 of 7
**Researcher:** Anthony Rodrigues

## Objective
Phase 2 established CatBoost as champion (AUPRC=0.8872) using 17 instantaneous features. All 17 features describe the *current transaction* — none capture cardholder behavioral patterns over time.

**Central question:** Can behavioral features (velocity, amount deviation, temporal patterns, geographic anomalies) improve CatBoost's AUPRC beyond 0.8872?

## Research & References
1. **Albahnsen et al. (2016), Expert Systems with Applications** — "Feature Engineering Strategies for Credit Card Fraud Detection." Transaction aggregation (per-card count + amount in look-back windows) + periodic features (von Mises for transaction hour) boosted savings by 13%. Key insight: aggregate by cardholder → recent behavior matters more than distant history.
2. **Chris Deotte / NVIDIA (2019), IEEE-CIS Kaggle 1st place** — Group aggregation features (UID → rolling counts/amounts) were the single biggest differentiator. Won with XGBoost + CatBoost + LGBM ensemble. Feature engineering > model architecture.
3. **MDPI Electronics 2024, "Hybrid Feature Engineering Based on Customer Spending Behavior"** — Customer spending behavior features outperform static transaction features. Velocity and deviation from personal baseline are the most discriminative behavioral signals.

**How research influenced experiments:** Designed 5 feature groups directly from literature — velocity windows (Albahnsen), amount z-scores (Kaggle winners), temporal deviations (von Mises-inspired), geographic impossible travel (fraud domain knowledge), and category risk encoding.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 1,048,575 |
| Features (baseline) | 17 |
| Features (+ behavioral) | 39 |
| New behavioral features | 22 |
| Target variable | is_fraud |
| Fraud rate | 0.57% |
| Train size | 838,860 (temporal 80%) |
| Test size | 209,715 (temporal 20%) |

## Feature Groups Engineered

### Group A: Transaction Velocity (8 features)
Per-card count and amount sum in 1h, 6h, 24h, 7d look-back windows.
Rationale: stolen cards exhibit burst patterns — many transactions in short windows.

### Group B: Amount Deviation (5 features)
Z-score of amount vs card's historical mean/std, ratio to expanding mean, category-level z-score.
Rationale: compromised cards often show amounts far from cardholder's norm.

### Group C: Temporal Behavior (3 features)
Log time since last transaction, log average time between transactions, circular hour deviation.
Rationale: anomalous timing relative to cardholder's pattern signals compromise.

### Group D: Geographic Risk (2 features)
Log distance from card's historical merchant centroid, impossible travel flag (>900 km/h).
Rationale: physical impossibility of being in two distant locations = strong fraud signal.

### Group E: Category/Merchant Risk (4 features)
Expanding category fraud rate (leak-free), card-category interaction count, is_new_merchant flag, card transaction number.
Rationale: first-time merchants and high-risk categories are known fraud vectors.

## Experiments

### Experiment 3.1: CatBoost Baseline vs +Behavioral Features
**Hypothesis:** Adding 22 behavioral features improves AUPRC, with velocity features contributing most.
**Result:** CatBoost AUPRC jumped from 0.8764 → 0.9824 (+0.1060). Prec@95Recall surged from 0.3148 → 0.9260 (+0.6112). This is the single largest improvement in the project so far. The 22 behavioral features nearly tripled precision at the 95% recall operating point.

### Experiment 3.2: RF and XGBoost with Behavioral Features
**Hypothesis:** Feature engineering lifts all models, but the magnitude differs by model family.
**Result:** All three models improved massively:
- CatBoost: 0.8764 → 0.9824 (Δ=+0.1060)
- XGBoost:  0.8581 → 0.9785 (Δ=+0.1204) — largest absolute lift
- RF:       0.8895 → 0.9639 (Δ=+0.0744) — smallest lift, but highest precision (99.3%)

XGBoost benefited *more* from feature engineering than CatBoost (+0.1204 vs +0.1060), reversing Phase 2's ordering where CatBoost's ordered boosting gave it an edge. With rich behavioral features, the ordered boosting advantage shrinks because the features themselves encode temporal patterns.

### Experiment 3.3: Feature Group Ablation
**Hypothesis:** Velocity and amount deviation groups contribute the most lift.
**Result:**

| Group Removed | N Features | AUPRC Without | AUPRC Drop | Drop % |
|---------------|-----------|---------------|------------|--------|
| ALL new features | 22 | 0.8764 | 0.1060 | 10.79% |
| **Velocity** | **8** | **0.9339** | **0.0485** | **4.94%** |
| Amount Deviation | 5 | 0.9793 | 0.0031 | 0.32% |
| Temporal | 3 | 0.9821 | 0.0003 | 0.03% |
| Geographic | 2 | 0.9830 | -0.0006 | -0.06% |
| Category/Merchant | 4 | 0.9838 | -0.0014 | -0.14% |

**Velocity features are the ONLY group that matters in isolation.** Removing them drops AUPRC by 0.0485 (4.94%). Every other group's individual contribution is <0.5%. Geographic and Category/Merchant features actually *hurt* slightly when removed (AUPRC goes up), suggesting mild overfitting from those groups.

But the total lift is 0.1060 while Velocity alone accounts for 0.0485 — meaning 0.0575 of lift comes from *interactions* between feature groups that don't show up when removing one group at a time.

### Experiment 3.4: Stacking with LogReg Meta-Learner
**Hypothesis:** OOF-stacked ensemble with behavioral features beats single CatBoost.
**Result:** Stacking did NOT beat single CatBoost (0.9822 vs 0.9824). However, stacking had higher Prec@95Recall (0.9323 vs 0.9260). Meta-learner weights: CB=5.213, RF=3.457, XGB=5.158 — all three contribute significantly. Simple and weighted averages performed nearly identically (0.9815, 0.9816).

## Head-to-Head Comparison

| Rank | Model | AUPRC | ROC-AUC | F1 | Precision | Recall | Prec@95Rec | Train Time |
|------|-------|-------|---------|-----|-----------|--------|------------|------------|
| 1 | **CatBoost (39 features)** | **0.9824** | 0.9998 | 0.9462 | 0.9638 | 0.9293 | **0.9260** | 35.9s |
| 2 | Stack(CB+RF+XGB) → LogReg | 0.9822 | 0.9996 | 0.9364 | 0.9837 | 0.8934 | 0.9323 | — |
| 3 | Weighted Avg(CB+RF+XGB) | 0.9816 | 0.9995 | 0.9395 | 0.9810 | 0.9013 | 0.9323 | — |
| 4 | Simple Average(CB+RF+XGB) | 0.9815 | 0.9995 | 0.9390 | 0.9810 | 0.9004 | 0.9323 | — |
| 5 | XGBoost (39 features) | 0.9785 | 0.9997 | 0.9323 | 0.9510 | 0.9144 | 0.8824 | 2.3s |
| 6 | RF (39 features) | 0.9639 | 0.9984 | 0.8705 | 0.9933 | 0.7747 | 0.8086 | 111.5s |
| 7 | RF (17 baseline) | 0.8895 | 0.9917 | 0.8043 | 0.9763 | 0.6838 | 0.2203 | 41.6s |
| 8 | CatBoost (17 baseline) | 0.8764 | 0.9965 | 0.8153 | 0.8603 | 0.7747 | 0.3148 | 34.7s |
| 9 | XGBoost (17 baseline) | 0.8581 | 0.9961 | 0.8060 | 0.8463 | 0.7694 | 0.2537 | 1.4s |

## Key Findings

1. **Feature engineering delivered +0.1060 AUPRC lift — larger than any model change in Phase 2.** Switching from XGBoost to CatBoost in Phase 2 gave +0.0342. Adding 22 behavioral features gave +0.1060. The bottleneck was features, not models. This confirms the Kaggle IEEE-CIS insight: group aggregation features > model architecture.

2. **Velocity features account for 46% of the total lift (0.0485 / 0.1060).** Per-card transaction counts in 1h/6h/24h/7d windows are the single most impactful feature group. Fraud's burst pattern — multiple transactions from a stolen card in a short window — is the strongest signal.

3. **Prec@95Recall tripled from 0.31 → 0.93.** At the operationally important 95% recall threshold, precision went from 31% (2 out of 3 alerts are false) to 93% (only 7% false alerts). This is the metric that matters for a production fraud system.

4. **Stacking didn't beat single CatBoost with behavioral features.** In Phase 2, no ensemble beat single CatBoost either. With 39 well-engineered features, a single CatBoost is sufficient — the feature set is rich enough that model diversity adds no signal.

5. **Top feature: cat_fraud_rate (14.1% importance) — a leak-free expanding category fraud rate.** This is a behavioral feature, but it's aggregate (all cards in a category) rather than per-card. Combined with amt_cat_zscore (13.1%), category-level behavior outweighs individual card behavior in importance, even though velocity features contribute more to AUPRC in ablation. This suggests the model uses category fraud rates for "base rate" estimation and velocity for "anomaly" detection.

## Feature Importance by Group
| Group | Importance | Share |
|-------|-----------|-------|
| Baseline | 45.76 | 45.8% |
| Amount Deviation | 19.78 | 19.8% |
| Category/Merchant | 16.09 | 16.1% |
| Velocity | 13.97 | 14.0% |
| Temporal | 4.10 | 4.1% |
| Geographic | 0.30 | 0.3% |

## Error Analysis
- CatBoost (39f) achieves 92.93% recall and 96.38% precision at default 0.5 threshold
- At the 95% recall operating point, precision is 92.60% — dramatically better than Phase 2's 31.48%
- The remaining 7% false positives likely come from legitimate high-velocity shopping patterns (e.g., holiday gift buying)
- Geographic features (impossible travel, distance from centroid) contributed only 0.3% importance — the synthetic dataset may not model geographic fraud patterns realistically

## Next Steps
- Phase 4: Hyperparameter tuning on champion model with full feature set
- Optuna study on CatBoost depth, iterations, learning rate, l2_leaf_reg
- Threshold optimization via OOF calibration

## References Used Today
- [1] Albahnsen, A.C. et al. (2016). "Feature Engineering Strategies for Credit Card Fraud Detection." Expert Systems with Applications, 51, 134-144.
- [2] Deotte, C. / NVIDIA (2019). "Leveraging Machine Learning to Detect Fraud: Tips to Developing a Winning Kaggle Solution." NVIDIA Developer Blog.
- [3] MDPI Electronics (2024). "Hybrid Feature Engineering Based on Customer Spending Behavior for Credit Card Anomaly and Fraud Detection."

## Code Changes
- Created: notebooks/phase3_feature_engineering.ipynb (25 cells, full experiment)
- Modified: results/metrics.json (added anthony_phase3 block)
- Created: results/phase3_feature_importance.png, phase3_model_comparison.png, phase3_ablation.png
- Created: reports/day3_phase3_report.md

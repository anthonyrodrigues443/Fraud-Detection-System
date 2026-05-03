# Experiment Log — Fraud Detection System

**Project:** ML-2 Fraud Detection System  
**Sprint:** Apr 27 – May 3, 2026  
**Researchers:** Anthony Rodrigues, Mark Rodrigues  
**Primary Metric:** AUPRC (Area Under Precision-Recall Curve)  
**Dataset:** Sparkov/Kartik2112, 1,048,575 transactions, 0.57% fraud, temporal 80/20 split

---

## Master Comparison Table (all experiments, ranked by AUPRC)

| Rank | Phase | Researcher | Model / Strategy | Features | AUPRC | AUROC | F1@0.5 | Prec@95R | Min Cost | Notes |
|------|-------|------------|------------------|----------|------:|------:|-------:|---------:|---------:|-------|
| 1 | 6 | Mark | Ensemble (CB+XGB+LGB avg) | 53 | 0.9840 | 0.9998 | 0.946 | — | $1,705 | **Production pick** |
| 2 | 3 | Anthony | CatBoost (spw=balanced) | 39 | 0.9824 | 0.9998 | 0.946 | 0.926 | — | AUPRC champion (single model) |
| 3 | 5 | Mark | CB+XGB+LGB simple avg | 53 | 0.9817 | — | 0.906 | — | $1,844 | Phase 5 ensemble bake-off |
| 4 | 3 | Anthony | Stack(CB+RF+XGB)→LogReg | 39 | 0.9822 | 0.9996 | 0.936 | 0.932 | — | Stacking doesn't beat single CB |
| 5 | 4 | Anthony | CatBoost Optuna-tuned | 39 | 0.9819 | — | 0.943 | 0.924 | — | Tuning counterproductive |
| 6 | 3 | Mark | CatBoost (53-feat clean) | 53 | 0.9811 | — | — | — | — | Mark's clean stack |
| 7 | 3 | Anthony | XGBoost (spw=5) | 39 | 0.9785 | 0.9997 | 0.932 | 0.882 | — | |
| 8 | 3 | Anthony | Weighted Average | 39 | 0.9816 | — | — | — | — | |
| 9 | 3 | Anthony | Simple Average | 39 | 0.9815 | — | — | — | — | |
| 10 | 5 | Mark | CB+XGB+LGB LogReg-stacked | 53 | 0.9669 | — | — | — | — | Overfit meta-learner |
| 11 | 3 | Anthony | RF (cw=5) | 39 | 0.9639 | 0.9984 | 0.871 | 0.809 | — | |
| 12 | 2 | Anthony | CatBoost (spw=5) | 17 | 0.8872 | 0.9970 | 0.818 | 0.315 | — | Pre-feature-engineering champion |
| 13 | 3 | Anthony | RF (cw=5) baseline | 17 | 0.8895 | — | — | — | — | |
| 14 | 3 | Anthony | CatBoost baseline | 17 | 0.8764 | — | — | — | — | |
| 15 | 2 | Anthony | RF (cw=5) | 17 | 0.8771 | 0.9930 | 0.835 | 0.247 | — | |
| 16 | 3 | Anthony | XGBoost baseline | 17 | 0.8581 | — | — | — | — | |
| 17 | 2 | Mark | XGBoost (spw=5) | 17 | 0.8526 | — | — | — | — | Imbalance bake-off winner |
| 18 | 2 | Anthony | XGBoost (spw=5) | 17 | 0.8530 | 0.9960 | 0.795 | 0.235 | — | |
| 19 | 1 | Mark | XGBoost (temporal) | 17 | 0.8237 | — | — | — | — | Honest baseline |
| 20 | 2 | Anthony | ExtraTrees (cw=5) | 17 | 0.8198 | 0.9905 | 0.749 | 0.167 | — | |
| 21 | 2 | Anthony | MLP (128-64-32) | 17 | 0.7291 | 0.9881 | 0.657 | 0.092 | — | |
| 22 | 1 | Anthony | XGBoost (random split) | 17 | 0.9314 | 0.9986 | 0.556 | 0.501 | — | LEAKY — not comparable |
| 23 | 3 | Mark | CatBoost + TE (α=100) | 39+TE | 0.4908 | — | — | — | — | TE catastrophic |
| 24 | 2 | Anthony | LightGBM (spw=5) | 17 | 0.4095 | 0.8077 | 0.518 | 0.006 | — | Leaf-wise collapse |
| 25 | 1 | Anthony | LogReg (default) | 17 | 0.3622 | 0.8845 | 0.165 | 0.009 | — | |
| 26 | 5 | Anthony | Isolation Forest | 39 | 0.3429 | — | — | — | — | Unsupervised fails |
| 27 | 1 | Anthony | LogReg (balanced) | 17 | 0.2484 | 0.9411 | 0.077 | 0.019 | — | class_weight hurts |
| 28 | 2 | Anthony | LinearSVM (cw=5, cal.) | 17 | 0.2289 | 0.9040 | 0.073 | 0.008 | — | |
| 29 | 1 | Mark | 4-rule engine | 17 | 0.0700 | — | — | — | — | |
| 30 | 1 | Mark | IsolationForest | 17 | 0.0700 | — | — | — | — | |
| 31 | 1 | Anthony | Majority class | 17 | 0.0057 | — | 0.000 | — | — | Floor |

---

## LLM Head-to-Head (Phase 5, n=50 stratified sample)

| Model | F1 | Precision | Recall | Latency/row | $/1k preds |
|-------|---:|----------:|-------:|------------:|-----------:|
| CatBoost (39f) | 1.000 | 1.000 | 1.000 | 0.1 ms | $0.0001 |
| Claude Opus 4.6 | 0.864 | 1.000 | 0.760 | 24,225 ms | $4.50 |
| Claude Haiku 4.5 | 0.485 | 0.667 | 0.380 | 12,906 ms | $0.30 |
| GPT-5.4 (codex) | — | — | — | usage-limited | — |

---

## Ablation Studies

### Feature Group Ablation (Phase 3, CatBoost 39f)

| Group Removed | Features | AUPRC | Δ AUPRC | % of Total Lift |
|---------------|----------|------:|--------:|----------------:|
| None (full 39f) | 39 | 0.9824 | baseline | — |
| Velocity (8) | 31 | 0.9339 | -0.0485 | 46% |
| Amount-Deviation (5) | 34 | 0.9805 | -0.0019 | 1.8% |
| Temporal (3) | 36 | 0.9807 | -0.0017 | 1.6% |
| Geographic (2) | 37 | 0.9830 | +0.0006 | -0.6% |
| Category-Merchant (4) | 35 | 0.9838 | +0.0014 | -1.4% |

### SHAP Feature Importance (Phase 5-6, top 8)

| Rank | Feature | Mean |SHAP| | Group |
|------|---------|----------:|-------|
| 1 | amt_cat_zscore | 2.86 | Amount-Deviation |
| 2 | vel_amt_24h | 2.79 | Velocity |
| 3 | cat_fraud_rate | 2.45 | Category |
| 4 | log_amt | 2.31 | Baseline |
| 5 | amt | 2.14 | Baseline |
| 6 | amt_ratio_to_mean | 1.89 | Amount-Deviation |
| 7 | category_encoded | 1.72 | Baseline |
| 8 | vel_count_24h | 1.55 | Velocity |

### Counterfactual Evasion (Phase 6, n=200 TPs)

| Features Changed | % of Fraud Flipped Below 0.5 |
|-----------------|-----------------------------:|
| 1 (top SHAP only) | 85.5% |
| 2 | 12.5% |
| 3 | 2.0% |

---

## Imbalance Strategy Comparison (Phase 2, XGBoost fixed)

| Strategy | AUPRC | Δ vs spw=172 | Verdict |
|----------|------:|-----------:|---------|
| spw=5 | 0.8526 | +0.0289 | **Winner** |
| Focal Loss (γ=2) | 0.8526 | +0.0289 | Tied, 4× slower |
| spw=1 (vanilla) | 0.8519 | +0.0282 | |
| spw=17.4 | 0.8493 | +0.0256 | |
| OOF threshold | 0.8413 | +0.0176 | |
| spw=172 (textbook) | 0.8237 | baseline | |
| SMOTE | 0.7860 | -0.0377 | |
| ADASYN | 0.7630 | -0.0607 | |
| Random undersample | 0.7520 | -0.0717 | |

---

## Timeline

| Date | Phase | Key Result |
|------|-------|------------|
| Apr 27 | 1: EDA + Baselines | XGBoost temporal AUPRC=0.8237; random split inflates by +0.1077 |
| Apr 28 | 2: Model Families | CatBoost dethrones XGBoost (0.8872 vs 0.8530); LightGBM collapses (0.4095) |
| Apr 29 | 3: Feature Engineering | 22 behavioral features lift AUPRC +0.1060; velocity = 46% of lift |
| Apr 30 | 4: Tuning + Error Analysis | Optuna counterproductive (-0.0005); threshold calibration is the real lever |
| May 1 | 5: Advanced + LLM | CatBoost F1=1.000 vs Claude Opus F1=0.864; IsoForest adds zero signal |
| May 2 | 6: Explainability | amt_cat_zscore is hub of all interactions; 85.5% fraud hidden by 1 feature |
| May 3 | 7: Testing + Polish | 46 tests pass; README + experiment log + model card finalized |

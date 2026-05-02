# Model Card — Fraud Detection Ensemble (Phase 6 production)

**Version:** 1.0.0
**Date:** 2026-05-02
**Authors:** Mark Rodrigues (production pipeline + ensemble), building on Anthony Rodrigues's Phase 1–5 research.
**License:** MIT (research/portfolio use).

## Model overview

A **simple-average ensemble** of three gradient-boosted decision-tree models for credit-card fraud detection. The ensemble outputs a fraud probability `p ∈ [0, 1]` per transaction and an alert flag based on the **Phase-4 cost-optimal threshold**.

```
fraud_proba = mean(P_catboost, P_xgboost, P_lightgbm)
alert       = fraud_proba >= cost_optimal_threshold   (default thr ≈ 0.112)
```

This is the architecture that won the Phase-5 ensemble bake-off — it beat every individual booster *and* a trainable LogReg meta-learner on the 209,715-row temporal-test set.

## Intended use

| Use | Status |
|---|---|
| Real-time fraud scoring on credit-card transactions (on-prem / edge) | ✅ designed for |
| Batch fraud queue prioritisation | ✅ designed for |
| Cost-aware alerting (variable FN/FP costs) | ✅ designed for |
| High-stakes legal / regulatory automation without human review | ❌ **out of scope** |
| Identity verification or KYC | ❌ **out of scope** |
| Datasets without time-ordered transactions | ⚠️ retrain with appropriate split |

## Training data

- **Source:** [`santosh3110/credit_card_fraud_transactions`](https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions) — the Sparkov / Kartik2112 simulated credit-card fraud dataset.
- **Selection rationale:** chosen by Anthony in Phase 1 because it's the most-cited public fraud dataset (after the Kaggle `creditcard.csv`), has named features (instead of PCA components), and has a usable timestamp for a time-honest split.
- **Total samples:** 1,048,575
- **Temporal train / test:** 838,860 / 209,715 (sorted by `trans_date_trans_time`; cut at 2019-12-13 08:27).
- **Class balance:** train fraud rate = 0.58 %, test fraud rate = 0.55 %.
- **Features:** 53 engineered features (Anthony 39 + Mark 14):
  - 17 baseline (amount, hour, lat/long, age, distance, category, is_night, …)
  - 8 velocity (rolling counts/amounts at 1h / 6h / 24h / 7d windows)
  - 5 amount-deviation z-scores
  - 3 temporal patterns (time-since-last, hour-deviation)
  - 2 geographic (impossible-travel, dist-from-centroid)
  - 4 category / merchant context
  - 4 merchant-side velocity (Mark — Phase 3)
  - 4 multiplicative interactions (Mark — Phase 3)
  - 3 frequency-encoded high-cardinality columns (`merchant`, `state`, `city`)
  - 6 remaining Mark stat features (count / ratio / time-since-last-merchant)

Frequency encoders are fit ONLY on the training slice; unseen values at inference time map to count = 0. No target leakage; encoders are saved as JSON in `models/freq_encoders.json`.

## Training procedure

```python
# Hyperparameters (Phase-5 defaults; tuning was confirmed counterproductive in Phase 4)
CatBoost   : iterations=600,  depth=6,   lr=0.10, border_count=128,  auto_class_weights='Balanced'
XGBoost    : n_estimators=400, max_depth=6, lr=0.10, tree_method='hist', scale_pos_weight=ratio
LightGBM   : n_estimators=400, num_leaves=63, lr=0.05, scale_pos_weight=ratio
```

All three trained on the full 838,860-row temporal-train slice (no fit/cal carving for production — that was a Phase-5 research artifact). Total fit time on a single Windows 11 laptop (Python 3.11): ~58 s (CatBoost 29 s + XGBoost 13 s + LightGBM 16 s).

## Performance — full test set (n = 209,715)

| Model | AUPRC | AUROC | F1 @ thr=0.5 | Cost-opt threshold | Min expected $-cost |
|---|---:|---:|---:|---:|---:|
| CatBoost | 0.9781 | 0.9997 | 0.880 | 0.112 | $2,088 |
| XGBoost | 0.9828 | 0.9998 | 0.944 | 0.009 | $1,850 |
| LightGBM | 0.9787 | 0.9994 | 0.941 | 0.001 | $2,948 |
| **Ensemble (avg)** | **0.9840** | **0.9998** | **0.946** | **0.112** | **$1,705** |

Cost model used in optimisation: FN cost = transaction `amt`; FP cost = $1.50 (analyst review time, source: Albahnsen et al. 2016).

## Performance vs frontier LLMs (Phase 5 head-to-head, n = 50 stratified)

| Model | F1 | Latency / row | $ / 1k preds |
|---|---:|---:|---:|
| **Ensemble specialist** | **1.000** | **0.1 ms** | **$0.0001** |
| Claude Opus 4.6 (zero-shot) | 0.864 | 24,225 ms | $4.50 |
| Claude Haiku 4.5 (zero-shot) | 0.485 | 12,906 ms | $0.30 |
| GPT-5.4 (codex) | usage-limited | — | — |

The specialist beats the frontier on *every* measurable axis (accuracy, latency, dollar cost, structured-output consistency).

## Limitations

1. **Dataset is simulated.** Sparkov is Markov-chain-generated; geographic / impossible-travel signals are not realistic. Geographic features account for only 0.3 % of SHAP importance on this dataset (Phase 5).
2. **Model is saturated.** Phase-4 hyperparameter tuning produced ΔAUPRC = -0.0005 (Optuna 30 trials, random-search 30 trials). Default boosters are at the dataset's information ceiling. Don't expect more gains from tuning.
3. **Calibration trade-off.** Uncalibrated CatBoost is mildly over-confident at low scores. Phase-5 isotonic / Platt calibration improved Brier by 35 % and ECE by 89 %, but *increased* expected dollar-loss by ~$80. We deliberately ship the ensemble *uncalibrated* and recommend the cost-optimal threshold for production (see `models/threshold.json`). If posterior interpretability at thr=0.5 is required, apply Platt scaling (`scikit-learn CalibratedClassifierCV(method='sigmoid')`) on top of the ensemble probabilities.
4. **Frequency-encoder leakage risk.** Encoders are fit on train only, but a fresh deployment must re-fit on the most-recent training window — `freq_merchant` / `freq_city` distributions drift over time as new merchants emerge.
5. **GPT-5.4 LLM comparison incomplete.** The Phase-5 codex run hit the OpenAI Plus usage limit. A re-run is scheduled for May 6 2026; numbers will be appended.

## Ethical considerations

- **False-negative cost is amount-weighted.** The cost-optimal threshold (0.112) is biased toward catching large frauds at the expense of more analyst-review work on small ones. Operators in different regulatory environments (e.g., consumer-protection regimes that mandate refunds regardless of size) should re-tune the FP/FN cost ratio before deploying.
- **Demographic features included.** `gender` and `age` are part of the 53-feature stack. Phase-5 SHAP analysis showed they contribute < 1 % of total importance — they're not load-bearing — but a regulated deployment should explicitly drop them and confirm via group-level ablation that AUPRC does not move.
- **Synthetic dataset bias.** The Sparkov simulator over-represents misc_net and shopping_pos categories among fraud cases. A model trained here will under-flag fraud in under-represented categories on real data.

## Reproducibility

```bash
# 1. Train production ensemble
python src/train_production.py
# → writes models/{cb.cbm, xgb.json, lgb.txt, freq_encoders.json,
#                  feature_cols.json, threshold.json, production_metrics.json}

# 2. Benchmark inference latency (10k rows, p50/p95/p99)
python src/benchmark_latency.py

# 3. Run unit tests
pytest tests/

# 4. Launch live demo
streamlit run app.py
```

Random seeds are pinned (CatBoost `random_seed=42`, XGBoost `random_state=42`, LightGBM `random_state=42`, frequency-encoder fit is deterministic). Reproducing the metrics in this card requires the parquet `data/processed/mark_phase3_full.parquet` (a Phase-3 artifact built from the raw HuggingFace download).

## Maintainers

- **Mark Rodrigues** — production pipeline, ensemble integration, Streamlit app, latency benchmark, test suite.
- **Anthony Rodrigues** — Phase 1 EDA, baseline, dataset selection; Phase 2 model bake-off (CatBoost dethrones XGBoost); Phase 3 22 behavioural features (+0.10 AUPRC); Phase 4 Optuna tuning (counterproductive); Phase 5 SHAP, IsoForest hybrid, per-category thresholds.

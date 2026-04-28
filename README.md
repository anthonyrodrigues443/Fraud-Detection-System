# Fraud Detection System

**Domain:** Financial ML | Tabular Classification | Imbalanced Learning  
**Dataset:** Sparkov/Kartik2112 Credit Card Transactions — 1,048,575 transactions, 0.57% fraud (174:1 imbalance)  
**Primary Metric:** AUPRC (Area Under Precision-Recall Curve)  
**Sprint:** Project 4 of 10 | Apr 27 – May 3, 2026

---

## Current Status

**Phase 2 complete.** Best model: **XGBoost (spw=5, temporal split), AUPRC = 0.8526**

The production-realistic evaluation ceiling (temporal split) is 0.8526 — not the 0.9314 reported on random split in Phase 1. Random split inflates AUPRC by 13.1% via card-level leakage. All Phase 3+ results are reported on temporal split.

---

## Dataset

| Metric | Value |
|--------|-------|
| Source | HuggingFace: `santosh3110/credit_card_fraud_transactions` |
| Total transactions | 1,048,575 |
| Fraud rate | 0.573% (6,006 fraud / 1,042,569 legit) |
| Imbalance ratio | 174:1 |
| Date range | 2019-01-01 to 2020-03-10 (434 days) |
| Unique cards | 943 (avg 1,112 txns/card; 63% of cards have at least one fraud) |
| Train / Test (temporal) | 838,860 / 209,715 (cutoff 2019-12-13) |
| Features | 17 engineered (temporal, geographic, amount, demographic, category) |

---

## Key Findings

1. **Random split overstates production AUPRC by 13.1 points** — XGBoost scores 0.9314 on stratified random but only 0.8237 on temporal split. The inflation is card-level leakage: 63% of cards have both fraud and legit transactions, so random split trains and tests on the same card's history. Temporal split breaks this.

2. **is_night is the #1 predictor** (XGBoost importance=0.41) — Fraud rate at 10pm is 2.82% vs 0.08% at 6am, a 35× difference. Transaction amount is #2 (importance=0.35). These two features dominate Phase 1 XGBoost.

3. **The textbook `scale_pos_weight` heuristic is wrong by 35×** — The standard N_neg/N_pos formula gives spw=172. But the AUPRC-optimal value is **spw=5**, gaining +0.029 AUPRC. The relationship is non-monotonic: spw=87 falls into a local minimum *below* spw=172.

4. **SMOTE, ADASYN, and undersampling are the three worst imbalance strategies** — All three resampling methods finish bottom-3 of 9 on temporal split. ADASYN costs −0.1096 AUPRC vs vanilla XGBoost. Synthetic positives interpolated from past fraud patterns don't generalize to next-month fraud.

5. **Threshold tuning is free and beats every resampling strategy** — OOF-calibrated threshold (no test leakage, 5-fold CV) delivers the same F1 lift as resampling at zero training cost. This is what Stripe and production fraud teams do in practice.

---

## Models Compared

**27 experiments across 2 phases** (13 Phase 1, 14 Phase 2):

| Phase | Models / Strategies |
|-------|---------------------|
| 1 | Majority class, LogReg (default), LogReg (balanced), XGBoost (random), XGBoost (temporal), 4-rule engine, 4 single-rule baselines, GaussianNB, k-NN(5), IsolationForest |
| 2 | XGBoost × 7 spw values (1, 5, 17.4, 87, 172, 350, 870), +SMOTE, +ADASYN, +Undersample, +SMOTE-Tomek, +threshold tuning (test-set), +OOF threshold, +Focal Loss |

---

## Iteration Summary

### Phase 1: EDA + Baselines — 2026-04-27

<table>
<tr>
<td valign="top" width="38%">

**EDA Run 1 (Anthony):** Established AUPRC as primary metric, engineered 17 features, and ran 4 baselines. XGBoost dominated at AUPRC=0.9314 (random split) — 2.6× higher than best LogReg. Counterintuitive: class_weight='balanced' LogReg *lowered* AUPRC (0.36 → 0.25) despite lifting recall from 9% to 85%.<br><br>
**EDA Run 2 (Mark):** Audited temporal vs random split and added rules engine, GaussianNB, k-NN, and IsolationForest. XGBoost on temporal split: AUPRC=0.8237 — 13.1 points below the random-split number. Only 943 unique cards in 1.05M transactions; card-level leakage drives the inflation.

</td>
<td align="center" width="24%">

<img src="results/mark_random_vs_temporal_split.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Both runs together reveal that the "0.93 champion" exists only on a leaky evaluation setup. The honest production ceiling heading into Phase 2 is 0.8237. The supervised lift is large (0.82 vs 0.07 for IsolationForest), confirming labels are essential — unsupervised anomaly detection alone is operationally useless on this dataset.<br><br>
**Surprise:** Adding a 4th rule to the rules engine *lowered* AUPRC from 0.13 (best single rule: amt > P99) to 0.07. More domain knowledge encoded as OR-style rules can be worse than less, when each rule has low individual precision.<br><br>
**Research:** Hassan & Wei (2025, arxiv:2506.02703) — random split inflates AUPRC via temporal leakage; Davis & Goadrich (2006) — AUPRC is the correct metric for imbalanced data.<br><br>
**Best Model So Far:** XGBoost, temporal split — AUPRC=0.8237

</td>
</tr>
</table>

---

### Phase 2: Imbalance Strategy Comparison — 2026-04-28

<table>
<tr>
<td valign="top" width="38%">

**Model Run 1 (Mark):** Fixed XGBoost (n_est=200, depth=6, lr=0.1) and swept 9 imbalance strategies on the temporal split: 7 scale_pos_weight values, SMOTE, ADASYN, random undersampling, SMOTE-Tomek, threshold tuning, and focal loss. spw=5 wins AUPRC=0.8526 — tied by focal loss (γ=2, α=0.25) at 4× the training time.

</td>
<td align="center" width="24%">

<img src="results/mark_phase2_leaderboard.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** The Phase 1 model (XGBoost spw=172) was not the optimal XGBoost configuration — just the textbook heuristic. Sweeping spw in {1, 5, 17.4} would have added +0.029 AUPRC for free. Phase 3 feature engineering and Phase 4 tuning must now beat AUPRC=0.8526, not 0.8237.<br><br>
**Surprise:** spw=87 falls into a local AUPRC minimum *below* spw=172 (0.7947 vs 0.8237). The spw → AUPRC relationship is non-monotonic with a local dip in the 50–200 range. Don't interpolate; sweep.<br><br>
**Research:** Hassan & Wei (2025) — SMOTE inflates under random split, collapses under temporal; Trisanto et al. — focal loss is sensitive to γ, sometimes underperforms weighted cross-entropy on tabular fraud.<br><br>
**Best Model So Far:** XGBoost spw=5, temporal split — AUPRC=0.8526

</td>
</tr>
</table>

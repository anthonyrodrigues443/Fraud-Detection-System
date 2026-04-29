# Fraud Detection System

**Domain:** Financial ML | Tabular Classification | Imbalanced Learning  
**Dataset:** Sparkov/Kartik2112 Credit Card Transactions — 1,048,575 transactions, 0.57% fraud (174:1 imbalance)  
**Primary Metric:** AUPRC (Area Under Precision-Recall Curve)  
**Sprint:** Project 4 of 10 | Apr 27 – May 3, 2026

---

## Current Status

**Phase 3 complete.** Best model: **CatBoost + 22 behavioral features (39 total), temporal split, AUPRC = 0.9824**

Feature engineering delivered the largest single-phase lift in the project: +0.1060 AUPRC over Phase 2's CatBoost baseline. Per-card velocity features (1h/6h/24h/7d count + amount windows) account for 46% of that lift. Mark's complementary statistical-FE pass surfaced a critical methodological warning: Bayesian target encoding — the canonical 2001 fraud-detection feature — costs −0.49 AUPRC under temporal split at every smoothing α tested.

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

2. **Behavioral feature engineering > model architecture** — Adding 22 behavioral features (velocity, amount z-score, temporal, geographic, category-merchant) to Phase 2's CatBoost lifts AUPRC from 0.8764 → 0.9824 (+0.1060). This is 3× larger than the model-family lift in Phase 2 (CatBoost vs XGBoost: +0.0342). Per-card velocity windows (1h/6h/24h/7d) alone account for 46% of the lift.

3. **The textbook `scale_pos_weight` heuristic is wrong by 35×** — The standard N_neg/N_pos formula gives spw=172. But the AUPRC-optimal value is **spw=5**, gaining +0.029 AUPRC. The relationship is non-monotonic: spw=87 falls into a local minimum *below* spw=172.

4. **Every target-aware feature technique fails on temporal split** — SMOTE, ADASYN, and Bayesian target encoding all finish in the bottom of their respective ablations. TE costs −0.4883 AUPRC at α=100 and never recovers above 0.84 across α∈{1,10,100,500,2000}. Target-aware techniques memorize training-period base rates that don't transfer; structural encodings (frequency counts, log transforms) survive.

5. **Prec@95Recall tripled from 0.31 → 0.93 with behavioral features** — At the operationally important 95% recall threshold, false-alert rate dropped from 69% → 7%. This is the metric production fraud systems are measured on, and it moved more in Phase 3 (feature engineering) than in any prior model-family or imbalance-strategy change.

---

## Models Compared

**52 experiments across 3 phases** (13 Phase 1, 14 Phase 2, 25 Phase 3):

| Phase | Models / Strategies |
|-------|---------------------|
| 1 | Majority class, LogReg (default), LogReg (balanced), XGBoost (random), XGBoost (temporal), 4-rule engine, 4 single-rule baselines, GaussianNB, k-NN(5), IsolationForest |
| 2 | XGBoost × 7 spw values (1, 5, 17.4, 87, 172, 350, 870), +SMOTE, +ADASYN, +Undersample, +SMOTE-Tomek, +threshold tuning (test-set), +OOF threshold, +Focal Loss |
| 3 | CatBoost/XGBoost/RF baseline-vs-+22-behavioral, 5-group ablation (velocity, amount-deviation, temporal, geographic, category-merchant), stacking (LogReg meta + simple/weighted avg), Mark's 5 statistical groups (Bayesian TE, per-merchant velocity, card×merchant repeat, frequency encoding, multiplicative interactions), TE α-sweep (1/10/100/500/2000), 53-feat clean stack, LogReg on 59 features |

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

---

### Phase 3: Feature Engineering — 2026-04-29

<table>
<tr>
<td valign="top" width="38%">

**Domain Features (Anthony):** Engineered 22 behavioral features in 5 groups (velocity, amount z-score, temporal, geographic, category-merchant) on top of Phase 2's 17. CatBoost AUPRC jumped 0.8764 → 0.9824 (+0.1060) and Prec@95Recall went 0.31 → 0.93. Group ablation: velocity alone accounts for 0.0485 of the +0.1060 lift (46%) — every other group's individual contribution is <0.5%.<br><br>
**Statistical Features (Mark):** Layered 5 automated FE families on Anthony's 39-feature pipeline — Bayesian target encoding (Micci-Barreca 2001), per-merchant velocity (BreachRadar), card×merchant repeat, frequency encoding, multiplicative interactions. TE catastrophically poisoned the model: AUPRC 0.9791 → 0.4908 at α=100, never recovering above 0.84 even at α=2000. Best clean addition: per-merchant velocity (+0.0024 AUPRC).

</td>
<td align="center" width="24%">

<img src="results/phase3_ablation.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Anthony's per-card velocity already captured nearly every signal worth capturing. Mark's clean additions (53-feat stack, no TE) reach 0.9811 — within 0.001 of the single best Mark group, confirming diminishing returns. Pair Mark's Phase 1 (random-split inflation), Phase 2 (SMOTE/ADASYN bottom-2), and Phase 3 (TE collapse) findings: every target-aware FE technique fails on temporal split.<br><br>
**Surprise:** Bayesian target encoding — invented in Micci-Barreca's 2001 paper *specifically for fraud detection* (ZIP/IP/SKU) — costs −0.49 AUPRC at every α tested. The cure (heavy smoothing) only "works" because it deletes the signal. Anthony's leak-free expanding `cat_fraud_rate` does the same job without the temporal-distribution-shift trap.<br><br>
**Research:** Albahnsen et al. (2016) — per-card transaction-aggregation windows; Deotte/NVIDIA (2019) IEEE-CIS Kaggle 1st place — group-aggregation features beat model architecture; Micci-Barreca (2001) — invented target encoding (now shown to fail under temporal split); Araujo et al. (CMU SDM 2017) BreachRadar — per-merchant rolling counts.<br><br>
**Best Model So Far:** CatBoost + 22 behavioral features (39 total), temporal split — AUPRC=0.9824, Prec@95Rec=0.9260

</td>
</tr>
</table>

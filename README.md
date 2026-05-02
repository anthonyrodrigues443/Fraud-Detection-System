# Fraud Detection System

**Domain:** Financial ML | Tabular Classification | Imbalanced Learning  
**Dataset:** Sparkov/Kartik2112 Credit Card Transactions — 1,048,575 transactions, 0.57% fraud (174:1 imbalance)  
**Primary Metric:** AUPRC (Area Under Precision-Recall Curve)  
**Sprint:** Project 4 of 10 | Apr 27 – May 3, 2026

---

## Current Status

**Phase 6 complete.** Best AUPRC unchanged: **CatBoost + 22 behavioral features (39 total) — AUPRC = 0.9824**. Best production cost candidate: **simple-average ensemble (CB+XGB+LGB) on 53-feat stack — AUPRC = 0.9817, min expected cost $1,844**.

Phase 6 went deep on model understanding: SHAP interaction values, fraud subtype profiling, LIME case studies, temporal-stability checks, counterfactual analysis, and FN/FP forensics. `amt_cat_zscore` is a hub feature appearing in ALL top 5 SHAP interactions — it doesn't work alone, it conditions on category identity and amount magnitude. Feature importance is rock-solid stable across 3 monthly windows (Spearman ρ > 0.986), supporting deployment without continuous retraining. The headline risk: 85.5% of caught fraud can be flipped by changing just 1 feature to the legitimate median — a single-point-of-failure vulnerability. Missed fraud is the inverse pattern: $49 median amount with NEGATIVE `amt_cat_zscore` (-0.07), blending into normal spending behavior.

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

3. **Every target-aware feature technique fails on temporal split** — SMOTE, ADASYN, and Bayesian target encoding all finish in the bottom of their respective ablations. TE costs −0.4883 AUPRC at α=100 and never recovers above 0.84 across α∈{1,10,100,500,2000}. Target-aware techniques memorize training-period base rates that don't transfer; structural encodings (frequency counts, log transforms) survive.

4. **Specialist beats frontier LLM on every axis** — On a stratified 50-sample test, CatBoost achieves F1=1.000 while Claude Opus 4.6 lands at F1=0.864 and Claude Haiku 4.5 at F1=0.485. CatBoost is ~242,000× faster (0.1ms vs 24.2s) and 45,000× cheaper ($0.0001 vs $4.50 per 1k predictions). Opus is conservative — zero false positives but misses 24% of small-amount, late-evening frauds that CatBoost catches via velocity and z-score signals.

5. **A simple uniform average beats a trainable meta-learner on a saturated model** — Uniform mean of CatBoost+XGBoost+LightGBM probabilities reaches AUPRC=0.9817 and min cost $1,844, beating every single booster *and* a LogReg-stacked meta (which overfit despite 125k calibration samples, degenerating to coefs CB=21.6/XGB=2.3/LGB=-2.6). Anthony's IsoForest-hybrid-weight=0 finding generalizes: any trainable combiner over a saturated CatBoost adds noise, not signal.

6. **85.5% of caught fraud can be hidden by changing just 1 feature** — Counterfactual analysis on 200 true-positive frauds: setting only the top-SHAP feature to the legitimate median flips 85.5% of predictions below 0.5; 12.5% need 2 features, 2% need 3. The model relies on a single dominant signal per transaction. Production should require multi-signal thresholds, not a single high-confidence score.

7. **Feature importance is rock-solid stable across 3 monthly windows** — Spearman rank correlation of per-window SHAP rankings: ρ=0.992 (W1↔W2), 0.987 (W1↔W3), 0.994 (W2↔W3). No concept drift in the test horizon, supporting deployment without continuous retraining. `amt_cat_zscore` is also a hub node in SHAP interactions — it appears in ALL top 5 interaction pairs (with `cat_fraud_rate`, `log_amt`, `category_encoded`, `amt`, `amt_ratio_to_mean`).

---

## Models Compared

**Experiments span Phases 1–6** (Phase 1 baselines, Phase 2 imbalance, Phase 3 feature engineering, Phase 5 advanced techniques + LLM head-to-head, Phase 6 deep explainability):

| Phase | Models / Strategies |
|-------|---------------------|
| 1 | Majority class, LogReg (default), LogReg (balanced), XGBoost (random), XGBoost (temporal), 4-rule engine, 4 single-rule baselines, GaussianNB, k-NN(5), IsolationForest |
| 2 | XGBoost × 7 spw values (1, 5, 17.4, 87, 172, 350, 870), +SMOTE, +ADASYN, +Undersample, +SMOTE-Tomek, +threshold tuning (test-set), +OOF threshold, +Focal Loss |
| 3 | CatBoost/XGBoost/RF baseline-vs-+22-behavioral, 5-group ablation (velocity, amount-deviation, temporal, geographic, category-merchant), stacking (LogReg meta + simple/weighted avg), Mark's 5 statistical groups (Bayesian TE, per-merchant velocity, card×merchant repeat, frequency encoding, multiplicative interactions), TE α-sweep (1/10/100/500/2000), 53-feat clean stack, LogReg on 59 features |
| 5 | TreeSHAP explainability, Isolation Forest standalone + CatBoost hybrid weight sweep, per-category threshold optimization, single-feature ablation (top 8), 7-group feature-family ablation (Velocity / Baseline / Temporal / Geographic / Category / Mark-stat / AmountDev), CB+XGB+LGB simple-average + LogReg-stacked meta, isotonic + Platt calibration, LLM head-to-head (Claude Haiku 4.5, Claude Opus 4.6 — GPT-5.4 usage-limited) on stratified 50-sample test |
| 6 | TreeSHAP interaction values (39×39 matrix on 500 stratified samples), fraud subtype profiling (high/low amount × night/day × new/repeat merchant), LIME local explanations on 3 case studies (borderline TP, near-miss FN, confident FP), temporal-stability Spearman correlation across 3 monthly windows, greedy counterfactual analysis on 200 TPs, FN/FP/TP/TN feature-median forensics |

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

---

### Phase 5: Advanced Techniques + Explainability — 2026-05-01

<table>
<tr>
<td valign="top" width="38%">

**SHAP + IsoForest (Anthony):** TreeSHAP on CatBoost names `amt_cat_zscore` the #1 feature (|SHAP|=2.86), with velocity at #2 (vel_amt_24h=2.79). Group-level SHAP: Baseline 33.0%, Velocity 30.7%, Amount-Dev 24.8%, Geographic only 0.3%. Isolation Forest standalone AUPRC=0.3429 (2.9× worse than CatBoost); CatBoost+IsoForest hybrid finds optimal weight at 0.0 — unsupervised anomaly detection adds zero signal on labeled data. Per-category optimal thresholds vary 0.11 (entertainment) to 0.66 (misc_net).<br><br>
**Stacking + LLM (Mark):** Group ablation confirms Velocity is load-bearing (-0.052 AUPRC, +$2,777 cost when dropped); Mark's 14 stat add-ons are redundant (+0.001 AUPRC, +$68 cost) — the 53-feat stack can be safely pruned to Anthony's 39-feat set. Simple uniform average of CB+XGB+LGB wins: AUPRC=0.9817, min cost $1,844 — beats every single learner *and* LogReg-stacked meta (which overfits with coefs CB=21.6 / XGB=2.3 / LGB=-2.6). Platt calibration lifts F1@0.5 by +3pp but raises cost by $80. LLM head-to-head on 50 stratified samples: CatBoost F1=1.000, Claude Opus 4.6 F1=0.864, Claude Haiku 4.5 F1=0.485 (codex/GPT-5.4 usage-limited).

</td>
<td align="center" width="24%">

<img src="results/mark_phase5_llm_vs_catboost.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Three independent angles converge on the same finding — the saturated CatBoost cannot be improved by *any* trainable combiner. Anthony's IsoForest hybrid weight collapsed to 0.0; Mark's LogReg meta over-weighted CatBoost (21.6) and inverted LightGBM (-2.6) without beating a uniform average. The frontier finding: an arithmetic mean of three decorrelated boosters is the only thing that beats single CatBoost, and CatBoost itself perfectly classifies the stratified 50-sample LLM benchmark while Opus misses 24% of frauds at 242,000× the latency.<br><br>
**Surprise:** Calibration *hurts* expected dollar loss. Both isotonic and Platt cut Brier by 35-37% and ECE by 86-89%, and lift F1@0.5 from 0.906 → 0.934 — yet min expected cost rises from $2,192 to $2,272-$2,278. Calibration compresses the high-end of the score distribution, collapsing the cost-optimal threshold from 0.081 to 0.011. Calibration is deployment ergonomics (interpretable posteriors at thr=0.5), not a cost lever.<br><br>
**Research:** Lundberg & Lee (2017, NeurIPS) — TreeSHAP exact Shapley values. Liu et al. (2008, ICDM) — Isolation Forest for unsupervised anomaly. Wolpert (1992) — original stacked generalization. Niculescu-Mizil & Caruana (2005, ICML) — isotonic dominates Platt for trees when N≥1k (we have 125k). Naeini et al. (2015, AAAI) — 20-bin ECE convention.<br><br>
**Best Model So Far:** AUPRC champion: CatBoost + 39 features — AUPRC=0.9824, Prec@95Rec=0.9260. Cost champion (Phase 6 production candidate): simple-average ensemble (CB+XGB+LGB) on 53-feat — AUPRC=0.9817, min expected cost $1,844 (-12.5% vs Phase 4 single CatBoost).

</td>
</tr>
</table>

---

### Phase 6: Deep Explainability & Model Understanding — 2026-05-02

<table>
<tr>
<td valign="top" width="38%">

**Explainability Run (Anthony):** Six diagnostics on the AUPRC champion (CatBoost + 39 features, AUPRC=0.9824) — TreeSHAP interaction values, fraud subtype profiling, LIME case studies, temporal stability, counterfactual analysis, and FN/FP forensics. SHAP interaction matrix (39×39 on 500 stratified samples): `amt_cat_zscore` is a hub node — it appears in ALL top 5 interaction pairs (strongest with `cat_fraud_rate` = 0.422). Counterfactual on 200 TPs: setting just 1 feature to the legitimate median flips 85.5% of fraud predictions below 0.5; 12.5% need 2 features, 2% need 3. Temporal stability is exceptional — Spearman ρ on per-window SHAP rankings is 0.992 / 0.987 / 0.994 across three monthly windows.

</td>
<td align="center" width="24%">

<img src="results/phase6_anthony_counterfactual.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** The same `amt_cat_zscore` finding from Phase 5 (top global SHAP) generalizes in two ways: it's the connective tissue of the model's reasoning (hub of all top interactions, dominates every fraud subtype with 1.7× higher reliance on high-amount fraud), AND it's the model's load-bearing weakness (85.5% of caught fraud is one-feature-flippable). The model has converged on a single dominant signal — strong when the signal fires, brittle when an adversary normalizes it.<br><br>
**Surprise:** Missed fraud (FN) has *negative* `amt_cat_zscore` (-0.07) and 1.9× HIGHER 24h velocity than caught fraud (1531 vs 812 transactions), but with $49 median amounts vs $733 for TPs. The blind spot isn't slow or low-volume — it's high-frequency, low-amount "blend-in" fraud that matches the stolen card's normal spending pattern. Also surprising: SHAP and LIME agree only 20-40% on individual cases despite agreeing globally.<br><br>
**Research:** Kong et al. (2024, CFTNet) — counterfactual XAI for fraud, "what minimum changes flip the prediction?". CEUR-WS Vol-4059 (2024) — temporal-stability metrics on SHAP attributions detect concept drift before AUPRC degrades. Lundberg et al. (2020) — TreeSHAP exact interaction values. Springer LNCS (2024) — SHAP vs LIME on tabular fraud: SHAP global, LIME local.<br><br>
**Best Model So Far:** Unchanged from Phase 5. AUPRC champion: CatBoost + 39 features — AUPRC=0.9824, Prec@95Rec=0.9260. Cost champion: simple-average ensemble (CB+XGB+LGB) on 53-feat — AUPRC=0.9817, min expected cost $1,844. Phase 6 added no model lift but produced two production-critical diagnostics: 1) the model is safe to deploy without continuous retraining (ρ>0.986 across 3 months), 2) the model needs a multi-signal threshold to defend against single-feature counterfactual evasion.

</td>
</tr>
</table>

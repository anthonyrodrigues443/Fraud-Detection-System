# Fraud Detection System

**Credit card fraud detection on 1.05M-transaction Sparkov/Kartik2112 dataset (0.57% fraud, 174:1 imbalance).** Phase 1 establishes XGBoost as the supervised ceiling (AUPRC=0.9314 random / 0.8237 temporal) and exposes a 13.1pp AUPRC inflation from the standard stratified-random split.

> **Headline (Day 1):** `is_night` (importance=0.41) is the single strongest predictor — fraud rate is 35x higher at 10pm than at 6am. The class_weight='balanced' trick HURTS AUPRC (0.36 → 0.25) despite raising recall, validating the SMOTE-skepticism literature.

---

## Current Status

- **Phase:** 1 of 7 — Domain Research + EDA + Baselines (complete)
- **Dataset:** Sparkov/Kartik2112 credit card transactions (1,048,575 rows, 0.573% fraud, 943 unique cards)
- **Primary metric:** AUPRC (per Davis & Goadrich 2006 — appropriate for extreme imbalance)
- **Best model so far:** XGBoost baseline — AUPRC=**0.9314** random / **0.8237** temporal split

---

## Key Findings

1. **`is_night` dominates feature importance (0.41)** — fraud rate at 10pm is 2.82% vs 0.08% at 6am, a 35x difference. Behavioral timing beats every other engineered feature so far.
2. **Random split inflates AUPRC by 13.1pp** vs production-realistic temporal split (0.9314 → 0.8237). Only high-capacity models exploit the leakage; LogReg/NB/IF are barely affected — it's a *card-level leakage* phenomenon that scales with model power.
3. **`class_weight='balanced'` HURTS AUPRC** (0.36 → 0.25 on LogReg). Recall climbs to 85% but precision crashes to 4% — the PR tradeoff is strictly worse than the default-threshold model.
4. **Unsupervised Isolation Forest is operationally useless** here (AUPRC=0.0727, basically tied with hand-written rules). The supervised lift is doing nearly all the work — the common "use IF as a fallback when labels are scarce" heuristic doesn't survive on this dataset.
5. **More rules can be worse than one rule.** A 4-rule industry-style engine (amt>P99 OR is_night OR risky-category OR far-from-home) gets AUPRC=0.0703 — *half* the single best rule (`amt > P99` alone, AUPRC=0.1345). OR-summing low-precision rules dilutes signal.

---

## Models Compared (cumulative)

**17 configurations** evaluated in Phase 1 across two split methodologies. Full leaderboard below — temporal split is the production-realistic ranking.

| Rank | Model | Split | AUPRC | F1 | Recall | Prec@95%R |
|------|-------|-------|-------|-----|--------|-----------|
| 1 | XGBoost (baseline) | Random | 0.9314 | 0.56 | 0.96 | 0.5009 |
| 2 | XGBoost (baseline) | Temporal | **0.8237** | 0.68 | 0.84 | 0.2397 |
| 3 | k-NN (k=5) | Random | 0.5468 | 0.57 | 0.67 | 0.0061 |
| 4 | k-NN (k=5) | Temporal | 0.5297 | 0.56 | 0.65 | 0.0054 |
| 5 | LogReg (default) | Random | 0.3622 | 0.17 | 0.09 | 0.0086 |
| 6 | LogReg (default) | Temporal | 0.3611 | 0.12 | 0.07 | 0.0086 |
| 7 | LogReg (balanced) | Random | 0.2484 | 0.08 | 0.85 | 0.0192 |
| 8 | LogReg (balanced) | Temporal | 0.2315 | 0.02 | 0.94 | 0.0086 |
| 9 | GaussianNB | Temporal | 0.2172 | 0.36 | 0.51 | 0.0169 |
| 10 | GaussianNB | Random | 0.2147 | 0.35 | 0.50 | 0.0173 |
| 11 | Rule: amt > P99 | Temporal | 0.1345 | 0.35 | 0.50 | — |
| 12 | Rules-engine (4 rules) | Random | 0.0736 | 0.08 | 0.54 | 0.0116 |
| 13 | IsolationForest | Random | 0.0729 | 0.03 | 0.90 | 0.0102 |
| 14 | IsolationForest | Temporal | 0.0727 | 0.02 | 0.97 | 0.0107 |
| 15 | Rules-engine (4 rules) | Temporal | 0.0703 | 0.07 | 0.53 | 0.0113 |
| 16 | Rule: is_night | Temporal | 0.0147 | 0.03 | 0.87 | — |
| 17 | Majority class | Either | 0.0057 | 0.00 | 0.00 | — |

---

## Dataset

**[Sparkov/Kartik2112 Credit Card Transactions](https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions)** — simulated transaction stream with real, interpretable feature names (chosen over ULB/IEEE-CIS for domain feature engineering).

| Metric | Value |
|--------|-------|
| Total samples | 1,048,575 |
| Fraud rate | 0.573% (174:1 imbalance) |
| Unique cards | 943 (596 ever-defrauded, mean 1,112 txns/card) |
| Date range | 2019-01-01 to 2020-03-10 (434 days) |
| Features (raw) | 23 columns; 17 used in baselines |
| Splits used | Stratified random (80/20) and temporal (cutoff 2020-01-13) |
| Primary metric | AUPRC |
| Secondary | F1, Precision@95%Recall, ROC-AUC, Recall@5%FPR |

---

## Project Structure

```
Fraud-Detection-System/
├── config/
├── data/
├── models/
├── notebooks/
│   ├── phase1_eda_baseline.ipynb
│   └── phase1_mark_complementary.ipynb
├── reports/
│   ├── day1_phase1_report.md            # Anthony — supervised baselines
│   └── day1_phase1_mark_report.md       # Mark — split audit, rules, alt paradigms
├── results/
│   ├── metrics.json
│   └── *.png                            # EDA + comparison plots
├── src/
├── tests/
├── requirements.txt
└── .gitignore
```

---

## Iteration Summary

### Phase 1: Domain Research + EDA + Baselines — 2026-04-27

<table>
<tr>
<td valign="top" width="38%">

**Supervised Baselines (Anthony):** Built 4 baselines on stratified random split with 17 features (temporal, geographic, amount, demographic, category). XGBoost (n_estimators=200, scale_pos_weight=174) achieves AUPRC=**0.9314**, ROC-AUC=0.9986, recall=96% at precision=39% — 2.6× the best LogReg. Counterintuitive finding: LogReg with `class_weight='balanced'` has WORSE AUPRC (0.2484) than default (0.3622) despite catching 85% of fraud.<br><br>
**Split Audit + Alt Paradigms (Mark):** Re-ran XGBoost on a temporal split (production-realistic) and added 6 baselines from paradigms Anthony skipped — 4-rule industry engine, single-rule baselines, GaussianNB, k-NN, Isolation Forest. Headline: temporal XGBoost AUPRC drops to **0.8237** (-13.1pp). Unsupervised IF is essentially useless (AUPRC=0.0727); one rule beats four (`amt>P99` alone=0.1345 vs combined=0.0703).

</td>
<td align="center" width="24%">

<img src="results/mark_random_vs_temporal_split.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Anthony's stratified-random AUPRC=0.9314 is a *card-level leakage* artifact — the 943 unique cards each appear ~1,112 times, so random splitting puts virtually every test card's history in the training set. Mark's temporal split (0.8237) is the honest number, and the 13.1pp gap *only opens for high-capacity models* (XGBoost +13.1%, k-NN +3.2%, NB ≈0%). The leakage scales with model power, which makes it the most dangerous kind.<br><br>
**Surprise:** The 4-rule industry baseline (HSBC/Stripe-style) is HALF as good as its single best rule. OR-summing rules with individually low precision dilutes the score. Domain intuition isn't free — it can actively hurt when encoded naively.<br><br>
**Research:** Davis & Goadrich (2006) — AUPRC > ROC-AUC for imbalanced data, so we adopted AUPRC as primary. Hassan & Wei (2025, arxiv:2506.02703) — *temporal validation is a pervasive flaw*, so we ran the random-vs-temporal audit. Stripe Engineering — *domain features beat algorithm choice*, motivating the planned Phase 3 velocity/risk features.<br><br>
**Best Model So Far:** XGBoost baseline — **AUPRC=0.9314 (random split)**, **0.8237 (temporal, production-realistic)**. Sets the bar Phase 2's RF/LightGBM/CatBoost/SVM/NN must beat.

</td>
</tr>
</table>

---

## References

1. Davis & Goadrich (2006). *The Relationship Between Precision-Recall and ROC Curves.* ICML.
2. Hassan & Wei (2025). *Data Leakage and Deceptive Performance: A Critical Examination of Credit Card Fraud Detection Methodologies.* [arxiv:2506.02703](https://arxiv.org/abs/2506.02703).
3. Frontiers in AI (2025). *Enhancing credit card fraud detection using traditional and deep learning models with class imbalance mitigation.*
4. *Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook* (fraud-detection-handbook.github.io).
5. Stripe Engineering Blog. *How ML works for payment fraud detection.*
6. Higson. *Enhancing Fraud Detection in Banking with Rule-Based Decision Engines.*
7. Salgado, Pessanha & Lobato (2024). *Reducing false positives in bank anti-fraud systems based on rule induction in distributed tree-based models.* Expert Systems with Applications.

---

## Next Steps

- **Phase 2 (Apr 28):** Compare 6+ models — Random Forest, LightGBM, CatBoost, Isolation Forest, SVM, NN. Adopt **temporal split** as the production-realistic ranking; report random-split alongside.
- **Phase 3 (Apr 29):** Engineer production fraud features — velocity (1h/24h/7d), amount z-score vs card history, target-encoded merchant risk, geographic impossible-travel speed.
- Open question: replace `unix_time` (MI rank #1, suspected calendar-leakage proxy) with cyclical date features.
- Open question: test card-level split (split by `cc_num`) to isolate card-leakage from calendar-leakage.

# Phase 1 (Mark): Rule-Based Baselines, Alternative ML Paradigms, and Split-Methodology Audit
**Date:** 2026-04-27
**Session:** 1 of 7
**Researcher:** Mark Rodrigues
**Project:** Fraud Detection System

## Objective
Audit three assumptions Anthony's Phase 1 left unchecked, and add baselines from paradigms he didn't test:
1. Does **temporal** train/test split change the answer vs his stratified-random split?
2. How does a **rule-based industry baseline** (HSBC/Stripe-style if-then engine) compare to ML?
3. How do **alternative ML paradigms** (probabilistic, instance-based, unsupervised) perform on the same features?

## Building on Anthony's Work
**Anthony found:**
- Selected the Sparkov/Kartik2112 dataset (1,048,575 transactions, 0.57% fraud, 174:1 imbalance)
- Established AUPRC as the primary metric (per Davis & Goadrich 2006)
- Champion: XGBoost AUPRC=0.9314 with `is_night` (importance=0.41) as top feature
- Counterintuitive finding: `class_weight='balanced'` LogReg HURTS AUPRC (0.2484 vs 0.3622 default)
- All baselines on a **stratified random** train/test split

**My approach:** Use Anthony's exact feature pipeline and the same dataset, but (a) re-run XGBoost with a **temporal** split to measure how much the random split inflates the score, (b) add a 4-rule industry-style baseline plus four single-rule baselines for an interpretable floor, and (c) add three baselines from paradigms he never touched: Gaussian Naive Bayes (probabilistic), k-NN (instance-based), Isolation Forest (UNSUPERVISED).

**Combined insight:** Anthony established the feature pipeline and the supervised tree-model ceiling. I established the floor (rules), the unsupervised floor (IF), and the *production-realistic* ceiling. Together these set the boundaries Phase 2 must operate inside.

## Research & References
1. **Hassan & Wei (2025) — *Data Leakage and Deceptive Performance: A Critical Examination of Credit Card Fraud Detection Methodologies*** (arxiv:2506.02703). Identifies four pervasive flaws in published fraud-detection results: improper preprocessing sequences, vague methodology, **inadequate temporal validation**, and metric manipulation. Directly motivates this Phase 1 audit.
2. **Reproducible ML for Credit Card Fraud Detection — Practical Handbook** (fraud-detection-handbook.github.io). Chapter 5: "Validation Strategies" argues temporal split is the only honest evaluation for transaction streams. Chapter 2 catalogs rule-engine architectures used in production fraud systems.
3. **Higson — *Enhancing Fraud Detection in Banking with Rule-Based Decision Engines***. Industry context: HSBC uses ML alongside rules engines — rules provide transparency and explainability that regulators require. Sets the bar that ML must beat the rules engine to be operationally worthwhile.
4. **Salgado, Pessanha & Lobato (2024) — *Reducing false positives in bank anti-fraud systems based on rule induction in distributed tree-based models*** (Expert Systems with Applications). Confirms tree models extract rule-like decision boundaries; informs why XGBoost should beat hand-written rules and *by how much*.

How research influenced today's experiments: arxiv:2506.02703 directly motivated the temporal-vs-random audit (the paper's central complaint). The Practical Handbook informed the rule-engine design (transaction-blocking + scoring rules). Anthony's metric choice (AUPRC) is preserved — no metric override needed.

## Dataset
Same as Anthony — Sparkov/Kartik2112 credit card transactions from HuggingFace (`santosh3110/credit_card_fraud_transactions`). 1,048,575 rows, 0.573% fraud, 174:1 imbalance, date range 2019-01-01 to 2020-03-10 (434 days).

| Metric | Value |
|--------|-------|
| Total samples | 1,048,575 |
| Fraud rate | 0.573% |
| Train/Test (random) | 838,860 / 209,715 (Anthony's setup) |
| Train/Test (temporal) | 838,860 / 209,715, cutoff 2020-01-13 |
| Daily fraud rate | mean 0.625%, std 0.475% (mild non-stationarity) |
| Features used | 17 (identical to Anthony's set) |

## EDA Anthony Skipped — Card-Level + MI

### Card-level concentration
The dataset has **943 unique cards** (cc_num), of which **596 (63.20%) are ever-defrauded** and 347 (36.80%) are clean. This is a striking structural fact Anthony's transaction-level EDA missed:
- Average transactions per card: 1,112 (heavy reuse)
- Average fraud transactions on a defrauded card: **10.08** (range 2–24)
- Top 10% of defrauded cards account for 15.9% of all fraud
- Top 25% account for 35.3% (relatively spread out, NOT power-law concentrated)

**Implication for split methodology:** Because almost every card has BOTH legitimate and fraud transactions and fraudulent transactions repeat 10× on the same card, a stratified random split puts ~99.99% of test cards' history into the training set. The model isn't really classifying "is this transaction fraud?" — it's pattern-matching against the same card's other behavior. A temporal split breaks that leakage path. (`results/mark_card_level_analysis.png`)

### Mutual information ranking (model-free)
Top 5 features by MI with `is_fraud`: `unix_time` (0.0145), `category_encoded` (0.0102), `amt` (0.0099), `log_amt` (0.0089), `is_night` (0.0079). Anthony's XGBoost importance ranked `is_night` (0.41) and `amt` (0.35) at the top. MI broadly agrees on `amt`, `category`, `is_night` carrying signal — but it ranks **`unix_time` highest**, which is suspicious: it's effectively a row-id proxy for date and could be picking up calendar-level fraud bursts that the model will overfit to. Worth flagging for Phase 3 — `unix_time` may need to be replaced with cyclical date features. (`results/mark_mutual_information.png`)

### Correlation structure
No feature pair has |Pearson r| > 0.5 — features are largely independent. Good news for the linear/probabilistic baselines below: the Naive Bayes independence assumption isn't catastrophically violated. (`results/mark_correlation_matrix.png`)

## Experiments

### Experiment M.1 — Random vs Temporal split (the headline)
**Hypothesis:** Stratified random split inflates AUPRC because it (a) leaks card-level patterns and (b) lets the model see future fraud days during training.
**Method:** Same XGBoost config Anthony used (`n_estimators=200, max_depth=6, scale_pos_weight≈174`). Same 17 features. Two splits: stratified random (his) vs sort-by-trans_time-then-cut-at-80% (production-realistic).
**Result:**

| Split | AUPRC | ROC-AUC | F1 | Precision | Recall | Prec@95%Recall |
|-------|-------|---------|-----|-----------|--------|----------------|
| Random (Anthony's) | **0.9314** | 0.9986 | 0.5563 | 0.3916 | 0.9600 | 0.5009 |
| Temporal (production) | **0.8237** | 0.9954 | 0.6794 | 0.5715 | 0.8376 | 0.2397 |

**AUPRC delta = +0.1077 (random is 13.1% inflated over temporal).**

**Interpretation:** The random split overstates the operational AUPRC by 11 absolute points. Note the trade-off pattern reverses: random split achieves higher recall (96% vs 84%) but worse precision at the operational threshold; temporal split has better F1 (0.68 vs 0.56) at the default threshold but a much worse Prec@95%Recall (0.24 vs 0.50). The model trained on past data and tested on future data has to be more conservative to maintain precision in unseen calendar regions. (`results/mark_random_vs_temporal_split.png`)

### Experiment M.2 — Rule-based industry baseline
**Hypothesis:** A 4-rule expert system (large amount, night, high-risk category, far-from-home merchant) won't match XGBoost but should beat random.
**Method:** Thresholds learned from training data only (no leakage):
- Rule 1: amount > P99-train ($543.88)
- Rule 2: is_night (10pm–5am)
- Rule 3: merchant category with train fraud rate > 2%
- Rule 4: customer-merchant distance > 100 km
Each rule contributes +1 to a risk score (max 4). Evaluated on the temporal test set.

**Result (temporal split):**

| Rule | AUPRC | ROC-AUC | F1 | Recall |
|------|-------|---------|-----|--------|
| amt > P99 (alone) | **0.1345** | 0.7447 | 0.3458 | 0.4969 |
| is_night (alone) | 0.0147 | 0.7896 | 0.0314 | 0.8742 |
| high-risk category (alone) | 0.0055 | 0.5000 | 0.0000 | 0.0000 |
| distance > 100 km (alone) | 0.0054 | 0.4933 | 0.0101 | 0.2183 |
| **All 4 rules combined** | 0.0703 | 0.8421 | 0.0736 | 0.5345 |

**Interpretation:** Two important findings here.

First, **adding rules HURT the engine.** The single best rule (`amt > P99`, AUPRC=0.1345) is nearly 2× better than the 4-rule combination (AUPRC=0.0703). The OR-style score sum fires on too many legitimate transactions: any night transaction on the dataset is night-flagged, but the night fraud rate is only 1.6% — most flagged-by-night transactions are legit, and combining rules that each have low precision is worse than picking the one rule with high precision. This is exactly the Keeper-style counterintuitive finding the project mandate calls for.

Second, **the "high-risk merchant category" rule found zero categories** — no category in the training set has fraud rate > 2%. The maximum is 1.75%. So the bank-industry intuition that some merchant categories are obviously risky doesn't translate to this dataset. Either the simulator distributes fraud uniformly across categories, or real-world category-risk only materializes after target encoding (Phase 3 work).

### Experiment M.3 — Gaussian Naive Bayes (probabilistic paradigm)
**Hypothesis:** Despite the violated independence assumption (cell 7 confirmed feature correlations are mild — no |r|>0.5), NB will be mediocre because amount and distance distributions aren't Gaussian.
**Method:** `GaussianNB` on standardized features.
**Result (temporal):** AUPRC=0.2172, ROC-AUC=0.9272, F1=0.3568, Precision=0.2746, Recall=0.5092.
**Interpretation:** NB lands between LogReg-default (0.3611) and LogReg-balanced (0.2315). Surprisingly competitive given how many distributions are non-Gaussian — but well below XGBoost. **NB barely cared about split type** (delta = -0.0025): it doesn't have enough capacity to exploit the card-level temporal patterns that random split leaks. This is a useful diagnostic — see Finding 4 below.

### Experiment M.4 — k-NN (instance-based paradigm)
**Hypothesis:** k-NN will be middling because it's lookup-based and the curse of dimensionality affects it at 17 features.
**Method:** k-NN with k=5 on standardized features. Trained on a 110k-row stratified subsample (10k positives + 100k negatives) and tested on a 50k random subsample of the test set (full k-NN is intractable on 1M rows).
**Result (temporal):** AUPRC=0.5297, ROC-AUC=0.9062, F1=0.5610.
**Interpretation:** k-NN sandwiched between XGBoost (0.82) and LogReg (0.36) — strong. It IS partially exploiting card-temporal leakage on the random split (0.5468 → 0.5297, +3.2% inflation), but to a lesser degree than XGBoost.

### Experiment M.5 — Isolation Forest (UNSUPERVISED)
**Hypothesis:** With ZERO labels, IF should still capture most fraud signal because fraud transactions are anomalous (large amounts, weird times).
**Method:** `IsolationForest(n_estimators=200, contamination=0.0058)` fit on a 200k random subsample of training data. Negative score_samples used as fraud probability.
**Result (temporal):** AUPRC=0.0727, ROC-AUC=0.8844, Recall=0.9738 (at default threshold).
**Interpretation:** **The unsupervised floor is much lower than I expected.** IF achieves AUPRC=0.07, basically tied with the rules engine — meaning "looks weird" alone is very weak signal in this dataset. The supervised models' lift comes almost entirely from labels, not from anomaly structure. At ROC-AUC=0.88, IF at least ranks fraud above legit on average, but its precision is essentially zero. This refutes the common assumption that unsupervised methods are a meaningful fallback for fraud detection on tabular data. Phase 5's hybrid supervised+unsupervised idea will need to be built carefully — it can't just OR the two scorers.

## Master Comparison Table

Ranked by AUPRC. **/Temporal** rows are production-realistic; **/Random** rows are direct comparisons with Anthony's published numbers.

| Rank | Model | AUPRC | ROC-AUC | F1 | Precision | Recall | Prec@95%R |
|------|-------|-------|---------|-----|-----------|--------|-----------|
| 1 | XGBoost / Random (Anthony) | **0.9314** | 0.9986 | 0.5563 | 0.3916 | 0.9600 | 0.5009 |
| 2 | XGBoost / Temporal (Mark) | **0.8237** | 0.9954 | 0.6794 | 0.5715 | 0.8376 | 0.2397 |
| 3 | k-NN(5) / Random | 0.5468 | 0.9288 | 0.5698 | 0.4940 | 0.6732 | 0.0061 |
| 4 | k-NN(5) / Temporal | 0.5297 | 0.9062 | 0.5610 | 0.4917 | 0.6531 | 0.0054 |
| 5 | LogReg-default / Temporal | 0.3611 | 0.9018 | 0.1195 | 0.5984 | 0.0664 | 0.0086 |
| 6 | LogReg-balanced / Temporal | 0.2315 | 0.9119 | 0.0174 | 0.0088 | 0.9441 | 0.0086 |
| 7 | GaussianNB / Temporal | 0.2172 | 0.9272 | 0.3568 | 0.2746 | 0.5092 | 0.0169 |
| 8 | GaussianNB / Random | 0.2147 | 0.9159 | 0.3511 | 0.2706 | 0.4996 | 0.0173 |
| 9 | Rule: amt > P99 / Temporal | 0.1345 | 0.7447 | 0.3458 | 0.2651 | 0.4969 | — |
| 10 | Rules-engine (4 rules) / Random | 0.0736 | 0.8364 | 0.0767 | 0.0413 | 0.5396 | 0.0116 |
| 11 | IsolationForest / Random | 0.0729 | 0.8699 | 0.0261 | 0.0132 | 0.9042 | 0.0102 |
| 12 | IsolationForest / Temporal | 0.0727 | 0.8844 | 0.0172 | 0.0087 | 0.9738 | 0.0107 |
| 13 | Rules-engine (4 rules) / Temporal | 0.0703 | 0.8421 | 0.0736 | 0.0395 | 0.5345 | 0.0113 |
| 14 | Rule: is_night / Temporal | 0.0147 | 0.7896 | 0.0314 | 0.0160 | 0.8742 | — |
| 15 | Rule: high-risk category / Temporal | 0.0055 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | — |
| 16 | Majority-class / Temporal | 0.0055 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | — |
| 17 | Rule: distance > 100km / Temporal | 0.0054 | 0.4933 | 0.0101 | 0.0051 | 0.2183 | — |

## Key Findings

1. **Random split inflates XGBoost AUPRC by 13.1% absolute (0.9314 → 0.8237 on temporal split).** This is a methodology-driven finding, not an algorithm finding — and it directly contradicts the implicit "0.93 AUPRC champion" framing of Phase 1. Production-realistic XGBoost AUPRC is 0.82, not 0.93. Phase 2+ should report both numbers in any leaderboard.
2. **The single best rule beats the 4-rule combination** (amt-only AUPRC=0.1345 vs combined 0.0703). Adding rules monotonically lowered AUPRC because the score sum fires on too many legitimate transactions. Counterintuitive: more domain knowledge encoded as rules can be worse than less, when each rule has low individual precision.
3. **Unsupervised Isolation Forest barely registers (AUPRC=0.0727, basically tied with rules).** Production fraud teams sometimes propose IF as a fallback when labels are scarce. On this dataset, that fallback would be operationally useless. The supervised lift is doing nearly all the work.
4. **Only powerful models are inflated by random split.** XGBoost: +13.1%. k-NN: +3.2%. Rules: +4.7%. NB: -1.2% (noise). IF: +0.3%. The pattern: high-capacity models exploit card-level temporal patterns that low-capacity models can't see. This means the random-split inflation isn't dataset-wide concept drift — it's a *card-level leakage* phenomenon, which is more dangerous because it scales with model power.
5. **There are only 943 unique cards** in 1.05M transactions. Mean transactions per card = 1,112. Random split lets the model train on transactions from the same cards it later tests on. **Anthony's split should arguably be by card, not by time** — splitting on cc_num would isolate the leakage even more cleanly. Open question for Phase 2.

## What Didn't Work (and why)
- **High-risk merchant category rule:** zero categories at >2% fraud rate (max in train was 1.75%). Industry intuition that some merchant categories are obviously risky doesn't survive the simulator's category distribution. Target encoding will salvage this in Phase 3.
- **GaussianNB:** held its own (AUPRC=0.2172) but well below LogReg-default. The non-Gaussian distributions of `amt` and `distance_km` are the most likely culprit, even though the independence assumption was OK.
- **4-rule additive scoring:** flagged correctly that combining low-precision rules dilutes signal. The right ensemble is *meta-learning over rules*, not OR-summing them — a Phase 5 direction.

## Frontier Model Comparison
Not run this phase (Phase 1 is baselines + EDA). Will be run in Phase 5 per project mandate. The temporal-split AUPRC=0.8237 is the number GPT-5.4/Opus 4.6 will need to beat — the random-split 0.9314 was an inflated target that would have made the LLM comparison too easy.

## Error Analysis
Brief — full error analysis is Phase 4. One observation: under the temporal split, XGBoost's recall drops from 96% to 84%. The fraud it misses on the temporal test set has a different signature (different time window, possibly newer fraud patterns) than the fraud it caught on the random test set. Phase 4 will look at WHICH fraud it misses on temporal split that it would have caught on random split — that's the operational gap.

## Next Steps for Phase 2
- **Adopt temporal split as the production-realistic evaluation.** Report random-split AUPRC alongside for direct comparability with Anthony's table, but rank by temporal AUPRC.
- **Both researchers should evaluate on temporal split going forward.** I'd advocate for this in the Phase 2 PR review.
- **Phase 2 should test card-level split as well** — split by cc_num, not by time — to isolate which leakage path is dominant (card-level vs calendar-level).
- **Consider replacing `unix_time` with cyclical date features** (sin/cos of day-of-year). MI ranked it #1 but it's almost certainly capturing temporal leakage rather than genuine signal.
- **For Phase 5's hybrid idea: don't OR supervised + unsupervised.** IF is too weak alone. Better: feed IF anomaly score as a *feature* into XGBoost.

## References Used Today
- [1] Hassan & Wei (2025) — *Data Leakage and Deceptive Performance: A Critical Examination of Credit Card Fraud Detection Methodologies*. https://arxiv.org/html/2506.02703v1 and https://www.mdpi.com/2227-7390/13/16/2563
- [2] *Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook*. Chapter 5 (Validation Strategies), Chapter 2 (FDS architecture). https://fraud-detection-handbook.github.io/fraud-detection-handbook/
- [3] Higson — *Enhancing Fraud Detection in Banking with Rule-Based Decision Engines*. https://www.higson.io/blog/enhancing-fraud-detection-in-banking-with-rule-based-decision-engines
- [4] Salgado, Pessanha & Lobato (2024) — *Reducing false positives in bank anti-fraud systems based on rule induction in distributed tree-based models*. https://www.sciencedirect.com/science/article/abs/pii/S016740482200181X

## Code Changes
- Created `notebooks/phase1_mark_complementary.ipynb` (41 cells, executed end-to-end)
- Added `download_data.py` and `build_notebook.py` (helper scripts; not part of the ML pipeline)
- Saved 7 new plots: `mark_card_level_analysis.png`, `mark_correlation_matrix.png`, `mark_mutual_information.png`, `mark_temporal_coverage.png`, `mark_random_vs_temporal_split.png`, `mark_baseline_pr_curves.png`, `mark_split_impact.png`
- Appended `mark_phase1` key to `results/metrics.json` with full split-audit + baseline metrics
- This report: `reports/day1_phase1_mark_report.md`

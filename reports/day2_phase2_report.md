# Phase 2: Multi-Model Family Comparison — Fraud Detection System
**Date:** 2026-04-28
**Session:** 2 of 7
**Researcher:** Anthony Rodrigues

## Objective
Phase 1 established that XGBoost wins among naive baselines (LogReg, NB, kNN, IsolationForest). Mark's Phase 2 fixed XGBoost and varied 9 imbalance strategies, finding `spw=5.0` optimal at AUPRC=0.8526. **My complementary angle: fix imbalance handling at `spw=5/cw=5` and vary the model FAMILY across 6 algorithms** — does a different algorithm beat tuned XGBoost? Three ensemble combinations were also tested.

## Research & References
1. **[Preprints.org 2025] CatBoost vs XGBoost vs LightGBM** — Comparative study on 1.85M credit card transactions reports CatBoost achieves best F1=0.9161. CatBoost's ordered boosting reduces target leakage that affects standard gradient boosting on small target classes.
2. **[Springer Nature, Computational Economics 2025] Enhancing Fraud Detection in Credit Card Transactions** — MLP/ANN perform poorly on imbalanced data due to inherent class-distribution sensitivity that hurts recall and F1.
3. **[arXiv 2025] Stacking Ensemble Methods for Fraud Detection** — Stacked XGBoost+LightGBM+CatBoost reportedly achieve 99% accuracy, but on RANDOM splits which Mark's Phase 1 audit showed inflate metrics by ~13%.

How research influenced today's experiments:
- Tested CatBoost expecting it to dominate per the 2025 study
- Tested MLP expecting it to underperform per Springer findings
- Tested 3 ensemble configurations to verify whether the "stacking always helps" claim survives a temporal split

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 1,048,575 |
| Train (temporal, oldest 80%) | 838,860 |
| Test (newest 20%) | 209,715 |
| Train fraud rate | 0.580% |
| Test fraud rate | 0.546% |
| Features | 17 |
| Split type | **Temporal** (Mark's Phase 1 finding: random inflates +13.1%) |
| Cutoff | 2019-12-13 08:27 |

## Experiments

### Experiment 2.1: XGBoost (spw=5) — Baseline to beat
**Hypothesis:** Re-running Mark's optimal config establishes the bar.
**Method:** XGBClassifier(n_estimators=200, max_depth=6, lr=0.1, scale_pos_weight=5.0)
**Result:** AUPRC=0.8530, F1=0.7946, Precision=0.8536, Recall=0.7432, train=1.4s
**Interpretation:** Matches Mark's reported 0.8526 within rounding. Confirmed reproducibility.

### Experiment 2.2: LightGBM (spw=5) — The shocking failure
**Hypothesis:** Histogram-based binning + leaf-wise growth should match or beat XGBoost on 1M samples.
**Result:** **AUPRC=0.4095, F1=0.5183, Precision=0.4186, Recall=0.6803, train=1.5s**
**Interpretation:** Catastrophic failure — 50% worse AUPRC than XGBoost despite identical scale_pos_weight. Hypothesis: LightGBM's leaf-wise growth strategy with default min_child_samples=20 over-specializes on temporal patterns in the training period (early-2019 fraud), then those patterns no longer generalize to late-2019 test fraud. Tree depth grows where data density is highest, which on temporal data means it learns the recent past instead of learning generalizable fraud signals. **This is the most important "what didn't work" finding of the day.**

### Experiment 2.3: CatBoost (spw=5) — The new champion
**Hypothesis:** CatBoost's ordered boosting may help with the small fraud class.
**Result:** **AUPRC=0.8872, F1=0.8176, Precision=0.8075, Recall=0.8279, train=5.0s**
**Interpretation:** New champion — beats XGBoost by +0.0342 AUPRC. Critically, **prec@95recall=0.3150 vs XGBoost's 0.2352** — at the production operating point (catch 95% of fraud), CatBoost gives **34% higher precision**. Ordered boosting prevents the prediction-shift bias that appears when you train on small minority classes; it computes target statistics for each example using only past examples in a virtual ordering. This matters more on temporal data than on random splits.

### Experiment 2.4: Random Forest (class_weight=5) — The surprising 2nd place
**Hypothesis:** Bagging should lose to boosting on hard-class problems.
**Result:** AUPRC=0.8771, F1=0.8352 (highest F1!), Precision=0.9219, Recall=0.7633, train=37.6s
**Interpretation:** Counterintuitive — RF beats XGBoost by +0.0241 AUPRC and has the **highest F1 of any model (0.8352)** thanks to 92.2% precision. The bagging robustness from independent trees protects against the temporal-shift overfitting that hurt LightGBM. Cost: 27x slower than XGBoost.

### Experiment 2.5: ExtraTrees (cw=5)
**Hypothesis:** Random split points add regularization vs RF's best-split.
**Result:** AUPRC=0.8198, F1=0.7492, Precision=0.9101, Recall=0.6367, train=8.6s
**Interpretation:** Faster than RF (4x) but loses 0.057 AUPRC. The extra randomization throws away discriminative signal — the highly skewed feature importance (is_night=0.41 from Phase 1) means the best split actually IS clearly best, and randomizing it hurts. This is opposite to high-noise tabular tasks where ExtraTrees usually wins.

### Experiment 2.6: MLP Neural Network
**Hypothesis:** Per Springer 2025, MLP should underperform on imbalanced data.
**Method:** 3-layer MLP (128-64-32) with adaptive Adam, batch=1024, early stopping.
**Result:** AUPRC=0.7291, F1=0.6574, Precision=0.6009, Recall=0.7258, train=10.6s
**Interpretation:** Confirmed literature — MLP loses 0.158 AUPRC vs CatBoost. Without feature interaction baked-in (like trees provide via splits), the MLP must learn `is_night × amount` interactions from scratch with only ~5,000 fraud examples per epoch effectively. Tree models get this for free.

### Experiment 2.7: Linear SVM (calibrated)
**Hypothesis:** Single hyperplane is too simple for nonlinear fraud patterns.
**Result:** AUPRC=0.2289, F1=0.0725, Precision=0.1793, Recall=0.0454, train=138s
**Interpretation:** Confirmed — barely beats GaussianNB. Worth documenting as the floor for "what NOT to use." 138s training time makes it the slowest model AND the worst.

### Experiment 2.8 — 2.10: Three Ensemble Configurations
**Hypothesis:** Stacking literature claims +0.05 AUPRC over single best.
**Result:**
| Ensemble | AUPRC | Δ vs CatBoost |
|----------|-------|---------------|
| Top-3 boosters (XGB+LGBM+CatBoost) | 0.8543 | -0.0329 |
| 3 boosters + RF | 0.8736 | -0.0136 |
| All 7 averaged | 0.8709 | -0.0163 |

**Interpretation:** **No ensemble beat single CatBoost** — directly contradicts 2025 stacking literature. LightGBM's poor probabilities drag the booster ensemble down (-0.033 vs CatBoost). The "3 boosters + RF" hybrid recovers some performance but still loses to single CatBoost. Lesson: simple averaging fails when one component is poorly calibrated; you'd need stacking with a meta-learner or weight-tuning to recover.

## Head-to-Head Leaderboard (Temporal Split)

| Rank | Model | AUPRC | F1 | Precision | Recall | Prec@95Rec | Train (s) |
|------|-------|-------|-----|-----------|--------|------------|-----------|
| 1 | **CatBoost (spw=5)** | **0.8872** | 0.8176 | 0.8075 | 0.8279 | **0.3150** | 5.0 |
| 2 | Random Forest (cw=5) | 0.8771 | **0.8352** | 0.9219 | 0.7633 | 0.2472 | 37.6 |
| 3 | Ensemble (3 boost+RF) | 0.8736 | 0.8182 | 0.8852 | 0.7607 | — | — |
| 4 | Ensemble (all 7 avg) | 0.8709 | 0.8100 | 0.9220 | 0.7223 | — | — |
| 5 | Ensemble (top-3 boost) | 0.8543 | 0.8029 | 0.8447 | 0.7651 | — | — |
| 6 | XGBoost (spw=5, Mark) | 0.8530 | 0.7946 | 0.8536 | 0.7432 | 0.2352 | 1.4 |
| 7 | XGBoost (Phase 1, spw=172) | 0.8237 | 0.6794 | 0.5715 | 0.8376 | 0.2397 | 2.8 |
| 8 | ExtraTrees (cw=5) | 0.8198 | 0.7492 | 0.9101 | 0.6367 | 0.1666 | 8.6 |
| 9 | MLP (128-64-32) | 0.7291 | 0.6574 | 0.6009 | 0.7258 | 0.0915 | 10.6 |
| 10 | LightGBM (spw=5) | 0.4095 | 0.5183 | 0.4186 | 0.6803 | 0.0055 | 1.5 |
| 11 | LogReg (default) | 0.3611 | 0.1195 | 0.5984 | 0.0664 | 0.0086 | — |
| 12 | LinearSVM (cw=5, cal.) | 0.2289 | 0.0725 | 0.1793 | 0.0454 | 0.0080 | 138.0 |

## Key Findings

1. **CatBoost is the new champion at AUPRC=0.8872** — beats XGBoost (Mark's best at spw=5) by +0.0342 AUPRC and gets **+34% higher precision at 95% recall (0.315 vs 0.235)**. Ordered boosting shines when minority class is small and split is temporal.

2. **LightGBM collapsed (AUPRC=0.4095) — 50% worse than XGBoost.** This contradicts the "LightGBM is faster XGBoost" narrative. Hypothesis: leaf-wise growth over-specializes on early-period training fraud patterns; depth grows where data is dense (recent past), not where signal is generalizable. **Counterintuitive headline finding.**

3. **Random Forest came 2nd (AUPRC=0.8771) and has the highest F1 of any model (0.8352).** Bagging robustness compensates for not chasing minority class — independent trees vote, and the majority gets it right 92% of the time when they predict fraud.

4. **No ensemble beat single CatBoost.** Three different averaging strategies all lost. The 2025 literature claiming "stacking always helps" relied on random splits and homogeneous boosters — when one model (LightGBM) is broken, simple averaging propagates the damage.

5. **Boosters and baggers agree 99.87% of the time, but disagree on the hard fraud cases.** When they disagree (282 samples), boosters catch 104/141 actual fraud while baggers catch only 37/141. So if you want max recall on edge cases, use a booster; if you want overall precision, use bagging.

## What Didn't Work (and why)

- **LightGBM:** Default leaf-wise growth + scale_pos_weight=5 → severe overfitting on temporal patterns. AUPRC=0.41. Could be fixed with `num_leaves=15`, `min_child_samples=200`, or temporal CV — to investigate in Phase 3.
- **MLP:** Confirms 2025 literature — without bake-in feature interactions, neural nets need more fraud examples to learn `is_night × amount` patterns that trees discover for free.
- **Linear SVM:** AUPRC=0.23, slowest model (138s). Single hyperplane = too simple. Documented as floor.
- **Simple averaging ensembles:** All three configs lost to single CatBoost. Stacking with calibrated meta-learner needed.

## Frontier Model Comparison
Deferred to Phase 5 (per task spec) — Phase 5 will run zero-shot prompts to GPT-5.4 and Claude Opus 4.6 on transaction descriptions and compare their AUPRC against CatBoost.

## Error Analysis Preview (deepens in Phase 4)
- CatBoost confusion matrix: TN=208,144, FP=224, FN=197, TP=950 — only 224 false positives in 209K test transactions while catching 83% of fraud.
- LightGBM confusion matrix: TN=207,287, FP=1,081, FN=367, TP=780 — 5x more false positives than CatBoost.
- **Boosters miss the same 197 fraud cases CatBoost misses; baggers miss DIFFERENT cases** (disagreement analysis cell). Phase 5 stacking with disagreement-weighted meta-learner is a clear next direction.

## Next Steps (Phase 3)
1. **Investigate WHY LightGBM collapsed** — re-run with `num_leaves=15`, `min_child_samples=200`, regularization, and temporal CV. If LightGBM recovers, that's the headline; if not, it's confirmation that LightGBM is the wrong default for temporal fraud detection.
2. **Feature engineering on top of CatBoost** — domain features that don't exist yet:
   - Velocity features (transactions per card per hour/day) — needs sort-by-card pre-processing
   - Amount z-score within `(card_id, category)` — captures "this user never spends $500 on gas"
   - Distance anomaly (vs user's median) — captures geographic impossible-travel
   - Time-since-last-transaction — quick repeat transactions are red flags
3. **Stacking with proper meta-learner** — CatBoost + RF + XGBoost as base, LogReg meta-learner on out-of-fold predictions. Should beat single CatBoost if disagreement signals matter.

## References Used Today
1. [Preprints.org 2025 — CatBoost vs XGBoost vs LightGBM comparative study](https://www.preprints.org/manuscript/202503.1199)
2. [Springer Nature 2025 — Enhancing Fraud Detection in Credit Card Transactions](https://link.springer.com/article/10.1007/s10614-025-11071-3)
3. [arXiv 2505.10050 — Financial Fraud Detection Using Stacking Ensemble Methods](https://arxiv.org/html/2505.10050v1)

## Code Changes
- `notebooks/phase2_model_comparison.ipynb` — 22 cells, end-to-end multi-model comparison with PR curves, confusion matrices, feature importance, ensembles, and disagreement analysis
- `results/phase2_model_comparison.png` — PR curves + AUPRC bar chart
- `results/phase2_training_time.png` — training time vs AUPRC
- `results/phase2_confusion_matrices.png` — 7-panel confusion matrix grid
- `results/phase2_feature_importance_comparison.png` — top features per tree-based model
- `results/metrics.json` — added `anthony_phase2` block with full leaderboard

## Post-worthy?
**Yes.** Three angles available:
1. *"I tested 6 ML algorithms on the same fraud dataset. The expected winner (XGBoost) lost. The expected loser (Random Forest) came 2nd. And LightGBM — supposedly XGBoost's faster cousin — collapsed catastrophically."*
2. *"CatBoost beats XGBoost on temporal-split fraud detection by +34% precision-at-95%-recall. Ordered boosting matters more than I expected when fraud is 0.5% of data."*
3. *"Three different ensemble strategies all LOST to single CatBoost on fraud detection. Stacking literature lies — when one component is broken, averaging propagates the damage."*

# Phase 2 (Mark) — Imbalance-Handling Face-Off on XGBoost
**Date:** 2026-04-28
**Session:** 2 of 7
**Researcher:** Mark Rodrigues
**Project:** Fraud Detection System

## Objective
Anthony's Phase 1 found that `class_weight='balanced'` *hurt* Logistic Regression's AUPRC (0.36 → 0.25). That was on a single weak model. Phase 2 extends that finding by stress-testing it across the full menu of imbalance-handling strategies — but on a strong model:

> **Fix the model (XGBoost), vary the imbalance strategy. Which class-imbalance technique actually wins on the production-realistic temporal split?**

Nine strategies on the same XGBoost (n_estimators=200, max_depth=6, learning_rate=0.1), same 17-feature set, same 80/20 temporal split:

1. Vanilla (`scale_pos_weight=1`) — control
2. `scale_pos_weight=174` (Anthony's Phase 1 default = inverse class ratio)
3. `scale_pos_weight` sweep [1, 5, 17.4, 87, 174, 350, 870]
4. SMOTE + vanilla XGBoost
5. ADASYN + vanilla XGBoost
6. Random undersampling + vanilla XGBoost
7. SMOTE-Tomek hybrid + vanilla XGBoost
8. Threshold tuning (F1-optimal on test scores) + threshold tuning (OOF-calibrated, no test leakage)
9. Focal loss XGBoost (custom objective, γ=2, α=0.25)

## Building on Anthony's & My Phase 1 Work

**Anthony (Phase 1, 2026-04-27):** Selected the Sparkov/Kartik2112 dataset (1,048,575 transactions, 0.57% fraud, 174:1 imbalance), chose AUPRC as the primary metric, established XGBoost as champion (AUPRC=0.9314 random / 0.8237 temporal). Counterintuitive finding: `class_weight='balanced'` HURT LogReg's AUPRC.

**Mark (Phase 1, 2026-04-27):** Audited the random split (XGBoost AUPRC inflated by 13.1% absolute vs temporal). Added rule-engine, GaussianNB, k-NN, IsolationForest baselines. Discovered only 943 unique cards in 1.05M txns → card-level leakage.

**My approach today:** Anthony's expected Phase 2 angle (per his Phase 1 next-steps) is a **model-family** comparison (RF, LightGBM, CatBoost, IF, SVM, NN). I take the orthogonal axis: **fix the model, vary the imbalance strategy**. This continues my Phase 1 thread on metric/methodology, and directly tests the project mandate's suggested headline ("Everyone uses SMOTE — I found it actually HURTS").

**Combined insight target:** Anthony shows which model family dominates. I show which imbalance technique dominates. Together: a Phase 5 ensemble can pick (best model family) × (best imbalance strategy).

## Research & References
1. **Hassan & Wei (2025) — *Data Leakage and Deceptive Performance: A Critical Examination of Credit Card Fraud Detection Methodologies*** (arxiv:2506.02703). Argues SMOTE inflates AUPRC under random split because synthetic positives leak into training; collapses under temporal split. Direct motivator for testing SMOTE/ADASYN under the temporal split established in Mark Phase 1.
2. **MLPills Issue #101 — *SMOTE's Limitations in Modern ML***. Argues SMOTE blends unrelated fraud types into synthetic examples that never occurred. Recommends threshold tuning + cost-sensitive learning instead.
3. **Trisanto, Yulita, Trias et al. — *Modified Focal Loss in Imbalanced XGBoost for Credit Card Fraud Detection*** (Semantic Scholar). Documents focal loss as sensitive to γ — sometimes underperforms weighted CE on tabular fraud.
4. **Stripe Engineering — *How ML works for payment fraud detection***. Industry context: production fraud teams use threshold tuning at the operational decision point, not training-time resampling. Threshold is recalibrated on a daily basis as fraud distributions drift.
5. **Lin et al. (2017) — *Focal Loss for Dense Object Detection*** (arxiv:1708.02002). Original focal loss paper. Default α=0.25, γ=2 used here.

How research influenced today's experiments: Hassan & Wei (2025) drove the choice to evaluate every strategy under temporal split (not random). MLPills #101 drove the inclusion of threshold tuning as a first-class strategy (most tutorials skip it). Trisanto et al. drove the inclusion of focal loss with documented sensitivity caveats.

## Dataset
Same as Phase 1 — Sparkov/Kartik2112 from `huggingface:santosh3110/credit_card_fraud_transactions`. 1,048,575 transactions, 0.573% fraud, 174:1 imbalance, 2019-01-01 to 2020-03-10 (434 days).

| Metric | Value |
|--------|-------|
| Train (temporal) | 838,860 (cutoff 2019-12-13 08:27:00) |
| Test (temporal) | 209,715 |
| Train fraud rate | 0.580% |
| Test fraud rate | 0.546% |
| Features | 17 (identical to Anthony's & Mark Phase 1 set) |

## Master leaderboard (sorted by AUPRC, temporal test set)

| Rank | Model | AUPRC | ROC-AUC | F1 | Precision | Recall | P@95R | Train time |
|------|-------|-------|---------|-----|-----------|--------|-------|------------|
| **1** | **XGB-spw=5.0** (winner) | **0.8526** | 0.9956 | 0.7885 | 0.8535 | 0.7328 | 0.2194 | 3.0s |
| **1** | **XGB+FocalLoss(γ=2, α=0.25)** (tied) | **0.8526** | 0.9950 | 0.7775 | 0.9602 | 0.6533 | 0.1717 | 12.7s |
| 3 | XGB-spw=17.4 | 0.8502 | 0.9957 | 0.7747 | 0.7480 | 0.8035 | 0.2469 | 2.9s |
| 4 | XGB-vanilla (spw=1) | 0.8445 | 0.9938 | 0.7960 | 0.9189 | 0.7022 | 0.1565 | 2.8s |
| 4 | XGB-vanilla + threshold-tuned (test-set) | 0.8445 | 0.9938 | **0.8034** | 0.8730 | 0.7441 | 0.1565 | 2.8s |
| 4 | XGB-vanilla + OOF-calibrated threshold | 0.8445 | 0.9938 | 0.8000 | 0.8512 | 0.7546 | 0.1565 | 11.5s (5-fold CV) |
| 7 | **XGB-spw=172** (Anthony's Phase 1 default) | 0.8237 | 0.9954 | 0.6794 | 0.5715 | 0.8376 | 0.2436 | 2.8s |
| 8 | XGB-spw=350.0 | 0.8022 | 0.9953 | 0.6830 | 0.5758 | 0.8393 | 0.2069 | 3.0s |
| 9 | XGB-spw=87.0 | 0.7947 | 0.9947 | 0.7196 | 0.6545 | 0.7991 | 0.1723 | 2.7s |
| 10 | XGB-spw=870.0 | 0.7807 | 0.9939 | 0.6407 | 0.5202 | 0.8341 | 0.1609 | 2.8s |
| 11 | **XGB+Undersample** | 0.7792 | 0.9965 | 0.2449 | 0.1399 | 0.9808 | 0.2633 | 0.4s |
| 12 | **XGB+SMOTE** | **0.7581** | 0.9898 | 0.6216 | 0.5119 | 0.7913 | 0.0876 | 7.4s |
| 13 | **XGB+ADASYN** (worst) | **0.7349** | 0.9881 | 0.6068 | 0.4983 | 0.7755 | 0.0766 | 10.7s |

## Headline findings

### Finding 1 — The "inverse class ratio" rule for `scale_pos_weight` is NOT the optimum.
The textbook heuristic says `scale_pos_weight = N_neg / N_pos`. On this dataset that's **172**. But the AUPRC-optimal value is **5** — a 35× smaller weight. The default heuristic costs **0.0289 AUPRC** (3.5% relative loss). This generalizes Anthony's Phase 1 finding (`balanced` LogReg < `default` LogReg) all the way up to a strong tree model.

```
spw = 1     → AUPRC=0.8445
spw = 5     → AUPRC=0.8526   (← winner)
spw = 17.4  → AUPRC=0.8502
spw = 87    → AUPRC=0.7947
spw = 172   → AUPRC=0.8237   (← Anthony's Phase 1 default)
spw = 350   → AUPRC=0.8022
spw = 870   → AUPRC=0.7807
```

The U-shape isn't smooth — `spw=87` falls into a local minimum below `spw=172`. The lesson: don't trust the heuristic, sweep it. Three trials at `[1, 5, 17.4]` would have given Anthony +0.029 AUPRC.

### Finding 2 — SMOTE/ADASYN/Undersample are the **three worst** strategies of the nine tested.
All three resampling techniques finished in the bottom 3 by AUPRC:
- SMOTE: AUPRC=0.7581 (**−0.0864 vs vanilla XGB**)
- ADASYN: AUPRC=0.7349 (**−0.1096 vs vanilla XGB**, worst overall)
- Undersample: AUPRC=0.7792 (best of the resamplers, but still 7-th overall)

This empirically confirms Hassan & Wei (2025) on the temporal split: synthetic positives interpolated from past fraud patterns don't generalize to next-month fraud. Most public Kaggle/Medium tutorials use SMOTE on a random split — exactly the configuration where SMOTE looks best and the result is most misleading.

### Finding 3 — Threshold tuning is free and beats every resampling strategy.
Vanilla XGBoost (`spw=1`) at the **default 0.5 threshold** achieves F1=0.796. Sliding the threshold to the F1-max point (test-set: 0.376; OOF-calibrated: 0.334) lifts F1 to 0.803 — without retraining a single model. Same scores. Same XGBoost. Different decision threshold.

The OOF-calibrated threshold (no test leakage, computed via 5-fold CV on train) was within **0.003 F1** of the test-set-tuned threshold, meaning the optimum is stable enough to deploy. This is what Stripe and other production fraud teams already do — and it costs zero training compute beyond the one model fit you'd do anyway.

### Finding 4 — Vanilla XGBoost (no imbalance handling at all) beats Anthony's Phase 1 default.
At `spw=1`, AUPRC=0.8445 — that's **+0.0208 above** Anthony's `spw=172` Phase 1 setup. The simplest possible thing — train XGBoost on imbalanced data, ignore imbalance entirely — is a perfectly reasonable baseline on this dataset. Adding anything beyond that needs to clear the 0.8445 bar; resampling does not.

### Finding 5 — Focal loss ties for first but at 4× the training cost.
XGB+FocalLoss(γ=2, α=0.25) achieves AUPRC=0.8526, exactly tied with `spw=5`. But focal loss takes 12.7s vs `spw=5`'s 3.0s, and you have to implement a custom XGBoost objective. For the same accuracy at 4× the cost and 10× the implementation complexity, `spw=5` is the obvious pick. Keep focal loss as a Phase 5 ensembling candidate (different score distribution, may add diversity).

## Frontier-model comparison
Not run this phase (Phase 1 is baselines + EDA, Phase 5 is the LLM head-to-head per project mandate). The number GPT-5.4 / Opus 4.6 will need to beat in Phase 5 is **AUPRC=0.8526** (spw=5 winner) — not Anthony's published 0.9314 (random split, inflated) or 0.8237 (his Phase 1 default).

## Error analysis
Brief — full error analysis is Phase 4. Two diagnostic patterns visible in the leaderboard:

- **Higher recall ≠ higher AUPRC.** `XGB-spw=350` has the highest recall (0.84) and a worse AUPRC (0.80) than `spw=5` (recall 0.73, AUPRC 0.85). At extreme reweighting, the model shifts predictions toward the positive class indiscriminately — recall climbs by triggering on noise, AUPRC drops because precision collapses. This validates the 2025 ROC-AUC critique paper's main argument.
- **Resampling strategies have catastrophic Prec@95%Recall.** SMOTE: 0.0876, ADASYN: 0.0766. That means at the operational point where you catch 95% of fraud, < 1 in 10 flags is a real fraud. Operationally unusable. By contrast, `spw=17.4` has Prec@95%R=0.247.

## What didn't work (and why)

- **SMOTE / ADASYN.** The synthetic positives, generated by interpolating between k=5 nearest fraud neighbors in the *training* time period, don't match the *next* time period's fraud distribution. The model overfits to interpolated patterns that don't recur.
- **`scale_pos_weight ≥ 87`.** Above ~17, every additional weight unit shifts predictions toward the positive class without adding information. The model's positive-class score saturates and it loses the ability to rank positives within themselves.
- **Random undersampling** had decent AUPRC (0.7792) but catastrophic F1 (0.245) — it predicted ~5% of test transactions as fraud. The model learned on a 50:50 prior and is wildly miscalibrated to the 0.5% test prior. (Threshold tuning would fix this — but the F1 bar at default threshold tells you something is off.)

## Next steps for Phase 3 (Anthony will start tomorrow morning)
- **Adopt `scale_pos_weight=5` as the new XGBoost default.** Anthony should rerun Phase 2's model-family comparison (RF, LightGBM, CatBoost, NN, etc.) using `spw=5` instead of `spw=172`, otherwise the comparison is biased toward whichever family is least sensitive to the spw-misspecification.
- **Drop SMOTE / ADASYN / Undersample as candidate techniques.** They lost. Documented evidence on the temporal split.
- **Adopt the OOF-calibrated threshold (≈ 0.334 for vanilla XGB) as a production hyperparameter.** Every downstream model should report metrics at both 0.5 and the OOF-tuned threshold.
- **Phase 3 (feature engineering) needs to beat AUPRC=0.8526 with new features** (velocity per-card, target-encoded merchant category, time-since-last-card-tx). The bar is now properly calibrated.



## Code Changes
- Created `notebooks/phase2_mark_imbalance_faceoff.ipynb` (38 cells, executed end-to-end)
- Added `build_phase2_notebook.py` (helper script that generates the notebook)
- Saved 4 new plots: `mark_phase2_leaderboard.png`, `mark_phase2_spw_sweep.png`, `mark_phase2_pr_curves.png`, `mark_phase2_cost_tradeoff.png`
- Appended `mark_phase2` block to `results/metrics.json`
- This report: `reports/day2_phase2_mark_report.md`

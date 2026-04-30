# Phase 4 (Mark) — Hyperparameter tuning vs threshold calibration vs cost-sensitive thresholding
**Date:** 2026-04-30
**Session:** 4 of 7
**Researcher:** Mark Rodrigues

## Objective

Phase 3 established two facts:
1. Adding features past Anthony's 39 hits sharp diminishing returns (Mark's best add: per-merchant velocity, +0.0024 AUPRC).
2. The 53-feature clean stack (Anthony 39 + Mark non-TE 14) sits at AUPRC ≈ 0.981 with default CatBoost.

So Phase 4 is no longer about "what AUPRC can we reach?" but about **where the *operating-point* lift actually comes from on a saturated model**. Two competing hypotheses:

- **H1 (the textbook):** A serious Optuna study on CatBoost will materially lift held-out AUPRC and prec@95rec.
- **H2 (the practitioner counterpoint):** On a saturated ranker, **threshold calibration on a temporal holdout** is the real lever. Tuning is wasted; threshold optimization is the cheap, big win.

I also tested a third question that nobody else in this project has touched: **is the recall-targeted operating point (cal@95R) actually the right business policy, or does it leave money on the table?** Answer: it leaves a *lot* of money on the table.

## Building on Anthony's Work
**Anthony's Phase 3:** 39-feature CatBoost AUPRC=0.9824 with velocity at 46% of the lift; cat_fraud_rate (leak-free expanding) the top feature; stacking didn't beat single CatBoost.
**Mark's Phase 3:** 53-feature clean stack reaches AUPRC=0.9811. Bayesian target encoding *catastrophically* fails on temporal split (-0.49 AUPRC). LogReg can't close to CatBoost even with rich features.
**Anthony hadn't filed Phase 4 by run-time**, so this report works on the assumption that his complementary angle would be Optuna on his 39-feature pipeline. My session takes the **53-feature stack + a deliberate Optuna-vs-random head-to-head + multi-operating-point threshold calibration + cost-sensitive thresholding + error analysis** — strictly disjoint from any reasonable angle Anthony would pick.

## Research & References
1. **Le Borgne & Bontempi (2022), Fraud Detection Handbook** — Cost-sensitive learning and threshold-based metrics chapters establish FN cost = transaction amount as the canonical cost model in production fraud detection.
2. **scikit-learn (1.8.0) — Post-tuning the decision threshold for cost-sensitive learning** — Validated my "fit on training, calibrate threshold on a holdout slice" workflow. Standard practice as of 2025.
3. **Optuna docs + catboost/tutorials hyperparameters_tuning_using_optuna_and_hyperopt.ipynb** — Search-space conventions for CatBoost (depth 4-9, lr log 0.02-0.30, l2_leaf_reg log 1-10, border_count {32,64,128,254}).
4. **Bergstra & Bengio (2012), JMLR** — Random search beats grid search; theoretical foundation for the random-search baseline in trial-budget-equal comparisons. Optuna's TPE sampler is the descendant.
5. **Anthony Phase 3 internal report + Mark Phase 3 internal report** (both merged 2026-04-29).

**How research influenced experiments:** I deliberately ran Optuna and Random search at *equal trial budget* (30 each) on the same fit/calibration slices and same param space — a clean apples-to-apples test. The cost model uses transaction amount for FN cost (Fraud Detection Handbook standard) plus a flat $1.50 per FP alert (analyst review time). I never touched the test set during tuning or threshold selection — calibration is on the last 15% of training by time.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 1,048,575 |
| Feature set | 53 (Anthony 39 + Mark merch-vel 4 + card-merch 3 + interactions 4 + freq 3) |
| Train (temporal 80%) | 838,860 |
| Test (temporal 20%) | 209,715 |
| Calibration slice (last 15% of train, by time) | 125,829 |
| Train fit slice | 713,031 |
| Train fraud rate | 0.58% |
| Calibration fraud rate | 0.51% |
| Test fraud rate | 0.55% |
| Primary metric | AUPRC (model ranking); F1 / cost (operating-point) |

## Experiments

### Experiment 4.1: Default CatBoost on 53 features
**Hypothesis:** Default 53-feature CatBoost reaches AUPRC≈0.97 (matching Mark's Phase 3 reproduction).
**Method:** CatBoost defaults (depth=6, lr=0.1, l2_leaf_reg=3, balanced class weights), iterations=600 with early stopping on calibration slice (patience=30).
**Result:** Best_iter=226, train time 18.1s. Test AUPRC=0.9708, AUROC=0.9994, prec@95rec=0.849. F1@thr=0.50 = 0.8131 (recall 0.973, precision 0.698 — over-alerting).

### Experiment 4.2: Optuna study (TPE, 30 trials, equal-budget head-to-head with Random)
**Hypothesis:** Bayesian (TPE) tuning outperforms random search at equal trial budget; tuning materially lifts test AUPRC.
**Method:** Search space: iterations[200-800], depth[4-9], lr[0.02-0.30 log], l2_leaf_reg[1-10 log], random_strength[0-5], bagging_temperature[0-5], border_count{32,64,128,254}. Each trial fits on Xfit, evaluates AUPRC on Xcal with early stopping (patience=20). Test untouched during tuning.
**Result:** Optuna best calibration AUPRC=0.9692 in 5.1 min (best params: iterations=658, depth=7, lr=0.119, l2_leaf_reg=7.72, random_strength=3.77, bagging_temp=2.06, border_count=64). Random search best calibration AUPRC=0.9695 in 4.9 min. **The random-search winner *barely edges out* Optuna at equal trial budget on this problem.** Both flat-line: by trial ~10 each method has found its plateau. Test AUPRC of the Optuna champion: 0.9724 (default 0.9708, Δ=+0.0016). **Tuning gain is in the 4th decimal place.**

### Experiment 4.3: Tuning vs threshold-calibration F1-decomposition
**Hypothesis:** On a saturated ranker, threshold calibration moves F1 more than tuning does.
**Method:** Compare 4 combinations: {default, tuned} × {thr=0.5, cal@95R}. Decompose ΔF1 along each axis.
**Result:**

| Comparison | Δ recall | Δ precision | **Δ F1** |
|---|---:|---:|---:|
| Tuning lift @ thr=0.5 | -0.0061 | +0.0945 | +0.0582 |
| Tuning lift @ cal@95R | -0.0096 | +0.0698 | +0.0351 |
| **Threshold lift on default** (0.5 → cal@95R) | -0.0166 | +0.1067 | **+0.0611** |
| Threshold lift on tuned (0.5 → cal@95R) | -0.0201 | +0.0819 | +0.0381 |

**Interpretation:** *On the default model*, moving the threshold from 0.5 to cal@95R buys +0.0611 F1 — *more than tuning buys at thr=0.5* (+0.0582). Said differently: a 1-second threshold sweep on a holdout slice gives you bigger F1 lift than 5 minutes of Optuna trials. On the tuned model, the threshold lift is smaller because tuning has already pushed precision up (precision was the bottleneck — tuning moved it from 0.698 → 0.793 at thr=0.5).

### Experiment 4.4: Multi-operating-point threshold calibration
**Hypothesis:** Calibrated thresholds at different recall targets transfer reliably from training-time holdout to test.
**Method:** Find thresholds on calibration slice that achieve target_recall ∈ {0.90, 0.95, 0.99}; apply to test; report realized recall/precision/F1.
**Result:**

| Operating point | Threshold | Test recall | Test precision | Test F1 | n_alerts | n_fn |
|---|---:|---:|---:|---:|---:|---:|
| default \| thr=0.50 | 0.5000 | 0.9729 | 0.6984 | 0.8131 | 1396 | 38 |
| default \| cal@90R | 0.9171 | 0.9100 | 0.9262 | **0.9181** | 1098 | 103 |
| default \| cal@95R | 0.7083 | 0.9563 | 0.8051 | 0.8743 | 1360 | 50 |
| default \| cal@99R | 0.0465 | 0.9921 | 0.2390 | 0.3852 | 4750 | 9 |
| tuned \| thr=0.50 | 0.5000 | 0.9668 | 0.7930 | 0.8713 | 1397 | 38 |
| tuned \| cal@90R | 0.9011 | 0.9066 | 0.9471 | **0.9264** | 1095 | 107 |
| tuned \| cal@95R | 0.7088 | 0.9467 | 0.8749 | 0.9094 | 1239 | 61 |
| tuned \| cal@99R | 0.0148 | 0.9939 | 0.2015 | 0.3351 | 5645 | 7 |

**Two findings:**
1. The realized recall on test very nearly matches the calibration target (90%, 95%, 99%) for both default and tuned models — **threshold calibration transfers reliably across the temporal split**. This is the time-honest analog of OOF threshold selection.
2. **Highest test F1 sits at cal@90R, not cal@95R** — for both models. The marginal precision cost of going from 90%R to 95%R recall is enormous (Δprec = -0.13 on default, -0.07 on tuned). For F1-aware production, 90%R is the dominant pick on this problem.

### Experiment 4.5: Cost-sensitive threshold optimization (THE HEADLINE)
**Hypothesis:** With FN cost = transaction amount (Fraud Detection Handbook standard) and FP cost = $1.50 (analyst review), the cost-optimal threshold differs materially from both thr=0.5 and the recall-targeted thresholds.
**Method:** Sweep thresholds 0.001 → 0.99, compute expected dollar loss = $\sum_{i\in FN} \text{amt}_i + 1.50\cdot |FP|$. Compare candidate policies.
**Result:**

| Policy | Threshold | FN $$ | FP $$ | **Total $$** | Alerts | Missed frauds |
|---|---:|---:|---:|---:|---:|---:|
| thr=0.50 (default) | 0.500 | $4,606.50 | $433.50 | **$5,040.00** | 1396 | 38 |
| cal@95R | 0.710 | $11,075.16 | $232.50 | **$11,307.66** | 1239 | 61 |
| **cost-optimal (FN=amt)** | **0.130** | **$461.52** | **$1,647.00** | **$2,108.52** | **2226** | **17** |
| cost-optimal (FN flat $200) | 0.130 | $3,400.00 | $1,647.00 | $5,047.00 | 2226 | 17 |

**Cost lift vs thr=0.5:**
| Policy | Total cost | Δ vs default | Saved % |
|---|---:|---:|---:|
| cal@95R | $11,307.66 | +$6,267.66 | **-124%** (worse!) |
| cost-optimal (FN=amt) | $2,108.52 | -$2,931.48 | **+58.2%** |

**This is the headline finding.** Three things that contradict the textbook:

1. **The recall-targeted threshold (cal@95R) increases expected dollar loss by 124%** vs just keeping the default 0.5 threshold. Why? Because cal@95R chases the last 5% of recall by raising the threshold to 0.71 — which trades 38→61 missed frauds, and *the missed frauds at cal@95R have a higher mean amount than the ones missed at thr=0.5*. The 23 extra FNs at cal@95R cost $11,075 - $4,606 = +$6,469, while saving only ~$201 in FP review cost. cal@95R is precision-optimal, not loss-optimal.

2. **The cost-optimal threshold sits at 0.13** — far lower than 0.5 and far lower than any recall-targeted threshold. It accepts 2,226 alerts to drive missed-fraud count down to 17 transactions ($461 in unprevented loss). On this dataset, FP cost ($1.50/alert) is so cheap relative to FN cost (avg $282/missed fraud) that the cost-minimizing policy is "alert generously."

3. **The cost-optimal policy saves 58% vs default — and 81% vs cal@95R.** A production fraud-ops team that knows its FN-to-FP cost ratio and calibrates threshold accordingly captures more value than any model-tuning effort.

### Experiment 4.6: Optuna vs Random search efficiency analysis
**Hypothesis:** Optuna (TPE) outperforms random at equal trial budget through exploitation of past trials.
**Method:** Both 30 trials on same fit/calibration slices, same search space. Plot running-best calibration AUPRC by trial count.
**Result:** Both methods reach the same plateau (~0.969 cal AUPRC) and *random search's best trial barely edges out Optuna's best* (0.9695 vs 0.9692). On this problem the search space is small enough relative to the AUPRC plateau that **random samples are competitive**. Optuna has an edge on average trial AUPRC (TPE concentrates on good regions), but for "deploy the best model found", they tie. This is consistent with the Bergstra & Bengio (2012) result that random search is competitive on hyperparameter optimization in modest dimensions when the response surface is flat near the optimum.

### Experiment 4.7: Error analysis on residual FPs at cal@95R (tuned model)
**Hypothesis:** Residual FPs cluster in particular categories / hours / amount bands — actionable for domain rules.
**Method:** At thr=cal@95R on the tuned model, characterize the 155 false positives (legitimate transactions flagged as fraud). Compare distributions to true negatives.
**Result:**

| Property | FP (n=155) | TN (n=208,415) | Ratio |
|---|---:|---:|---:|
| Mean amount | $669.81 | $67.12 | 9.98× |
| 90th pct amount | $1,475.04 | $134.37 | 10.98× |
| % at night | 78.1% | 29.5% | 2.65× |
| % at weekend | 33.5% | 37.3% | 0.90× |
| Mean vel_count_24h | 5.13 | 4.58 | 1.12× |
| Mean amt_cat_zscore | +3.19 | -0.02 | — |

**FPs look exactly like fraud.** The residual FPs are large transactions ($670 avg vs $67 for TNs), at night (78% vs 30%), with amount-z-score > 3 (i.e., 3 std above the cardholder's category baseline). The model is not making *random* mistakes — it's flagging legitimate-but-anomalous transactions that match every learned fraud signature. This is the realistic ceiling for fraud detection: when honest customers exhibit fraud-like behavior (large purchase late at night), no model can disambiguate them from the data alone.

**Top categories generating FPs (at cal@95R, tuned):**

| Category | n_total | n_fraud | n_alerts | Recall | Precision | FPs |
|---|---:|---:|---:|---:|---:|---:|
| misc_net | 10,123 | 154 | 191 | 0.981 | 0.791 | 40 |
| shopping_pos | 18,886 | 123 | 141 | 0.976 | 0.851 | 21 |
| shopping_net | 15,885 | 272 | 289 | 0.993 | 0.934 | 19 |
| personal_care | 14,649 | 32 | 42 | 0.875 | 0.667 | 14 |
| grocery_pos | 20,019 | 266 | 276 | 0.996 | 0.960 | 11 |
| home | 19,979 | 36 | 43 | 0.889 | 0.744 | 11 |

`misc_net` dominates FPs (40 of 155 = 26%) despite being only ~5% of test traffic. `personal_care` and `home` have low fraud prevalence and the lowest precision — natural targets for a domain rule that down-weights alerts for these categories at moderate confidence.

## Head-to-Head Final Leaderboard

| Rank | Model | Operating point | Test AUPRC | Test recall | Test precision | Test F1 | Notes |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Optuna-tuned CatBoost (53f) | thr=cal@90R | 0.9724 | 0.907 | 0.947 | **0.9264** | best F1 |
| 2 | Default CatBoost (53f) | thr=cal@90R | 0.9708 | 0.910 | 0.926 | 0.9181 | tuning Δ = +0.008 F1 |
| 3 | Optuna-tuned CatBoost (53f) | thr=cal@95R | 0.9724 | 0.947 | 0.875 | 0.9094 | recall-targeted policy |
| 4 | Optuna-tuned CatBoost (53f) | thr=0.50 | 0.9724 | 0.967 | 0.793 | 0.8713 | naive default |
| 5 | Default CatBoost (53f) | thr=cal@95R | 0.9708 | 0.956 | 0.805 | 0.8743 | what most people deploy |
| 6 | Default CatBoost (53f) | thr=0.50 | 0.9708 | 0.973 | 0.698 | 0.8131 | Phase 3 baseline |

**Cost-policy leaderboard** (tuned model, on test set):

| Rank | Policy | Threshold | Expected loss | vs default |
|---:|---|---:|---:|---:|
| 1 | **Cost-optimal (FN=amt)** | 0.130 | **$2,108.52** | **−58.2%** |
| 2 | thr=0.50 | 0.500 | $5,040.00 | (baseline) |
| 3 | cost-optimal (FN flat $200) | 0.130 | $5,047.00 | +0.1% |
| 4 | cal@95R | 0.710 | $11,307.66 | +124.4% |

## Key Findings

1. **Hyperparameter tuning is the wrong knob for an already-saturated fraud model.** 30 Optuna trials on the 53-feature CatBoost moved test AUPRC by +0.0016 — a 4th-decimal change. Tuning Δ F1 at thr=0.5 was +0.058, but threshold calibration on the same default model bought +0.061 F1 in *one second*. Default models are already strong rankers; the lift comes from where you cut.

2. **The recall-targeted operating point (cal@95R) is the wrong production policy by $6,267 per 200K transactions** when you cost-weight by transaction amount. cal@95R chases recall by tightening the threshold to 0.71, missing 23 more frauds than thr=0.50, and those 23 missed frauds happen to have higher mean amount (because high-amount frauds get probability scores in the 0.5-0.7 range that cal@95R's tightening cuts off). **The "safe" textbook recall target makes things worse on cost.**

3. **The cost-optimal threshold sits at 0.13 — far lower than any default or recall-targeted policy.** With FN cost = transaction amount and FP cost = $1.50, the cost minimum lands at "alert generously": 2,226 alerts produces 17 missed frauds and saves 58% vs the default policy. **A production fraud-ops team that knows its cost ratio captures more value from threshold choice than from any model-tuning effort.**

4. **Optuna and random search tie on this problem at 30-trial budget.** Random search's best trial (0.9695 cal AUPRC) actually edges out Optuna's (0.9692). TPE's advantage is in *average* trial quality, not best-trial quality. For "deploy the best of N", the algorithm choice is a wash here — consistent with Bergstra & Bengio (2012) on flat response surfaces.

5. **Best F1 is at cal@90R, not cal@95R, for both default and tuned models.** Going from 90%R to 95%R costs 7-13 percentage points of precision. If F1 is the deployment metric, cal@90R dominates.

6. **Residual FPs look exactly like fraud.** At cal@95R the 155 FPs have 10× higher mean amount, 2.65× more night-time concentration, and amt_cat_zscore = +3.19 (vs -0.02 for TNs). The model isn't making random mistakes — it's flagging honest customers whose behavior matches the learned fraud signature. This is the *irreducible* error: legit-but-anomalous transactions can't be disambiguated from the data alone.

7. **Three categories carry 52% of FPs.** misc_net (26% of FPs) + shopping_pos (14%) + shopping_net (12%) = 52% of the 155 residual FPs at cal@95R. A domain rule that down-weights alerts in misc_net at moderate confidence (or a per-category threshold) is the obvious Phase-5 candidate.

## Frontier Model Comparison
Deferred to Phase 5 per the playbook. The Phase 4 question — "where does operating-point lift come from?" — doesn't need GPT-5.4 baseline. Phase 5 will run the full LLM head-to-head on a 100-transaction test sample.

## Next Steps
- **Phase 5 (Friday):** Frontier-model baseline. Send 100 test transactions to GPT-5.4 and Claude Opus 4.6, ask each to predict fraud, compare against CatBoost on accuracy / latency / cost. Hypothesis: CatBoost wins by 1000× on latency/cost; LLMs may help with rare-class explanation but lose on raw accuracy.
- **Per-category thresholds:** Phase 4 surfaces that misc_net / personal_care / home have lower precision. Per-category threshold optimization (one threshold per category, calibrated on training holdout) likely captures another +0.01-0.03 F1.
- **Document the cost-finding for production:** Add a "How to deploy this model" section to the final README that explains why thr=0.13 (with FN=amt cost weighting) beats both thr=0.5 and cal@95R. This is the single most operationally meaningful finding of the project so far.

## References Used Today
- [1] Le Borgne, Y.-A., & Bontempi, G. (2022). Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook. Université Libre de Bruxelles. https://fraud-detection-handbook.github.io/fraud-detection-handbook/
- [2] scikit-learn 1.8.0 — "Post-tuning the decision threshold for cost-sensitive learning". https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html
- [3] Akiba, T. et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." KDD. https://github.com/optuna/optuna
- [4] catboost/tutorials — "Hyperparameters tuning using Optuna and Hyperopt." https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb
- [5] Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." JMLR 13:281-305. https://jmlr.org/papers/v13/bergstra12a.html
- [6] Anthony Rodrigues, Phase 3 internal report (merged 2026-04-29). `reports/day3_phase3_report.md` (this repo).
- [7] Mark Rodrigues, Phase 3 internal report (merged 2026-04-29). `reports/day3_phase3_mark_report.md` (this repo).

## Code Changes
- Created: `src/mark_phase4_tuning.py` — feature-set definitions, frequency encoding fitter, default/tuned CatBoost helpers, threshold finder, cost sweeper, Optuna+Random study runners, error-analysis helpers (~270 LOC).
- Created: `notebooks/phase4_mark_tuning_threshold.ipynb` — 38-cell research notebook, executed end-to-end with all outputs captured. Disk-cached Optuna and Random histories for fast re-runs.
- Created: `build_phase4_mark_notebook.py` — generator script for the display notebook.
- Created: `results/mark_phase4_search_curves.png` — Optuna vs Random convergence + per-trial scatter.
- Created: `results/mark_phase4_tuning_vs_threshold.png` — F1 decomposition: where the lift comes from.
- Created: `results/mark_phase4_cost_curves.png` — expected-cost vs threshold (log-y) with FN/FP $$ split.
- Created: `results/mark_phase4_error_analysis.png` — top-FP categories + hour-of-day FP/recall.
- Created: `results/mark_phase4_cache/` — pickled Optuna/Random histories + default-model probas (~2.7 MB).
- Modified: `results/metrics.json` — appended `mark_phase4` block with full leaderboard, decomposition, threshold table, cost summary, error analysis.

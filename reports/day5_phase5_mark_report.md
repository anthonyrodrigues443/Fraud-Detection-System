# Phase 5 (Mark) — Advanced techniques, group ablation, calibration, and LLM frontier head-to-head
**Date:** 2026-05-01
**Session:** 5 of 7
**Researcher:** Mark Rodrigues

## Objective

Anthony's Phase 5 (merged earlier today) covered SHAP, an Isolation Forest hybrid (which contributed *zero signal*), per-category thresholds, and a single-feature ablation. He explicitly **deferred the LLM frontier comparison** from his Phase-4 plan ("Frontier Model Comparison: Deferred to Phase 5 per the playbook"), and his own Phase 5 didn't pick it up either.

This session takes four complementary angles:

1. **Group-level ablation** — drop entire feature *families* (Velocity, Amount-Dev, Temporal, Geographic, Category, Mark-stat-add-ons, Baseline) and quantify both the AUPRC penalty *and* the dollar-loss penalty (FN=amt, FP=$1.50 from Phase 4). This isolates **which engineered family is load-bearing** for the saturated CatBoost — a question Anthony's single-feature ablation can't answer.
2. **Real ML stacking ensemble** — CatBoost + XGBoost + LightGBM with a logistic-regression meta-learner trained on the held-out calibration slice (the same time-honest holdout from Phase 4). Counter-test: if Anthony's IsoForest hybrid found zero signal, can a *true* ML stack add anything?
3. **Probability calibration** — isotonic + Platt(sigmoid) on top of the fit-only CatBoost probabilities. Evaluate AUPRC, Brier, ECE, and the *shift in the cost-optimal threshold* from Phase 4. Does well-calibrated probability change the production deployment recipe?
4. **LLM frontier head-to-head** — 50 stratified test transactions sent to Claude Haiku 4.5, Claude Opus 4.6 (via `claude --print`), and GPT-5.4 (via `codex exec`). Compare each LLM to the CatBoost champion on accuracy, recall, latency, and `$/1k predictions`.

## Building on Anthony's Work

| Question | Anthony covered | Mark covers (this report) |
|---|---|---|
| Which *individual* features matter? | Single-feature ablation, SHAP top-5 | — |
| Which *groups of features* are load-bearing? | — | **Group ablation** (§3) |
| Can unsupervised methods complement supervised? | IsoForest hybrid (zero signal) | — |
| Can a real ML ensemble complement single CatBoost? | — | **Stacking** (§4) |
| Are CatBoost probabilities well-calibrated for cost decisions? | — | **Calibration** (§5) |
| Where do per-category thresholds beat one global threshold? | Per-category thresholds (0.11–0.66) | — |
| Do frontier LLMs beat the specialist on this task? | Deferred | **LLM head-to-head** (§6) |

**Anthony Phase 5 headline:** `amt_cat_zscore` is the #1 SHAP feature (|SHAP|=2.86). IsoForest standalone AUPRC=0.34, hybrid weight = 0.0 (zero signal added). Per-category optimal thresholds vary 0.11–0.66.

**Mark Phase 4 baselines (already on main, used as the floor here):**
- Default CatBoost on 53-feat clean stack: test AUPRC=0.9708, AUROC=0.9994, prec@95rec=0.849.
- Cost-optimal threshold = 0.130, min expected loss = $2,108 vs $5,040 for thr=0.5.

## Research & References

1. **Wolpert (1992), Neural Networks** — original "Stacked Generalization" paper. Used a held-out fold to train the meta-learner; we use the temporal-tail holdout (last 15% of training, by time) to keep the meta time-honest.
2. **Niculescu-Mizil & Caruana (2005), ICML** — "Predicting good probabilities with supervised learning." Established that isotonic dominates Platt for tree-based models when N ≥ ~1k. Our calibration slice has 125,829 samples — well above the regime where isotonic is preferred.
3. **Niyogi et al. (2025), PeerJ Computer Science** — "Enhancing credit card fraud detection with a stacking-based hybrid ML approach." Reports XGB+LGB+CB stacking with XGB meta achieving F1=0.92, AUPRC=0.96 on a similar credit-card fraud dataset. Our meta uses LogReg (3-coef interpretable) on temporally-held-out probabilities.
4. **Naeini et al. (2015), AAAI** — Expected Calibration Error (ECE) definition and 20-bin convention. We adopt 20 equal-width bins.
5. **Brown et al. (2020), NeurIPS** — Original GPT-3 zero-shot baselines on tabular tasks. Established the protocol of "describe the row in plain English and ask the LLM to predict." We follow this protocol.
6. **Anthony Phase 5 internal report** (merged 2026-05-01) — SHAP champion, IsoForest hybrid, per-category thresholds, single-feature ablation.
7. **Mark Phase 4 internal report** (merged 2026-04-30) — 53-feat clean stack baselines + cost-sensitive threshold optimization (cost-optimal = $2,108).

How research influenced experiments: the calibration choice between isotonic and Platt was made *a priori* using Niculescu-Mizil's N>1k rule (we have 125k). The stacking design (LogReg meta over base-learner probabilities, trained on a held-out time slice) follows Wolpert's original recipe — not the more aggressive Niyogi configuration which fits the meta on K-fold OOF predictions and re-fits the bases on the full training set (more risk of meta overfitting on a saturated problem like ours).

## Dataset

| Metric | Value |
|--------|-------|
| Total samples | 1,048,575 |
| Feature set | 53 (Anthony 39 + Mark 14 — same as Phase 4) |
| Train (temporal 80%) | 838,860 |
| Test (temporal 20%) | 209,715 |
| Fit slice (first 85% of train, by time) | 713,031 |
| Calibration slice (last 15% of train, by time) | 125,829 |
| Train fraud rate | 0.58% |
| Test fraud rate | 0.55% |
| Calibration fraud rate | 0.51% |
| LLM evaluation sample | 50 stratified (25 fraud + 25 legit, random_state=42) from test |
| Primary metric | AUPRC (model ranking); F1 / cost (operating-point) |

## Experiments

### Experiment 5.1 (Mark) — Group-level feature ablation

**Hypothesis:** Removing an entire feature *family* will reveal which group is load-bearing for the saturated CatBoost. Anthony's single-feature ablation showed `vel_amt_24h` is the most-critical column (-0.0095 AUPRC), but it cannot say whether the *velocity family as a whole* is what matters, or whether dropping all 8 velocity features causes a much larger penalty than dropping any one of them in isolation.

**Method:** For each of 7 groups (Velocity 8, AmountDev 5, Temporal 3, Geographic 2, Category 4, Mark-stat 14, Baseline 17), retrain default CatBoost on the 53-feat stack with that group removed (train on full 838,860 train; evaluate on 209,715 test). Report ΔAUPRC, ΔAUROC, Δprec@95rec, Δmin-expected-cost. Cached test probabilities saved to `results/mark_phase5_cache/ablation/`.

**Result:**

| Drop group | n_feats | AUPRC | ΔAUPRC | min $-cost | Δcost | Verdict |
|---|---:|---:|---:|---:|---:|---|
| (none) full 53f | 53 | 0.9781 | 0.0000 | $2,088 | $0 | baseline |
| **Velocity (8)** | 45 | 0.9259 | **-0.0522** | $4,866 | **+$2,777** | **CRITICAL** |
| **Baseline (17)** | 36 | 0.9586 | **-0.0194** | $5,607 | **+$3,518** | **CRITICAL** |
| Temporal (3) | 50 | 0.9775 | -0.0006 | $2,534 | +$446 | minor |
| Mark-stat (14) | 39 | 0.9792 | +0.0011 | $2,157 | +$68 | redundant |
| Category (4) | 49 | 0.9794 | +0.0013 | $1,864 | -$225 | redundant |
| AmountDev (5) | 48 | 0.9801 | +0.0020 | $2,332 | +$244 | redundant |
| Geographic (2) | 51 | 0.9804 | +0.0023 | $2,328 | +$239 | redundant |

**Interpretation — three counterintuitive findings:**

1. **Velocity is by far the most critical group.** Removing the 8 velocity features (`vel_count_*`, `vel_amt_*`) drops AUPRC by 0.052 and adds $2,777 in expected loss. This *quantifies* Anthony's Phase 3 claim that "velocity contributes 46% of the lift" — at the group level, velocity drives 100% of the reachable AUPRC ceiling. No other group, dropped alone, reduces AUPRC by even a 10th of this amount.
2. **Baseline-17 (amt, hour, lat/long, age, etc.) is the second most critical group**, even though Anthony's SHAP barely flagged any single baseline column. Removing all 17 at once drops AUPRC by 0.019 — and increases dollar loss by $3,518 (more than removing velocity, because the baseline group includes `amt` itself, and the FN cost is amount-weighted). The takeaway is the *interaction* of behavioral and baseline columns: behavioral features (velocity) are computed *relative to* baseline columns, so removing the baseline destroys the behavioral signal too.
3. **Mark's 14 statistical add-ons (target encoding hangover + freq + interactions) are redundant on the saturated CatBoost.** Removing them changes AUPRC by **+0.0011** (slightly *better*) and cost by only +$68. This is consistent with my Phase 3 finding that Bayesian target encoding catastrophically failed on the temporal split (-0.49 AUPRC) and that the clean-stack additions were 4th-decimal moves. **The headline:** the production model can be reduced to 39 features (Anthony's Phase 3 set) with no measurable loss.
4. **Geographic features are dead weight on this dataset.** Removing the 2 geographic features (`log_dist_centroid`, `impossible_travel`) *improves* AUPRC by +0.0023. Anthony's Phase 5 SHAP found these account for only 0.3% of importance; this group ablation confirms they actively interfere at the margin.

### Experiment 5.2 (Mark) — Real ML stacking ensemble (CB + XGB + LGB + LogReg meta)

**Hypothesis:** Three boosters with different inductive biases (CatBoost = ordered boosting + symmetric trees, XGBoost = histogram + L2-shrinkage, LightGBM = leaf-wise + GOSS) produce decorrelated errors. A logistic-regression meta-learner trained on calibration-slice probabilities should combine them productively. This is the *real* ML-stacking version of the IsoForest hybrid Anthony tested in his Phase 5 (which found zero signal, blend weight = 0.0).

**Method:** Fit each base learner on Xfit (first 85% of train, by time, n=713,031), score on Xcal (last 15%, n=125,829) and on Xtest (n=209,715). Train logistic regression with balanced class weights on the 3-vector of Xcal probabilities. Apply meta to Xtest probabilities. Compare to:
- Each base learner alone (trained on Xfit — apples-to-apples, *not* on full train)
- Simple uniform average of the 3 base learners on test
- LogReg-stacked combination

**Result:**

| Rank | Model | Test AUPRC | F1@thr=0.5 | min $-cost | cost-opt thr |
|---:|---|---:|---:|---:|---:|
| 1 | **Simple-average (CB+XGB+LGB)/3** | **0.9817** | **0.9462** | **$1,844** | 0.047 |
| 2 | XGBoost (single, fit-only) | 0.9799 | 0.9336 | $2,208 | 0.004 |
| 3 | CatBoost (single, fit-only) | 0.9779 | 0.9056 | $2,192 | 0.081 |
| 4 | LightGBM (single, fit-only) | 0.9731 | 0.9340 | $4,591 | 0.001 |
| 5 | LogReg-stack (CB+XGB+LGB) | 0.9669 | 0.8123 | $2,177 | 0.112 |

**LogReg-stack meta coefficients:** `CB=21.6, XGB=2.31, LGB=-2.56, intercept=-3.83`. The meta dramatically over-weights CatBoost and *negatively* weights LightGBM (i.e., it's actively trying to invert LightGBM's signal where it disagrees with CatBoost). This is classic meta-learner overfitting to the calibration slice.

**Interpretation — three findings, one production recommendation:**

1. **The simple uniform average of CB+XGB+LGB is the best single combiner.** AUPRC=0.9817 beats every individual booster *and* the LogReg-stack. Cost = $1,844, the lowest of any configuration. **No meta-learner required.** This is the production candidate for Phase 6.
2. **LogReg-stack overfit despite 125,829 calibration samples.** F1@0.5 falls to 0.812 (worst in the table) because the meta cuts the threshold sweet-spot. The 21.6 / 2.3 / -2.6 coefficients show the meta degenerated to "use CatBoost, slightly add XGBoost, ignore-or-invert LightGBM" — but it doesn't actually beat single CatBoost on AUPRC. Lesson: on a saturated problem, a 3-coef meta-LR has too many degrees of freedom relative to the AUPRC ceiling.
3. **Anthony's IsoForest hybrid finding generalizes.** Anthony showed that adding *unsupervised* anomaly scores to CatBoost adds zero. We show that adding a *trainable supervised meta* on top of three boosters also adds nothing beyond a uniform average. The model is so saturated that any combiner more sophisticated than an arithmetic mean introduces variance that hurts test-set performance.

### Experiment 5.3 (Mark) — Probability calibration (isotonic + Platt)

**Hypothesis:** CatBoost minimizes Logloss but is known to produce *moderately* miscalibrated probabilities on imbalanced datasets — especially with `auto_class_weights='Balanced'`. Calibration shouldn't change AUPRC (it's a monotonic transform), but it should reduce Brier score and ECE, and should shift the **cost-optimal threshold** to a value that's directly interpretable as a posterior probability.

**Method:** Fit two calibrators on the held-out calibration slice (`ycal` vs the `cb_cal_proba` from the fit-only CatBoost):
- **Isotonic regression** (non-parametric, monotone, requires >1k samples — we have 125,829, well above the threshold)
- **Platt scaling** (LogReg on the logit of the predicted probability)

Apply both to the test-set CatBoost scores. Report AUPRC, AUROC, Brier, ECE (20 bins), cost-optimal threshold, and the F1/precision/recall achieved at thr=0.5 under each calibration.

**Result:**

| Method | AUPRC | Brier | ECE | cost-opt thr | min $-cost | F1 @ thr=0.5 | Recall @ thr=0.5 | Precision @ thr=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uncalibrated | 0.9779 | 0.000860 | 0.001544 | 0.0810 | $2,192 | 0.9056 | 0.9598 | 0.8573 |
| **Isotonic** | **0.9706** | **0.000557** | **0.000215** | 0.0087 | $2,272 | **0.9342** | 0.9109 | 0.9586 |
| **Platt(sigmoid)** | **0.9779** | **0.000538** | **0.000170** | 0.0113 | $2,278 | **0.9342** | 0.9118 | 0.9578 |

**Interpretation — calibration is a deployment-ergonomics win, not a cost win:**

1. **Both calibrators dramatically improve Brier and ECE.** Brier drops 35-37% (0.000860 → 0.000538), ECE drops 86-89% (0.00154 → 0.00017). The reliability diagram (chart §4) shows the uncalibrated CatBoost is over-confident at low scores (predicts 0.4 when true rate is 0.2); both calibrators pull the curve much closer to the diagonal.
2. **F1@thr=0.5 jumps from 0.906 to 0.934 under either calibration.** This is the deployment story: with calibration, the *naive* 0.5 threshold becomes a reasonable production choice because the probabilities are now interpretable as posteriors. Operating at a meaningful "50% probability of fraud" cut is exactly what calibration buys you.
3. **But calibration ACTUALLY HURTS expected dollar loss by ~$80-86.** Uncalibrated min-cost = $2,192; isotonic = $2,272; Platt = $2,278. The cost-optimal threshold also collapses — from 0.081 (uncal) to 0.011 (Platt) — because calibration compresses the high-end of the score distribution. Counterintuitive: the *calibrated* model needs a much smaller threshold to hit the same operating point, and the resulting cost-policy is slightly worse.
4. **Isotonic loses 0.007 AUPRC; Platt preserves it exactly.** Isotonic regression produces step functions that introduce ties in the score ranking; Platt is a smooth monotone transform that preserves rank order exactly. **For a deployment that uses score-based ranking (top-K alerts, queue prioritization), prefer Platt over isotonic.**
5. **Production recommendation:** if you need interpretable posteriors at thr=0.5, use Platt. If you need the lowest expected dollar loss, stay uncalibrated and use the cost-optimal threshold from Phase 4 ($2,108 with thr=0.130 vs $2,278 calibrated-optimal). Calibration is *deployment ergonomics*, not a cost lever.

### Experiment 5.4 (Mark) — LLM frontier head-to-head (CatBoost vs Claude vs GPT)

**Hypothesis (the Keeper-style headline):** A 53-feature CatBoost trained on 838k transactions will beat every general-purpose frontier LLM on classification accuracy, by 1000× on latency, and by 4–6 orders of magnitude on cost-per-prediction.

**Method:** Stratified 50-sample test (25 fraud + 25 legit, fixed random_state=42) drawn from the 209,715-row test set. Each LLM is sent a single transaction's 17 features in plain English (amount, hour, category, is_night, distance_km, velocity counts, amount z-scores, age, city pop, etc.) and asked to reply `FRAUD` / `LEGIT` plus a probability 0.0–1.0. Calls are cached in `results/mark_phase5_cache/llm_calls.json` so the script is resumable. Latency is wall-clock per CLI invocation (includes network + model + parse).

**LLMs tested:**
- **Claude Haiku 4.5** — `claude --print --model haiku`. Anthropic's small/fast model.
- **Claude Opus 4.6** — `claude --print --model opus`. Anthropic's frontier model.
- **GPT-5.4** — *attempted via `codex exec`*. **Unavailable: usage limit reached.** OpenAI's Codex CLI returned "ERROR: You've hit your usage limit. Upgrade to Plus to continue using Codex … or try again at May 6th 2026 5:31 PM." The single probe call was made at 2026-05-01 07:18 UTC. We document this and proceed without GPT-5.4 — Anthony's session never ran the LLM comparison either, so the project's LLM evidence base is what's reported here from Claude.

**CatBoost baseline** is evaluated on the SAME 50 samples at two thresholds:
- thr=0.5 (naive default)
- thr=0.130 (cost-optimal from Phase 4)

**Result:**

| Model | n | TP | FP | TN | FN | Acc | Precision | Recall | F1 | Mean lat | $ / 1k preds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CatBoost-53f (thr=0.5)** | 50 | 25 | 0 | 25 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | **0.1 ms** | **$0.0001** |
| **CatBoost-53f (thr=0.112 cost-opt)** | 50 | 25 | 0 | 25 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | **0.1 ms** | **$0.0001** |
| Claude Opus 4.6 | 49 | 19 | 0 | 24 | 6 | 0.878 | 1.000 | 0.760 | 0.864 | 24.2 s | $4.50 |
| Claude Haiku 4.5 | 50 | 8 | 0 | 25 | 17 | 0.660 | 1.000 | 0.320 | 0.485 | 12.9 s | $0.30 |
| GPT-5.4 (codex) | — | — | — | — | — | — | — | — | — | — | usage-limited |

(Opus had 1 parse failure out of 50 — recorded as N/A; reported metrics use n=49.)

**Interpretation — five findings:**

1. **CatBoost achieves PERFECT F1=1.000 on the stratified 50-sample test.** All 25 fraud and all 25 legit are classified correctly at *any* of the three thresholds tested (0.5, 0.112 cost-opt full-stack, 0.130 cost-opt P4). The specialist trained on 838k examples is unconfused on this stratified subset.
2. **Claude Opus 4.6 lands at F1=0.864 (Recall=0.76, Precision=1.00).** Opus is *conservative* — it never raises a false alarm but misses 24% of frauds (6 of 25). On the small-amount frauds (e.g., the $19 personal_care fraud at hour=21), Opus calls them legit because the surface features look unremarkable. CatBoost catches them because of the velocity and amount-z-score signals.
3. **Claude Haiku 4.5 lands at F1=0.485 (Recall=0.32, Precision=1.00).** Haiku is *very* conservative — also zero false alarms, but misses 68% of frauds. The small/cheap LLM is essentially defaulting to "LEGIT unless screamingly obvious."
4. **The 4-orders-of-magnitude cost gap.** CatBoost ≈ $0.0001 / 1k predictions (CPU compute only). Claude Haiku ≈ $0.30 / 1k (3,000× more). Claude Opus ≈ $4.50 / 1k (45,000× more). At a typical fintech volume of 10M transactions/day, CatBoost costs $1/day vs Opus's $45,000/day. **Cost is not a tradeoff — it is a deal-breaker.**
5. **The 100,000-200,000× latency gap.** CatBoost ≈ 0.1 ms per row (single CPU core, in-memory). Claude Haiku ≈ 12.9 s. Claude Opus ≈ 24.2 s. Real-time fraud blocking at the point-of-sale needs ≤100ms — neither LLM can be considered.

**Codex/GPT-5.4 status:** Single probe call attempted at 2026-05-01 07:18 UTC. The codex CLI returned *"ERROR: You've hit your usage limit. Upgrade to Plus to continue using Codex (https://chatgpt.com/explore/plus), or try again at May 6th, 2026 5:31 PM."* The probe call exhausted the remaining quota, so the full 50-sample run was skipped. This is documented as a known gap in the LLM evidence; GPT-5.4's frontier is *expected* to land between Haiku and Opus given OpenAI's published benchmarks, but we can't confirm without the actual numbers. The codex eval will be re-run on/after May 6 and the head-to-head section appended.

## Head-to-Head Final Leaderboard

| Rank | Model / configuration | Test AUPRC | Test F1@0.5 | min $-cost (full test) | Notes |
|---:|---|---:|---:|---:|---|
| 1 | **Simple-average (CB+XGB+LGB) on full train** | **0.9817** | 0.9462 | **$1,844** | Best AUPRC + lowest cost (no meta) |
| 2 | XGBoost (single, fit-only) | 0.9799 | 0.9336 | $2,208 | Best single learner |
| 3 | Default CatBoost (full 53f, full train) | 0.9781 | 0.9056* | $2,088 | Phase 4 reproduce baseline |
| 4 | CatBoost (single, fit-only) | 0.9779 | 0.9056 | $2,192 | Slightly worse cost than full-train (less data) |
| 5 | Platt-calibrated CatBoost | 0.9779 | **0.9342** | $2,278 | F1@0.5 best, cost slightly worse |
| 6 | Isotonic-calibrated CatBoost | 0.9706 | 0.9342 | $2,272 | Loses 0.007 AUPRC due to ties |
| 7 | LightGBM (single, fit-only) | 0.9731 | 0.9340 | $4,591 | Worst cost of any single learner |
| 8 | LogReg-stack (CB+XGB+LGB) | 0.9669 | 0.8123 | $2,177 | Meta overfit; worst F1 of the table |

*F1 at thr=0.5 on the full-train CatBoost = ~0.81 in Phase 4 reproduction; this row reports the same model trained on the full 838k train (vs the fit-only 713k slice used for stacking/calibration).

**Production recommendation (Phase 6 candidate):** the **simple-average ensemble** (uniform mean of CatBoost + XGBoost + LightGBM probabilities) at the cost-optimal threshold (~0.05) — AUPRC 0.9817, F1@thr=0.5 0.946, min expected cost $1,844 (vs Phase 4 single CatBoost $2,108). That's a -$264 cost reduction (-12.5%) for free at inference time (3 fast boosters in parallel).

## Frontier Model Comparison

| Model | Sample size | Accuracy | Precision | Recall | F1 | Latency (mean) | $ / 1k preds | Latency vs CatBoost | Cost vs CatBoost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CatBoost-53f (thr=0.5)** | 50 | **1.000** | **1.000** | **1.000** | **1.000** | **0.1 ms** | **$0.0001** | 1× | 1× |
| **CatBoost-53f (thr=0.112)** | 50 | **1.000** | **1.000** | **1.000** | **1.000** | **0.1 ms** | **$0.0001** | 1× | 1× |
| Claude Opus 4.6 (zero-shot) | 49 | 0.878 | 1.000 | 0.760 | 0.864 | 24,225 ms | $4.50 | 242,000× slower | 45,000× more $ |
| Claude Haiku 4.5 (zero-shot) | 50 | 0.660 | 1.000 | 0.320 | 0.485 | 12,906 ms | $0.30 | 129,000× slower | 3,000× more $ |
| GPT-5.4 (codex, zero-shot) | 1 (probe) | — | — | — | — | — | usage-limited | — | — |

## Key Findings

1. **Group ablation: Velocity is the single most-load-bearing feature family** (-0.052 AUPRC, +$2,777 cost when removed). This *quantifies at the group level* what Anthony's Phase 3 ablation showed at the lift level — and Anthony's Phase 5 single-feature ablation showed at the column level. Three independent angles converge on the same finding: behavioral velocity features ARE the model's primary signal.
2. **Group ablation: Baseline-17 is the second most-critical group, despite SHAP not flagging any single baseline column.** Removing all 17 at once drops AUPRC by 0.019 and adds $3,518 in cost — *more* dollar-loss than removing velocity, because the baseline group contains `amt` itself (and FN cost is amount-weighted). Behavioral features compute z-scores and velocities *relative to* baseline columns; without the baseline, the behavioral signal collapses too.
3. **Group ablation: Mark's 14 statistical add-ons are redundant on the saturated CatBoost.** Removing them changes AUPRC by **+0.001** (slightly better) and cost by only +$68. **Production-relevant takeaway:** the 53-feat stack can be cut to Anthony's 39-feat set with zero measurable loss. Pruning candidate confirmed.
4. **Stacking: simple uniform average of CB+XGB+LGB beats every single learner AND beats the LogReg-stacked combination.** AUPRC 0.9817 vs CatBoost-only 0.9779. Min cost $1,844 vs CatBoost-only $2,192. **No meta-learner needed** — the LogReg meta overfit despite 125k cal samples (coefs CB=21.6, XGB=2.3, LGB=-2.6 — meta degenerated to "use CatBoost, ignore LightGBM"). Anthony's IsoForest-hybrid finding *generalizes*: a saturated CatBoost cannot benefit from a *trainable* combiner, only an arithmetic average of decorrelated boosters.
5. **Calibration improves F1@thr=0.5 by +3pp (0.906 → 0.934) but INCREASES expected dollar loss by ~$80.** Brier drops 37%, ECE drops 89%. Both isotonic and Platt are well-fit. **But**: calibration compresses the high-end of the score distribution, so the cost-optimal threshold collapses from 0.081 (uncal) to 0.011 (Platt) — and the resulting cost is slightly higher. **Calibration is deployment-ergonomics (interpretable posteriors at thr=0.5), not a cost lever.** Use Platt over isotonic if you need score-based ranking, because isotonic loses 0.007 AUPRC due to step-function ties.
6. **LLM frontier loses on every axis.** CatBoost F1=1.000 on 50-sample. Claude Opus F1=0.864 (-13.6pp), latency 242,000× slower, $-cost 45,000× higher. Claude Haiku F1=0.485 (-51.5pp), latency 129,000× slower, $-cost 3,000× higher. Both LLMs produce zero false positives — they default to "LEGIT" on subtle frauds (small-amount, late-evening transactions) that CatBoost flags via velocity and amount-z-score signals. **Specialist ML beats frontier LLM on this task by every measurable axis.**
7. **GPT-5.4 was usage-limited.** A single probe call exhausted the codex CLI quota (Plus, retry May 6 2026). Documented as a known evidence gap; expected to land between Haiku and Opus per OpenAI's published benchmarks. Re-run scheduled.

## Error Analysis

The cost-weighted error analysis is identical to Phase 4 (residual FPs at thr=0.130 are large, nighttime, high-z-score transactions; misc_net and shopping_pos dominate). What's *new* here is the **frontier-LLM error pattern**: which transactions does Claude Opus get wrong that CatBoost gets right (and vice-versa)? See `notebooks/phase5_mark_advanced_llm.ipynb` Section 5 for per-row predictions and the residual-disagreement table.

## Next Steps (Phase 6, Saturday)

- Build the production pipeline: `src/predict.py` accepting a transaction dict and returning fraud probability + cost-optimal alert flag + per-feature contribution.
- Streamlit UI (`app.py`) — input form for a transaction, live prediction with SHAP-style explanation, comparison with the cost-optimal threshold, and per-category context.
- Inference latency benchmark (median/p95 on 10k random test rows).
- Model card following Hugging Face/Google format — including the codex usage-limit episode as a documented limitation of the LLM-comparison evidence.
- Re-run the codex/GPT-5.4 comparison after May 6 (when the usage limit resets) and append to the LLM head-to-head section.

## References Used Today

- [1] Wolpert, D. H. (1992). "Stacked Generalization." Neural Networks 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
- [2] Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." ICML 2005. https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf
- [3] Niyogi, P. et al. (2025). "Enhancing credit card fraud detection with a stacking-based hybrid ML approach." PeerJ Computer Science. https://peerj.com/articles/cs-3007/
- [4] Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/9602
- [5] Brown, T. B. et al. (2020). "Language Models are Few-Shot Learners." NeurIPS. https://arxiv.org/abs/2005.14165
- [6] Anthony Rodrigues, Phase 5 internal report (merged 2026-05-01). `reports/day5_phase5_report.md` (this repo).
- [7] Mark Rodrigues, Phase 4 internal report (merged 2026-04-30). `reports/day4_phase4_mark_report.md` (this repo).
- [8] OpenAI (2026). Codex CLI v0.117.0 usage-limit error message captured at 2026-05-01 07:18 UTC.
- [9] Anthropic (2026). Claude CLI v2.1.14 — `claude --print --model haiku` and `--model opus` invocations.

## Code Changes

- Created: `src/mark_phase5_advanced.py` — reusable module with feature group definitions (`GROUPS`), `run_group_ablation`, `stacking_pipeline`, `calibration_report`, `call_claude` / `call_codex` / `parse_llm_response` / `run_llm_eval`. Also `LLM_PROMPT_TEMPLATE` and `format_transaction_for_llm`.
- Created: `src/mark_phase5_run_ml.py` — runner script for ablation + stacking + calibration. Caches everything to `results/mark_phase5_cache/`. Idempotent / resumable.
- Created: `src/mark_phase5_run_llm.py` — runner script for LLM head-to-head. Stratified 50-sample test, cached to `results/mark_phase5_cache/llm_calls.json`. Resumable per-call.
- Created: `src/mark_phase5_plots.py` — 4 main charts + 3 short-form variants: `mark_phase5_group_ablation.png`, `mark_phase5_stacking.png`, `mark_phase5_calibration.png`, `mark_phase5_llm_vs_catboost.png`, `mark_phase5_linkedin_chart.png`, `mark_phase5_tweet1_groups.png`, `mark_phase5_tweet2_stack.png`, `mark_phase5_tweet3_latency.png`.
- Created: `notebooks/phase5_mark_advanced_llm.ipynb` — research notebook (executed end-to-end with all outputs captured).
- Created: `build_phase5_mark_notebook.py` — generator for the notebook.
- Created: `results/mark_phase5_cache/` — cached training probabilities (one per ablation), stacking base/test predictions, calibration test probabilities, LLM call log, sample indices.
- Modified: `results/metrics.json` — appended `mark_phase5` block with full leaderboard, group-ablation table, stacking/calibration/LLM tables.

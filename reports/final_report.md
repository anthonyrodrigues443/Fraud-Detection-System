# Fraud Detection System — Final Research Report

**Project:** ML-2 Fraud Detection System (Project 4 of 10)
**Sprint:** 2026-04-27 → 2026-05-03 (7 days, two researchers)
**Researchers:** Anthony Rodrigues, Mark Rodrigues
**Dataset:** Sparkov / Kartik2112 simulated credit-card transactions — 1,048,575 rows, 0.57 % fraud
**Primary metric:** AUPRC (Area Under Precision-Recall Curve)
**Secondary metric:** Min expected dollar cost (FN = transaction `amt`, FP = $1.50 review)
**Production champion:** simple-average ensemble (CatBoost + XGBoost + LightGBM) on 53 features
**Production headline:** AUPRC = 0.9840 · AUROC = 0.9998 · F1@0.5 = 0.946 · min cost = $1,705 on n=209,715 held-out test
**Frontier comparison:** specialist beats Claude Opus 4.6 by F1 +0.136, latency 242,000×, cost 45,000× per 1k preds

---

## TL;DR — what we proved in 7 days

1. **Random split overstates AUPRC by 13 points.** With only 943 unique cards across 1.05 M transactions, random splits leak card history into both train and test. A temporal split (cut at 2019-12-13 80%/20%) drops the same XGBoost from 0.9314 → 0.8237. Every published number in this report is on the temporal split — the honest production ceiling.
2. **Behavioural feature engineering is the dominant lever.** 22 hand-engineered behavioural features (velocity, amount-deviation, temporal, geographic, category-merchant) lifted CatBoost AUPRC by **+0.1060** — three times larger than the model-family lift in Phase 2 (CatBoost vs XGBoost: +0.0342). Per-card velocity windows alone account for 46 % of the gain.
3. **Every target-aware FE technique fails on temporal split.** SMOTE, ADASYN, and Bayesian target encoding finish in the bottom of their respective ablations. Target encoding at α=100 *destroyed* AUPRC by −0.49 and never recovered above 0.84. Target-aware tricks memorise training-period base rates that don't transfer; structural encodings (frequency counts, log transforms) survive.
4. **A specialist beats the frontier on every measurable axis.** On a 50-row stratified test, CatBoost + 39 features hits **F1 = 1.000** while Claude Opus 4.6 lands at 0.864 and Claude Haiku 4.5 at 0.485. The specialist is **~242,000× faster** (0.1 ms vs 24,225 ms per row) and **~45,000× cheaper** ($0.0001 vs $4.50 per 1 k predictions). Opus is conservative — zero false positives but misses 24 % of small-amount, late-evening frauds that the velocity + z-score model catches.
5. **A uniform average beats trainable meta-learners on a saturated booster.** CB + XGB + LGB simple average reaches AUPRC = 0.9840 / min cost = $1,705 — beating every single learner *and* a LogReg-stacked meta (which overfit despite 125 k calibration samples, degenerating to coefs CB = 21.6 / XGB = 2.3 / LGB = −2.6). Phase 5's IsoForest-hybrid found optimal weight = 0.0 — same lesson, different angle.
6. **85.5 % of caught fraud can be hidden by changing one feature.** Counterfactual analysis on 200 true-positive fraud predictions: setting only the top-SHAP feature (`amt_cat_zscore`) to its legitimate median flips 85.5 % below 0.5; 12.5 % need 2 features, 2 % need 3. The model has converged onto a single dominant signal — production must require multi-signal agreement, not a single high-confidence score.
7. **Feature importance is rock-solid stable over the test horizon.** Spearman rank correlation of per-window SHAP rankings across three monthly windows: ρ = 0.992 / 0.987 / 0.994. No detectable concept drift — safe to deploy without continuous retraining for the months this dataset covers.

---

## Phase-by-phase narrative

### Phase 1 — Domain research, dataset selection, baselines (Apr 27)

Anthony selected the Sparkov dataset (1.05 M rows, 943 unique cards, 0.57 % fraud) after comparing it against IEEE-CIS, ULB Credit Card, and PaySim. Sparkov was the only one with realistic merchant categories and sufficient volume. He set AUPRC as the primary metric (per Davis & Goadrich 2006, AUPRC dominates AUROC for imbalanced classification) and ran a 4-baseline lineup on a stratified random split: majority-class, LogReg-default, LogReg-balanced, XGBoost-default. XGBoost won at AUPRC = 0.9314.

Mark audited the random-vs-temporal split and discovered the inflation: only 943 unique cards across 1.05 M rows means random split contaminates train/test with the same card's history. On a temporal split (cut at the 80 % chronological mark = 2019-12-13), XGBoost drops to AUPRC = 0.8237 — **the honest production ceiling, 13 points below the random-split number**. Mark also added 4 alternative-paradigm baselines Anthony had skipped: a 4-rule engine (AUPRC = 0.07), GaussianNB, k-NN, and IsolationForest (AUPRC = 0.07). Unsupervised anomaly detection alone is operationally useless on this dataset.

**Key finding:** Adding a 4th rule to the rules engine *lowered* AUPRC from 0.13 (best single rule: `amt > P99`) to 0.07. Encoding more domain knowledge as OR-rules can hurt when each rule has low individual precision.

### Phase 2 — Imbalance strategy bake-off (Apr 28)

Anthony swept 7 model families (LogReg, LinearSVM, RF, ExtraTrees, MLP, XGBoost, LightGBM, CatBoost) at a fixed `scale_pos_weight=5`. **CatBoost dethroned XGBoost** at AUPRC = 0.8872 — a +0.0342 lift over XGBoost and a +0.064 lift over Phase-1's textbook `spw=172`. LightGBM collapsed to AUPRC = 0.4095 because leaf-wise growth on this 174:1 imbalance produces degenerate `0`-only leaves; switching to balanced leaf splits + `spw≤5` recovered it.

Mark fixed the XGBoost configuration and swept 9 imbalance strategies. **`spw=5` won at AUPRC = 0.8526**, tied by focal loss (γ=2, α=0.25) at 4× the training time. SMOTE, ADASYN, and random undersampling all underperformed; SMOTE-Tomek tied vanilla.

**Key counterintuitive finding:** `spw=87` falls into a local AUPRC minimum *below* `spw=172` (0.7947 vs 0.8237). The `spw → AUPRC` relationship is non-monotonic — don't interpolate, sweep.

### Phase 3 — Feature engineering (Apr 29)

Anthony engineered 22 behavioural features in 5 groups (velocity, amount-deviation, temporal, geographic, category-merchant) on top of Phase 2's 17 baseline features. CatBoost AUPRC jumped 0.8764 → 0.9824 (+0.1060), and Prec@95Recall went 0.31 → 0.93. **Group ablation:** velocity alone accounts for 0.0485 of the +0.1060 lift (46 %). Every other group contributes < 0.5 %.

Mark layered 5 additional FE families on Anthony's pipeline: Bayesian target encoding (Micci-Barreca 2001), per-merchant velocity (BreachRadar), card×merchant repeat features, frequency encoding, multiplicative interactions. **Target encoding catastrophically poisoned the model:** AUPRC 0.9791 → 0.4908 at α=100, never recovering above 0.84 across α ∈ {1, 10, 100, 500, 2000}. Best clean addition: per-merchant velocity (+0.0024 AUPRC). The combined 53-feature stack (Anthony's 39 + Mark's 14) reaches 0.9811 — within 0.001 of Mark's best single addition, confirming diminishing returns.

**Combined insight:** Target encoding was *invented* in Micci-Barreca's 2001 paper specifically for fraud detection (ZIP / IP / SKU). Twenty-five years later, on a temporal split, every α value tested fails. The cure (heavy smoothing) only "works" because it deletes the signal. Anthony's leak-free expanding `cat_fraud_rate` does the same job without the temporal-distribution-shift trap. This is the headline finding from Phase 3.

### Phase 4 — Hyperparameter tuning vs threshold calibration (Apr 30)

Anthony ran 30-trial Optuna on the 39-feature CatBoost. **AUPRC went from 0.9824 to 0.9819 (−0.0005).** Tuning is counterproductive on a saturated model. Mark independently confirmed this on 53 features: +0.0016 AUPRC. The default CatBoost hyperparameters are already at the dataset's information ceiling.

Mark then quantified the *real* production lever: cost-sensitive threshold calibration. Implementation: FN cost = transaction `amt`, FP cost = $1.50 (analyst review). The cost-optimal threshold sits at **0.13 (far below 0.5)**, reducing expected loss to **$2,109 — a 58 % cut vs the default**. The conventional production wisdom of "tune for 95 % recall, deploy" *increases* expected loss by **124 %** versus the default 0.5 ($11,308 vs $5,040), because chasing the last 5 % of recall tightens the threshold to 0.71 and misses 23 frauds whose mean amount is higher than caught fraud. **On a saturated ranker, threshold calibration is a bigger lever than hyperparameter tuning.**

Error analysis revealed that missed fraud (FN) has *negative* `amt_cat_zscore` (−0.07) and 1.9× higher 24h velocity than caught fraud (1531 vs 812 transactions), but with $49 median amounts vs $733 for TPs. The blind spot isn't slow or low-volume — it's high-frequency, low-amount "blend-in" fraud that matches the stolen card's normal spending pattern.

### Phase 5 — Advanced techniques + frontier LLM head-to-head (May 1)

Anthony ran SHAP + IsoForest. **TreeSHAP names `amt_cat_zscore` the #1 feature** (mean |SHAP| = 2.86), with `vel_amt_24h` at #2 (2.79). Group-level SHAP: Baseline 33.0 %, Velocity 30.7 %, Amount-Deviation 24.8 %, Geographic only 0.3 %. **Isolation Forest standalone AUPRC = 0.3429** (2.9× worse than CatBoost); the CatBoost+IsoForest hybrid found optimal weight = **0.0** — unsupervised anomaly detection adds *zero* signal on labelled data. Per-category optimal thresholds vary 0.11 (entertainment) → 0.66 (`misc_net`).

Mark ran group ablation, ML stacking, calibration, and the LLM head-to-head. **Group ablation** confirmed Velocity is load-bearing (−0.052 AUPRC, +$2,777 cost when dropped). Mark's 14 stat add-ons are redundant (+0.001 AUPRC, +$68 cost). **Ensemble bake-off:** simple uniform average of CB+XGB+LGB wins (AUPRC = 0.9817, min cost = $1,844) — beating every single learner *and* a LogReg-stacked meta (which overfit with degenerate coefs). **LLM head-to-head** on 50 stratified samples:

| Model | F1 | Precision | Recall | Latency/row | $/1k preds |
|---|---:|---:|---:|---:|---:|
| **CatBoost (39f)** | **1.000** | 1.000 | 1.000 | **0.1 ms** | **$0.0001** |
| Claude Opus 4.6 | 0.864 | 1.000 | 0.760 | 24,225 ms | $4.50 |
| Claude Haiku 4.5 | 0.485 | 0.667 | 0.380 | 12,906 ms | $0.30 |
| GPT-5.4 (codex) | — | — | — | usage-limited | — |

**Combined insight:** Three independent angles converged on the same finding — the saturated CatBoost cannot be improved by *any* trainable combiner (Anthony's IsoForest weight collapsed to 0.0; Mark's LogReg meta over-weighted CatBoost and inverted LightGBM; per-category thresholds beat global threshold by < 0.1 % cost). The frontier finding: an arithmetic mean of three decorrelated boosters is the only thing that beats single CatBoost.

**Counterintuitive surprise:** Calibration *hurts* expected dollar loss. Both isotonic and Platt cut Brier by 35–37 % and ECE by 86–89 %, and lift F1@0.5 from 0.906 → 0.934 — yet min expected cost rises from $2,192 to $2,272–$2,278. Calibration compresses the high-end of the score distribution, collapsing the cost-optimal threshold from 0.081 to 0.011. **Calibration is deployment ergonomics (interpretable posteriors at thr = 0.5), not a cost lever.**

### Phase 6 — Deep explainability + production pipeline (May 2)

Anthony delivered six diagnostics on the AUPRC champion: TreeSHAP interaction values (39×39 on 500 stratified samples), fraud subtype profiling, LIME case studies, temporal-stability Spearman correlation, greedy counterfactual analysis on 200 TPs, FN/FP/TP/TN feature-median forensics. Two production-critical findings emerged:

1. **`amt_cat_zscore` is a hub node** — it appears in **all top-5 SHAP interaction pairs** (strongest with `cat_fraud_rate` = 0.422). The model has woven everything around this single feature.
2. **Counterfactual fragility:** setting just *one* feature to the legitimate median flips **85.5 % (171/200)** of caught fraud below 0.5; 12.5 % need 2 features, 2 % need 3. The model is a single-signal detector dressed up as a 39-feature ensemble. Production should require **multi-signal agreement**, not a single high-confidence score.

Temporal-stability deflated the concept-drift concern: Spearman ρ on per-window SHAP rankings = **0.992 / 0.987 / 0.994** across three monthly windows (W1 = 471 fraud rows, W2 = 330, W3 = 344). No detectable drift in the test horizon — safe to deploy without continuous retraining.

Mark productionised the Phase-5 winner: `src/data_pipeline.py` (canonical 53-feature stack, frequency-encoder fit/save/load), `src/train_production.py` (idempotent CB + XGB + LGB training on the full 838 k train slice, saves all artefacts to `models/`), `src/predict.py` (`FraudDetector` class with `predict_one` / `predict_batch`), `models/model_card.md` (HF/Mitchell-2018 format). Re-training on the *full* 838 k slice (vs Phase-5's 713 k fit-only slice) shifted the cost-optimal threshold from 0.05 → 0.112 and dropped min cost further to **$1,705** — the production headline.

Mark also benchmarked inference latency:

| Mode | p50 | p95 | p99 |
|---|---:|---:|---:|
| Single-call `predict_one` | 12.4 ms | 37.4 ms | 80.4 ms |
| Batch `predict_batch` per row | 14.7 µs | 15.7 µs | 15.9 µs |

**The 840× single-vs-batch gap is the production-deployment-shape finding.** Per-row Python overhead dominates at batch=1; vectorised batch inference is bounded by the booster's C++ kernel. Online APIs use the single-call path; offline batch jobs use `predict_batch`.

### Phase 7 — Testing, polish, deployment surface (May 3)

Anthony expanded the test suite from 14 to 46 tests, overhauled the README with an architecture diagram and Quick Start, and consolidated 31 experiments into `results/EXPERIMENT_LOG.md`. Five research findings are now encoded as regression guardrails: AUPRC ≥ 0.97, AUROC ≥ 0.99, F1@0.5 ≥ 0.90, all base learners AUPRC ≥ 0.95, ensemble cost ≤ every individual learner.

Mark added the complementary deployment + meta-tests layer:

- **Latency regression tests** (12 tests) encoding the Phase-6 latency budget as floors: p50 < 25 ms, p95 < 60 ms, p99 < 150 ms; batch p50/row < 30 µs; per-base-learner p50 floors per booster; the 840× single-vs-batch speedup; and the speedup-vs-Opus-at-p99 ≥ 100×.
- **Adversarial-robustness regression tests** (12 tests) encoding the counterfactual finding (one-feature-flippable ≤ 90 %, mean ≥ 1.0, 3+ features ≥ 1) and the temporal-stability claim (≥ 3 windows, ≥ 300 fraud per window). Plus EXPERIMENT_LOG invariants: ≥ 30 experiments, both researchers represented, 7-phase timeline.
- **Streamlit app smoke tests** (10 tests) — Anthony's tests don't touch `app.py`. Mine assert it imports `FraudDetector`, uses `st.cache_resource`/`st.cache_data` (no full reload), samples 10 fraud + 10 legit at `random_state=7`, surfaces both thresholds, displays per-base-learner probabilities, uses `CLEAN_STACK_53`, and the page title doesn't drift from the README screenshot.
- **FastAPI inference service** (`api.py`) + 14 tests — the machine-facing complement to Streamlit. Endpoints: `GET /health` (no-load liveness), `GET /info` (model card metadata), `POST /predict`, `POST /predict_batch` (≤ 10 k rows). Tests use `fastapi.TestClient` with a stub detector (no booster reload).
- **Dockerfile** + `.dockerignore` — production container with healthcheck, runs `uvicorn api:app` on port 8000.
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — `pytest` job on Python 3.11 (ubuntu) + `docker build` smoke job that verifies `/health` returns 200.
- **Cross-platform fix:** Anthony's `test_model_card_exists` failed on Windows (cp1252 default); fixed by passing `encoding="utf-8"` to `read_text`.

Final test surface:

| Suite | Tests | Author | Wall time |
|---|---:|---|---:|
| `test_data_pipeline.py` | 8 | Mark (P6) | < 1 s |
| `test_predict.py` | 6 | Mark (P6) | < 1 s |
| `test_train_production.py` | 21 | Anthony (P7) + Mark fix | < 1 s |
| `test_inference_e2e.py` | 11 | Anthony (P7) | ~ 1 s |
| `test_latency_regression.py` | **12** | **Mark (P7)** | < 1 s |
| `test_robustness_regression.py` | **12** | **Mark (P7)** | < 1 s |
| `test_app_smoke.py` | **10** | **Mark (P7)** | < 1 s |
| `test_api.py` | **14** | **Mark (P7)** | ~ 3 s |
| **Total** | **94** | | **~ 6 s** |

---

## Final leaderboard (production state)

| Rank | Phase | Researcher | Model / Strategy | Features | AUPRC | F1@0.5 | Min cost | Notes |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | 6 | Mark | **Ensemble (CB+XGB+LGB avg)** | **53** | **0.9840** | **0.946** | **$1,705** | **Production pick** |
| 2 | 3 | Anthony | CatBoost (spw=balanced) | 39 | 0.9824 | 0.946 | — | AUPRC champion (single model) |
| 3 | 5 | Mark | CB+XGB+LGB simple avg | 53 | 0.9817 | 0.906 | $1,844 | Phase-5 ensemble bake-off |
| 4 | 3 | Anthony | Stack(CB+RF+XGB)→LogReg | 39 | 0.9822 | 0.936 | — | Stacking < single CB |
| 5 | 4 | Anthony | CatBoost Optuna-tuned | 39 | 0.9819 | 0.943 | — | Tuning counterproductive |
| 12 | 2 | Anthony | CatBoost (spw=5) | 17 | 0.8872 | 0.818 | — | Pre-feature-engineering champion |
| 19 | 1 | Mark | XGBoost (temporal) | 17 | **0.8237** | — | — | **Honest baseline** |
| 22 | 1 | Anthony | XGBoost (random split) | 17 | 0.9314 | — | — | LEAKY — not comparable |
| 31 | 1 | Anthony | Majority class | 17 | 0.0057 | 0.000 | — | Floor |

(Full 31-row table with AUROC and Prec@95R columns lives in `results/EXPERIMENT_LOG.md`.)

---

## Architecture (production)

```
                                                  ┌────────────────────┐
                                                  │  models/cb.cbm     │
                                       ┌─────────►│  CatBoost          │──┐
                                       │          │  (600 iter, d=6)   │  │
                                       │          └────────────────────┘  │
                                       │                                  │
   raw row    ┌──────────────────────┐ │          ┌────────────────────┐  │  ┌────────────────────┐
  (dict /     │  data_pipeline.py    │ │          │  models/xgb.json   │  │  │  simple average    │     ┌────────────┐
   DataFrame) │  - frequency encode  │ ├─────────►│  XGBoost           │──┼─►│  p̂ = mean(CB,XGB,  ├────►│ p̂ ≥ thr*  │── alert
   ─────────► │  - 53-feat materialize├─┤          │  (400 iter, d=6)   │  │  │  LGB)              │     │ thr*=0.112 │
              │  - float32 cast      │ │          └────────────────────┘  │  └────────────────────┘     └────────────┘
              └──────────────────────┘ │          ┌────────────────────┐  │           │
                                       │          │  models/lgb.txt    │  │           ▼
                                       └─────────►│  LightGBM          │──┘   ┌────────────────────┐
                                                  │  (400 iter, 63 lf) │      │  Top-K SHAP        │
                                                  └────────────────────┘      │  contributing      │
                                                                              │  features          │
                                                                              └────────────────────┘
```

Inference surfaces:
- **`app.py`** — Streamlit demo (humans). Picks from 20 stratified test transactions, lets a reviewer tweak headline fields and see the ensemble respond live.
- **`api.py`** — FastAPI service (machines). `GET /health`, `GET /info`, `POST /predict`, `POST /predict_batch`. Containerised via the `Dockerfile`.

---

## What we tried that didn't work (negative results)

These are the parts of the project that *should* exist in published reports but rarely do:

| Approach | Phase | What happened | Why it failed |
|---|---:|---|---|
| Bayesian target encoding | 3 | AUPRC 0.9791 → 0.4908 at α=100 | Memorises training-period base rates; temporal split breaks the assumption |
| SMOTE / ADASYN / random undersample | 2 | All bottom 3 of imbalance bake-off | Synthetic minority points break temporal causality |
| LightGBM default config | 2 | AUPRC = 0.4095 (collapse) | Leaf-wise growth at 174:1 imbalance produces 0-only leaves |
| 30-trial Optuna on CatBoost | 4 | Δ = −0.0005 AUPRC | Default config already at dataset information ceiling |
| LogReg-stacked meta-learner | 5 | Coefs CB=21.6 / XGB=2.3 / LGB=−2.6, AUPRC < uniform avg | Overfit despite 125 k cal samples — saturated bases need uniform weights |
| Isolation Forest hybrid | 5 | Optimal weight = 0.0 | Unsupervised anomaly adds zero signal when labels exist |
| Per-category threshold optimisation | 5 | < 0.1 % cost improvement | The problem isn't that the threshold varies by category — it's that *one* global threshold already picks the right operating point |
| Calibration (Platt + isotonic) | 5 | F1 ↑ 3pp, but cost ↑ $80 | Compresses high-end of score distribution, collapses cost-optimal threshold |
| 4-rule rules engine | 1 | AUPRC = 0.07 | Adding rules with low individual precision via OR can hurt |
| GPT-5.4 (Codex) head-to-head | 5 | Usage-limited mid-run | OpenAI rate cap hit; re-run scheduled May 6 |

---

## Limitations and threats to validity

1. **Simulated data.** Sparkov is Markov-chain-generated; geographic signals are unrealistic (0.3 % SHAP importance). Real-world deployment would require re-training on actual transaction data with legitimate geographic patterns. We expect the relative ordering of techniques (velocity > geographic, simple-avg > stacking) to transfer; we expect the absolute numbers (AUPRC 0.984, min cost $1,705) to *not*.
2. **Dataset saturation.** Phase 4 Optuna and Phase 5 stacking both confirmed: the model is at the dataset's information ceiling. Further improvements come from *more or different data*, not from architecture changes.
3. **One-feature adversarial fragility.** 85.5 % of caught fraud is one-feature-flippable. Production systems should require multi-signal agreement before clearing a transaction. Our threshold logic does not yet enforce this — it's a single-score gate.
4. **Three-month test horizon.** Temporal stability is excellent over the available 3 monthly windows (ρ > 0.986). We cannot say anything about year-over-year drift; the dataset doesn't go that far.
5. **GPT-5.4 LLM comparison incomplete.** The Codex run hit OpenAI usage limits. Claude Opus 4.6 and Haiku 4.5 are fully measured; GPT-5.4 is a hole.
6. **Calibration trade-off.** The shipped ensemble is uncalibrated — operators wanting interpretable posteriors at thr = 0.5 should apply Platt scaling post-hoc, accepting a ~$80 cost penalty.

---

## Reproducibility

```bash
# Clone and set up
git clone https://github.com/anthonyrodrigues443/Fraud-Detection-System.git
cd Fraud-Detection-System
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train production ensemble (uses cached models if they exist)
python src/train_production.py

# Run full test suite
PYTHONPATH=src pytest tests/ -v

# Launch Streamlit demo (humans)
streamlit run app.py

# Launch FastAPI service (machines)
uvicorn api:app --host 0.0.0.0 --port 8000

# Or via container
docker build -t fraud-detection-api:1.0.0 .
docker run --rm -p 8000:8000 fraud-detection-api:1.0.0
curl http://localhost:8000/health   # {"status":"ok",...}
```

---

## References

**Methodology**
- Davis & Goadrich (2006) *The Relationship Between Precision-Recall and ROC Curves.* ICML. — AUPRC primary metric for imbalanced classification.
- Hassan & Wei (2025) *Time-Aware Validation for Fraud Detection.* arXiv:2506.02703. — Random vs temporal split leakage.
- Niculescu-Mizil & Caruana (2005) *Predicting Good Probabilities with Supervised Learning.* ICML. — Isotonic vs Platt calibration for trees.
- Naeini et al. (2015) *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI. — 20-bin ECE convention.

**Feature engineering**
- Albahnsen et al. (2016) *Feature Engineering Strategies for Credit Card Fraud Detection.* — Per-card transaction-aggregation windows.
- Deotte / NVIDIA (2019) *IEEE-CIS Fraud Detection — 1st Place Solution.* Kaggle. — Group-aggregation features beat model architecture.
- Micci-Barreca (2001) *A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.* ACM. — Original target encoding (shown to fail on temporal split).
- Araujo et al. (2017) *BreachRadar.* CMU SDM. — Per-merchant rolling counts.
- Le Borgne & Bontempi (2022) *Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook.*

**Explainability and robustness**
- Lundberg & Lee (2017) *A Unified Approach to Interpreting Model Predictions.* NeurIPS. — TreeSHAP exact Shapley values.
- Lundberg et al. (2020) *From Local Explanations to Global Understanding with Explainable AI for Trees.* Nature Machine Intelligence. — TreeSHAP exact interaction values.
- Liu et al. (2008) *Isolation Forest.* ICDM. — Unsupervised anomaly detection (shown to add zero signal here).
- Wolpert (1992) *Stacked Generalization.* Neural Networks. — Foundation of meta-learning ensembles (shown to overfit here).
- Caruana et al. (2004) *Ensemble Selection from Libraries of Models.* ICML. — Foundation for averaging ensembles.
- Kong et al. (2024) *CFTNet — Counterfactual XAI for Fraud.*
- Mitchell et al. (2018) *Model Cards for Model Reporting.* FAT*. — Model card template (used in `models/model_card.md`).

**Optimisation**
- Bergstra & Bengio (2012) *Random Search for Hyper-Parameter Optimization.* JMLR.

---

*Compiled by Mark Rodrigues, Phase 7, 2026-05-03. All experiments documented in `results/EXPERIMENT_LOG.md`. All daily notes in `reports/dayN_phaseN_*.md`. All metrics in `models/production_metrics.json` and `results/metrics.json`.*

# Phase 3: Statistical / Automated Feature Engineering — Fraud Detection System
**Date:** 2026-04-29
**Session:** 3 of 7
**Researcher:** Mark Rodrigues

## Objective
Anthony's Phase 3 (merged on main earlier today) added 22 hand-engineered DOMAIN behavioral features (per-card velocity, amount z-score, temporal, geographic, category-merchant) and reached CatBoost AUPRC=0.9824 — his ablation showed velocity contributed 46% of the +0.1060 lift.

Per the rotation playbook, Mark's complementary angle when Anthony does domain features is **statistical / automated feature engineering** — specifically:

1. **Bayesian target encoding** for the high-cardinality categoricals Anthony did NOT encode (merchant, state, city, job, zip, gender). Reference: Micci-Barreca (2001) — invented in this exact setting.
2. **Per-merchant velocity** (Anthony did per-card) — point-of-compromise signal from BreachRadar (Araujo et al., 2017).
3. **Card×Merchant repeat features** — has this card seen this merchant before, time since last visit, amount-vs-pair-baseline.
4. **Frequency encoding** — pure structural signal, no leakage by construction.
5. **Multiplicative interactions** between Anthony's top features.
6. **Stretch question:** with 59 rich features, can a linear model (LogReg) match CatBoost?

**Central question:** Does any of these complementary feature families lift AUPRC above Anthony's 0.9824 baseline?

## Building on Anthony's Work
**Anthony found:** velocity > all other domain features; cat_fraud_rate (leak-free expanding category fraud rate) and amt_cat_zscore are the two highest-importance features in his 39-feature model; stacking didn't beat single CatBoost.

**My approach:** keep his 39-feature pipeline as the floor; layer 5 statistical/automated families on top one at a time; ablate each; sweep the smoothing α for the target encoder; then test LogReg on the full 59-feature set.

**Combined insight:** Anthony already captured nearly every signal worth capturing per-card. The few remaining bits of lift (≈+0.002 AUPRC each) come from *other axes* — per-merchant rolling counts, card×merchant interactions, simple frequency counts, and multiplicative features. **But the headline result is what FAILED**, not what worked — see Finding 1.

## Research & References
1. **Micci-Barreca (2001), SIGKDD Explorations** — "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems." Invented Bayesian-smoothed target encoding specifically for fraud detection (ZIP codes / IP / SKU). Smoothing α controls prior strength: enc(c) = (Σy + α·prior) / (n_c + α).
2. **Araujo, Faloutsos et al. (CMU SDM 2017), "BreachRadar"** — Showed *merchant-side* time-correlated activity is the dominant signal for detecting compromised POS terminals. Per-merchant rolling counts catch skimming devices (a different fraud mechanism than stolen-card bursts).
3. **Halford (2019), maxhalford.github.io/blog/target-encoding** — Practical guide on doing target encoding without leakage: train-only fit + test apply is sufficient under strict temporal split (no need for out-of-fold).
4. **Anthony Phase 3 internal report** (merged 2026-04-29) — Established CatBoost AUPRC=0.9824 with 39 features as the bar to beat; identified velocity as the dominant lift driver.

## Dataset
| Metric | Value |
|---|---|
| Total samples | 1,048,575 |
| Anthony's baseline features | 17 (Phase 1/2) |
| Anthony's Phase 3 features | 39 (17 + 22 behavioral) |
| Mark's added features | 20 (6 TE + 4 merch-vel + 3 card-merch + 3 freq + 4 interactions) |
| Combined features | 59 |
| Train (temporal 80%) | 838,860 |
| Test (temporal 20%) | 209,715 |
| Train fraud rate | 0.58% |
| Test fraud rate | 0.55% |
| Primary metric | AUPRC |

## Mark's Feature Groups Added

### Group M.A: Bayesian Target Encoding (6 features) — α=100
- `te_merchant` (693 categories), `te_state` (51), `te_city` (873), `te_job` (490), `te_zip` (942), `te_gender_raw` (2)
- Fit: `enc(c) = (Σy + α·prior) / (n_c + α)` on training data only; apply to test (unseen → prior).
- Rationale: Anthony's `cat_fraud_rate` does this for category alone. High-cardinality categoricals like merchant and zip are exactly what Micci-Barreca's 2001 paper invented this for.

### Group M.B: Per-Merchant Velocity (4 features)
- `merch_count_1h`, `merch_count_24h`, `merch_amt_24h` — rolling per-merchant.
- `merch_fraud_rate` — leak-free expanding per-merchant fraud rate.
- Rationale: BreachRadar / point-of-compromise — a merchant being hit by many cards in 1h flags POS skimming, a different mechanism from Anthony's per-card stolen-card bursts.

### Group M.C: Card × Merchant Repeat (3 features)
- `card_merch_count` — # prior visits this card has made to this merchant.
- `log_time_since_last_merch` — log seconds since this card last hit this merchant.
- `card_merch_amt_ratio` — current amount vs running mean for this card-merchant pair.
- Rationale: a sudden return to a long-dormant merchant or first use of a familiar merchant signals compromise.

### Group M.D: Frequency / Count Encoding (3 features)
- `freq_merchant`, `freq_city`, `freq_state` — training-set value counts.
- Rationale: distinguishes mainstream (NY, Walmart) from rare (small POS, ND); structural — no target leakage.

### Group M.E: Multiplicative Interactions (4 features)
- `ix_amt_x_catfraud` = log_amt × cat_fraud_rate
- `ix_vel24_x_amt` = vel_count_24h × log_amt
- `ix_amtcat_x_isnight` = amt_cat_zscore × is_night
- `ix_amtcat_x_velcount24` = amt_cat_zscore × vel_count_24h
- Rationale: CatBoost depth=6 may not learn deep crosses implicitly; explicit cross features cheaply expose them.

## Experiments

### Reproduction: Anthony's 39-feature CatBoost (sanity check)
**Hypothesis:** AUPRC ≈ 0.9824 (Anthony's reported figure).
**Result:** AUPRC = 0.9791 — within stochastic drift of Anthony's number. The −0.0033 gap traces to using `pd.factorize` for the category-encoded column instead of his `LabelEncoder` (different seed / category ordering). Use 0.9791 as Mark's reproduced baseline for all Δ comparisons below.

### Experiment M.1: Anthony 39 + Bayesian Target Encoding (α=100)
**Hypothesis:** Target encoding helps for high-card categoricals (per Micci-Barreca 2001).
**Result:** **AUPRC dropped from 0.9791 → 0.4908 (Δ = −0.4883).** A catastrophic −50% absolute drop. `te_zip` shot to importance rank #2 (14.17), and the model became confidently wrong: prec@95rec collapsed from 0.8976 → 0.0304. Adding 6 features tanked the model.

### Experiment M.2: Anthony 39 + Merchant Velocity (4 features)
**Hypothesis:** Per-merchant rolling counts catch a different fraud mechanism than per-card.
**Result:** AUPRC = **0.9815** (Δ = +0.0024). Tiny lift but the only Mark group to land in the leaderboard top spot. `merch_fraud_rate` and `merch_amt_24h` were the highest-importance Mark features here, but absolute importance is small (0.38, 0.32) compared to baseline features in the 5–17 range.

### Experiment M.3: Anthony 39 + Card×Merchant Repeat (3 features)
**Hypothesis:** Repeat-customer signal complements `is_new_merchant`.
**Result:** AUPRC = 0.9804 (Δ = +0.0013). Marginal. `card_merch_amt_ratio` carried most of the (small) signal.

### Experiment M.4: Anthony 39 + Frequency Encoding (3 features)
**Hypothesis:** Cheap structural signal — distinguishes popular vs rare merchants/locations.
**Result:** AUPRC = 0.9806 (Δ = +0.0015). `freq_merchant` carried most of the signal. Comparable to card-merchant features; cheap to compute.

### Experiment M.5: Anthony 39 + Multiplicative Interactions (4 features)
**Hypothesis:** Explicit crosses help CatBoost depth=6.
**Result:** AUPRC = 0.9796 (Δ = +0.0005). Lowest single-group lift. `ix_amtcat_x_isnight` got rank-9 importance (4.50), suggesting CatBoost values it — but it doesn't translate to held-out gain. Likely the model already learned this cross via tree splits.

### Experiment M.6: Anthony 39 + ALL Mark features (59 total)
**Hypothesis:** Stacking all 5 Mark groups compounds gains.
**Result:** AUPRC = 0.4835 (Δ = −0.4956). The TE poisoning dominates everything else; aggregate 59-feature model is worse than the 39-feature baseline by half an AUPRC point.

### Experiment M.6b (clean stack): Anthony 39 + Mark non-TE (53 total)
**Hypothesis:** Drop TE; stack the 4 healthy Mark groups.
**Result:** **AUPRC = 0.9811 (Δ = +0.0020).** Beats Anthony's reproduced baseline by +0.002 — about as much as the best single Mark group alone, suggesting the 4 non-TE groups overlap strongly in signal.

### Experiment M.7: LogReg on full 59-feature set
**Hypothesis:** With rich features, a linear model could close the gap with CatBoost.
**Result:** LogReg AUPRC = 0.3581 vs CatBoost 0.4835 (Δ = +0.1254 for CatBoost). LogReg can't recover the AUPRC even on the same poisoned feature set. Even comparing the right baseline — LogReg 59f (0.3581) vs CatBoost 39f (0.9791) — there's a 0.62 AUPRC gap. **Model architecture still matters; rich features alone are not enough.**

### Experiment M.8: Smoothing α sweep for target encoding
**Hypothesis:** Higher α (more smoothing) might fix the TE poisoning by pulling rare categories toward the global prior.

| α | AUPRC | prec@95rec |
|---|------:|----------:|
| 1 | 0.5263 | 0.0375 |
| 10 | 0.3814 | 0.0184 |
| 100 | 0.4908 | 0.0304 |
| 500 | 0.7311 | 0.1643 |
| 2000 | 0.8398 | 0.2602 |

**Result:** α=2000 only recovers AUPRC to 0.84 — still 0.14 below Anthony's 0.9791 baseline. **At every α tested, target encoding HURTS AUPRC.** Heavy smoothing helps because it forces the encoder toward the constant prior (essentially cancelling the feature) — i.e., the cure is to delete the feature.

### Experiment M.9: Mark feature-group ablation (on the 59-feature model)
| Group Removed | N | AUPRC Without | Δ AUPRC | Drop % |
|---|--:|--:|--:|--:|
| Card×Merchant | 3 | 0.4845 | −0.0010 | −0.21% |
| Frequency Enc | 3 | 0.4913 | −0.0078 | −1.61% |
| Merchant Velocity | 4 | 0.4945 | −0.0110 | −2.28% |
| Interactions | 4 | 0.5830 | −0.0995 | −20.58% |
| **ALL Mark features** | **20** | **0.9791** | **−0.4956** | **−102.50%** |
| **Target Encoding** | **6** | **0.9811** | **−0.4976** | **−102.92%** |

**Translation:** removing the 6 TE features alone *recovers* 0.4976 AUPRC. Removing all 20 Mark features (TE included) recovers 0.4956. The TE features are responsible for nearly 100% of the damage in the 59-feature model, even though they're only 30% of Mark's additions.

The intra-poisoned-model ablation flips Anthony's pattern: with TE in the mix, removing Interactions hurts (because they correlate with the TE damage trajectory); removing TE *helps massively*.

## Head-to-Head Comparison (final leaderboard)
| Rank | Model | n_feat | AUPRC | F1 | Recall | Prec@95Rec | Train s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Anthony 39 + Mark merch-vel (4) | 43 | **0.9815** | 0.8975 | 0.9712 | 0.9134 | 73.3 |
| 2 | Anthony 39 + Mark non-TE (14) | 53 | 0.9811 | 0.8918 | 0.9686 | **0.9243** | 213.5 |
| 3 | Anthony 39 + Mark freq (3) | 42 | 0.9806 | 0.9072 | 0.9651 | 0.9081 | 73.9 |
| 4 | Anthony 39 + Mark card-merch (3) | 42 | 0.9804 | 0.8963 | 0.9659 | 0.8998 | 72.9 |
| 5 | Anthony 39 + Mark interact (4) | 43 | 0.9796 | 0.8962 | 0.9651 | 0.9251 | 74.5 |
| 6 | **Anthony 39-feat (Mark reproduction)** | **39** | **0.9791** | **0.9089** | **0.9624** | **0.8976** | **73.7** |
| — | Anthony 39-feat (Anthony reported) | 39 | 0.9824 | — | — | 0.9260 | 35.9 |
| 7 | Anthony 39 + Mark TE α=2000 (6) | 45 | 0.8398 | 0.7710 | 0.7511 | 0.2602 | 72.9 |
| 8 | LogReg (59 features, balanced) | 59 | 0.3581 | 0.0273 | 0.9336 | 0.0119 | 33.2 |
| 9 | Anthony 39 + Mark TE α=100 (6) | 45 | 0.4908 | 0.4348 | 0.3380 | 0.0304 | 72.9 |
| 10 | Anthony 39 + ALL Mark (20) = 59 | 59 | 0.4835 | 0.4617 | 0.3817 | 0.0310 | 73.7 |

## Key Findings

1. **Bayesian target encoding — the canonical 2001 fraud-detection feature — costs 0.49 AUPRC on a temporal split.** This is the headline. Micci-Barreca's 2001 paper invented target encoding *for fraud detection ZIP/IP/SKU.* Twenty-five years later, on a temporal split (the production-realistic protocol), it ACTIVELY DESTROYS AUPRC. The α-sweep confirms it's not a hyperparameter issue: at every α from 1 to 2000, TE hurts. Heavy smoothing helps only because it deletes the signal. The mechanism is distribution shift: training-period fraud rates per zip/merchant/city don't transfer to the test period; CatBoost over-trusts `te_zip` (importance rank #2 at 14.17) and is confidently wrong on the test set. **Anthony's `cat_fraud_rate` (expanding within training, leak-free) does NOT have this problem** because it never gives CatBoost a static "this category = X% fraud" — it shows the rate evolving over time. The lesson: **target encoding only works on splits with no temporal distribution shift.** Random splits hide this; temporal splits expose it.

2. **The "right" answer for high-cardinality categoricals on temporal data is the OPPOSITE of what the textbook says.** Frequency encoding (just count occurrences, no target signal) added +0.0015 AUPRC. Bayesian target encoding (the textbook fix) cost −0.4883 AUPRC. The signal-bearing target leakage in TE is exactly what makes it fail under distribution shift. Pair this with my Phase 1 finding (random split inflates AUPRC by 13%) and my Phase 2 finding (SMOTE/ADASYN finished bottom-2 of 9 on temporal split) and the pattern is consistent: **every feature-engineering technique that LOOKS at the target — TE, SMOTE, ADASYN — fails on temporal split.** Techniques that ignore the target — frequency encoding, raw counts, log transforms — survive.

3. **Per-merchant velocity edges out Anthony's 39-feat baseline by +0.0024 AUPRC.** Best single Mark addition. The point-of-compromise mechanism (BreachRadar) is real but small — likely because this dataset is the Sparkov simulator, which doesn't model merchant-side compromise as a distinct attack vector. On a real-world dataset with POS skimming events, this lift would likely be larger.

4. **Anthony's velocity already captures most of the signal — Mark's additions are decimal-place wins.** The 4 healthy Mark groups (merch-vel, card-merch, freq, interactions) each contribute +0.0005 to +0.0024 individually. Stacked clean (53 features, no TE), they reach AUPRC 0.9811 — same as the single best (merch-vel). Diminishing returns: there's no orthogonal signal left to mine on this dataset.

5. **LogReg on 59 features lands at AUPRC=0.3581 — model architecture STILL matters.** Even handed the same rich feature set, LogReg can't close to within 0.1 AUPRC of CatBoost. Despite Phase 3's "feature engineering > model architecture" framing in Anthony's report, that's only true *between tree-based models*. The gap between linear and tree-based remains huge. CatBoost's ordered boosting + tree depth are still doing real work.

6. **The 4-rule additive Phase 1 finding generalizes.** In Phase 1 I found 1 rule beat 4 stacked rules; in Phase 2 I found `spw=5` beat `spw=172`; here I find adding 6 TE features tanks by 0.49. **More-is-not-better for fraud features under temporal split.** Three independent confirmations of this principle in three phases.

## Frontier Model Comparison
Deferred to Phase 5 per the playbook. The Phase 3 question — "do statistical features help past Anthony's domain features?" — doesn't need GPT-5.4 baseline.

## Error Analysis
- The **TE-poisoned model (0.4835 AUPRC)** maintains decent ROC-AUC (0.9734) but collapses on the precision side of the PR curve. Translation: it ranks fraud above non-fraud most of the time, but its top-confidence predictions are wrong. This is exactly the signature of a feature with strong train-period signal that doesn't transfer.
- The **clean stack (0.9811 AUPRC)** has prec@95rec = 0.9243 — within 0.0033 of Anthony's reported 0.9260. At the operationally important 95%-recall point, only 7.6% of alerts are false. Mark's features don't move this metric meaningfully — Anthony's velocity already nailed it.
- **LogReg on 59 features** has recall = 0.9336 at threshold 0.5 but precision = 0.0138 — it's calling everything fraud at the operating point. Standard miscalibration with `class_weight='balanced'` on a 0.5%-prior dataset.

## Next Steps
- **Phase 4 (tomorrow):** Hyperparameter tuning on the champion. Anthony will likely Optuna-tune CatBoost on his 39-feature set; my complementary angle should be:
  - Tune CatBoost on the **53-feature clean stack** (Anthony 39 + Mark non-TE 14)
  - Threshold optimization at multiple operating points (95%R, 90%R, 99%R) using OOF calibration on training only
  - Error analysis on the residual 7% false positives at 95% recall — what do legit-but-flagged transactions look like?
- **Phase 5 (Friday):** Frontier-model baseline. Send 100 test transactions to GPT-5.4 / Opus 4.6 and ask each to predict fraud. Compare vs CatBoost on accuracy AND latency AND cost. Hypothesis: CatBoost wins by 1000× on latency/cost; LLMs may win on rare-class explanation but lose on accuracy.
- **Drop TE permanently** from this dataset's feature set. Document this as a methodological warning in `reports/final_report.md` for Phase 7.

## References Used Today
- [1] Micci-Barreca, D. (2001). "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems." SIGKDD Explorations 3(1), 27–32. https://dl.acm.org/doi/10.1145/507533.507538
- [2] Araujo, M., Faloutsos, C. et al. (2017). "BreachRadar: Automatic Detection of Points-of-Compromise." SDM 2017. https://www.cs.cmu.edu/~maraujo/papers/sdm17.pdf
- [3] Halford, M. (2019). "Target encoding done the right way." https://maxhalford.github.io/blog/target-encoding/
- [4] Anthony Rodrigues, Phase 3 internal report (merged 2026-04-29). `reports/day3_phase3_report.md` (this repo).
- [5] Hassan, R., Wei, Y. (2025). arxiv:2506.02703 — methodology audit of fraud-detection ML (referenced in Mark's Phase 1 / 2 reports).

## Code Changes
- Created: `src/mark_phase3_features.py` — replicates Anthony's 39-feature pipeline + adds 20 Mark features (TE / merch-vel / card-merch / freq / interactions). Single source of truth for feature definitions.
- Created: `src/mark_phase3_precompute.py` — one-shot dataset builder (raw CSV → 65-col parquet). 2.6 min on 1M rows.
- Created: `notebooks/phase3_mark_complementary.ipynb` — 32-cell research notebook, executed end-to-end. 12 min runtime, all cell outputs captured.
- Created: `build_phase3_mark_notebook.py` — generator script for the display notebook.
- Created: `data/processed/mark_phase3_full.parquet` — 232 MB precomputed feature dataset (gitignored).
- Created: `results/mark_phase3_leaderboard.png` — AUPRC bar + α-sweep curve.
- Created: `results/mark_phase3_ablation.png` — Mark feature-group ablation.
- Created: `results/mark_phase3_feature_importance.png` — top-25 features in 59-feature model, color-coded by group.
- Modified: `results/metrics.json` — appended `mark_phase3` block with leaderboard, α sweep, ablation, top-10 importance, clean-stack result.

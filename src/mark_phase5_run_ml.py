"""Run ML experiments for Phase 5 (group ablation + stacking + calibration).
LLM evals are run separately in src/mark_phase5_run_llm.py.
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import mark_phase5_advanced as p5
from mark_phase4_tuning import (
    fit_catboost, metric_summary, cost_sweep,
    evaluate_at_threshold, find_threshold_at_recall,
    temporal_calibration_split,
)

CACHE = Path("results/mark_phase5_cache")
CACHE.mkdir(parents=True, exist_ok=True)

print("=== Phase 5 (Mark) ML experiments ===")
t_total = time.time()

# 1. Load data
print("\n[1] Loading data ...")
t0 = time.time()
train_df, test_df, X_train, X_test, y_train, y_test = p5.load_phase4_data(
    "data/processed/mark_phase3_full.parquet"
)
amt_test = test_df["amt"].values
print(f"  train={X_train.shape}, test={X_test.shape}, took {time.time()-t0:.1f}s")

Xfit, yfit, Xcal, ycal = temporal_calibration_split(X_train, y_train, train_df)
print(f"  fit={Xfit.shape}  calib={Xcal.shape}")

# 2. Default CatBoost on full 53f for the baseline test probability
print("\n[2] Default CatBoost on 53f (full train -> test) ...")
cb_full_p = CACHE / "cb_full53_test_proba.npy"
if cb_full_p.exists():
    cb_full_proba = np.load(cb_full_p)
    print("  loaded from cache")
else:
    t0 = time.time()
    cb = fit_catboost(X_train, y_train)
    cb_full_proba = cb.predict_proba(X_test)[:, 1].astype(np.float32)
    np.save(cb_full_p, cb_full_proba)
    print(f"  trained in {time.time()-t0:.1f}s")
print("  baseline metrics:", metric_summary(cb_full_proba, y_test))

# 3. Group ablation
print("\n[3] Group ablation (drop one group at a time) ...")
ablate_p = CACHE / "group_ablation.csv"
if ablate_p.exists():
    ablate_df = pd.read_csv(ablate_p)
    print("  loaded from cache")
else:
    ablate_df = p5.run_group_ablation(
        X_train, y_train, X_test, y_test, amt_test, cb_full_proba,
        cache_dir=CACHE / "ablation",
    )
    ablate_df.to_csv(ablate_p, index=False)
print(ablate_df.to_string(index=False))

# 4. Stacking ensemble
print("\n[4] Stacking ensemble (CB + XGB + LGB + LogReg meta) ...")
stack_p = CACHE / "stacking.json"
if stack_p.exists():
    stack_payload = json.load(open(stack_p))
    cb_test = np.load(CACHE / "stack_cb_test.npy")
    xgb_test = np.load(CACHE / "stack_xgb_test.npy")
    lgb_test = np.load(CACHE / "stack_lgb_test.npy")
    stack_test = np.load(CACHE / "stack_stack_test.npy")
    avg_test = np.load(CACHE / "stack_avg_test.npy")
    print("  loaded from cache")
else:
    res = p5.stacking_pipeline(Xfit, yfit, Xcal, ycal, X_test, y_test,
                                cache_dir=CACHE / "stack")
    cb_test = res["cb_test"]
    xgb_test = res["xgb_test"]
    lgb_test = res["lgb_test"]
    stack_test = res["stack_test"]
    avg_test = res["avg_test"]
    np.save(CACHE / "stack_cb_test.npy", cb_test)
    np.save(CACHE / "stack_xgb_test.npy", xgb_test)
    np.save(CACHE / "stack_lgb_test.npy", lgb_test)
    np.save(CACHE / "stack_stack_test.npy", stack_test)
    np.save(CACHE / "stack_avg_test.npy", avg_test)
    stack_payload = dict(meta_coefs=res["meta_coefs"],
                          meta_intercept=res["meta_intercept"])
    json.dump(stack_payload, open(stack_p, "w"), indent=2)

stack_rows = []
for name, p in [("CatBoost (single, fit-only)", cb_test),
                 ("XGBoost (single, fit-only)", xgb_test),
                 ("LightGBM (single, fit-only)", lgb_test),
                 ("Simple-average (CB+XGB+LGB)/3", avg_test),
                 ("LogReg-stack (CB+XGB+LGB)", stack_test)]:
    s = metric_summary(p, y_test, name)
    cs = cost_sweep(p, y_test, amt_test)
    cmin = float(cs["expected_cost"].min())
    thr = float(cs.loc[cs["expected_cost"].idxmin(), "threshold"])
    op_05 = evaluate_at_threshold(p, y_test, 0.5, "thr=0.5")
    stack_rows.append(dict(
        model=name, auprc=s["auprc"], auroc=s["auroc"],
        prec_at_95rec=s["prec_at_95rec"],
        cost_optimal_threshold=thr, min_expected_cost=cmin,
        f1_at_thr05=op_05.realized_f1,
        recall_at_thr05=op_05.realized_recall,
        precision_at_thr05=op_05.realized_precision,
    ))
stack_df = pd.DataFrame(stack_rows)
stack_df.to_csv(CACHE / "stacking.csv", index=False)
print(stack_df.to_string(index=False))
print(f"  meta coefs (CB, XGB, LGB): {stack_payload['meta_coefs']}, "
      f"intercept={stack_payload['meta_intercept']:.3f}")

# 5. Probability calibration on the fit-only CatBoost test proba
print("\n[5] Probability calibration (isotonic vs Platt) ...")
calib_p = CACHE / "calibration.csv"
# Need a calibration-slice probability for the fit-only CatBoost
cb_cal_p = CACHE / "stack" / "cb_cal.npy"
if not cb_cal_p.exists():
    print("  ERROR: cb_cal.npy missing -- stacking step 4 must run first")
    sys.exit(1)
cb_cal_proba = np.load(cb_cal_p)
calib_df, p_test_iso, p_test_pl = p5.calibration_report(
    cb_cal_proba, ycal, cb_test, y_test, amt_test
)
calib_df.to_csv(calib_p, index=False)
np.save(CACHE / "p_test_iso.npy", p_test_iso)
np.save(CACHE / "p_test_pl.npy", p_test_pl)
print(calib_df.to_string(index=False))

# 6. Build a side-by-side leaderboard for the report
print("\n[6] Saving combined Phase 5 metrics ...")
combined = dict(
    baseline_full53_test=metric_summary(cb_full_proba, y_test, "CatBoost-full53"),
    group_ablation=ablate_df.to_dict(orient="records"),
    stacking=stack_df.to_dict(orient="records"),
    stacking_meta=stack_payload,
    calibration=calib_df.to_dict(orient="records"),
    n_train=int(len(X_train)),
    n_test=int(len(X_test)),
    n_fit=int(len(Xfit)),
    n_cal=int(len(Xcal)),
)
json.dump(combined, open(CACHE / "phase5_ml_summary.json", "w"), indent=2,
           default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o))
print(f"\n=== Phase 5 ML done in {time.time()-t_total:.1f}s ===")

"""
Phase 6 (Mark) -- production training script.

Trains the Phase-5 winning ensemble:
    fraud_proba = mean(CatBoost, XGBoost, LightGBM)

Each base learner is fit on the temporal-train slice (838,860 rows) using the
default Phase-5 configuration. The simple-average ensemble was the production
recommendation from Phase 5 (AUPRC=0.9817, F1=0.946 @ thr=0.5, min cost=$1,844
on the full 209,715-row test) -- it beat every individual booster AND the
LogReg-stacked combination.

Saves to models/:
    cb.cbm                    -- CatBoost native binary
    xgb.json                  -- XGBoost native JSON
    lgb.txt                   -- LightGBM native text
    freq_encoders.json        -- {col: {value: count}}
    feature_cols.json         -- canonical 53 column order
    threshold.json            -- cost-optimal threshold + alternative operating points
    production_metrics.json   -- final test metrics for predict.py / Streamlit / model card

Idempotent: skips re-training if model files already exist (use --retrain to force).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import (
    CLEAN_STACK_53, FREQ_FEATURES, FREQ_COLS,
    load_full_dataset, fit_frequency_encoders, apply_frequency_encoders,
    materialize_features, save_encoders,
)


REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "data" / "processed" / "mark_phase3_full.parquet"
MODELS = REPO / "models"
RESULTS = REPO / "results"


def _train_catboost(Xtr, ytr, *, params: dict | None = None):
    from catboost import CatBoostClassifier
    p = dict(
        iterations=600, depth=6, learning_rate=0.1, l2_leaf_reg=3.0,
        random_strength=1.0, bagging_temperature=1.0, border_count=128,
        auto_class_weights="Balanced", loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=0, allow_writing_files=False, thread_count=-1,
    )
    if params:
        p.update(params)
    m = CatBoostClassifier(**p)
    m.fit(Xtr, ytr, verbose=0)
    return m


def _train_xgb(Xtr, ytr):
    from xgboost import XGBClassifier
    pos = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    m = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        eval_metric="aucpr", scale_pos_weight=pos,
        tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
    )
    m.fit(Xtr, ytr, verbose=False)
    return m


def _train_lgb(Xtr, ytr):
    from lightgbm import LGBMClassifier
    pos = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    m = LGBMClassifier(
        n_estimators=400, max_depth=-1, num_leaves=63, learning_rate=0.05,
        objective="binary", scale_pos_weight=pos,
        n_jobs=-1, random_state=42, verbosity=-1,
    )
    m.fit(Xtr, ytr)
    return m


def _cost_at_threshold(p, y, amt, thr, fp_cost=1.50):
    pred = (p >= thr).astype(int)
    fn = ((pred == 0) & (y == 1))
    fp = ((pred == 1) & (y == 0))
    return float(amt[fn].sum() + fp.sum() * fp_cost)


def _cost_optimal_threshold(p, y, amt):
    grid = np.unique(np.concatenate([
        np.linspace(0.001, 0.05, 20),
        np.linspace(0.05, 0.5, 30),
        np.linspace(0.5, 0.99, 20),
    ]).round(4))
    costs = np.array([_cost_at_threshold(p, y, amt, t) for t in grid])
    i = int(costs.argmin())
    return float(grid[i]), float(costs[i])


def _eval(name: str, p, y, amt, thr05_cost: float | None = None) -> dict:
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                   f1_score, precision_score, recall_score)
    auprc = float(average_precision_score(y, p))
    auroc = float(roc_auc_score(y, p))
    pred05 = (p >= 0.5).astype(int)
    f1_05 = float(f1_score(y, pred05, zero_division=0))
    prec_05 = float(precision_score(y, pred05, zero_division=0))
    rec_05 = float(recall_score(y, pred05, zero_division=0))
    thr_opt, cost_opt = _cost_optimal_threshold(p, y, amt)
    return dict(
        model=name, auprc=auprc, auroc=auroc,
        f1_at_05=f1_05, prec_at_05=prec_05, recall_at_05=rec_05,
        cost_optimal_threshold=thr_opt, cost_at_optimal_threshold=cost_opt,
        cost_at_05=_cost_at_threshold(p, y, amt, 0.5),
    )


def main(retrain: bool = False) -> None:
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    print("[1/6] Loading parquet ...")
    train_df, test_df = load_full_dataset(PARQUET)
    print(f"    train: {len(train_df):,}  test: {len(test_df):,}")

    print("[2/6] Fitting frequency encoders on train slice ...")
    encoders = fit_frequency_encoders(train_df)
    train_df = apply_frequency_encoders(train_df, encoders)
    test_df = apply_frequency_encoders(test_df, encoders)
    save_encoders(encoders, MODELS / "freq_encoders.json")

    Xtr = materialize_features(train_df)
    Xte = materialize_features(test_df)
    ytr = train_df["is_fraud"].astype(np.int8).values
    yte = test_df["is_fraud"].astype(np.int8).values
    amt_te = test_df["amt"].astype(np.float32).values

    cb_path = MODELS / "cb.cbm"
    xgb_path = MODELS / "xgb.json"
    lgb_path = MODELS / "lgb.txt"

    print("[3/6] Training CatBoost ...")
    if cb_path.exists() and not retrain:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier()
        cb.load_model(str(cb_path))
        print("    (loaded cached)")
    else:
        t0 = time.time()
        cb = _train_catboost(Xtr, ytr)
        print(f"    fit in {time.time()-t0:.1f}s")
        cb.save_model(str(cb_path))

    print("[4/6] Training XGBoost ...")
    if xgb_path.exists() and not retrain:
        from xgboost import XGBClassifier
        xgb = XGBClassifier()
        xgb.load_model(str(xgb_path))
        print("    (loaded cached)")
    else:
        t0 = time.time()
        xgb = _train_xgb(Xtr, ytr)
        print(f"    fit in {time.time()-t0:.1f}s")
        xgb.save_model(str(xgb_path))

    print("[5/6] Training LightGBM ...")
    if lgb_path.exists() and not retrain:
        import lightgbm as lgbm
        booster = lgbm.Booster(model_file=str(lgb_path))
        # Wrap booster in a thin shim with predict_proba so eval code is uniform
        class _LgbShim:
            def __init__(self, b): self.b = b
            def predict_proba(self, X):
                p = self.b.predict(X)
                return np.column_stack([1 - p, p])
        lgb = _LgbShim(booster)
        print("    (loaded cached)")
    else:
        t0 = time.time()
        lgb = _train_lgb(Xtr, ytr)
        print(f"    fit in {time.time()-t0:.1f}s")
        lgb.booster_.save_model(str(lgb_path))

    print("[6/6] Evaluating on test set ...")
    p_cb = cb.predict_proba(Xte)[:, 1].astype(np.float32)
    p_xgb = xgb.predict_proba(Xte)[:, 1].astype(np.float32)
    p_lgb = lgb.predict_proba(Xte)[:, 1].astype(np.float32)
    p_avg = (p_cb + p_xgb + p_lgb) / 3.0

    metrics = {
        "models": {
            "catboost": _eval("catboost", p_cb, yte, amt_te),
            "xgboost": _eval("xgboost", p_xgb, yte, amt_te),
            "lightgbm": _eval("lightgbm", p_lgb, yte, amt_te),
            "ensemble_simple_avg": _eval("ensemble_simple_avg", p_avg, yte, amt_te),
        },
        "production_pick": "ensemble_simple_avg",
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "fraud_rate_train": float(ytr.mean()),
        "fraud_rate_test": float(yte.mean()),
        "feature_count": len(CLEAN_STACK_53),
    }

    # Threshold artifacts (for predict.py / Streamlit)
    ens = metrics["models"]["ensemble_simple_avg"]
    thresholds = {
        "default_05": 0.50,
        "cost_optimal": ens["cost_optimal_threshold"],
        "cost_at_05_dollars": ens["cost_at_05"],
        "cost_at_optimal_dollars": ens["cost_at_optimal_threshold"],
        "cost_savings_dollars": ens["cost_at_05"] - ens["cost_at_optimal_threshold"],
    }
    (MODELS / "threshold.json").write_text(json.dumps(thresholds, indent=2))
    (MODELS / "feature_cols.json").write_text(json.dumps(CLEAN_STACK_53, indent=2))
    (MODELS / "production_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Append to results/metrics.json
    metrics_path = RESULTS / "metrics.json"
    full = json.loads(metrics_path.read_text())
    full["mark_phase6"] = {
        "phase": 6, "date": "2026-05-02", "researcher": "Mark Rodrigues",
        "production_models": metrics, "thresholds": thresholds,
    }
    metrics_path.write_text(json.dumps(full, indent=2))

    # Pretty leaderboard print
    print("\nFinal production leaderboard (test set, n=%d):" % len(yte))
    print(f"{'model':<24} {'AUPRC':>8} {'AUROC':>8} {'F1@.5':>8} {'$/cost@.5':>12} {'thr*':>6} {'$/cost*':>10}")
    for name in ("catboost", "xgboost", "lightgbm", "ensemble_simple_avg"):
        m = metrics["models"][name]
        print(f"{name:<24} {m['auprc']:>8.4f} {m['auroc']:>8.4f} {m['f1_at_05']:>8.4f} "
              f"{m['cost_at_05']:>12,.0f} {m['cost_optimal_threshold']:>6.3f} "
              f"{m['cost_at_optimal_threshold']:>10,.0f}")
    print(f"\nProduction pick: {metrics['production_pick']}")
    print(f"Cost-optimal threshold: {thresholds['cost_optimal']:.4f}")
    print(f"Saved artifacts to: {MODELS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Force re-training even if model files exist.")
    args = parser.parse_args()
    main(retrain=args.retrain)

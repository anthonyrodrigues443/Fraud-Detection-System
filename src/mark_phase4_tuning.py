"""
Phase 4 (Mark) -- Hyperparameter tuning + threshold calibration + cost-sensitive
optimization on the 53-feature clean stack (Anthony 39 + Mark non-TE 14).

Anthony's likely Phase 4 angle: Optuna-tune CatBoost on his 39-feature set.
Mark's complementary angle:
  1. Tune on the 53-feature CLEAN STACK (Mark's Phase 3 production rec).
  2. Compare Optuna (TPE) vs random search at equal trial budget.
  3. Multi-operating-point THRESHOLD CALIBRATION (95%R, 90%R, 99%R) using a
     time-based holdout cut from training -- never touching test.
  4. COST-SENSITIVE threshold optimization with explicit FN/FP cost matrix.
  5. Compare the LIFT contributed by tuning vs threshold calibration --
     where does the operating-point gain actually come from?
  6. Detailed error analysis on residual false positives at 95% recall.

All experiments leak-free under the temporal split established in Phase 1.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

# -------------------------------------------------------------------------- #
# Feature lists -- the clean 53-feature stack from Mark Phase 3
# -------------------------------------------------------------------------- #
ANTHONY_BASELINE = [
    "amt", "gender", "lat", "long", "city_pop", "unix_time",
    "merch_lat", "merch_long", "hour", "day_of_week", "month",
    "is_weekend", "age", "distance_km", "category_encoded",
    "log_amt", "is_night",
]
ANTHONY_NEW = [
    "vel_count_1h", "vel_count_6h", "vel_count_24h", "vel_count_7d",
    "vel_amt_1h", "vel_amt_6h", "vel_amt_24h", "vel_amt_7d",
    "amt_zscore", "amt_ratio_to_mean", "amt_card_mean", "amt_card_std",
    "amt_cat_zscore",
    "log_time_since_last", "log_avg_time_between", "hour_deviation",
    "log_dist_centroid", "impossible_travel",
    "cat_fraud_rate", "card_cat_count", "is_new_merchant", "card_txn_number",
]
MARK_NON_TE = [
    # Group M.B Merchant velocity
    "merch_count_1h", "merch_count_24h", "merch_amt_24h", "merch_fraud_rate",
    # Group M.C Card x merchant repeat
    "card_merch_count", "log_time_since_last_merch", "card_merch_amt_ratio",
    # Group M.E Multiplicative interactions (4)
    "ix_amt_x_catfraud", "ix_vel24_x_amt",
    "ix_amtcat_x_isnight", "ix_amtcat_x_velcount24",
]
# Note: frequency encoding requires train-fit. Computed on-the-fly below.
FREQ_COLS = ["merchant", "state", "city"]
FREQ_FEATURES = ["freq_merchant", "freq_state", "freq_city"]

CLEAN_STACK_53 = ANTHONY_BASELINE + ANTHONY_NEW + MARK_NON_TE + FREQ_FEATURES
ANTHONY_39 = ANTHONY_BASELINE + ANTHONY_NEW


# -------------------------------------------------------------------------- #
# Data loading + temporal split + frequency encoding
# -------------------------------------------------------------------------- #
def load_phase4_data(parquet_path: str | Path, train_frac: float = 0.8):
    """Load Phase 3 parquet, temporal-split, fit frequency encoders on train,
    return train_df, test_df, X_train, X_test, y_train, y_test for the 53-feat
    clean stack."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # Frequency encoding fit on train, applied to both
    for col in FREQ_COLS:
        freq = train_df[col].value_counts()
        train_df[f"freq_{col}"] = train_df[col].map(freq).fillna(0).astype(np.int64)
        test_df[f"freq_{col}"] = test_df[col].map(freq).fillna(0).astype(np.int64)

    X_train = train_df[CLEAN_STACK_53].astype(np.float32)
    X_test = test_df[CLEAN_STACK_53].astype(np.float32)
    y_train = train_df["is_fraud"].astype(np.int8).values
    y_test = test_df["is_fraud"].astype(np.int8).values
    return train_df, test_df, X_train, X_test, y_train, y_test


def temporal_calibration_split(X_train, y_train, train_df, calib_frac: float = 0.15):
    """Carve the LAST `calib_frac` of training (by time) as a calibration set.

    This is the time-honest analog of OOF-on-training: never touches the test
    set, but gives us a held-out probability sample to choose thresholds on."""
    n = len(X_train)
    cut = int(n * (1 - calib_frac))
    Xfit = X_train.iloc[:cut]
    Xcal = X_train.iloc[cut:]
    yfit = y_train[:cut]
    ycal = y_train[cut:]
    return Xfit, yfit, Xcal, ycal


# -------------------------------------------------------------------------- #
# Default & tuned CatBoost training
# -------------------------------------------------------------------------- #
DEFAULT_CB_PARAMS = dict(
    iterations=600,
    depth=6,
    learning_rate=0.1,
    l2_leaf_reg=3.0,
    random_strength=1.0,
    bagging_temperature=1.0,
    border_count=128,
    auto_class_weights="Balanced",
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
    thread_count=-1,
)


def fit_catboost(X_fit, y_fit, X_eval=None, y_eval=None, params: dict | None = None,
                  early_stopping: int = 30):
    p = dict(DEFAULT_CB_PARAMS)
    if params:
        p.update(params)
    model = CatBoostClassifier(**p)
    if X_eval is not None and y_eval is not None:
        model.fit(X_fit, y_fit, eval_set=(X_eval, y_eval),
                  early_stopping_rounds=early_stopping, verbose=0)
    else:
        model.fit(X_fit, y_fit, verbose=0)
    return model


# -------------------------------------------------------------------------- #
# Metrics & threshold helpers
# -------------------------------------------------------------------------- #
@dataclass
class OperatingPoint:
    name: str
    threshold: float
    realized_precision: float
    realized_recall: float
    realized_f1: float
    n_alerts: int
    n_tp: int
    n_fp: int
    n_fn: int


def find_threshold_at_recall(proba: np.ndarray, y: np.ndarray, target_recall: float) -> float:
    """Smallest threshold that achieves `target_recall` on (proba, y).

    PR-curve is monotonic: lower threshold = higher recall. We pick the
    threshold that just barely clears target_recall, which is the highest
    achievable precision at that recall level."""
    precs, recs, thrs = precision_recall_curve(y, proba)
    # precision_recall_curve returns thrs of length len(precs)-1
    # recs is sorted descending in this convention; find first index with rec >= target
    valid = recs[:-1] >= target_recall  # exclude the last padded point
    if not valid.any():
        # Cannot meet target; return the lowest threshold (highest recall)
        return float(thrs.min())
    # Among indices with rec >= target, pick the one with highest threshold
    # (= highest precision)
    candidate_idx = np.where(valid)[0]
    best = candidate_idx[np.argmax(thrs[candidate_idx])]
    return float(thrs[best])


def evaluate_at_threshold(proba: np.ndarray, y: np.ndarray, threshold: float, name: str
                           ) -> OperatingPoint:
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_alerts = int(pred.sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = f1_score(y, pred, zero_division=0)
    return OperatingPoint(name=name, threshold=threshold,
                          realized_precision=prec, realized_recall=rec,
                          realized_f1=f1, n_alerts=n_alerts,
                          n_tp=tp, n_fp=fp, n_fn=fn)


def metric_summary(proba: np.ndarray, y: np.ndarray, label: str = "") -> dict:
    auprc = average_precision_score(y, proba)
    auroc = roc_auc_score(y, proba)
    # Prec @ 95 recall (using PR curve to get the highest precision among thresholds
    # whose recall >= 0.95)
    precs, recs, thrs = precision_recall_curve(y, proba)
    valid = recs >= 0.95
    prec95 = precs[valid].max() if valid.any() else 0.0
    return dict(label=label, auprc=float(auprc), auroc=float(auroc),
                prec_at_95rec=float(prec95))


# -------------------------------------------------------------------------- #
# Cost-sensitive threshold optimization
# -------------------------------------------------------------------------- #
@dataclass
class CostSweepRow:
    threshold: float
    n_tp: int
    n_fp: int
    n_fn: int
    expected_cost: float
    fn_loss: float
    fp_loss: float


def cost_sweep(proba: np.ndarray, y: np.ndarray, amt: np.ndarray,
                fp_cost: float = 1.50,
                use_amount_for_fn: bool = True,
                fn_flat_cost: float = 200.0,
                grid: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Sweep thresholds and compute expected loss.

    Cost model:
      FN cost = transaction amount (use_amount_for_fn=True) OR fn_flat_cost.
      FP cost = fp_cost (analyst review time -- ~$1.50 per alert).
      TP/TN cost = 0.

    Returns a DataFrame indexed by threshold."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.001, 0.05, 20),
            np.linspace(0.05, 0.5, 30),
            np.linspace(0.5, 0.99, 20),
        ])
        grid = np.unique(grid.round(4))

    rows = []
    y_arr = np.asarray(y)
    amt_arr = np.asarray(amt)
    for t in grid:
        pred = (proba >= t).astype(int)
        is_fp = (pred == 1) & (y_arr == 0)
        is_fn = (pred == 0) & (y_arr == 1)
        is_tp = (pred == 1) & (y_arr == 1)
        n_tp = int(is_tp.sum())
        n_fp = int(is_fp.sum())
        n_fn = int(is_fn.sum())
        if use_amount_for_fn:
            fn_loss = float(amt_arr[is_fn].sum())
        else:
            fn_loss = float(n_fn * fn_flat_cost)
        fp_loss = float(n_fp * fp_cost)
        rows.append(CostSweepRow(threshold=float(t), n_tp=n_tp, n_fp=n_fp,
                                  n_fn=n_fn, expected_cost=fn_loss + fp_loss,
                                  fn_loss=fn_loss, fp_loss=fp_loss))
    return pd.DataFrame([asdict(r) for r in rows])


# -------------------------------------------------------------------------- #
# Optuna search & random search (same param space, equal trial budget)
# -------------------------------------------------------------------------- #
TRIAL_PARAM_BOUNDS = dict(
    iterations=(200, 800),
    depth=(4, 9),
    learning_rate=(0.02, 0.30),
    l2_leaf_reg=(1.0, 10.0),
    random_strength=(0.0, 5.0),
    bagging_temperature=(0.0, 5.0),
    border_count=[32, 64, 128, 254],
)


def trial_to_params(rng_or_trial, mode: str) -> dict:
    if mode == "optuna":
        t = rng_or_trial
        params = dict(
            iterations=t.suggest_int("iterations", *TRIAL_PARAM_BOUNDS["iterations"]),
            depth=t.suggest_int("depth", *TRIAL_PARAM_BOUNDS["depth"]),
            learning_rate=t.suggest_float("learning_rate",
                                           *TRIAL_PARAM_BOUNDS["learning_rate"], log=True),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg",
                                         *TRIAL_PARAM_BOUNDS["l2_leaf_reg"], log=True),
            random_strength=t.suggest_float("random_strength",
                                              *TRIAL_PARAM_BOUNDS["random_strength"]),
            bagging_temperature=t.suggest_float("bagging_temperature",
                                                  *TRIAL_PARAM_BOUNDS["bagging_temperature"]),
            border_count=t.suggest_categorical("border_count",
                                                 TRIAL_PARAM_BOUNDS["border_count"]),
        )
    else:
        rng = rng_or_trial
        params = dict(
            iterations=int(rng.integers(*TRIAL_PARAM_BOUNDS["iterations"])),
            depth=int(rng.integers(*TRIAL_PARAM_BOUNDS["depth"])),
            learning_rate=float(np.exp(rng.uniform(np.log(TRIAL_PARAM_BOUNDS["learning_rate"][0]),
                                                     np.log(TRIAL_PARAM_BOUNDS["learning_rate"][1])))),
            l2_leaf_reg=float(np.exp(rng.uniform(np.log(TRIAL_PARAM_BOUNDS["l2_leaf_reg"][0]),
                                                    np.log(TRIAL_PARAM_BOUNDS["l2_leaf_reg"][1])))),
            random_strength=float(rng.uniform(*TRIAL_PARAM_BOUNDS["random_strength"])),
            bagging_temperature=float(rng.uniform(*TRIAL_PARAM_BOUNDS["bagging_temperature"])),
            border_count=int(rng.choice(TRIAL_PARAM_BOUNDS["border_count"])),
        )
    return params


def run_optuna_study(X_fit, y_fit, X_cal, y_cal, n_trials: int = 30, seed: int = 42):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    history = []

    def objective(trial):
        params = trial_to_params(trial, mode="optuna")
        t0 = time.time()
        m = fit_catboost(X_fit, y_fit, X_cal, y_cal, params=params, early_stopping=20)
        proba = m.predict_proba(X_cal)[:, 1]
        auprc = float(average_precision_score(y_cal, proba))
        elapsed = time.time() - t0
        history.append(dict(trial=trial.number, mode="optuna", auprc=auprc,
                              elapsed_s=elapsed, **params))
        return auprc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study, pd.DataFrame(history)


def run_random_search(X_fit, y_fit, X_cal, y_cal, n_trials: int = 30, seed: int = 13):
    rng = np.random.default_rng(seed)
    history = []
    for i in range(n_trials):
        params = trial_to_params(rng, mode="random")
        t0 = time.time()
        m = fit_catboost(X_fit, y_fit, X_cal, y_cal, params=params, early_stopping=20)
        proba = m.predict_proba(X_cal)[:, 1]
        auprc = float(average_precision_score(y_cal, proba))
        elapsed = time.time() - t0
        history.append(dict(trial=i, mode="random", auprc=auprc, elapsed_s=elapsed, **params))
    return pd.DataFrame(history)


# -------------------------------------------------------------------------- #
# Error analysis helpers
# -------------------------------------------------------------------------- #
def subgroup_metrics(test_df: pd.DataFrame, proba: np.ndarray, threshold: float,
                       group_col: str, top_k: int = 12) -> pd.DataFrame:
    pred = (proba >= threshold).astype(int)
    test_df = test_df.copy()
    test_df["_pred"] = pred
    test_df["_proba"] = proba
    rows = []
    grouped = test_df.groupby(group_col)
    for key, sub in grouped:
        if len(sub) < 50:
            continue
        y = sub["is_fraud"].values
        p = sub["_pred"].values
        n_total = len(sub)
        n_fraud = int(y.sum())
        if n_fraud == 0:
            recall = np.nan
        else:
            recall = float(((p == 1) & (y == 1)).sum() / n_fraud)
        n_alerts = int(p.sum())
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        precision = tp / max(tp + fp, 1)
        rows.append(dict(group=key, n_total=n_total, n_fraud=n_fraud,
                         n_alerts=n_alerts, recall=recall, precision=precision,
                         fp=fp))
    out = pd.DataFrame(rows).sort_values("fp", ascending=False)
    return out.head(top_k)


def fp_profile(test_df: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    """Return a one-row summary of false-positive characteristics at a threshold."""
    pred = (proba >= threshold).astype(int)
    is_fp = (pred == 1) & (test_df["is_fraud"].values == 0)
    is_tn = (pred == 0) & (test_df["is_fraud"].values == 0)
    fp_rows = test_df.loc[is_fp]
    tn_rows = test_df.loc[is_tn]
    summary = {
        "n_fp": int(is_fp.sum()),
        "n_tn": int(is_tn.sum()),
        "fp_amt_mean": float(fp_rows["amt"].mean()),
        "tn_amt_mean": float(tn_rows["amt"].mean()),
        "fp_amt_p90": float(fp_rows["amt"].quantile(0.9)),
        "tn_amt_p90": float(tn_rows["amt"].quantile(0.9)),
        "fp_pct_night": float(fp_rows["is_night"].mean()),
        "tn_pct_night": float(tn_rows["is_night"].mean()),
        "fp_pct_weekend": float(fp_rows["is_weekend"].mean()),
        "tn_pct_weekend": float(tn_rows["is_weekend"].mean()),
        "fp_vel_count_24h_mean": float(fp_rows["vel_count_24h"].mean()),
        "tn_vel_count_24h_mean": float(tn_rows["vel_count_24h"].mean()),
        "fp_amt_cat_zscore_mean": float(fp_rows["amt_cat_zscore"].mean()),
        "tn_amt_cat_zscore_mean": float(tn_rows["amt_cat_zscore"].mean()),
    }
    return pd.DataFrame([summary])

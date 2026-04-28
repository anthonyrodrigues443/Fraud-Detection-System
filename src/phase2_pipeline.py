"""Mark Phase 2 — imbalance-handling face-off pipeline.

Each strategy is a function that:
  - Checks results/mark_phase2_cache.json — if already computed, returns cached result.
  - Otherwise runs the experiment, saves both the metric record and the test-set
    predicted probabilities (npy) to disk, then returns.

This means re-executing the notebook never re-trains models we already have.

Functions:
  - load_split()                 -> X_train, X_test, y_train, y_test, FEATURES
  - run_vanilla_xgb()            -> dict (Strategy 0)
  - run_spw_default()            -> dict (Strategy 1, spw=174)
  - run_spw_sweep()              -> list[dict] (Strategy 2)
  - run_smote()                  -> dict (Strategy 3)
  - run_adasyn()                 -> dict (Strategy 4)
  - run_undersample()            -> dict (Strategy 5)
  - run_threshold_tuning()       -> dict (Strategy 6 — uses Strategy 0 scores)
  - run_oof_threshold()          -> dict (Strategy 7 — 5-fold OOF tuning)
  - run_focal_loss()             -> dict (Strategy 8)
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             precision_recall_curve, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix, XGBClassifier
from xgboost import train as xgb_train

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "fraud_transactions.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CACHE_PATH = RESULTS_DIR / "mark_phase2_cache.json"
PROBA_DIR = RESULTS_DIR / "mark_phase2_proba"
PROBA_DIR.mkdir(parents=True, exist_ok=True)

RNG = 42

FEATURES = [
    "amt", "gender", "lat", "long", "city_pop", "unix_time",
    "merch_lat", "merch_long", "hour", "day_of_week", "month",
    "is_weekend", "age", "distance_km", "category_encoded",
    "log_amt", "is_night",
]


# ----------------------------------------------------------------------
# Cache I/O
# ----------------------------------------------------------------------
def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"results": {}, "proba_paths": {}}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _save_proba(name: str, y_proba: np.ndarray) -> str:
    fname = PROBA_DIR / (name.replace(" ", "_").replace("/", "_") + ".npy")
    np.save(fname, y_proba.astype(np.float32))
    return str(fname.relative_to(RESULTS_DIR.parent))


def _load_proba(rel_path: str) -> np.ndarray:
    return np.load(RESULTS_DIR.parent / rel_path)


# ----------------------------------------------------------------------
# Eval harness
# ----------------------------------------------------------------------
def eval_proba(name, y_true, y_proba, threshold=None, train_time=None) -> dict:
    auprc = float(average_precision_score(y_true, y_proba))
    roc = float(roc_auc_score(y_true, y_proba))
    if threshold is None:
        threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    p, r, _ = precision_recall_curve(y_true, y_proba)
    mask = r[:-1] >= 0.95
    p95 = float(p[:-1][mask].max()) if mask.any() else 0.0
    return {
        "model": name,
        "auprc": round(auprc, 4),
        "roc_auc": round(roc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "prec@95recall": round(p95, 4),
        "threshold": round(float(threshold), 5),
        "train_time_s": round(float(train_time), 2) if train_time is not None else None,
    }


def make_xgb(spw=1.0, **extra) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, random_state=RNG,
        eval_metric="aucpr", n_jobs=-1, verbosity=0, tree_method="hist",
        **extra,
    )


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_split():
    """Load CSV, build features, return temporal split. Cached as parquet."""
    cache_train = RESULTS_DIR.parent / "data" / "processed" / "phase2_train.parquet"
    cache_test = RESULTS_DIR.parent / "data" / "processed" / "phase2_test.parquet"
    if cache_train.exists() and cache_test.exists():
        train = pd.read_parquet(cache_train)
        test = pd.read_parquet(cache_test)
        X_train = train[FEATURES].reset_index(drop=True)
        X_test = test[FEATURES].reset_index(drop=True)
        y_train = train["is_fraud"].reset_index(drop=True)
        y_test = test["is_fraud"].reset_index(drop=True)
        return X_train, X_test, y_train, y_test

    print("Building feature matrix from raw CSV (one time setup)...")
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    df["distance_km"] = _haversine(df["lat"].values, df["long"].values,
                                   df["merch_lat"].values, df["merch_long"].values)
    df["gender"] = (df["gender"] == "M").astype(int)
    df["category_encoded"] = LabelEncoder().fit_transform(df["category"])
    df["log_amt"] = np.log1p(df["amt"])
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx][FEATURES + ["is_fraud"]].reset_index(drop=True)
    test = df.iloc[split_idx:][FEATURES + ["is_fraud"]].reset_index(drop=True)

    cache_train.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(cache_train)
    test.to_parquet(cache_test)
    print(f"  built+cached in {time.time()-t0:.1f}s")
    print(f"  train: {len(train):,}  test: {len(test):,}")

    X_train = train[FEATURES].reset_index(drop=True)
    X_test = test[FEATURES].reset_index(drop=True)
    y_train = train["is_fraud"].reset_index(drop=True)
    y_test = test["is_fraud"].reset_index(drop=True)
    del df
    gc.collect()
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# Strategy runners (all checkpoint-aware)
# ----------------------------------------------------------------------
def _strategy(name, runner):
    """Decorator-ish helper: run only if BOTH the metric AND the proba file are cached.

    The .npy proba files are gitignored; on a fresh clone the cache.json may have
    metrics but no probas. In that case re-run so the proba file is restored.
    """
    cache = _load_cache()
    proba_rel = cache["proba_paths"].get(name)
    proba_present = proba_rel is not None and (RESULTS_DIR.parent / proba_rel).exists()
    if name in cache["results"] and proba_present:
        print(f"[CACHED] {name}: {cache['results'][name]}")
        return cache["results"][name]
    print(f"[RUN]    {name} ...")
    res, y_proba = runner()
    if y_proba is not None:
        cache["proba_paths"][name] = _save_proba(name, y_proba)
    cache["results"][name] = res
    _save_cache(cache)
    print(f"[DONE]   {name}: AUPRC={res['auprc']} F1={res['f1']} ({res.get('train_time_s', '?')}s)")
    return res


def run_vanilla_xgb(X_train, X_test, y_train, y_test):
    name = "XGB-vanilla (spw=1)"
    def _run():
        t0 = time.time()
        m = make_xgb(spw=1.0).fit(X_train, y_train)
        tt = time.time() - t0
        yp = m.predict_proba(X_test)[:, 1]
        return eval_proba(name, y_test, yp, train_time=tt), yp
    return _strategy(name, _run)


def run_spw_default(X_train, X_test, y_train, y_test):
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    name = f"XGB-spw={spw:.0f} (Anthony default)"
    def _run():
        t0 = time.time()
        m = make_xgb(spw=spw).fit(X_train, y_train)
        tt = time.time() - t0
        yp = m.predict_proba(X_test)[:, 1]
        return eval_proba(name, y_test, yp, train_time=tt), yp
    return _strategy(name, _run)


def run_spw_sweep(X_train, X_test, y_train, y_test, values=None):
    if values is None:
        spw_default = (y_train == 0).sum() / (y_train == 1).sum()
        values = [1.0, 5.0, 17.4, 87.0, float(spw_default), 350.0, 870.0]
    out = []
    for spw in values:
        name = f"XGB-spw={spw:.1f}"
        def _run(spw=spw):
            t0 = time.time()
            m = make_xgb(spw=spw).fit(X_train, y_train)
            tt = time.time() - t0
            yp = m.predict_proba(X_test)[:, 1]
            return eval_proba(name, y_test, yp, train_time=tt), yp
        out.append({"spw": spw, **_strategy(name, _run)})
    return out


def run_smote(X_train, X_test, y_train, y_test):
    from imblearn.over_sampling import SMOTE
    name = "XGB+SMOTE"
    def _run():
        t0 = time.time()
        sm = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=RNG)
        X_sm, y_sm = sm.fit_resample(X_train, y_train)
        tt_sm = time.time() - t0
        t1 = time.time()
        m = make_xgb(spw=1.0).fit(X_sm, y_sm)
        tt = time.time() - t1
        yp = m.predict_proba(X_test)[:, 1]
        del X_sm, y_sm; gc.collect()
        return eval_proba(name, y_test, yp, train_time=tt_sm + tt), yp
    return _strategy(name, _run)


def run_adasyn(X_train, X_test, y_train, y_test):
    from imblearn.over_sampling import ADASYN
    name = "XGB+ADASYN"
    def _run():
        t0 = time.time()
        ad = ADASYN(sampling_strategy="auto", n_neighbors=5, random_state=RNG)
        X_ad, y_ad = ad.fit_resample(X_train, y_train)
        tt_ad = time.time() - t0
        t1 = time.time()
        m = make_xgb(spw=1.0).fit(X_ad, y_ad)
        tt = time.time() - t1
        yp = m.predict_proba(X_test)[:, 1]
        del X_ad, y_ad; gc.collect()
        return eval_proba(name, y_test, yp, train_time=tt_ad + tt), yp
    return _strategy(name, _run)


def run_undersample(X_train, X_test, y_train, y_test):
    from imblearn.under_sampling import RandomUnderSampler
    name = "XGB+Undersample"
    def _run():
        t0 = time.time()
        rus = RandomUnderSampler(sampling_strategy="auto", random_state=RNG)
        X_us, y_us = rus.fit_resample(X_train, y_train)
        tt_us = time.time() - t0
        t1 = time.time()
        m = make_xgb(spw=1.0).fit(X_us, y_us)
        tt = time.time() - t1
        yp = m.predict_proba(X_test)[:, 1]
        del X_us, y_us; gc.collect()
        return eval_proba(name, y_test, yp, train_time=tt_us + tt), yp
    return _strategy(name, _run)


def run_threshold_tuning(y_test):
    cache = _load_cache()
    base_name = "XGB-vanilla (spw=1)"
    if base_name not in cache["proba_paths"]:
        raise RuntimeError("Run vanilla XGB first to get base scores.")
    yp_vanilla = _load_proba(cache["proba_paths"][base_name])
    p, r, t = precision_recall_curve(y_test, yp_vanilla)
    f1c = 2 * p * r / (p + r + 1e-12)
    best_idx = int(f1c[:-1].argmax())
    best_thr = float(t[best_idx])

    name = "XGB-vanilla + threshold-tuned (test-set)"
    def _run():
        return eval_proba(name, y_test, yp_vanilla, threshold=best_thr,
                          train_time=cache["results"][base_name].get("train_time_s")), yp_vanilla
    return _strategy(name, _run)


def run_oof_threshold(X_train, X_test, y_train, y_test, n_splits=5):
    cache = _load_cache()
    base_name = "XGB-vanilla (spw=1)"
    yp_vanilla = _load_proba(cache["proba_paths"][base_name])
    name = "XGB-vanilla + OOF-calibrated threshold"

    def _run():
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
        oof = np.zeros(len(X_train))
        t0 = time.time()
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            m = make_xgb(spw=1.0).fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
            print(f"    fold {fold_i+1}/{n_splits} done")
        tt = time.time() - t0
        p, r, t = precision_recall_curve(y_train, oof)
        f1c = 2 * p * r / (p + r + 1e-12)
        best_idx = int(f1c[:-1].argmax())
        best_thr = float(t[best_idx])
        # Save OOF threshold for downstream use
        meta_path = RESULTS_DIR / "mark_phase2_oof_threshold.json"
        meta_path.write_text(json.dumps({"oof_threshold": best_thr}, indent=2))
        return eval_proba(name, y_test, yp_vanilla, threshold=best_thr,
                          train_time=tt), yp_vanilla
    return _strategy(name, _run)


def run_focal_loss(X_train, X_test, y_train, y_test, gamma=2.0, alpha=0.25):
    name = f"XGB+FocalLoss(g={gamma},a={alpha})"

    def _run():
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        def focal_obj(y_pred, dtrain):
            y = dtrain.get_label()
            p = sigmoid(y_pred)
            eps = 1e-12
            p_clip = np.clip(p, eps, 1 - eps)
            pt = np.where(y == 1, p_clip, 1 - p_clip)
            alpha_t = np.where(y == 1, alpha, 1 - alpha)
            grad = (alpha_t * (1 - pt) ** (gamma - 1) *
                    (gamma * pt * np.log(pt) + pt - 1) *
                    np.where(y == 1, 1, -1))
            hess = alpha_t * (1 - pt) ** gamma * np.maximum(
                pt * (1 - pt) * (gamma + 1), 1e-6)
            return grad, hess

        dtrain = DMatrix(X_train.values, label=y_train.values)
        dtest = DMatrix(X_test.values, label=y_test.values)
        params = dict(max_depth=6, eta=0.1, tree_method="hist", verbosity=0)
        t0 = time.time()
        booster = xgb_train(params=params, dtrain=dtrain, num_boost_round=200, obj=focal_obj)
        tt = time.time() - t0
        yp = sigmoid(booster.predict(dtest))
        return eval_proba(name, y_test, yp, train_time=tt), yp
    return _strategy(name, _run)


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def run_all():
    print("=" * 70)
    print("Mark Phase 2 — imbalance-handling face-off (checkpointed)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = load_split()
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  features: {len(FEATURES)}")
    spw_default = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight (inverse class ratio) = {spw_default:.1f}\n")

    print("--- Strategy 0: vanilla XGBoost (spw=1) ---")
    run_vanilla_xgb(X_train, X_test, y_train, y_test)

    print("\n--- Strategy 1: spw = inverse class ratio (Anthony default) ---")
    run_spw_default(X_train, X_test, y_train, y_test)

    print("\n--- Strategy 2: spw sweep ---")
    run_spw_sweep(X_train, X_test, y_train, y_test)

    print("\n--- Strategy 3: SMOTE + vanilla XGBoost ---")
    run_smote(X_train, X_test, y_train, y_test)

    print("\n--- Strategy 4: ADASYN + vanilla XGBoost ---")
    run_adasyn(X_train, X_test, y_train, y_test)

    print("\n--- Strategy 5: Random undersampling + vanilla XGBoost ---")
    run_undersample(X_train, X_test, y_train, y_test)

    print("\n--- Strategy 6: Threshold tuning on vanilla scores (test-set) ---")
    run_threshold_tuning(y_test)

    print("\n--- Strategy 7: OOF-calibrated threshold ---")
    run_oof_threshold(X_train, X_test, y_train, y_test)

    print("\n--- Strategy 8: Focal loss XGBoost (g=2, a=0.25) ---")
    run_focal_loss(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 70)
    print("DONE — all strategies cached in", CACHE_PATH)
    cache = _load_cache()
    print(f"Total strategies in cache: {len(cache['results'])}")


if __name__ == "__main__":
    run_all()

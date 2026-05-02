"""
Production data pipeline (Phase 6, Mark).

Loads the Phase-3 feature-engineered parquet (the same artifact every prior
phase uses) and exposes a single clean interface used by training, inference,
and the Streamlit app.

The 53-feature CLEAN_STACK_53 list is canonical (Phase 3-5). Frequency-encoded
columns (freq_merchant, freq_state, freq_city) are fit on the temporal-train
slice ONLY -- never on test -- and the encoder maps are saved as artifacts so
inference reproduces them exactly.

Public API:
    load_full_dataset(parquet_path)          -> train_df, test_df
    fit_frequency_encoders(train_df)          -> dict[str, dict[value -> count]]
    apply_frequency_encoders(df, encoders)    -> df with freq_* columns
    materialize_features(df)                  -> X (DataFrame, dtype float32)
    sample_test_transactions(test_df, ...)    -> list of dicts for Streamlit demo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Canonical feature lists (mirrored from src/mark_phase4_tuning.py so this
# module has zero dependency on the research scripts).
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
    "merch_count_1h", "merch_count_24h", "merch_amt_24h", "merch_fraud_rate",
    "card_merch_count", "log_time_since_last_merch", "card_merch_amt_ratio",
    "ix_amt_x_catfraud", "ix_vel24_x_amt",
    "ix_amtcat_x_isnight", "ix_amtcat_x_velcount24",
]
FREQ_COLS = ["merchant", "state", "city"]
FREQ_FEATURES = ["freq_merchant", "freq_state", "freq_city"]

CLEAN_STACK_53 = ANTHONY_BASELINE + ANTHONY_NEW + MARK_NON_TE + FREQ_FEATURES


def load_full_dataset(parquet_path: str | Path, train_frac: float = 0.8):
    """Load the Phase-3 parquet and split temporally (80/20 by trans time)."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()
    return train_df, test_df


def fit_frequency_encoders(train_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Fit frequency encoders on the train slice. Returns {col: {value: count}}.

    Uses str() on values for JSON-serialisable artifacts."""
    encoders: Dict[str, Dict[str, int]] = {}
    for col in FREQ_COLS:
        counts = train_df[col].value_counts()
        encoders[col] = {str(k): int(v) for k, v in counts.items()}
    return encoders


def apply_frequency_encoders(df: pd.DataFrame, encoders: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Add freq_merchant / freq_state / freq_city columns. Unseen values map to 0."""
    out = df.copy()
    for col in FREQ_COLS:
        m = encoders[col]
        out[f"freq_{col}"] = out[col].astype(str).map(m).fillna(0).astype(np.int64)
    return out


def materialize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 53-feature matrix (float32) in canonical column order."""
    missing = [c for c in CLEAN_STACK_53 if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[CLEAN_STACK_53].astype(np.float32)


def save_encoders(encoders: Dict[str, Dict[str, int]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(encoders))


def load_encoders(path: str | Path) -> Dict[str, Dict[str, int]]:
    return json.loads(Path(path).read_text())


def sample_test_transactions(test_df: pd.DataFrame,
                              n_fraud: int = 10,
                              n_legit: int = 10,
                              random_state: int = 7) -> List[dict]:
    """Stratified sample of test rows for the Streamlit demo. Keeps a small
    set of human-readable fields plus the engineered features. Each row is
    returned as a plain dict (JSON-serialisable)."""
    rng = np.random.default_rng(random_state)
    fraud_idx = test_df.index[test_df["is_fraud"] == 1].to_numpy()
    legit_idx = test_df.index[test_df["is_fraud"] == 0].to_numpy()
    fraud_pick = rng.choice(fraud_idx, size=min(n_fraud, len(fraud_idx)), replace=False)
    legit_pick = rng.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)

    keep_meta = [
        "trans_date_trans_time", "amt", "category", "merchant",
        "city", "state", "gender", "age", "hour", "is_night",
        "distance_km", "is_fraud",
    ]
    feat_cols = [c for c in CLEAN_STACK_53 if c not in FREQ_FEATURES]

    out: List[dict] = []
    for idx in np.concatenate([fraud_pick, legit_pick]):
        row = test_df.loc[idx]
        meta = {k: (row[k].isoformat() if hasattr(row[k], "isoformat") else _to_native(row[k]))
                for k in keep_meta if k in row.index}
        feats = {c: _to_native(row[c]) for c in feat_cols if c in row.index}
        # Frequency-encoded columns are reproducible via the saved encoders, but
        # we also embed them so the Streamlit app can predict without re-encoding.
        freq_block = {c: _to_native(row[c]) for c in FREQ_FEATURES if c in row.index}
        out.append({
            "id": int(idx),
            "label": int(row["is_fraud"]),
            "meta": meta,
            "features": feats,
            "freq_features": freq_block,
        })
    rng.shuffle(out)
    return out


def _to_native(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


__all__ = [
    "ANTHONY_BASELINE", "ANTHONY_NEW", "MARK_NON_TE", "FREQ_COLS", "FREQ_FEATURES",
    "CLEAN_STACK_53",
    "load_full_dataset", "fit_frequency_encoders", "apply_frequency_encoders",
    "materialize_features", "save_encoders", "load_encoders",
    "sample_test_transactions",
]

"""Phase 6 (Mark) -- tests for src/data_pipeline.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from data_pipeline import (
    CLEAN_STACK_53, FREQ_COLS, FREQ_FEATURES,
    fit_frequency_encoders, apply_frequency_encoders,
    materialize_features, save_encoders, load_encoders,
    sample_test_transactions, _to_native,
)


def _toy_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(0, 1, n).astype(np.float32)
                        for c in CLEAN_STACK_53 if c not in FREQ_FEATURES})
    df["merchant"] = rng.choice(["a", "b", "c"], n)
    df["state"] = rng.choice(["CA", "NY"], n)
    df["city"] = rng.choice(["foo", "bar"], n)
    df["is_fraud"] = (rng.uniform(size=n) < 0.1).astype(np.int8)
    df["trans_date_trans_time"] = pd.date_range("2020-01-01", periods=n, freq="h")
    df["amt"] = rng.uniform(1, 500, n)
    df["category"] = rng.choice(["entertainment", "grocery_pos", "misc_net"], n)
    df["age"] = rng.integers(18, 80, n)
    df["hour"] = rng.integers(0, 24, n)
    df["is_night"] = (df["hour"] >= 22) | (df["hour"] <= 5)
    df["distance_km"] = rng.uniform(0, 100, n)
    df["gender"] = rng.choice([0, 1], n)
    return df


def test_clean_stack_53_has_53_columns():
    assert len(CLEAN_STACK_53) == 53
    assert all(isinstance(c, str) for c in CLEAN_STACK_53)


def test_fit_frequency_encoders_returns_serialisable_dict():
    df = _toy_df()
    enc = fit_frequency_encoders(df)
    assert set(enc.keys()) == set(FREQ_COLS)
    # Values are int counts that sum to len(df) for each col
    for col in FREQ_COLS:
        assert sum(enc[col].values()) == len(df)
    # Round-trip JSON
    s = json.dumps(enc)
    enc2 = json.loads(s)
    assert enc2.keys() == enc.keys()


def test_apply_frequency_encoders_handles_unseen_values():
    df = _toy_df()
    enc = fit_frequency_encoders(df)
    new = df.head(3).copy()
    new["merchant"] = ["a", "totally_new", "another_unseen"]
    enc_df = apply_frequency_encoders(new, enc)
    assert "freq_merchant" in enc_df.columns
    # Unseen values map to 0
    assert enc_df["freq_merchant"].iloc[1] == 0
    assert enc_df["freq_merchant"].iloc[2] == 0
    # Seen value maps to its count
    assert enc_df["freq_merchant"].iloc[0] > 0


def test_materialize_features_returns_53_float32_columns():
    df = _toy_df()
    enc = fit_frequency_encoders(df)
    df = apply_frequency_encoders(df, enc)
    X = materialize_features(df)
    assert list(X.columns) == CLEAN_STACK_53
    assert X.dtypes.unique().tolist() == [np.float32]
    assert X.shape == (len(df), 53)


def test_materialize_features_raises_on_missing_columns():
    df = _toy_df().drop(columns=["amt"])
    with pytest.raises(KeyError, match="amt"):
        materialize_features(df)


def test_save_and_load_encoders_round_trip(tmp_path: Path):
    enc = {"merchant": {"a": 1, "b": 2}, "state": {"CA": 5}, "city": {"foo": 7}}
    p = tmp_path / "enc.json"
    save_encoders(enc, p)
    loaded = load_encoders(p)
    assert loaded == enc


def test_sample_test_transactions_returns_stratified_dicts():
    df = _toy_df(n=200)
    enc = fit_frequency_encoders(df)
    df = apply_frequency_encoders(df, enc)
    samples = sample_test_transactions(df, n_fraud=3, n_legit=3, random_state=0)
    assert 1 <= len(samples) <= 6  # bounded by fraud count in toy df
    for s in samples:
        assert "id" in s and "label" in s and "meta" in s and "features" in s
        assert s["label"] in (0, 1)
        # All 50 non-freq feature columns are present in features dict
        for c in CLEAN_STACK_53:
            if c not in FREQ_FEATURES:
                assert c in s["features"]


def test_to_native_handles_numpy_scalars():
    assert _to_native(np.int64(7)) == 7
    assert isinstance(_to_native(np.float32(3.5)), float)
    assert _to_native(np.bool_(True)) is True
    assert _to_native(pd.Timestamp("2020-01-01")) == "2020-01-01T00:00:00"

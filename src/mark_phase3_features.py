"""
Phase 3 (Mark) — complementary feature engineering.

Anthony's Phase 3 (merged on main) added 22 DOMAIN behavioral features per cardholder
(velocity, amount z-score, temporal, geographic, category-merchant). His top finding:
velocity features account for 46% of the lift; CatBoost AUPRC went 0.8764 -> 0.9824.

Mark's Phase 3 takes the COMPLEMENTARY angle from the rotation playbook: when the
collaborator's session focused on hand-engineered domain features, the next session
should test STATISTICAL / AUTOMATED features and INTERACTIONS. Specifically:

  Group M.A: Bayesian target encoding (smoothing alpha) on high-cardinality
             categoricals Anthony did NOT encode -- merchant, state, city, job, zip,
             gender. Reference: Micci-Barreca (2001) "A preprocessing scheme for
             high-cardinality categorical attributes" -- introduced in a fraud
             setting for ZIP/IP/SKU.

  Group M.B: Merchant-side velocity. Anthony engineered per-CARD rolling counts.
             Mark engineers per-MERCHANT rolling counts and per-merchant fraud
             rate (leak-free, expanding) to surface point-of-compromise patterns
             (Araujo et al., CMU SDM 2017, "BreachRadar").

  Group M.C: Card x Merchant repeat features -- has this card transacted with
             this merchant before? Time since last visit? Repeat-customer signal.

  Group M.D: Frequency / count encoding for high-cardinality categoricals
             (cheap signal that distinguishes popular vs rare merchants/states).

  Group M.E: Multiplicative interactions between Anthony's top-importance features
             (cat_fraud_rate, amt_cat_zscore, vel_count_24h, log_amt, hour).

All features are computed leak-free (expanding / shifted / training-fit-only).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- #
# Anthony's Phase 3 features (replicated so we can rebuild the 39-feature
# dataset deterministically from the raw CSV).
# ----------------------------------------------------------------------------- #


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


BASELINE_FEATURES = [
    "amt", "gender", "lat", "long", "city_pop", "unix_time",
    "merch_lat", "merch_long", "hour", "day_of_week", "month",
    "is_weekend", "age", "distance_km", "category_encoded",
    "log_amt", "is_night",
]

VELOCITY_FEATURES = [
    "vel_count_1h", "vel_count_6h", "vel_count_24h", "vel_count_7d",
    "vel_amt_1h", "vel_amt_6h", "vel_amt_24h", "vel_amt_7d",
]
AMOUNT_DEV_FEATURES = [
    "amt_zscore", "amt_ratio_to_mean", "amt_card_mean",
    "amt_card_std", "amt_cat_zscore",
]
TEMPORAL_FEATURES = ["log_time_since_last", "log_avg_time_between", "hour_deviation"]
GEO_FEATURES = ["log_dist_centroid", "impossible_travel"]
CATEGORY_FEATURES = ["cat_fraud_rate", "card_cat_count", "is_new_merchant", "card_txn_number"]

ANTHONY_NEW_FEATURES = (
    VELOCITY_FEATURES + AMOUNT_DEV_FEATURES + TEMPORAL_FEATURES + GEO_FEATURES + CATEGORY_FEATURES
)
ALL_ANTHONY_FEATURES = BASELINE_FEATURES + ANTHONY_NEW_FEATURES


def build_anthony_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate Anthony's 22-feature behavioral set on the raw CSV (already
    loaded with parsed dates and sorted by trans_date_trans_time)."""
    df = df.copy()

    # Baseline features
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    df["distance_km"] = _haversine(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values,
    )
    df["gender"] = (df["gender"] == "M").astype(int)
    df["category_encoded"] = pd.factorize(df["category"])[0]
    df["log_amt"] = np.log1p(df["amt"])
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # Per-card velocity / amount-dev / temporal / geo features
    df["ts"] = df["trans_date_trans_time"].astype(np.int64) // 10**9
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    windows = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800}
    n = len(df)
    velocity_count = {w: np.zeros(n) for w in windows}
    velocity_amt = {w: np.zeros(n) for w in windows}
    amt_exp_mean = np.zeros(n)
    amt_exp_std = np.zeros(n)
    amt_ratio_to_mean = np.zeros(n)
    time_since_last = np.zeros(n)
    avg_time_between = np.zeros(n)
    hour_deviation = np.zeros(n)
    dist_from_centroid = np.zeros(n)
    impossible_travel = np.zeros(n)

    card_groups = df.groupby("cc_num", sort=False)
    n_cards = card_groups.ngroups
    t0 = time.time()
    for i, (_, grp) in enumerate(card_groups):
        if i % 200 == 0:
            print(f"  per-card features: {i:,}/{n_cards:,} ({i/n_cards*100:.0f}%)", end="\r")
        idx = grp.index.values
        ts_arr = grp["ts"].values
        amt_arr = grp["amt"].values
        hour_arr = grp["hour"].values.astype(float)
        lat_arr = grp["lat"].values  # noqa: F841 (kept for parity if reused)
        lon_arr = grp["long"].values  # noqa: F841
        mlat_arr = grp["merch_lat"].values
        mlon_arr = grp["merch_long"].values
        m = len(ts_arr)

        # --- Velocity (4 windows, count + amt) -------------------------------
        for wname, wsec in windows.items():
            counts = np.zeros(m)
            sums = np.zeros(m)
            left = 0
            running_sum = 0.0
            for j in range(m):
                while left < j and ts_arr[j] - ts_arr[left] > wsec:
                    running_sum -= amt_arr[left]
                    left += 1
                counts[j] = j - left
                sums[j] = running_sum
                running_sum += amt_arr[j]
            velocity_count[wname][idx] = counts
            velocity_amt[wname][idx] = sums

        # --- Amount deviation expanding ------------------------------------
        running_sum = 0.0
        running_sq = 0.0
        for j in range(m):
            if j == 0:
                amt_exp_mean[idx[j]] = 0.0
                amt_exp_std[idx[j]] = 0.0
                amt_ratio_to_mean[idx[j]] = 0.0
            else:
                me = running_sum / j
                amt_exp_mean[idx[j]] = me
                if j > 1:
                    var = (running_sq / j) - me ** 2
                    s = np.sqrt(max(var, 0))
                else:
                    s = 0.0
                amt_exp_std[idx[j]] = s
                amt_ratio_to_mean[idx[j]] = amt_arr[j] / (me + 1e-6)
            running_sum += amt_arr[j]
            running_sq += amt_arr[j] ** 2

        # --- Temporal ------------------------------------------------------
        running_hour_sum = 0.0
        for j in range(m):
            if j == 0:
                time_since_last[idx[j]] = -1
                avg_time_between[idx[j]] = -1
                hour_deviation[idx[j]] = 0.0
            else:
                time_since_last[idx[j]] = ts_arr[j] - ts_arr[j - 1]
                avg_time_between[idx[j]] = (ts_arr[j] - ts_arr[0]) / j
                mean_hour = running_hour_sum / j
                diff = abs(hour_arr[j] - mean_hour)
                hour_deviation[idx[j]] = min(diff, 24 - diff)
            running_hour_sum += hour_arr[j]

        # --- Geographic ----------------------------------------------------
        running_lat = 0.0
        running_lon = 0.0
        for j in range(m):
            if j == 0:
                dist_from_centroid[idx[j]] = 0.0
                impossible_travel[idx[j]] = 0
            else:
                mean_lat = running_lat / j
                mean_lon = running_lon / j
                dist_from_centroid[idx[j]] = _haversine(
                    np.array([mean_lat]), np.array([mean_lon]),
                    np.array([mlat_arr[j]]), np.array([mlon_arr[j]]),
                )[0]
                d = _haversine(
                    np.array([mlat_arr[j - 1]]), np.array([mlon_arr[j - 1]]),
                    np.array([mlat_arr[j]]), np.array([mlon_arr[j]]),
                )[0]
                dt = max(ts_arr[j] - ts_arr[j - 1], 1)
                speed_kmh = d / (dt / 3600)
                impossible_travel[idx[j]] = int(speed_kmh > 900)
            running_lat += mlat_arr[j]
            running_lon += mlon_arr[j]

    print(f"  per-card features: done in {time.time() - t0:.1f}s" + " " * 40)

    for wname in windows:
        df[f"vel_count_{wname}"] = velocity_count[wname]
        df[f"vel_amt_{wname}"] = velocity_amt[wname]
    df["amt_card_mean"] = amt_exp_mean
    df["amt_card_std"] = amt_exp_std
    df["amt_zscore"] = np.where(
        df["amt_card_std"] > 0,
        (df["amt"] - df["amt_card_mean"]) / df["amt_card_std"],
        0.0,
    )
    df["amt_ratio_to_mean"] = amt_ratio_to_mean
    df["time_since_last_txn"] = time_since_last
    df["avg_time_between_txns"] = avg_time_between
    df["hour_deviation"] = hour_deviation
    df["dist_from_centroid"] = dist_from_centroid
    df["impossible_travel"] = impossible_travel

    median_tsl = df.loc[df["time_since_last_txn"] > 0, "time_since_last_txn"].median()
    median_atb = df.loc[df["avg_time_between_txns"] > 0, "avg_time_between_txns"].median()
    df["time_since_last_txn"] = df["time_since_last_txn"].replace(-1, median_tsl)
    df["avg_time_between_txns"] = df["avg_time_between_txns"].replace(-1, median_atb)
    df["log_time_since_last"] = np.log1p(df["time_since_last_txn"])
    df["log_avg_time_between"] = np.log1p(df["avg_time_between_txns"])
    df["log_dist_centroid"] = np.log1p(df["dist_from_centroid"])

    # Now re-sort by time for the cross-card / category features
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    # Per-category amount z-score, expanding shifted by 1 (no leakage)
    df["amt_cat_mean"] = df.groupby("category")["amt"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["amt_cat_std"] = df.groupby("category")["amt"].transform(
        lambda x: x.expanding().std().shift(1)
    )
    df["amt_cat_zscore"] = np.where(
        df["amt_cat_std"] > 0,
        (df["amt"] - df["amt_cat_mean"]) / df["amt_cat_std"],
        0.0,
    )
    df.drop(columns=["amt_cat_mean", "amt_cat_std"], inplace=True)

    # Category fraud rate (leak-free expanding shifted)
    df["cat_fraud_rate"] = (
        df.groupby("category")["is_fraud"].transform(lambda x: x.expanding().mean().shift(1))
        .fillna(df["is_fraud"].mean())
    )
    df["card_cat_count"] = df.groupby(["cc_num", "category"]).cumcount()
    df["is_new_merchant"] = (df.groupby(["cc_num", "merchant"]).cumcount() == 0).astype(int)
    df["card_txn_number"] = df.groupby("cc_num").cumcount()

    return df


# ----------------------------------------------------------------------------- #
# Mark's complementary features.
# ----------------------------------------------------------------------------- #


# Group M.A: Bayesian target encoding (smoothing-alpha)
TARGET_ENCODE_COLS = ["merchant", "state", "city", "job", "zip", "gender_raw"]
TE_FEATURES = [f"te_{c}" for c in TARGET_ENCODE_COLS]


def fit_target_encoding(
    df_train: pd.DataFrame, col: str, target: str = "is_fraud", alpha: float = 100.0
) -> tuple[pd.Series, float]:
    """Bayesian smoothed mean encoding fit on training only.

    encoding(c) = (sum_c + alpha * prior) / (count_c + alpha)
    where prior = global training fraud rate. alpha controls smoothing strength;
    100 is a moderate default that pulls rare categories toward the prior.
    """
    prior = df_train[target].mean()
    stats = df_train.groupby(col)[target].agg(["sum", "count"])
    enc = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
    return enc, prior


def apply_target_encoding(df: pd.DataFrame, col: str, enc: pd.Series, prior: float) -> pd.Series:
    return df[col].map(enc).fillna(prior)


# Group M.B: Per-merchant rolling velocity + leak-free per-merchant fraud rate
MERCHANT_VEL_FEATURES = [
    "merch_count_1h", "merch_count_24h",
    "merch_amt_24h",
    "merch_fraud_rate",
]


def build_merchant_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-merchant rolling counts in 1h and 24h windows + per-merchant
    expanding fraud rate (leak-free, shifted by 1).

    Surface point-of-compromise: a merchant being hit by many cards in a short
    window is a different signal than a card hitting many merchants. CMU's
    BreachRadar (Araujo et al., SDM 2017) uses this exact construct."""
    df = df.copy()
    df["ts"] = df["trans_date_trans_time"].astype(np.int64) // 10 ** 9
    df = df.sort_values(["merchant", "trans_date_trans_time"]).reset_index(drop=True)

    n = len(df)
    windows = {"1h": 3600, "24h": 86400}
    counts = {w: np.zeros(n) for w in windows}
    amts = {w: np.zeros(n) for w in windows}

    t0 = time.time()
    merch_groups = df.groupby("merchant", sort=False)
    n_merch = merch_groups.ngroups
    for i, (_, grp) in enumerate(merch_groups):
        if i % 100 == 0:
            print(f"  per-merchant velocity: {i:,}/{n_merch:,}", end="\r")
        idx = grp.index.values
        ts_arr = grp["ts"].values
        amt_arr = grp["amt"].values
        m = len(ts_arr)
        for wname, wsec in windows.items():
            left = 0
            running_sum = 0.0
            for j in range(m):
                while left < j and ts_arr[j] - ts_arr[left] > wsec:
                    running_sum -= amt_arr[left]
                    left += 1
                counts[wname][idx[j]] = j - left
                amts[wname][idx[j]] = running_sum
                running_sum += amt_arr[j]
    print(f"  per-merchant velocity: done in {time.time() - t0:.1f}s" + " " * 40)

    df["merch_count_1h"] = counts["1h"]
    df["merch_count_24h"] = counts["24h"]
    df["merch_amt_24h"] = amts["24h"]

    # Per-merchant fraud rate, expanding-shifted (leak-free)
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    df["merch_fraud_rate"] = (
        df.groupby("merchant")["is_fraud"].transform(lambda x: x.expanding().mean().shift(1))
        .fillna(df["is_fraud"].mean())
    )
    return df


# Group M.C: Card x Merchant repeat / time-since-last-merchant-visit
CARD_MERCHANT_FEATURES = [
    "card_merch_count",
    "log_time_since_last_merch",
    "card_merch_amt_ratio",
]


def build_card_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """How many times has this card transacted with this merchant before?
    How long since the last visit? How does this transaction's amount compare
    to the average for this card-merchant pair so far?
    All computed leak-free (cumcount / shift)."""
    df = df.copy()
    df["ts"] = df["trans_date_trans_time"].astype(np.int64) // 10 ** 9

    # cumcount on (cc_num, merchant) -> repeat number (0 = first visit)
    df["card_merch_count"] = df.groupby(["cc_num", "merchant"]).cumcount()

    # Time since last visit to same merchant (per card-merchant pair)
    df = df.sort_values(["cc_num", "merchant", "trans_date_trans_time"]).reset_index(drop=True)
    prev_ts = df.groupby(["cc_num", "merchant"])["ts"].shift(1)
    delta = df["ts"] - prev_ts
    df["time_since_last_merch"] = delta.fillna(delta.median())  # first visit -> median
    df["log_time_since_last_merch"] = np.log1p(df["time_since_last_merch"].clip(lower=0))

    # Amount ratio vs running mean for this card-merchant pair (shifted to avoid leak)
    running_mean = df.groupby(["cc_num", "merchant"])["amt"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["card_merch_amt_ratio"] = df["amt"] / (running_mean.fillna(df["amt"]) + 1e-6)

    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    df.drop(columns=["time_since_last_merch"], inplace=True)
    return df


# Group M.D: Frequency / count encoding
FREQ_ENCODE_COLS = ["merchant", "state", "city"]
FREQ_FEATURES = [f"freq_{c}" for c in FREQ_ENCODE_COLS]


def fit_frequency_encoding(df_train: pd.DataFrame, col: str) -> pd.Series:
    return df_train[col].value_counts()


def apply_frequency_encoding(df: pd.DataFrame, col: str, freq: pd.Series) -> pd.Series:
    return df[col].map(freq).fillna(0).astype(np.int64)


# Group M.E: Multiplicative interactions between Anthony's top features
INTERACTION_FEATURES = [
    "ix_amt_x_catfraud",      # high-amount * high-cat-risk
    "ix_vel24_x_amt",         # burst pattern * amount
    "ix_amtcat_x_isnight",    # cat-z-score * night-hour
    "ix_amtcat_x_velcount24", # cat-z-score * card velocity
]


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ix_amt_x_catfraud"] = df["log_amt"] * df["cat_fraud_rate"]
    df["ix_vel24_x_amt"] = df["vel_count_24h"] * df["log_amt"]
    df["ix_amtcat_x_isnight"] = df["amt_cat_zscore"] * df["is_night"]
    df["ix_amtcat_x_velcount24"] = df["amt_cat_zscore"] * df["vel_count_24h"]
    return df


ALL_MARK_FEATURES = (
    TE_FEATURES + MERCHANT_VEL_FEATURES + CARD_MERCHANT_FEATURES + FREQ_FEATURES + INTERACTION_FEATURES
)


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same temporal 80/20 split Anthony / Phase 2 used."""
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def build_full_phase3_dataset(raw_csv_path: Path | str) -> pd.DataFrame:
    """End-to-end: load raw csv -> Anthony's 39 features -> Mark's complementary
    features -> return a single dataframe with all features defined."""
    df = pd.read_csv(raw_csv_path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    # Keep raw gender for target encoding (Anthony's `gender` column gets numeric 0/1)
    df["gender_raw"] = df["gender"]

    print("[1/3] Anthony's 39-feature pipeline ...")
    df = build_anthony_features(df)

    print("[2/3] Mark merchant velocity ...")
    df = build_merchant_velocity(df)

    print("[3/3] Mark card-merchant + interactions ...")
    df = build_card_merchant_features(df)
    df = build_interaction_features(df)
    return df

"""
Phase 6 (Mark) -- production inference module.

A clean Python interface around the saved CatBoost + XGBoost + LightGBM
ensemble (Phase 5 production winner) and the Phase-4 cost-optimal threshold.

Usage:
    from src.predict import FraudDetector
    det = FraudDetector.load()
    out = det.predict_one(transaction_dict)
    # {"prob": 0.0123, "alert": False, "threshold": 0.0473,
    #  "cost_optimal_threshold": 0.0473, "default_threshold": 0.5,
    #  "top_features": [("vel_amt_24h", 0.082, 312.5), ...],
    #  "individual_probs": {"catboost": 0.011, "xgboost": 0.012, "lightgbm": 0.014}}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from data_pipeline import (
    CLEAN_STACK_53, FREQ_FEATURES, FREQ_COLS,
    apply_frequency_encoders, materialize_features, load_encoders,
)


REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"


@dataclass
class PredictionResult:
    prob: float
    alert: bool
    threshold: float
    default_threshold: float
    cost_optimal_threshold: float
    individual_probs: Dict[str, float]
    top_features: List[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class FraudDetector:
    """Production inference wrapper for the simple-average ensemble."""

    def __init__(self, cb, xgb, lgb_booster, encoders, feature_cols, thresholds):
        self.cb = cb
        self.xgb = xgb
        self.lgb = lgb_booster  # native Booster (not Sklearn shim)
        self.encoders = encoders
        self.feature_cols = feature_cols
        self.thresholds = thresholds
        # CatBoost feature importance is the cheapest interpretable signal we
        # can ship without re-fitting; cache it once at load time.
        try:
            self._feat_importance = dict(zip(
                feature_cols, cb.get_feature_importance().tolist()
            ))
        except Exception:
            self._feat_importance = {c: 0.0 for c in feature_cols}

    @classmethod
    def load(cls, models_dir: Path | str = MODELS) -> "FraudDetector":
        models_dir = Path(models_dir)
        from catboost import CatBoostClassifier
        from xgboost import XGBClassifier
        import lightgbm as lgbm

        cb = CatBoostClassifier()
        cb.load_model(str(models_dir / "cb.cbm"))

        xgb = XGBClassifier()
        xgb.load_model(str(models_dir / "xgb.json"))

        lgb_booster = lgbm.Booster(model_file=str(models_dir / "lgb.txt"))

        encoders = load_encoders(models_dir / "freq_encoders.json")
        feature_cols = json.loads((models_dir / "feature_cols.json").read_text())
        thresholds = json.loads((models_dir / "threshold.json").read_text())
        return cls(cb, xgb, lgb_booster, encoders, feature_cols, thresholds)

    def _featurize(self, txn: dict) -> pd.DataFrame:
        """Convert one transaction dict to a 53-column float32 row.

        The dict can pass either:
            (a) raw merchant/state/city + all 50 engineered features, or
            (b) a fully prepared row including freq_* (sample_test_transactions).
        Missing freq_* columns are filled by applying the saved encoders to
        the raw merchant/state/city fields (unseen values map to 0)."""
        row = pd.DataFrame([txn])
        if not all(c in row.columns for c in FREQ_FEATURES):
            for col in FREQ_COLS:
                if col not in row.columns:
                    row[col] = ""  # encoder will return 0 for unseen
            row = apply_frequency_encoders(row, self.encoders)
        return materialize_features(row)

    def _ensemble_probs(self, X: pd.DataFrame):
        p_cb = float(self.cb.predict_proba(X)[:, 1][0])
        p_xgb = float(self.xgb.predict_proba(X)[:, 1][0])
        p_lgb = float(self.lgb.predict(X.values)[0])
        p_avg = (p_cb + p_xgb + p_lgb) / 3.0
        return p_avg, dict(catboost=p_cb, xgboost=p_xgb, lightgbm=p_lgb)

    def predict_one(self, transaction: dict, use_cost_optimal: bool = True,
                     top_k: int = 5) -> PredictionResult:
        t0 = time.perf_counter()
        X = self._featurize(transaction)
        p_avg, inds = self._ensemble_probs(X)
        thr = self.thresholds["cost_optimal"] if use_cost_optimal else self.thresholds["default_05"]
        latency_ms = (time.perf_counter() - t0) * 1000

        # Top-K contributing features by CatBoost importance, weighted by the
        # row's z-scored value so the explanation is row-specific. This is a
        # cheap proxy for SHAP that returns in <1ms (real SHAP would be ~50ms).
        x_row = X.iloc[0]
        contribs = []
        for c in self.feature_cols:
            imp = self._feat_importance.get(c, 0.0)
            val = float(x_row[c])
            score = imp * abs(val) / (abs(val) + 1.0)  # tanh-like smoothing
            contribs.append({"feature": c, "value": val, "importance": float(imp),
                             "contribution_score": float(score)})
        contribs.sort(key=lambda d: d["contribution_score"], reverse=True)
        top = contribs[:top_k]

        return PredictionResult(
            prob=float(p_avg),
            alert=bool(p_avg >= thr),
            threshold=float(thr),
            default_threshold=float(self.thresholds["default_05"]),
            cost_optimal_threshold=float(self.thresholds["cost_optimal"]),
            individual_probs=inds,
            top_features=top,
            latency_ms=latency_ms,
        )

    def predict_batch(self, df_with_features: pd.DataFrame) -> np.ndarray:
        """Vectorized batch inference. Expects a DataFrame already containing
        the 53 feature columns (or raw + merchant/state/city if encoders need
        to be applied)."""
        if not all(c in df_with_features.columns for c in FREQ_FEATURES):
            df_with_features = apply_frequency_encoders(df_with_features, self.encoders)
        X = materialize_features(df_with_features)
        p_cb = self.cb.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict(X.values)
        return ((p_cb + p_xgb + p_lgb) / 3.0).astype(np.float32)


__all__ = ["FraudDetector", "PredictionResult"]

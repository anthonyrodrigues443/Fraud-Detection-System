"""
Phase 7 (Mark) -- FastAPI inference service.

Streamlit (`app.py`) is the human-facing demo; this is the machine-facing
production surface. Endpoints:

    GET  /health              -> liveness probe (no model load required)
    GET  /info                -> model card metadata: features, thresholds, metrics
    POST /predict             -> single-transaction prediction
    POST /predict_batch       -> batch prediction (up to 10_000 rows)

Run locally:

    uvicorn api:app --host 0.0.0.0 --port 8000

The lazy `_get_detector()` indirection lets tests stub out FraudDetector
without touching the on-disk booster artefacts (which avoids the Windows
LightGBM model-format incompatibility surfaced during Phase 7 testing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parent
MODELS = REPO / "models"
sys.path.insert(0, str(REPO / "src"))

from data_pipeline import CLEAN_STACK_53  # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Production fraud-detection ensemble (CatBoost + XGBoost + LightGBM "
        "simple-average) served as a JSON HTTP API. Companion to the "
        "Streamlit demo at `app.py`."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Lazy detector (cached after first call)
# ---------------------------------------------------------------------------

_DETECTOR_SINGLETON: Optional[object] = None


def _get_detector():
    """Lazy-load the FraudDetector. Tests can monkeypatch this function."""
    global _DETECTOR_SINGLETON
    if _DETECTOR_SINGLETON is None:
        from predict import FraudDetector  # noqa: WPS433
        _DETECTOR_SINGLETON = FraudDetector.load(MODELS)
    return _DETECTOR_SINGLETON


def _reset_detector():
    """Test helper - clears the singleton between tests."""
    global _DETECTOR_SINGLETON
    _DETECTOR_SINGLETON = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Feature dict. Pass any subset of the 53 production features; "
            "missing values default to 0.0. Pass merchant/state/city as "
            "raw strings if you want frequency encoding to apply."
        ),
    )
    use_cost_optimal: bool = Field(
        True,
        description="If true, use the Phase-4 cost-optimal threshold (~0.11). Else 0.5.",
    )
    top_k: int = Field(
        5,
        ge=0,
        le=53,
        description="Number of top-K contributing features to return per prediction.",
    )


class PredictResponse(BaseModel):
    prob: float
    alert: bool
    threshold: float
    default_threshold: float
    cost_optimal_threshold: float
    individual_probs: Dict[str, float]
    top_features: List[Dict] = []
    latency_ms: float = 0.0


class BatchPredictRequest(BaseModel):
    rows: List[Dict[str, float]] = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Up to 10,000 transaction rows. Same schema as /predict.features.",
    )
    use_cost_optimal: bool = True


class BatchPredictResponse(BaseModel):
    n: int
    probs: List[float]
    alerts: List[bool]
    threshold: float


class InfoResponse(BaseModel):
    name: str
    version: str
    feature_count: int
    feature_names: List[str]
    thresholds: Dict[str, float]
    metrics: Dict[str, Dict[str, float]]
    production_pick: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe. Does NOT load the predictor."""
    return {"status": "ok", "service": "fraud-detection-api"}


@app.get("/info", response_model=InfoResponse)
def info():
    """Model card metadata. Reads JSON artefacts from disk; no booster load."""
    metrics_path = MODELS / "production_metrics.json"
    threshold_path = MODELS / "threshold.json"
    feature_cols_path = MODELS / "feature_cols.json"

    if not all(p.exists() for p in (metrics_path, threshold_path, feature_cols_path)):
        raise HTTPException(503, detail="Model artefacts missing - run train_production.py first")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    feature_names = json.loads(feature_cols_path.read_text(encoding="utf-8"))

    return InfoResponse(
        name="Fraud Detection Ensemble",
        version="1.0.0",
        feature_count=len(feature_names),
        feature_names=feature_names,
        thresholds={k: v for k, v in thresholds.items() if isinstance(v, (int, float))},
        metrics={
            name: {
                k: v for k, v in m.items()
                if isinstance(v, (int, float)) and k != "model"
            }
            for name, m in metrics["models"].items()
        },
        production_pick=metrics.get("production_pick", "ensemble_simple_avg"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    detector = _get_detector()
    # Fill missing features with 0.0 so callers can send minimal payloads
    feats = {c: 0.0 for c in CLEAN_STACK_53}
    feats.update(req.features)
    try:
        result = detector.predict_one(feats, use_cost_optimal=req.use_cost_optimal, top_k=req.top_k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, detail=f"prediction failed: {e}")
    return JSONResponse(content=result.to_dict())


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    import pandas as pd

    detector = _get_detector()

    rows = []
    for r in req.rows:
        full = {c: 0.0 for c in CLEAN_STACK_53}
        full.update(r)
        # Pass-through merchant/state/city for freq encoding
        for k in ("merchant", "state", "city"):
            if k in r:
                full[k] = r[k]
        rows.append(full)

    df = pd.DataFrame(rows)
    try:
        probs = detector.predict_batch(df)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, detail=f"batch prediction failed: {e}")

    thr = detector.thresholds["cost_optimal"] if req.use_cost_optimal else 0.5
    probs_list = [float(p) for p in probs]
    alerts = [p >= thr for p in probs_list]

    return BatchPredictResponse(n=len(probs_list), probs=probs_list, alerts=alerts, threshold=thr)


# ---------------------------------------------------------------------------
# Module-level helper for testing/CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

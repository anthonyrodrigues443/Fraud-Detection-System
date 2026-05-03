# Phase 7 (Mark) -- production container for the FastAPI inference service.
#
# This image bundles:
#   - the trained CatBoost / XGBoost / LightGBM artefacts (under models/)
#   - the FastAPI app (api.py)
#   - the data_pipeline + predict modules (src/)
#
# It does NOT bundle the raw data, notebooks, or training scripts - those
# stay out via .dockerignore.
#
# Build:
#   docker build -t fraud-detection-api:1.0.0 .
#
# Run:
#   docker run --rm -p 8000:8000 fraud-detection-api:1.0.0
#
# Health probe:
#   curl http://localhost:8000/health

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install only the runtime dependencies needed for the API + boosters
RUN pip install --upgrade pip && \
    pip install \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "pydantic>=2.5" \
        "pandas>=2.0" \
        "numpy>=1.24" \
        "scikit-learn>=1.3" \
        "catboost>=1.2" \
        "xgboost>=2.0" \
        "lightgbm>=4.0"

# Copy only the runtime surface (models, src, api). .dockerignore filters out
# tests, notebooks, raw data, and processed parquet.
COPY src/ /app/src/
COPY models/ /app/models/
COPY api.py /app/api.py

EXPOSE 8000

# Healthcheck calls /health (no model load required, ~1ms response).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
                   r=urllib.request.urlopen('http://localhost:8000/health',timeout=3); \
                   sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

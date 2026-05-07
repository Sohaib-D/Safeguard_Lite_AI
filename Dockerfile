FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app


FROM base AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip wheel --wheel-dir /wheels -r requirements.txt


FROM base AS runtime

COPY --from=builder /wheels /wheels
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY . .

RUN mkdir -p /app/data /app/logs /app/models/cache/predictions


FROM runtime AS backend

EXPOSE 8000

ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    SAFEGUARD_DB_PATH=/app/data/safeguard_ai.db \
    BACKEND_LOG_FILE=/app/logs/backend.log \
    MODEL_BUNDLE_PATH=/app/models/trained_multiclass_smoke/best_model.pkl \
    PREDICTION_CACHE_DIR=/app/models/cache/predictions

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM runtime AS frontend

EXPOSE 8501

ENV SAFEGUARD_API_BASE_URL=http://backend:8000 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

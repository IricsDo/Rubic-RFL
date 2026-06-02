FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/app backend/app
COPY rl/pyproject.toml rl/pyproject.toml
COPY rl/rubic_rl rl/rubic_rl

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels ./rl ./backend

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUBIC_SESSION_STORE_DIR=/data/sessions \
    RUBIC_RL_MODEL_PATH=/models/mlp-baseline-depth-1-2-3.npz \
    RUBIC_RL_POLICY_TYPE=auto \
    RUBIC_RL_POLICY_DEVICE=cpu \
    RUBIC_RL_MAX_STEPS=30 \
    RUBIC_RL_SEARCH_WIDTH=5 \
    RUBIC_RL_SEARCH_TOP_K=5 \
    RUBIC_RL_SEARCH_TRACE_LIMIT=200

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app --home /app app \
    && mkdir -p /data/sessions /models \
    && chown -R app:app /app /data /models

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

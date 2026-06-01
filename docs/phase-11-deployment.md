# Phase 11 Deployment

Phase 11 makes the current MVP reproducible in containers. The stack runs the static frontend, FastAPI backend, PostgreSQL, and Redis with named volumes for durable local state.

## Services

- `frontend`: Nginx-served static UI on `http://127.0.0.1:5173`.
- `backend`: FastAPI API on `http://127.0.0.1:8000` with Prometheus-compatible metrics at `/metrics`.
- `postgres`: local development database initialized from `backend/app/database/migrations/`.
- `redis`: queue/cache placeholder for future worker orchestration.

The dedicated worker service is deferred until a real background queue module exists.

## Local Run

Create a local environment file, then start the stack:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Verify the backend:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Verify the metrics endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

Open `http://127.0.0.1:5173` and keep the UI API base set to `http://127.0.0.1:8000`.

## RL Checkpoints

The backend mounts `./checkpoints` as `/models`. Set `RUBIC_RL_MODEL_FILE` in `.env` to the checkpoint filename to use for RL solving. If the file is missing, classical solving remains available and RL solving reports an unavailable checkpoint.

## Persistence

Replay sessions are stored in the `replay_sessions` Docker volume at `/data/sessions`. PostgreSQL data is stored in `postgres_data`, and Redis append-only data is stored in `redis_data`.

The PostgreSQL migration directory runs only when the database volume is first created. For schema resets, recreate the development volume intentionally.

## Monitoring

The backend exposes plain Prometheus text at `/metrics`. The current in-process registry tracks:

- HTTP request count, server error count, and request duration by method, route, and status.
- WebSocket accepted connections, active connections, and handler errors by endpoint.
- Solver runs, solved runs, solver duration, move totals, and model version usage.

Useful local checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics | Select-String "rubic_"
```

These metrics are enough for a first Grafana dashboard over API latency, error rate, active streams, solve rate, inference duration, and model usage. A dedicated Prometheus/Grafana service can be added to Compose once dashboards and alert rules are ready.

## CI

`.github/workflows/test.yml` now keeps Python tests and the performance smoke benchmark, then validates Compose configuration and builds both Docker images.

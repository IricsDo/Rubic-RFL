# Phase 11 Deployment

Phase 11 makes the current MVP reproducible in containers. The stack runs the static frontend, FastAPI backend, PostgreSQL, and Redis with named volumes for durable local state. An optional monitoring profile adds Prometheus and Grafana.

## Services

- `frontend`: Nginx-served static UI on `http://127.0.0.1:5173`.
- `backend`: FastAPI API on `http://127.0.0.1:8000` with Prometheus-compatible metrics at `/metrics`.
- `postgres`: local development database initialized from `backend/app/database/migrations/`.
- `redis`: queue/cache placeholder for future worker orchestration.
- `prometheus`: optional metrics scraper on `http://127.0.0.1:9090`.
- `grafana`: optional dashboard UI on `http://127.0.0.1:3000`.

The dedicated worker service is deferred until a real background queue module exists.

## Local Run

Create a local environment file, then start the stack:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

PostgreSQL and Redis stay private to the Compose network by default. If you
need host access to the container database, add the host-port override:

```powershell
docker compose -f docker-compose.yml -f docker-compose.host-ports.yml up --build
```

That override binds PostgreSQL to `127.0.0.1:5433`, while containers still use
`postgres:5432` internally. This avoids conflicts with a local PostgreSQL
service already using host port `5432`.

Verify the backend:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Verify the metrics endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

Open `http://127.0.0.1:5173` and keep the UI API base set to `http://127.0.0.1:8000`.

Run with local monitoring:

```powershell
docker compose --profile monitoring up --build
```

Grafana loads the `Rubic RFL Overview` dashboard automatically from
`monitoring/grafana/dashboards/rubic-rfl-overview.json`.

## RL Checkpoints

The backend mounts `./checkpoints` as `/models`. Set `RUBIC_RL_MODEL_FILE` in `.env` to the checkpoint filename to use for RL solving. If the file is missing, classical solving remains available and RL solving reports an unavailable checkpoint.

## Persistence

Replay sessions use PostgreSQL by default in Compose through `DATABASE_URL` and
`RUBIC_SESSION_STORAGE_BACKEND=postgres`. The backend applies the Phase 9
migration before the first session operation and writes both the canonical
replay JSONB artifact and structured solve rows. Set
`RUBIC_SESSION_STORAGE_BACKEND=files` to use the JSON fallback in the
`replay_sessions` Docker volume at `/data/sessions`.

PostgreSQL data is stored in `postgres_data`, and Redis append-only data is
stored in `redis_data`.

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

Prometheus is configured by `monitoring/prometheus/prometheus.yml`, with local
alert rules in `monitoring/prometheus/rules/rubic-rfl.yml`. The Grafana
dashboard covers API traffic, server error rate, average latency, active
WebSockets, solver run rate, solver success rate, solver latency, and model
version usage.

## Production Path

The `Publish Images` workflow in `.github/workflows/publish-images.yml`
publishes backend and frontend images to GitHub Container Registry when run
manually or when a `v*` tag is pushed.

For a single VPS or VM deployment:

```powershell
Copy-Item deploy\production.env.example deploy\production.env
docker compose -f deploy\compose.production.yml --env-file deploy\production.env up -d
```

Edit `deploy\production.env` before deployment with real GHCR image names,
a strong PostgreSQL password, public ports, and checkpoint settings. Put a TLS
reverse proxy in front of the frontend and backend before exposing the service.

## CI

`.github/workflows/test.yml` keeps Python tests and performance smoke
benchmarks, validates the default Compose file, monitoring profile, host-port
override, production Compose file, and Prometheus rules, then builds both Docker
images.

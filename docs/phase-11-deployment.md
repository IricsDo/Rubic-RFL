# Phase 11 Deployment

Phase 11 makes the current MVP reproducible in containers. The stack runs the static frontend, FastAPI backend, PostgreSQL, and Redis with named volumes for durable local state. Optional monitoring profiles add Prometheus and Grafana through the production Compose file.

## Services

- `frontend`: Nginx-served static UI on `http://127.0.0.1`.
- `backend`: FastAPI API on `http://127.0.0.1:8000` with Prometheus-compatible metrics at `/metrics`.
- `postgres`: local development database initialized from `backend/app/database/migrations/`.
- `redis`: queue/cache placeholder for future worker orchestration.
- `prometheus`: optional metrics scraper on `http://127.0.0.1:9090`.
- `grafana`: optional dashboard UI on `http://127.0.0.1:3000`.

The dedicated worker service is deferred until a real background queue module exists.

## Local Run

Create the production-style local environment file, then start the active stack:

```powershell
.\.venv\Scripts\python.exe tools\prepare_production_env.py --generate-secrets --force
docker compose -f deploy\compose.production.yml --env-file deploy\production.env up -d
```

PostgreSQL and Redis stay private to the Compose network by default. Host-port
override Compose files were removed so local runs use the same `rubic-rfl-prod`
project name and do not accidentally start an extra stack.

Verify the backend:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Verify the metrics endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

Open `http://127.0.0.1` and keep the UI API base set to `http://127.0.0.1:8000`.

Run with monitoring:

```powershell
docker compose -f deploy\compose.production.yml --env-file deploy\production.env --profile monitoring up -d
```

Grafana loads the `Rubic RFL Overview` dashboard automatically from
`monitoring/grafana/dashboards/rubic-rfl-overview.json`.

## RL Checkpoints

The backend mounts `./checkpoints` as `/models`. Set `RUBIC_RL_MODEL_FILE` in
`.env` to the checkpoint filename to use for RL solving, and keep
`RUBIC_RL_POLICY_TYPE=auto` for normal `.npz` policies. For `.pt` Torch
checkpoints, set `RUBIC_RL_POLICY_TYPE=torch` and use an image/runtime with
PyTorch installed. `RUBIC_RL_POLICY_DEVICE` defaults to `cpu`; use `cuda` only
with a CUDA-capable backend runtime. If the file is missing, classical solving
remains available and RL solving reports an unavailable checkpoint.

After the backend is healthy, verify the configured RL checkpoint:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/solve/rl/status?load=true"
```

The response should show `status` as `loaded`, `checkpoint_exists` as `true`,
and `load_error` as `null`.

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
.\.venv\Scripts\python.exe tools\prepare_production_env.py --generate-secrets --force
docker compose -f deploy\compose.production.yml --env-file deploy\production.env up -d
```

Use `--image-namespace ghcr.io/<owner>/<repo> --image-tag <sha>` when the
deployment host is not checked out at the exact published revision. Edit
`deploy\production.env` only for public ports, checkpoint settings, or managed
service connection details. Use immutable image tags from the publish workflow
rather than `:latest` for production rollouts.

Validate release settings before starting the stack:

```powershell
.\.venv\Scripts\python.exe tools\verify_deployment.py --env-file deploy\production.env --strict --require-model-file
docker compose -f deploy\compose.production.yml --env-file deploy\production.env config
docker compose -f deploy\compose.production.yml --env-file deploy\production.env --profile monitoring config
```

To include Prometheus and Grafana in a single-host production deployment, add
the monitoring profile:

```powershell
docker compose -f deploy\compose.production.yml --env-file deploy\production.env --profile monitoring up -d
```

Put a TLS reverse proxy in front of the frontend, backend, and any exposed
monitoring UI before exposing the service.

## Production Smoke Checks

After deployment, run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod "http://127.0.0.1:8000/solve/rl/status?load=true"
.\.venv\Scripts\python.exe tools\load_test_network.py --base-url http://127.0.0.1:8000 --requests 60 --concurrency 6 --warmups 6 --out reports\network-load-smoke.json --enforce-budgets
```

When monitoring is enabled, open Grafana and confirm the Rubic RFL dashboard is
receiving HTTP, WebSocket, solver, latency, and model-version metrics.

## CI

`.github/workflows/test.yml` keeps Python tests and performance smoke
benchmarks, validates the default Compose file, monitoring profile, host-port
override, production Compose file, production monitoring profile, deployment env
schema, and Prometheus rules, then builds both Docker images and runs an
external Docker/Uvicorn network smoke test.

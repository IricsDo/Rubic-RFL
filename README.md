# Rubic RFL

Real-time Rubik's Cube solver platform based on the implementation plan in
`Real-Time Rubik's Cube Solver Using Reinforcement Learning.md`.

The first milestone focuses on a working MVP foundation:

- Python cube engine with legal move application, scrambles, inverse moves, and cubie legality checks.
- FastAPI backend scaffold for validation, scrambling, move application, and solver endpoints.
- Dependency-light browser UI for cube manipulation, backend-backed scrambling, validation, Kociemba solving, and replay.
- Gymnasium-compatible RL environment that reuses the shared cube engine.
- Deterministic JSONL and canonical short-depth dataset generators for supervised RL warm-start data.
- NumPy supervised baseline policy for predicting the next solving action.
- NumPy MLP supervised baseline for nonlinear next-action prediction.
- Policy evaluation rollouts that report solve rates by scramble depth.
- Policy comparison reports for ranking checkpoints on identical scrambles.
- One-command baseline experiment runner for dataset, training, comparison, and reporting.
- Backend RL policy-guided beam search endpoint available from the UI solver mode switch.
- RL search benchmark reports that compare greedy rollout against beam search.
- RL decision trace panel with model metadata, Top-K confidence bars, and active move highlighting.
- WebSocket RL solve streaming with live move updates and REST fallback.
- Bounded RL branch trace viewer for kept, pruned, skipped, and solved search paths.
- Replay package generation for reproducible RL solve inspection and import/export.
- Runtime solve session persistence with PostgreSQL when configured and JSON-file fallback for local use.
- Historical solve analytics API over persisted replay sessions.
- PostgreSQL migration and runtime adapter for sessions, cube states, solves, moves, solver runs, model versions, and metrics.
- GitHub Actions test pipeline plus Python coverage and API/WebSocket/frontend FPS performance smoke benchmarks.
- Docker Compose deployment baseline for frontend, backend, PostgreSQL, and Redis.
- Prometheus-compatible backend `/metrics` endpoint for API, WebSocket, solver, and model-version monitoring.
- Optional Prometheus and Grafana monitoring profile with a preprovisioned Phase 11 dashboard.

## Project Layout

```text
backend/   Python API, cube engine, solver interfaces, tests
frontend/  Static MVP web app
rl/        Reinforcement-learning research placeholders
docs/      Architecture decisions and implementation notes
docker-compose.yml  Local container orchestration for Phase 11
deploy/    Production Compose baseline and image publishing notes
```

## Development Commands

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```powershell
cd backend
python -m pip install -e ".[dev]"
cd ..
```

Install RL research dependencies:

```powershell
cd rl
python -m pip install -e ".[dev]"
cd ..
```

Run backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend/tests
```

Run frontend static smoke tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider frontend/tests
```

Install and run frontend E2E tests:

```powershell
$env:PATH="D:\AppPrograms\nodejs;$env:PATH"
npm.cmd install
npx.cmd playwright install chromium
npm.cmd run test:e2e
```

Run only the frontend animation FPS smoke check:

```powershell
npm.cmd run test:e2e:fps
```

Run RL environment tests:

```powershell
cd rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
cd ..
```

Run the combined backend and RL coverage report:

```powershell
.\.venv\Scripts\python.exe tools\coverage_report.py --out-dir reports
```

Run PostgreSQL integration and runtime storage tests when a local database is available:

```powershell
$env:RUBIC_TEST_DATABASE_URL="postgresql://postgres:admin@localhost:5432/rubic_rfl_test"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend\tests\test_postgres_integration.py backend\tests\test_postgres_runtime_store.py
```

Run the API performance smoke benchmark:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_api.py --iterations 10 --warmups 2 --out reports\performance-smoke.json
```

Run the WebSocket performance smoke benchmark:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_websocket.py --iterations 10 --warmups 2 --out reports\websocket-performance-smoke.json
```

Run the concurrent backend load smoke test:

```powershell
.\.venv\Scripts\python.exe tools\load_test_backend.py --requests 60 --concurrency 6 --warmups 6 --out reports\backend-load-smoke.json
```

The CI workflow in `.github/workflows/test.yml` installs both Python packages,
runs backend, PostgreSQL integration, frontend static smoke, and RL tests,
writes combined backend/RL coverage XML and JSON reports, executes API,
WebSocket, backend load, and frontend FPS smoke checks, and uploads the
generated report files as artifacts. It also validates the Docker Compose
configuration, monitoring profile, host-port override, production Compose
baseline, and Prometheus rules, builds the backend and frontend images, and
runs Playwright Chromium E2E tests against the static UI.

Run the containerized stack:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The frontend is available at `http://127.0.0.1:5173`, and the backend health
endpoint is available at `http://127.0.0.1:8000/health`. See
`docs/phase-11-deployment.md` for service details and persistence notes.
PostgreSQL and Redis stay private to the Compose network by default. If you
need host access to the container database, add the host-port override:

```powershell
docker compose -f docker-compose.yml -f docker-compose.host-ports.yml up --build
```

That override binds PostgreSQL to `127.0.0.1:5433`, avoiding the common local
PostgreSQL conflict on `5432`.

Run the stack with the monitoring dashboard:

```powershell
docker compose --profile monitoring up --build
```

Prometheus is available at `http://127.0.0.1:9090`, and Grafana is available at
`http://127.0.0.1:3000` with the credentials from `.env`.

Generate a small supervised dataset:

```powershell
cd rl
..\.venv\Scripts\python.exe -m rubic_rl.datasets.generator --out ..\datasets\smoke.jsonl --depths 1 2 3 --samples-per-depth 10 --seed 20260531 --include-solved
cd ..
```

Generate a canonical short-depth dataset:

```powershell
cd rl
..\.venv\Scripts\python.exe -m rubic_rl.datasets.canonical --out ..\datasets\canonical-depth-1-2.jsonl --depths 1 2
cd ..
```

Train the supervised baseline:

```powershell
cd rl
..\.venv\Scripts\python.exe -m rubic_rl.training.supervised --dataset ..\datasets\smoke.jsonl --model-out ..\checkpoints\linear-policy-smoke.npz --epochs 200 --learning-rate 0.2 --batch-size 128 --validation-split 0.2 --seed 20260531
cd ..
```

Train the MLP supervised baseline:

```powershell
cd rl
..\.venv\Scripts\python.exe -m rubic_rl.training.mlp_supervised --dataset ..\datasets\smoke.jsonl --model-out ..\checkpoints\mlp-policy-smoke.npz --hidden-units 64 --epochs 300 --learning-rate 0.05 --batch-size 128 --validation-split 0.2 --seed 20260531
cd ..
```

Evaluate a trained policy:

```powershell
cd rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.policy_eval --model ..\checkpoints\mlp-policy-smoke.npz --policy-type auto --depths 1 2 3 --samples-per-depth 100 --max-steps 30 --seed 20260531
cd ..
```

Compare trained policies on the same seeded rollouts:

```powershell
cd rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.compare_policies --model linear=..\checkpoints\linear-policy-smoke.npz --model mlp=..\checkpoints\mlp-policy-smoke.npz --depths 1 2 3 --samples-per-depth 100 --max-steps 30 --seed 20260531 --out ..\reports\policy-comparison.json
cd ..
```

Benchmark greedy policy rollout against policy-guided beam search:

```powershell
cd rl
..\.venv\Scripts\python.exe -m rubic_rl.evaluation.search_eval --model ..\checkpoints\mlp-baseline-depth-1-2-3.npz --policy-type auto --depths 1 2 3 --samples-per-depth 100 --max-depth 30 --beam-width 5 --top-k 5 --seed 20260531 --out ..\reports\search-evaluation.json
cd ..
```

Run the reproducible baseline experiment pipeline:

```powershell
cd rl
..\.venv\Scripts\python.exe -m rubic_rl.training.baseline_experiment --depths 1 2 3 --samples-per-depth 100 --evaluation-samples-per-depth 100 --evaluation-max-steps 30 --seed 20260531
cd ..
```

This prepends canonical depth 1-2 records by default, removes duplicate cube
states, and writes ignored dataset, checkpoint, and report files under
`datasets/`, `checkpoints/`, and `reports/`. The report includes the top-ranked
checkpoint path to use with the backend.

Run the FastAPI backend:

```powershell
$env:RUBIC_RL_MODEL_PATH="checkpoints\mlp-baseline-depth-1-2-3.npz"
$env:RUBIC_RL_MAX_STEPS="30"
$env:RUBIC_RL_SEARCH_WIDTH="5"
$env:RUBIC_RL_SEARCH_TOP_K="5"
$env:RUBIC_RL_SEARCH_TRACE_LIMIT="200"
$env:RUBIC_SESSION_DATABASE_URL="postgresql://postgres:admin@localhost:5432/rubic_rfl_test"
cd backend
..\.venv\Scripts\uvicorn.exe app.main:app --reload
```

In a second terminal, serve the frontend locally:

```powershell
python -m http.server 5173 -d frontend
```

Open the static UI at `http://localhost:5173`. Use the Backend panel to check
`http://127.0.0.1:8000`; if the API is offline, the UI falls back to local cube
logic. Use the Replay panel's Classical/RL mode switch to choose the solver.
Classical mode calls `/solve/classical`; RL mode opens `/ws/solve/{session_id}`
first for live decision and move streaming, then falls back to
`/solve/rl/replay-package` if the stream is unavailable. RL solves show the
policy-guided search trace and branch records beside the replay and enable
`Export Replay` for a reloadable solve artifact. The Saved Sessions panel calls
`/sessions`, loads stored replay packages, and saves the current replay package
to the configured session store. The backend uses PostgreSQL when
`RUBIC_SESSION_DATABASE_URL` or `DATABASE_URL` is set; otherwise it falls back
to JSON files under `RUBIC_SESSION_STORE_DIR` or `replays/sessions`.

Verify the API directly:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected result: `status` is `ok`.

List persisted sessions:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/sessions?limit=10"
```

Inspect historical solve analytics:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/analytics/solves?limit=100"
```

Inspect Prometheus-compatible metrics:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/metrics"
```

The backend applies the first production PostgreSQL migration automatically when
PostgreSQL session storage is active. To preapply it to a development database,
run:

```powershell
psql $env:DATABASE_URL -f backend/app/database/migrations/0001_session_persistence.sql
```

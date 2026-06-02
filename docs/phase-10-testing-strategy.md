# Phase 10 Testing Strategy

## Scope

Phase 10 establishes a repeatable verification baseline for the backend API,
cube engine, static frontend contract, RL research package, and basic solver
performance. The project is still a static frontend plus Python services, so the
first CI pipeline stays focused on deterministic checks and a lightweight API
benchmark.

## CI Pipeline

The GitHub Actions workflow lives at:

```text
.github/workflows/test.yml
```

It runs on pushes, pull requests, and manual dispatches. The job installs both
editable Python packages with development extras, then runs:

```bash
python -m pytest -p no:cacheprovider backend/tests
RUBIC_TEST_DATABASE_URL=postgresql://rubic:rubic@localhost:5432/rubic_rfl_test python -m pytest -p no:cacheprovider backend/tests/test_postgres_integration.py backend/tests/test_postgres_runtime_store.py
python -m pytest -p no:cacheprovider frontend/tests
cd rl && python -m pytest -p no:cacheprovider tests
python tools/coverage_report.py --out-dir reports
python tools/benchmark_api.py --iterations 5 --warmups 1 --out reports/performance-smoke.json --enforce-budgets
python tools/benchmark_websocket.py --iterations 5 --warmups 1 --out reports/websocket-performance-smoke.json --enforce-budgets
python tools/load_test_backend.py --requests 24 --concurrency 4 --warmups 4 --out reports/backend-load-smoke.json --enforce-budgets
python tools/load_test_network.py --base-url http://127.0.0.1:8000 --requests 24 --concurrency 4 --warmups 4 --out reports/network-load-smoke.json --enforce-budgets
npm run test:e2e
```

Coverage and benchmark reports are uploaded as workflow artifacts when
available. The frontend E2E job also uploads
`reports/frontend-fps-smoke.json` when the animation FPS check runs.

## Local Verification

From the repository root, use the same checks locally:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend/tests
$env:RUBIC_TEST_DATABASE_URL="postgresql://rubic:rubic@localhost:5432/rubic_rfl_test"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend\tests\test_postgres_integration.py backend\tests\test_postgres_runtime_store.py
Remove-Item Env:\RUBIC_TEST_DATABASE_URL
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider frontend/tests
cd rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
cd ..
.\.venv\Scripts\python.exe tools\coverage_report.py --out-dir reports
.\.venv\Scripts\python.exe tools\benchmark_api.py --iterations 10 --warmups 2 --out reports\performance-smoke.json
.\.venv\Scripts\python.exe tools\benchmark_websocket.py --iterations 10 --warmups 2 --out reports\websocket-performance-smoke.json
.\.venv\Scripts\python.exe tools\load_test_backend.py --requests 60 --concurrency 6 --warmups 6 --out reports\backend-load-smoke.json
.\.venv\Scripts\python.exe tools\load_test_network.py --base-url http://127.0.0.1:8000 --requests 60 --concurrency 6 --warmups 6 --out reports\network-load-smoke.json
npm.cmd run test:e2e:fps
```

Generated reports are written under `reports/`. Coverage outputs are ignored by
source control and benchmark JSON files are intended as regenerated CI
artifacts.

## Coverage Reporting

`tools/coverage_report.py` runs backend and RL tests under `coverage.py` with
branch coverage enabled. It writes:

- `reports/coverage.xml` for CI services that ingest Cobertura-style reports
- `reports/coverage.json` for local inspection and future dashboards

No minimum threshold is enforced yet. Add `--fail-under <percent>` once the
baseline stabilizes and the team agrees on a gate.

## PostgreSQL Integration Tests

`backend/tests/test_postgres_integration.py` applies the Phase 9 migration to a
real PostgreSQL database, verifies the planned persistence tables are present,
and exercises inserts across users, sessions, cube states, solves, moves, model
versions, solver runs, and metrics. The test is skipped unless
`RUBIC_TEST_DATABASE_URL` or `DATABASE_URL` is set.

`backend/tests/test_postgres_runtime_store.py` verifies the runtime PostgreSQL
session adapter through the REST API. It saves, lists, loads, summarizes, and
deletes a replay package, then checks that structured rows and cascade cleanup
match the Phase 9 schema.

## Frontend Static Smoke Tests

`frontend/tests/test_static_contract.py` verifies the static UI contract without
requiring Node or a browser runtime. It checks that:

- `index.html` references existing CSS and module JavaScript assets
- every `document.querySelector("#...")` ID in `app.js` exists in the HTML
- core cube, replay, RL decision trace, and saved-session controls are present
- frontend route references still match FastAPI route declarations
- responsive, replay, trace, and saved-session CSS selectors exist

## Frontend E2E Tests

Playwright tests live under:

```text
frontend/e2e/
```

Install Node dependencies and the Chromium browser once:

```powershell
npm.cmd install
npx.cmd playwright install chromium
```

Run the browser-driven checks:

```powershell
npm.cmd run test:e2e
```

The E2E suite serves `frontend/` with `python -m http.server`, blocks backend
requests to verify local fallback behavior, and covers workspace load,
scramble/replay controls, RL replay-package trace rendering, and the cube
animation FPS smoke check.

Run only the animation performance check with:

```powershell
npm.cmd run test:e2e:fps
```

The FPS spec samples `requestAnimationFrame` while the cube viewport is
animated with CSS 3D transforms. It writes `reports/frontend-fps-smoke.json`.
The target is 60 FPS. The default local budgets require average FPS >= 55 and
10th-percentile FPS >= 40; CI uses average FPS >= 45 and 10th-percentile FPS >=
30 to avoid noisy failures on shared runners. Override with
`RUBIC_FRONTEND_MIN_AVERAGE_FPS`, `RUBIC_FRONTEND_MIN_P10_FPS`, or
`RUBIC_FRONTEND_FPS_SAMPLE_MS`.

## Performance Smoke Benchmark

`tools/benchmark_api.py` uses FastAPI `TestClient` and does not require a
running server. It measures:

- `GET /health`
- `POST /cube/validate` for a solved cube
- `POST /cube/scramble` at depth 4
- `POST /solve/classical` for a deterministic depth-3 scramble
- `GET /analytics/solves`

Budgets are conservative p95 smoke thresholds, not production SLOs. They exist
to catch severe regressions while remaining stable in CI.

`tools/benchmark_websocket.py` uses the same in-process app with a deterministic
one-move RL policy. It measures:

- time to first `/ws/solve/{session_id}` event
- time to completed RL solve stream
- expected event sequence: `started`, `decision`, `move`, `completed`

The WebSocket benchmark writes `reports/websocket-performance-smoke.json`.

`tools/load_test_backend.py` runs a concurrent in-process mixed-route load test
against health, metrics, cube validation, scramble, apply-move, classical solve,
session listing, and solve analytics. It reports throughput, error rate,
overall latency percentiles, and per-scenario p95/p99 values to
`reports/backend-load-smoke.json`. The CI budget enforces zero request errors,
minimum throughput, overall p95 latency, and route-level p95 ceilings.

`tools/load_test_network.py` runs the same style of mixed-route check through a
real HTTP client against a running backend, normally the Docker/Uvicorn service
on `http://127.0.0.1:8000`. It also probes `/solve/rl/status` so container
runtime model configuration is covered by the external smoke test. It writes
`reports/network-load-smoke.json`.

`frontend/e2e/animation-fps.spec.js` records the frontend animation smoke
metrics for the MVP smoothness criterion and writes
`reports/frontend-fps-smoke.json`.

## Follow-Up Work

Next testing additions should add longer staging-scale load profiles and
production-like data volumes once deployment ports and environment
configuration are stable.

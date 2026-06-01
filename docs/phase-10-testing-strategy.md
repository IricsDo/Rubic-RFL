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
python -m pytest -p no:cacheprovider frontend/tests
cd rl && python -m pytest -p no:cacheprovider tests
python tools/benchmark_api.py --iterations 5 --warmups 1 --out reports/performance-smoke.json --enforce-budgets
```

The benchmark report is uploaded as a workflow artifact when available.

## Local Verification

From the repository root, use the same checks locally:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend/tests
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider frontend/tests
cd rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
cd ..
.\.venv\Scripts\python.exe tools\benchmark_api.py --iterations 10 --warmups 2 --out reports\performance-smoke.json
```

Generated reports are written under `reports/`, which is ignored by source
control.

## Frontend Static Smoke Tests

`frontend/tests/test_static_contract.py` verifies the static UI contract without
requiring Node or a browser runtime. It checks that:

- `index.html` references existing CSS and module JavaScript assets
- every `document.querySelector("#...")` ID in `app.js` exists in the HTML
- core cube, replay, RL decision trace, and saved-session controls are present
- frontend route references still match FastAPI route declarations
- responsive, replay, trace, and saved-session CSS selectors exist

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

## Follow-Up Work

Next testing additions should cover browser-driven frontend E2E behavior,
WebSocket latency, coverage reporting, and database-backed integration tests
once PostgreSQL is wired into runtime storage.

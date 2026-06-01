# Repository Guidelines

## Project Structure & Module Organization

This repository implements the first milestone of a real-time Rubik's Cube solver with a 3D web UI and reinforcement-learning research track. Keep files organized by responsibility:

- `frontend/` for the static MVP cube UI, replay controls, and browser assets.
- `backend/app/` for API routes, cube validation, solver orchestration, and WebSocket scaffolding.
- `backend/tests/` for cube engine and backend tests.
- `rl/` for future Rubik's Cube environments, training scripts, checkpoints, evaluation, and experiment configs.
- `tools/` for local verification utilities such as API performance benchmarks.
- `.github/workflows/` for CI pipelines.
- `docker-compose.yml`, `backend/Dockerfile`, and `frontend/Dockerfile` for containerized local deployment.
- `docs/` for architecture notes, research findings, and implementation plans.

Keep generated datasets, model checkpoints, and build outputs out of source control unless explicitly documented.

## Build, Test, and Development Commands

Run backend tests from the repository root:

- `.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend/tests` runs cube and API tests.
- `.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider frontend/tests` runs static frontend smoke tests.
- `npm.cmd run test:e2e` runs Playwright Chromium E2E tests for the static UI after Node dependencies and browsers are installed.
- `npm.cmd run test:e2e:fps` runs the Playwright cube animation FPS smoke check and writes `reports/frontend-fps-smoke.json`.
- `cd rl; ..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests` runs RL research tests.
- `.\.venv\Scripts\python.exe tools\coverage_report.py --out-dir reports` writes combined backend/RL coverage XML and JSON reports.
- `$env:RUBIC_TEST_DATABASE_URL="postgresql://postgres:admin@localhost:5432/rubic_rfl_test"; .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend\tests\test_postgres_integration.py backend\tests\test_postgres_runtime_store.py` runs PostgreSQL migration and runtime storage tests when a database is available.
- `.\.venv\Scripts\python.exe tools\benchmark_api.py --iterations 10 --warmups 2 --out reports\performance-smoke.json` records API p95 smoke metrics.
- `.\.venv\Scripts\python.exe tools\benchmark_websocket.py --iterations 10 --warmups 2 --out reports\websocket-performance-smoke.json` records WebSocket streaming p95 smoke metrics.
- `.\.venv\Scripts\python.exe tools\load_test_backend.py --requests 60 --concurrency 6 --warmups 6 --out reports\backend-load-smoke.json` records concurrent mixed-route backend load metrics.
- `docker compose up --build` starts the containerized frontend, backend, PostgreSQL, and Redis stack.
- `docker compose -f docker-compose.yml -f docker-compose.host-ports.yml up --build` additionally exposes PostgreSQL on `127.0.0.1:5433` for host-side database inspection.
- `docker compose --profile monitoring up --build` starts the same stack plus Prometheus and Grafana.
- `docker compose config` validates the Compose file before CI or deployment edits.
- `Invoke-RestMethod http://127.0.0.1:8000/metrics` inspects Prometheus-compatible backend metrics after the API is running.
- `python -m http.server 5173 -d frontend` serves the static MVP UI at `http://localhost:5173`.
- `cd backend; python -m pip install -e ".[dev]"` installs FastAPI backend dependencies when network access is available.
- `cd backend; ..\.venv\Scripts\uvicorn.exe app.main:app --reload` runs the FastAPI API after dependencies are installed.

If a command requires environment variables, add `.env.example` with safe placeholders.
Session storage uses PostgreSQL when `RUBIC_SESSION_DATABASE_URL` or `DATABASE_URL` is set; use `RUBIC_SESSION_STORAGE_BACKEND=files` for the JSON fallback.

## Coding Style & Naming Conventions

Use clear boundaries between UI, cube logic, API code, and RL experiments. Prefer descriptive names such as `cube_state`, `move_sequence`, `solver_session`, and `training_config`. Use `PascalCase` for UI components, `camelCase` for JavaScript/TypeScript values, and `snake_case` for Python files, functions, and variables.

Format future TypeScript with Prettier and Python with Black/Ruff when those tools are added. Avoid mixing formatting-only changes with behavior changes.

## Testing Guidelines

Prioritize tests around cube state validation, legal moves, scramble generation, solver output validity, API contracts, frontend UI contracts, Playwright E2E flows, metrics instrumentation, and replay serialization. Name tests by behavior, for example `test_single_edge_flip_is_rejected`, `test_app_dom_selectors_have_matching_html_ids`, or `static-ui.spec.js`.

Before submitting changes, run the relevant test suite and include any known gaps in the pull request.

## Commit & Pull Request Guidelines

This folder is not currently initialized as a Git repository, so no local commit convention is available. Use short imperative messages such as `Add cube state validator`.

Pull requests should include a summary, testing performed, linked issues, and screenshots or recordings for UI changes. For RL changes, include metrics, config details, and reproducibility notes.

## Agent-Specific Instructions

Keep generated contributor docs concise and update this file when the repository gains real tooling, directories, or conventions.

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
- `cd rl; ..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests` runs RL research tests.
- `.\.venv\Scripts\python.exe tools\benchmark_api.py --iterations 10 --warmups 2 --out reports\performance-smoke.json` records API p95 smoke metrics.
- `docker compose up --build` starts the containerized frontend, backend, PostgreSQL, and Redis stack.
- `docker compose config` validates the Compose file before CI or deployment edits.
- `Invoke-RestMethod http://127.0.0.1:8000/metrics` inspects Prometheus-compatible backend metrics after the API is running.
- `python -m http.server 5173 -d frontend` serves the static MVP UI at `http://localhost:5173`.
- `cd backend; python -m pip install -e ".[dev]"` installs FastAPI backend dependencies when network access is available.
- `cd backend; ..\.venv\Scripts\uvicorn.exe app.main:app --reload` runs the FastAPI API after dependencies are installed.

If a command requires environment variables, add `.env.example` with safe placeholders.

## Coding Style & Naming Conventions

Use clear boundaries between UI, cube logic, API code, and RL experiments. Prefer descriptive names such as `cube_state`, `move_sequence`, `solver_session`, and `training_config`. Use `PascalCase` for UI components, `camelCase` for JavaScript/TypeScript values, and `snake_case` for Python files, functions, and variables.

Format future TypeScript with Prettier and Python with Black/Ruff when those tools are added. Avoid mixing formatting-only changes with behavior changes.

## Testing Guidelines

Prioritize tests around cube state validation, legal moves, scramble generation, solver output validity, API contracts, frontend UI contracts, metrics instrumentation, and replay serialization. Name tests by behavior, for example `test_single_edge_flip_is_rejected` or `test_app_dom_selectors_have_matching_html_ids`.

Before submitting changes, run the relevant test suite and include any known gaps in the pull request.

## Commit & Pull Request Guidelines

This folder is not currently initialized as a Git repository, so no local commit convention is available. Use short imperative messages such as `Add cube state validator`.

Pull requests should include a summary, testing performed, linked issues, and screenshots or recordings for UI changes. For RL changes, include metrics, config details, and reproducibility notes.

## Agent-Specific Instructions

Keep generated contributor docs concise and update this file when the repository gains real tooling, directories, or conventions.

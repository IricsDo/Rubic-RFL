# Phase 9 Data Persistence

## Scope

The backend persists solver sessions behind a shared storage interface. Runtime
uses PostgreSQL when `RUBIC_SESSION_DATABASE_URL` or `DATABASE_URL` is set, and
falls back to replay-package JSON files for lightweight local runs.

## Storage Configuration

PostgreSQL runtime storage:

```powershell
$env:RUBIC_SESSION_DATABASE_URL="postgresql://postgres:admin@localhost:5432/rubic_rfl_test"
```

The backend applies the Phase 9 migration automatically before the first
session operation. Force JSON-file storage even when a database URL exists:

```powershell
$env:RUBIC_SESSION_STORAGE_BACKEND="files"
```

Without a database URL, sessions are stored under:

```text
replays/sessions/
```

Override the JSON location for local runs or tests:

```powershell
$env:RUBIC_SESSION_STORE_DIR="D:\path\to\sessions"
```

Each session file is keyed by a SHA-256 hash of `session_id`, so arbitrary user
session IDs cannot escape the storage directory.

## API Endpoints

```text
GET    /sessions?limit=50&solver=rl-policy&status=solved
GET    /sessions/{session_id}
POST   /sessions
DELETE /sessions/{session_id}
GET    /analytics/solves?limit=500&solver=rl-policy&status=solved
```

`GET /sessions` returns summaries with solver, status, move count, timestamps,
model metadata, and initial/final sticker fingerprints. `GET /sessions/{id}`
returns the full replay package. `POST /sessions` accepts either a replay
package directly or `{ "replay_package": { ... } }`.

`GET /analytics/solves` returns historical solve summaries from persisted
sessions:

- `totals` for session count, solved count, unsolved count, and solve rate.
- `performance` for average/best/max move count and duration.
- `by_status`, `by_solver`, `by_model`, and `by_day` grouped breakdowns.
- `recent_sessions` for the newest saved solve summaries.

Solver endpoints automatically persist replay packages:

- `/solve/classical`
- `/solve/rl`
- `/solve/rl/replay-package`
- `/ws/solve/{session_id}` on `completed`

## Frontend Workflow

The Replay panel now includes `Saved Sessions` controls:

- `Refresh` loads recent sessions from the API.
- `Save Replay` stores the currently loaded replay package.
- `Load` imports a saved replay package into the cube and timeline.
- `Delete` hides a stored session by writing a tombstone record.

## PostgreSQL Schema

The production schema is defined and used by the runtime adapter:

```text
backend/app/database/migrations/0001_session_persistence.sql
```

It creates `users`, `sessions`, `cube_states`, `solves`, `moves`,
`solver_runs`, `model_versions`, and `metrics`, plus indexes for session
history, solver status filters, model-version lookups, and metric time series.
The `sessions.replay_package` JSONB column remains the canonical replay
artifact, while related tables support filtering, analytics, and future
dashboards.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend\tests
$env:RUBIC_TEST_DATABASE_URL="postgresql://postgres:admin@localhost:5432/rubic_rfl_test"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend\tests\test_postgres_integration.py backend\tests\test_postgres_runtime_store.py
```

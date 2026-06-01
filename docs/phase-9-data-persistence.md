# Phase 9 Data Persistence

## Scope

The backend now persists solver sessions as replay-package JSON files. This is a
lightweight local store that keeps the product usable while the PostgreSQL
schema and migrations from the long-term plan are still pending.

## Storage Location

By default, sessions are stored under:

```text
replays/sessions/
```

The directory is ignored by source control. Override it for local runs or tests:

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

## PostgreSQL Follow-Up

The production schema is defined in:

```text
backend/app/database/migrations/0001_session_persistence.sql
```

It creates `users`, `sessions`, `cube_states`, `solves`, `moves`,
`solver_runs`, `model_versions`, and `metrics`, plus indexes for session
history, solver status filters, model-version lookups, and metric time series.
The eventual database-backed store should replace the JSON implementation
behind the same API contract.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend\tests
```

# Database Migrations

This directory holds PostgreSQL schema migrations for the production
persistence target. The application still runs against the JSON replay store by
default, so these migrations are safe to keep as forward-looking infrastructure
until a local `DATABASE_URL` and migration runner are added.

Apply the first migration manually against a development database:

```powershell
psql $env:DATABASE_URL -f backend/app/database/migrations/0001_session_persistence.sql
```

The baseline schema covers users, sessions, cube states, solves, moves, solver
runs, model versions, metrics, replay package JSON, and indexes for historical
analytics queries.

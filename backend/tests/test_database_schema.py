from pathlib import Path


def test_phase_9_postgres_migration_defines_planned_tables_and_indexes():
    migration = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "database"
        / "migrations"
        / "0001_session_persistence.sql"
    )

    sql = migration.read_text(encoding="utf-8").lower()

    for table in [
        "users",
        "sessions",
        "cube_states",
        "solves",
        "moves",
        "solver_runs",
        "model_versions",
        "metrics",
    ]:
        assert f"create table if not exists {table}" in sql

    assert "replay_package jsonb not null" in sql
    assert "create index if not exists idx_sessions_created_at" in sql
    assert "create index if not exists idx_solver_runs_model_version_id" in sql
    assert "create index if not exists idx_metrics_name_created_at" in sql

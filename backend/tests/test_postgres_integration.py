import os
from pathlib import Path
from uuid import uuid4

import pytest


MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "database"
    / "migrations"
    / "0001_session_persistence.sql"
)

TABLES = [
    "users",
    "sessions",
    "cube_states",
    "solves",
    "moves",
    "solver_runs",
    "model_versions",
    "metrics",
]

SOLVED_STICKERS = (
    "U" * 9
    + "R" * 9
    + "F" * 9
    + "D" * 9
    + "L" * 9
    + "B" * 9
)


def _database_url() -> str:
    url = os.environ.get("RUBIC_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("set RUBIC_TEST_DATABASE_URL to run PostgreSQL integration tests")
    return url


def test_postgres_migration_supports_core_session_persistence_flow():
    database_url = _database_url()
    psycopg = pytest.importorskip("psycopg")
    from psycopg.types.json import Jsonb

    session_key = f"pg-integration-{uuid4().hex}"
    user_external_id = f"user-{session_key}"
    model_version = f"deepcubea-test-{uuid4().hex}"
    session_uuid = None
    user_uuid = None
    model_version_uuid = None
    solve_uuid = None

    with psycopg.connect(database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))

            for table in TABLES:
                cursor.execute("SELECT to_regclass(%s)", (table,))
                assert cursor.fetchone()[0] == table

            try:
                cursor.execute(
                    """
                    INSERT INTO users (external_id, display_name)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (user_external_id, "PostgreSQL Integration"),
                )
                user_uuid = cursor.fetchone()[0]

                replay_package = {
                    "schema_version": "rubic-rfl-replay-v1",
                    "session_id": session_key,
                    "solver": "rl",
                    "moves": ["R", "R'"],
                }
                cursor.execute(
                    """
                    INSERT INTO sessions (
                        session_id,
                        user_id,
                        source,
                        replay_schema_version,
                        replay_package
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        session_key,
                        user_uuid,
                        "integration-test",
                        "rubic-rfl-replay-v1",
                        Jsonb(replay_package),
                    ),
                )
                session_uuid = cursor.fetchone()[0]

                cursor.execute(
                    """
                    INSERT INTO cube_states (
                        session_id,
                        role,
                        stickers,
                        faces,
                        history,
                        validation,
                        is_solved
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        session_uuid,
                        "initial",
                        SOLVED_STICKERS,
                        Jsonb({"U": ["U"] * 9}),
                        Jsonb([]),
                        Jsonb({"valid": True}),
                        True,
                    ),
                )
                initial_state_uuid = cursor.fetchone()[0]

                cursor.execute(
                    """
                    INSERT INTO cube_states (
                        session_id,
                        role,
                        stickers,
                        history,
                        validation,
                        is_solved
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        session_uuid,
                        "final",
                        SOLVED_STICKERS,
                        Jsonb(["R", "R'"]),
                        Jsonb({"valid": True}),
                        True,
                    ),
                )
                final_state_uuid = cursor.fetchone()[0]

                cursor.execute(
                    """
                    INSERT INTO model_versions (
                        version,
                        checkpoint,
                        policy_type,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        model_version,
                        "checkpoints/integration-test.npz",
                        "mlp",
                        Jsonb({"hidden_units": 64}),
                    ),
                )
                model_version_uuid = cursor.fetchone()[0]

                cursor.execute(
                    """
                    INSERT INTO solves (
                        session_id,
                        solver,
                        status,
                        move_count,
                        duration_ms,
                        moves
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        session_uuid,
                        "rl",
                        "solved",
                        2,
                        42,
                        Jsonb(["R", "R'"]),
                    ),
                )
                solve_uuid = cursor.fetchone()[0]

                cursor.execute(
                    """
                    INSERT INTO moves (
                        solve_id,
                        step,
                        move,
                        before_state_id,
                        after_state_id,
                        decision
                    )
                    VALUES
                        (%s, 1, %s, %s, NULL, %s),
                        (%s, 2, %s, NULL, %s, %s)
                    """,
                    (
                        solve_uuid,
                        "R",
                        initial_state_uuid,
                        Jsonb({"confidence": 0.92}),
                        solve_uuid,
                        "R'",
                        final_state_uuid,
                        Jsonb({"confidence": 0.88}),
                    ),
                )

                cursor.execute(
                    """
                    INSERT INTO solver_runs (
                        solve_id,
                        model_version_id,
                        strategy,
                        max_depth,
                        beam_width,
                        top_k,
                        depth_reached,
                        expanded_states,
                        visited_states,
                        search_trace,
                        details
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        solve_uuid,
                        model_version_uuid,
                        "beam-search",
                        30,
                        5,
                        5,
                        2,
                        7,
                        9,
                        Jsonb([{"move": "R", "status": "kept"}]),
                        Jsonb({"source": "integration-test"}),
                    ),
                )

                cursor.execute(
                    """
                    INSERT INTO metrics (
                        session_id,
                        solve_id,
                        model_version_id,
                        name,
                        value,
                        unit,
                        tags
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_uuid,
                        solve_uuid,
                        model_version_uuid,
                        "solve.duration",
                        42.0,
                        "ms",
                        Jsonb({"solver": "rl"}),
                    ),
                )

                cursor.execute(
                    """
                    SELECT
                        s.replay_package->>'schema_version',
                        so.status,
                        sr.strategy,
                        mv.version,
                        mt.name,
                        COUNT(m.id)
                    FROM sessions s
                    JOIN solves so ON so.session_id = s.id
                    JOIN moves m ON m.solve_id = so.id
                    JOIN solver_runs sr ON sr.solve_id = so.id
                    JOIN model_versions mv ON mv.id = sr.model_version_id
                    JOIN metrics mt ON mt.solve_id = so.id
                    WHERE s.session_id = %s
                    GROUP BY
                        s.replay_package->>'schema_version',
                        so.status,
                        sr.strategy,
                        mv.version,
                        mt.name
                    """,
                    (session_key,),
                )
                row = cursor.fetchone()

                assert row == (
                    "rubic-rfl-replay-v1",
                    "solved",
                    "beam-search",
                    model_version,
                    "solve.duration",
                    2,
                )

                cursor.execute(
                    "DELETE FROM sessions WHERE id = %s RETURNING id",
                    (session_uuid,),
                )
                assert cursor.fetchone()[0] == session_uuid

                for table, column, identifier in [
                    ("cube_states", "session_id", session_uuid),
                    ("solves", "id", solve_uuid),
                    ("moves", "solve_id", solve_uuid),
                    ("solver_runs", "solve_id", solve_uuid),
                    ("metrics", "solve_id", solve_uuid),
                ]:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE {column} = %s",
                        (identifier,),
                    )
                    assert cursor.fetchone()[0] == 0

                session_uuid = None
            finally:
                if session_uuid is not None:
                    cursor.execute("DELETE FROM sessions WHERE id = %s", (session_uuid,))
                if user_uuid is not None:
                    cursor.execute("DELETE FROM users WHERE id = %s", (user_uuid,))
                if model_version_uuid is not None:
                    cursor.execute(
                        "DELETE FROM model_versions WHERE id = %s",
                        (model_version_uuid,),
                    )

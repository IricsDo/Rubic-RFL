import os
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.cube import Cube
from app.main import app
from app.replay import build_replay_package
from app.solvers import SolverResult
from app.storage import (
    PostgresReplaySessionStore,
    ReplaySessionStore,
    create_session_store_from_environment,
)
import app.main as main_module


def _database_url() -> str:
    url = os.environ.get("RUBIC_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("set RUBIC_TEST_DATABASE_URL to run PostgreSQL runtime store tests")
    return url


def _replay_package(session_id: str, model_version: str) -> dict:
    cube = Cube.solved().apply_move("R")
    result = SolverResult(
        solver="rl-policy",
        status="solved",
        moves=("R'",),
        move_count=1,
        duration_ms=17,
        message="Solved with test policy",
        details={
            "model_version": model_version,
            "model_checkpoint": "checkpoints/postgres-runtime-test.npz",
            "strategy": "policy-guided-beam-search",
            "max_depth": 30,
            "beam_width": 5,
            "top_k": 5,
            "depth_reached": 1,
            "expanded_states": 2,
            "visited_states": 3,
            "search_trace": {
                "trace_limit": 10,
                "records": [{"node_id": 1, "outcome": "solution"}],
            },
            "steps": [
                {
                    "step": 1,
                    "selected_move": "R'",
                    "confidence": 1.0,
                    "top_candidates": [{"move": "R'", "confidence": 1.0}],
                }
            ],
        },
    )
    return build_replay_package(cube, result, session_id=session_id)


def test_session_store_factory_selects_postgres_from_database_url(monkeypatch):
    monkeypatch.delenv("RUBIC_SESSION_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv(
        "RUBIC_SESSION_DATABASE_URL",
        "postgresql://postgres:admin@localhost:5432/rubic_rfl_test",
    )

    store = create_session_store_from_environment()

    assert isinstance(store, PostgresReplaySessionStore)
    assert store.storage_info()["backend"] == "postgres"


def test_session_store_factory_can_force_file_fallback(monkeypatch):
    store_dir = (
        Path(__file__).resolve().parents[2]
        / ".tmp"
        / "test-sessions"
        / f"factory-{uuid4().hex}"
    )
    monkeypatch.setenv("RUBIC_SESSION_STORAGE_BACKEND", "files")
    monkeypatch.setenv(
        "RUBIC_SESSION_DATABASE_URL",
        "postgresql://postgres:admin@localhost:5432/rubic_rfl_test",
    )
    monkeypatch.setenv("RUBIC_SESSION_STORE_DIR", str(store_dir))

    store = create_session_store_from_environment()

    assert isinstance(store, ReplaySessionStore)
    assert store.storage_info()["backend"] == "json-files"


def test_postgres_runtime_store_powers_session_api_and_structured_rows():
    database_url = _database_url()
    psycopg = pytest.importorskip("psycopg")
    session_id = f"pg-runtime-{uuid4().hex}"
    model_version = f"runtime-model-{uuid4().hex}"
    package = _replay_package(session_id, model_version)
    store = PostgresReplaySessionStore(database_url)
    original_store = main_module.session_store

    main_module.session_store = store
    try:
        client = TestClient(app)

        save_response = client.post("/sessions", json={"replay_package": package})
        assert save_response.status_code == 200
        assert save_response.json()["session"]["session_id"] == session_id

        list_response = client.get("/sessions?limit=10&solver=rl-policy&status=solved")
        assert list_response.status_code == 200
        list_payload = list_response.json()
        assert list_payload["storage"]["backend"] == "postgres"
        assert any(item["session_id"] == session_id for item in list_payload["sessions"])

        load_response = client.get(f"/sessions/{session_id}")
        assert load_response.status_code == 200
        assert load_response.json()["session_id"] == session_id
        assert load_response.json()["model"]["version"] == model_version

        analytics_response = client.get("/analytics/solves?solver=rl-policy&status=solved")
        assert analytics_response.status_code == 200
        analytics_payload = analytics_response.json()
        assert analytics_payload["storage"]["backend"] == "postgres"
        assert analytics_payload["totals"]["session_count"] >= 1
        assert analytics_payload["totals"]["solved_count"] >= 1

        with psycopg.connect(database_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(DISTINCT s.id),
                        COUNT(DISTINCT cs.id),
                        COUNT(DISTINCT so.id),
                        COUNT(DISTINCT m.id),
                        COUNT(DISTINCT sr.id),
                        COUNT(DISTINCT mt.id)
                    FROM sessions s
                    LEFT JOIN cube_states cs ON cs.session_id = s.id
                    LEFT JOIN solves so ON so.session_id = s.id
                    LEFT JOIN moves m ON m.solve_id = so.id
                    LEFT JOIN solver_runs sr ON sr.solve_id = so.id
                    LEFT JOIN metrics mt ON mt.solve_id = so.id
                    WHERE s.session_id = %s
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                assert row == (1, 2, 1, 1, 1, 2)

        delete_response = client.delete(f"/sessions/{session_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted"]

        missing_response = client.get(f"/sessions/{session_id}")
        assert missing_response.status_code == 404

        with psycopg.connect(database_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(DISTINCT s.id),
                        COUNT(DISTINCT cs.id),
                        COUNT(DISTINCT so.id),
                        COUNT(DISTINCT m.id),
                        COUNT(DISTINCT sr.id),
                        COUNT(DISTINCT mt.id)
                    FROM sessions s
                    LEFT JOIN cube_states cs ON cs.session_id = s.id
                    LEFT JOIN solves so ON so.session_id = s.id
                    LEFT JOIN moves m ON m.solve_id = so.id
                    LEFT JOIN solver_runs sr ON sr.solve_id = so.id
                    LEFT JOIN metrics mt ON mt.solve_id = so.id
                    WHERE s.session_id = %s
                    """,
                    (session_id,),
                )
                assert cursor.fetchone() == (0, 0, 0, 0, 0, 0)
    finally:
        main_module.session_store = original_store
        with psycopg.connect(database_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
                cursor.execute(
                    "DELETE FROM model_versions WHERE version = %s",
                    (model_version,),
                )

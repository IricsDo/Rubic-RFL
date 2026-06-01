from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

from app.replay import REPLAY_SCHEMA_VERSION


class ReplayPackageError(ValueError):
    """Raised when a replay package cannot be persisted safely."""


class ReplayStoreError(RuntimeError):
    """Raised when the configured replay/session store is unavailable."""


class ReplaySessionStore:
    """File-backed replay/session store for local development and fallback use."""

    backend_name = "json-files"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = RLock()

    @classmethod
    def from_environment(cls) -> "ReplaySessionStore":
        root = os.environ.get("RUBIC_SESSION_STORE_DIR")
        if root and root.strip():
            return cls(root)
        return cls(_project_root() / "replays" / "sessions")

    def storage_info(self) -> dict[str, str]:
        return {"backend": self.backend_name}

    def save(self, replay_package: dict[str, Any]) -> dict[str, Any]:
        package = _validated_package(replay_package)
        session_id = package["session_id"]
        payload = json.dumps(package, indent=2, sort_keys=True)

        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._path_for(session_id).write_text(payload + "\n", encoding="utf-8")

        return _summary(package)

    def list(
        self,
        *,
        limit: int = 50,
        solver: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(0, min(int(limit), 500))
        solver_filter = solver.lower() if solver else None
        status_filter = status.lower() if status else None
        summaries: list[dict[str, Any]] = []

        with self._lock:
            if not self.root.exists():
                return []
            files = sorted(self.root.glob("*.json"))

        for path in files:
            try:
                package = self._read_path(path)
                summary = _summary(package)
            except (OSError, ReplayPackageError, json.JSONDecodeError):
                continue
            if solver_filter and summary["solver"].lower() != solver_filter:
                continue
            if status_filter and summary["status"].lower() != status_filter:
                continue
            summaries.append(summary)

        summaries.sort(key=lambda item: item["created_at"], reverse=True)
        return summaries[:limit]

    def get(self, session_id: str) -> dict[str, Any]:
        path = self._path_for(session_id)
        if not path.exists():
            raise KeyError(session_id)
        try:
            return self._read_path(path)
        except ReplayPackageError as error:
            raise KeyError(session_id) from error

    def delete(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        with self._lock:
            if not path.exists():
                return False
            try:
                self._read_path(path)
            except ReplayPackageError:
                return False
            tombstone = {"deleted": True, "session_id": _validated_session_id(session_id)}
            path.write_text(json.dumps(tombstone, sort_keys=True) + "\n", encoding="utf-8")
        return True

    def _read_path(self, path: Path) -> dict[str, Any]:
        with self._lock:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        return _validated_package(raw_payload)

    def _path_for(self, session_id: str) -> Path:
        session_key = _validated_session_id(session_id)
        digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"


class PostgresReplaySessionStore:
    """PostgreSQL-backed replay/session store for production-like runtime use."""

    backend_name = "postgres"

    def __init__(self, database_url: str, *, apply_migrations: bool = True):
        self.database_url = str(database_url or "").strip()
        if not self.database_url:
            raise ReplayStoreError("PostgreSQL replay store requires a database URL")
        self.apply_migrations = apply_migrations
        self._lock = RLock()
        self._schema_ready = False

    @classmethod
    def from_environment(cls) -> "PostgresReplaySessionStore":
        database_url = _database_url_from_environment()
        if not database_url:
            raise ReplayStoreError(
                "Set RUBIC_SESSION_DATABASE_URL or DATABASE_URL for PostgreSQL storage"
            )
        return cls(database_url)

    def storage_info(self) -> dict[str, str]:
        return {"backend": self.backend_name}

    def save(self, replay_package: dict[str, Any]) -> dict[str, Any]:
        package = _validated_package(replay_package)
        self._ensure_schema()

        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    session_uuid = self._upsert_session(cursor, package)
                    cursor.execute("DELETE FROM metrics WHERE session_id = %s", (session_uuid,))
                    cursor.execute("DELETE FROM solves WHERE session_id = %s", (session_uuid,))
                    cursor.execute("DELETE FROM cube_states WHERE session_id = %s", (session_uuid,))

                    self._insert_cube_state(cursor, session_uuid, "initial", package.get("initial_state"))
                    self._insert_cube_state(cursor, session_uuid, "final", package.get("final_state"))
                    model_version_uuid = self._upsert_model_version(cursor, package)
                    solve_uuid = self._insert_solve(cursor, session_uuid, package)
                    self._insert_moves(cursor, solve_uuid, package)
                    self._insert_solver_run(
                        cursor,
                        solve_uuid,
                        model_version_uuid,
                        package,
                    )
                    self._insert_metrics(
                        cursor,
                        session_uuid,
                        solve_uuid,
                        model_version_uuid,
                        package,
                    )
                connection.commit()
        except Exception as error:
            raise ReplayStoreError(
                f"PostgreSQL replay package could not be stored: {error}"
            ) from error

        return _summary(package)

    def list(
        self,
        *,
        limit: int = 50,
        solver: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_schema()
        limit = max(0, min(int(limit), 500))
        where, params = self._filter_clause(solver=solver, status=status)
        params.append(limit)

        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT replay_package
                        FROM sessions
                        {where}
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        params,
                    )
                    packages = [row[0] for row in cursor.fetchall()]
        except Exception as error:
            raise ReplayStoreError(f"PostgreSQL sessions could not be listed: {error}") from error

        summaries: list[dict[str, Any]] = []
        for raw_package in packages:
            try:
                summaries.append(_summary(_validated_package(raw_package)))
            except ReplayPackageError:
                continue
        return summaries

    def get(self, session_id: str) -> dict[str, Any]:
        self._ensure_schema()
        session_key = _validated_session_id(session_id)

        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT replay_package
                        FROM sessions
                        WHERE session_id = %s
                        """,
                        (session_key,),
                    )
                    row = cursor.fetchone()
        except Exception as error:
            raise ReplayStoreError(f"PostgreSQL session could not be loaded: {error}") from error

        if row is None:
            raise KeyError(session_key)
        try:
            return _validated_package(row[0])
        except ReplayPackageError as error:
            raise KeyError(session_key) from error

    def delete(self, session_id: str) -> bool:
        self._ensure_schema()
        session_key = _validated_session_id(session_id)

        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE FROM sessions
                        WHERE session_id = %s
                        RETURNING session_id
                        """,
                        (session_key,),
                    )
                    deleted = cursor.fetchone() is not None
                connection.commit()
        except Exception as error:
            raise ReplayStoreError(f"PostgreSQL session could not be deleted: {error}") from error

        return deleted

    def _ensure_schema(self) -> None:
        if self._schema_ready or not self.apply_migrations:
            return

        with self._lock:
            if self._schema_ready:
                return
            try:
                with self._connect(autocommit=True) as connection:
                    connection.execute(_migration_sql())
            except Exception as error:
                raise ReplayStoreError(
                    f"PostgreSQL migration could not be applied: {error}"
                ) from error
            self._schema_ready = True

    def _connect(self, *, autocommit: bool = False):
        try:
            import psycopg
        except ImportError as error:
            raise ReplayStoreError(
                "Install psycopg to use PostgreSQL replay storage"
            ) from error
        return psycopg.connect(self.database_url, autocommit=autocommit)

    def _jsonb(self, value: Any):
        from psycopg.types.json import Jsonb

        return Jsonb(value)

    def _upsert_session(self, cursor, package: dict[str, Any]):
        created_at = _created_at_from_package(package)
        cursor.execute(
            """
            INSERT INTO sessions (
                session_id,
                source,
                replay_schema_version,
                replay_package,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, now())
            ON CONFLICT (session_id) DO UPDATE SET
                source = EXCLUDED.source,
                replay_schema_version = EXCLUDED.replay_schema_version,
                replay_package = EXCLUDED.replay_package,
                created_at = EXCLUDED.created_at,
                updated_at = now()
            RETURNING id
            """,
            (
                package["session_id"],
                "api",
                package["schema_version"],
                self._jsonb(package),
                created_at,
            ),
        )
        return cursor.fetchone()[0]

    def _insert_cube_state(
        self,
        cursor,
        session_uuid: Any,
        role: str,
        snapshot: object,
    ) -> Any | None:
        if not isinstance(snapshot, dict):
            return None
        stickers = snapshot.get("stickers")
        if not isinstance(stickers, str) or len(stickers) != 54:
            return None

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
                role,
                stickers,
                _jsonb_or_none(self, snapshot.get("faces"), dict),
                self._jsonb(snapshot.get("history") if isinstance(snapshot.get("history"), list) else []),
                _jsonb_or_none(self, snapshot.get("validation"), dict),
                bool(snapshot.get("is_solved")),
            ),
        )
        return cursor.fetchone()[0]

    def _upsert_model_version(self, cursor, package: dict[str, Any]) -> Any | None:
        model = package.get("model") if isinstance(package.get("model"), dict) else {}
        raw_version = model.get("version")
        version = str(raw_version).strip() if raw_version not in {None, ""} else ""
        if not version:
            return None

        raw_checkpoint = model.get("checkpoint")
        checkpoint = (
            str(raw_checkpoint).strip() if raw_checkpoint not in {None, ""} else None
        )
        cursor.execute(
            """
            SELECT id
            FROM model_versions
            WHERE version = %s
              AND COALESCE(checkpoint, '') = %s
            """,
            (version, checkpoint or ""),
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0]

        cursor.execute(
            """
            INSERT INTO model_versions (version, checkpoint, policy_type, metadata)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                version,
                checkpoint,
                _policy_type_from_package(package),
                self._jsonb({"model": model, "solver": package.get("solver")}),
            ),
        )
        return cursor.fetchone()[0]

    def _insert_solve(self, cursor, session_uuid: Any, package: dict[str, Any]) -> Any:
        solver_result = (
            package.get("solver_result")
            if isinstance(package.get("solver_result"), dict)
            else {}
        )
        cursor.execute(
            """
            INSERT INTO solves (
                session_id,
                solver,
                status,
                move_count,
                duration_ms,
                moves,
                message
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_uuid,
                str(package.get("solver") or "unknown"),
                str(package.get("status") or "unknown"),
                _int_or_zero(package.get("move_count")),
                _int_or_zero(package.get("duration_ms")),
                self._jsonb(package.get("moves") if isinstance(package.get("moves"), list) else []),
                solver_result.get("message"),
            ),
        )
        return cursor.fetchone()[0]

    def _insert_moves(self, cursor, solve_uuid: Any, package: dict[str, Any]) -> None:
        steps = package.get("steps") if isinstance(package.get("steps"), list) else []
        if steps:
            rows = [
                (
                    solve_uuid,
                    index,
                    str(step.get("move") or ""),
                    self._jsonb(step.get("decision") if isinstance(step.get("decision"), dict) else {}),
                )
                for index, step in enumerate(steps, start=1)
                if isinstance(step, dict) and str(step.get("move") or "")
            ]
        else:
            moves = package.get("moves") if isinstance(package.get("moves"), list) else []
            rows = [
                (solve_uuid, index, str(move), self._jsonb({}))
                for index, move in enumerate(moves, start=1)
                if str(move)
            ]

        for row in rows:
            cursor.execute(
                """
                INSERT INTO moves (solve_id, step, move, decision)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (solve_id, step) DO NOTHING
                """,
                row,
            )

    def _insert_solver_run(
        self,
        cursor,
        solve_uuid: Any,
        model_version_uuid: Any | None,
        package: dict[str, Any],
    ) -> None:
        search = package.get("search") if isinstance(package.get("search"), dict) else {}
        solver_result = (
            package.get("solver_result")
            if isinstance(package.get("solver_result"), dict)
            else {}
        )
        details = solver_result.get("details") if isinstance(solver_result.get("details"), dict) else {}
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
                search.get("strategy"),
                _int_or_none(search.get("max_depth")),
                _int_or_none(search.get("beam_width")),
                _int_or_none(search.get("top_k")),
                _int_or_none(search.get("depth_reached")),
                _int_or_none(search.get("expanded_states")),
                _int_or_none(search.get("visited_states")),
                self._jsonb(search.get("trace")) if search.get("trace") is not None else None,
                self._jsonb(details),
            ),
        )

    def _insert_metrics(
        self,
        cursor,
        session_uuid: Any,
        solve_uuid: Any,
        model_version_uuid: Any | None,
        package: dict[str, Any],
    ) -> None:
        for name, value, unit in [
            ("solve.duration", _float_or_none(package.get("duration_ms")), "ms"),
            ("solve.move_count", _float_or_none(package.get("move_count")), "moves"),
        ]:
            if value is None:
                continue
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
                    name,
                    value,
                    unit,
                    self._jsonb({"solver": package.get("solver"), "status": package.get("status")}),
                ),
            )

    @staticmethod
    def _filter_clause(
        *,
        solver: str | None,
        status: str | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if solver:
            clauses.append("LOWER(COALESCE(replay_package->>'solver', 'unknown')) = %s")
            params.append(solver.lower())
        if status:
            clauses.append("LOWER(COALESCE(replay_package->>'status', 'unknown')) = %s")
            params.append(status.lower())
        if not clauses:
            return "", params
        return "WHERE " + " AND ".join(clauses), params


def create_session_store_from_environment() -> ReplaySessionStore | PostgresReplaySessionStore:
    backend = str(os.environ.get("RUBIC_SESSION_STORAGE_BACKEND") or "").strip().lower()
    if backend in {"file", "files", "json", "json-files"}:
        return ReplaySessionStore.from_environment()
    if backend in {"postgres", "postgresql"}:
        return PostgresReplaySessionStore.from_environment()

    database_url = _database_url_from_environment()
    if database_url:
        return PostgresReplaySessionStore(database_url)
    return ReplaySessionStore.from_environment()


def _validated_package(replay_package: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(replay_package, dict):
        raise ReplayPackageError("Replay package must be a JSON object")
    if replay_package.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ReplayPackageError(
            f"Replay package schema_version must be {REPLAY_SCHEMA_VERSION}"
        )

    try:
        package = json.loads(json.dumps(replay_package))
    except (TypeError, ValueError) as error:
        raise ReplayPackageError("Replay package must be JSON serializable") from error
    package["session_id"] = _validated_session_id(package.get("session_id"))
    if not isinstance(package.get("created_at"), str) or not package["created_at"]:
        raise ReplayPackageError("Replay package must include created_at")
    return package


def _validated_session_id(value: object) -> str:
    session_id = str(value or "").strip()
    if not session_id:
        raise ReplayPackageError("Replay package must include session_id")
    if len(session_id) > 160:
        raise ReplayPackageError("session_id cannot exceed 160 characters")
    return session_id


def _summary(package: dict[str, Any]) -> dict[str, Any]:
    model = package.get("model") if isinstance(package.get("model"), dict) else {}
    initial = (
        package.get("initial_state")
        if isinstance(package.get("initial_state"), dict)
        else {}
    )
    final = (
        package.get("final_state")
        if isinstance(package.get("final_state"), dict)
        else {}
    )
    steps = package.get("steps") if isinstance(package.get("steps"), list) else []

    return {
        "session_id": package.get("session_id"),
        "created_at": package.get("created_at"),
        "solver": str(package.get("solver") or "unknown"),
        "status": str(package.get("status") or "unknown"),
        "move_count": int(package.get("move_count") or 0),
        "duration_ms": int(package.get("duration_ms") or 0),
        "model_version": model.get("version"),
        "model_checkpoint": model.get("checkpoint"),
        "initial_stickers": initial.get("stickers"),
        "final_stickers": final.get("stickers"),
        "final_is_solved": bool(final.get("is_solved")),
        "step_count": len(steps),
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _database_url_from_environment() -> str | None:
    url = os.environ.get("RUBIC_SESSION_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if url and url.strip():
        return url.strip()
    return None


def _migration_sql() -> str:
    migration_name = "0001_session_persistence.sql"
    path = _project_root() / "backend" / "app" / "database" / "migrations" / migration_name
    if path.exists():
        return path.read_text(encoding="utf-8")

    try:
        from importlib.resources import files

        return (
            files("app.database.migrations")
            .joinpath(migration_name)
            .read_text(encoding="utf-8")
        )
    except Exception as error:
        raise ReplayStoreError(f"Cannot load database migration {migration_name}") from error


def _created_at_from_package(package: dict[str, Any]) -> datetime:
    value = str(package.get("created_at") or "")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _jsonb_or_none(
    store: PostgresReplaySessionStore,
    value: object,
    expected_type: type,
):
    if isinstance(value, expected_type):
        return store._jsonb(value)
    return None


def _policy_type_from_package(package: dict[str, Any]) -> str | None:
    solver_result = (
        package.get("solver_result")
        if isinstance(package.get("solver_result"), dict)
        else {}
    )
    details = solver_result.get("details") if isinstance(solver_result.get("details"), dict) else {}
    raw_policy_type = details.get("policy_type")
    if raw_policy_type not in {None, ""}:
        return str(raw_policy_type)
    if str(package.get("solver") or "").startswith("rl"):
        return "rl-policy"
    return None


def _int_or_zero(value: object) -> int:
    number = _int_or_none(value)
    return number if number is not None and number >= 0 else 0


def _int_or_none(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None

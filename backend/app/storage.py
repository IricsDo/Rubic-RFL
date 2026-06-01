from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

from app.replay import REPLAY_SCHEMA_VERSION


class ReplayPackageError(ValueError):
    """Raised when a replay package cannot be persisted safely."""


class ReplaySessionStore:
    """File-backed replay/session store used until the PostgreSQL phase lands."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = RLock()

    @classmethod
    def from_environment(cls) -> "ReplaySessionStore":
        root = os.environ.get("RUBIC_SESSION_STORE_DIR")
        if root and root.strip():
            return cls(root)
        return cls(_project_root() / "replays" / "sessions")

    def save(self, replay_package: dict[str, Any]) -> dict[str, Any]:
        package = self._validated_package(replay_package)
        session_id = package["session_id"]
        payload = json.dumps(package, indent=2, sort_keys=True)

        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._path_for(session_id).write_text(payload + "\n", encoding="utf-8")

        return self._summary(package)

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
                summary = self._summary(package)
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
            tombstone = {"deleted": True, "session_id": self._validated_session_id(session_id)}
            path.write_text(json.dumps(tombstone, sort_keys=True) + "\n", encoding="utf-8")
        return True

    def _read_path(self, path: Path) -> dict[str, Any]:
        with self._lock:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        return self._validated_package(raw_payload)

    def _path_for(self, session_id: str) -> Path:
        session_key = self._validated_session_id(session_id)
        digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def _validated_package(self, replay_package: dict[str, Any]) -> dict[str, Any]:
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
        package["session_id"] = self._validated_session_id(package.get("session_id"))
        if not isinstance(package.get("created_at"), str) or not package["created_at"]:
            raise ReplayPackageError("Replay package must include created_at")
        return package

    @staticmethod
    def _validated_session_id(value: object) -> str:
        session_id = str(value or "").strip()
        if not session_id:
            raise ReplayPackageError("Replay package must include session_id")
        if len(session_id) > 160:
            raise ReplayPackageError("session_id cannot exceed 160 characters")
        return session_id

    @staticmethod
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

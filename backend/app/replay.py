from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.cube import Cube, validate_stickers
from app.solvers import SolverResult

REPLAY_SCHEMA_VERSION = "rubic-rfl-replay-v1"


def build_replay_package(
    cube: Cube,
    result: SolverResult,
    *,
    session_id: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Create a reproducible artifact for replaying and inspecting a solve."""
    timestamp = created_at or datetime.now(timezone.utc)
    result_payload = result.to_dict()
    details = result.details if isinstance(result.details, dict) else {}
    decision_steps = _decision_steps_by_number(details)
    moves = list(result.moves)

    current_cube = cube
    replay_steps: list[dict[str, Any]] = []
    for index, move in enumerate(moves, start=1):
        before_cube = current_cube
        after_cube = before_cube.apply_move(move, record_history=False)
        decision = decision_steps.get(index, {})
        replay_steps.append(
            {
                "step": index,
                "move": move,
                "before_stickers": before_cube.to_string(),
                "after_stickers": after_cube.to_string(),
                "is_solved_after": after_cube.is_solved(),
                "decision": {
                    "selected_move": decision.get("selected_move", move),
                    "confidence": decision.get("confidence"),
                    "top_candidates": decision.get("top_candidates", []),
                    "estimated_distance_to_solution": max(len(moves) - index + 1, 0),
                },
            }
        )
        current_cube = after_cube

    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "session_id": session_id or f"replay-{uuid4().hex}",
        "created_at": _format_timestamp(timestamp),
        "solver": result.solver,
        "status": result.status,
        "moves": moves,
        "move_count": result.move_count,
        "duration_ms": result.duration_ms,
        "initial_state": _cube_snapshot(cube),
        "final_state": _cube_snapshot(current_cube),
        "model": {
            "version": details.get("model_version"),
            "checkpoint": details.get("model_checkpoint"),
            "policy_type": details.get("policy_type"),
            "device": details.get("policy_device"),
        },
        "search": {
            "strategy": details.get("strategy"),
            "max_depth": details.get("max_depth"),
            "beam_width": details.get("beam_width"),
            "top_k": details.get("top_k"),
            "depth_reached": details.get("depth_reached"),
            "expanded_states": details.get("expanded_states"),
            "visited_states": details.get("visited_states"),
            "trace": details.get("search_trace"),
        },
        "steps": replay_steps,
        "solver_result": result_payload,
    }


def _cube_snapshot(cube: Cube) -> dict[str, Any]:
    validation = validate_stickers(cube.stickers)
    return {
        "stickers": cube.to_string(),
        "faces": cube.to_faces(),
        "history": list(cube.history),
        "is_solved": cube.is_solved(),
        "validation": {
            "valid": validation.valid,
            "errors": list(validation.errors),
            "details": validation.details,
        },
    }


def _decision_steps_by_number(details: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw_steps = details.get("steps")
    if not isinstance(raw_steps, list):
        return {}

    steps: dict[int, dict[str, Any]] = {}
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        step_number = int(raw_step.get("step") or index)
        steps[step_number] = raw_step
    return steps


def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

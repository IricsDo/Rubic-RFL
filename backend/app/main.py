from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.analytics import build_solve_analytics
from app.cube import Cube, generate_scramble, parse_moves, validate_stickers
from app.metrics import MetricsRegistry
from app.replay import build_replay_package
from app.solvers import KociembaSolver, RLSolver
from app.storage import (
    ReplayPackageError,
    ReplayStoreError,
    create_session_store_from_environment,
)

app = FastAPI(title="Rubic RFL API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
solver = KociembaSolver()
rl_solver = RLSolver.from_environment()
session_store = create_session_store_from_environment()
metrics = MetricsRegistry()

PROMETHEUS_MEDIA_TYPE = "text/plain; version=0.0.4; charset=utf-8"
WS_SOLVE_ENDPOINT = "/ws/solve/{session_id}"
WS_LOGS_ENDPOINT = "/ws/logs/{session_id}"


class CubePayload(BaseModel):
    stickers: str | None = None
    faces: dict[str, list[str]] | None = None
    history: list[str] = Field(default_factory=list)


class MovePayload(CubePayload):
    move: str


class ScramblePayload(BaseModel):
    depth: int = Field(default=20, ge=0, le=100)
    seed: int | str | None = None


class ReplayPackagePayload(CubePayload):
    session_id: str | None = None


def _cube_from_payload(payload: CubePayload) -> Cube:
    if payload.faces is not None:
        return Cube.from_faces(payload.faces, history=payload.history)
    if payload.stickers is not None:
        return Cube.from_string(payload.stickers, history=payload.history)
    return Cube.solved().apply_sequence(payload.history)


def _cube_response(cube: Cube) -> dict[str, Any]:
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


def _solver_mode_from_payload(payload: dict[str, Any]) -> str:
    requested = str(
        payload.get("solver") or payload.get("solver_mode") or "classical"
    ).strip().lower()
    if requested in {"rl", "rl-policy", "reinforcement-learning"}:
        return "rl"
    return "classical"


def _stream_solver_for_mode(mode: str) -> KociembaSolver | RLSolver:
    return rl_solver if mode == "rl" else solver


def _store_replay_package(package: dict[str, Any]) -> dict[str, Any]:
    try:
        return session_store.save(package)
    except ReplayPackageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (OSError, ReplayStoreError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Replay package could not be stored: {error}",
        ) from error


def _try_store_replay_package(package: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return session_store.save(package)
    except (OSError, ReplayPackageError, ReplayStoreError):
        return None


def _storage_info() -> dict[str, Any]:
    if hasattr(session_store, "storage_info"):
        return session_store.storage_info()
    return {"backend": "unknown"}


def _decision_steps_by_number(details: object) -> dict[int, dict[str, Any]]:
    if not isinstance(details, dict):
        return {}
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


def _decision_event(
    *,
    session_id: str,
    result_payload: dict[str, Any],
    details: dict[str, Any],
    step: int,
    step_details: dict[str, Any],
    replay_cube: Cube,
) -> dict[str, Any]:
    moves = result_payload.get("moves")
    total_moves = len(moves) if isinstance(moves, list) else 0
    return {
        "session_id": session_id,
        "event": "decision",
        "solver": result_payload.get("solver"),
        "step": step,
        "selected_move": step_details.get("selected_move"),
        "confidence": step_details.get("confidence"),
        "top_candidates": step_details.get("top_candidates", []),
        "stickers": replay_cube.to_string(),
        "search_depth": details.get("depth_reached"),
        "expanded_states": details.get("expanded_states"),
        "visited_states": details.get("visited_states"),
        "estimated_distance_to_solution": max(total_moves - step + 1, 0),
        "model_version": details.get("model_version"),
        "model_checkpoint": details.get("model_checkpoint"),
        "policy_type": details.get("policy_type"),
        "policy_device": details.get("policy_device"),
        "strategy": details.get("strategy"),
        "max_depth": details.get("max_depth"),
        "beam_width": details.get("beam_width"),
        "top_k": details.get("top_k"),
    }


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path or request.url.path)


def _number_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _model_version_from_result(result_payload: dict[str, Any]) -> str | None:
    details = result_payload.get("details")
    if isinstance(details, dict):
        version = details.get("model_version")
        if version not in {None, ""}:
            return str(version)
    version = result_payload.get("model_version")
    if version not in {None, ""}:
        return str(version)
    return None


def _record_solver_metrics(result_payload: dict[str, Any]) -> None:
    metrics.record_solver_run(
        solver=str(result_payload.get("solver") or "unknown"),
        status=str(result_payload.get("status") or "unknown"),
        duration_ms=_number_or_none(result_payload.get("duration_ms")),
        move_count=_number_or_none(result_payload.get("move_count")),
        model_version=_model_version_from_result(result_payload),
    )


@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    started_at = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        metrics.record_http_request(
            method=request.method,
            path=_route_template(request),
            status_code=status_code,
            duration_seconds=perf_counter() - started_at,
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(
        metrics.render_prometheus(),
        media_type=PROMETHEUS_MEDIA_TYPE,
    )


@app.get("/sessions")
def list_sessions(
    limit: int = Query(default=50, ge=0, le=500),
    solver: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    try:
        sessions = session_store.list(limit=limit, solver=solver, status=status)
    except ReplayStoreError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {
        "sessions": sessions,
        "count": len(sessions),
        "storage": _storage_info(),
    }


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    try:
        return session_store.get(session_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Session not found") from error
    except ReplayStoreError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/analytics/solves")
def solve_analytics(
    limit: int = Query(default=500, ge=0, le=500),
    solver: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    try:
        sessions = session_store.list(limit=limit, solver=solver, status=status)
    except ReplayStoreError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {
        "storage": _storage_info(),
        "filters": {"limit": limit, "solver": solver, "status": status},
        **build_solve_analytics(sessions),
    }


@app.post("/sessions")
def save_session(payload: dict[str, Any]) -> dict[str, Any]:
    replay_package = payload.get("replay_package") if "replay_package" in payload else payload
    if not isinstance(replay_package, dict):
        raise HTTPException(status_code=400, detail="Replay package must be an object")
    return {"saved": True, "session": _store_replay_package(replay_package)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    try:
        deleted = session_store.delete(session_id)
    except ReplayStoreError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": session_id}


@app.post("/cube/validate")
def validate_cube(payload: CubePayload) -> dict[str, Any]:
    return _cube_response(_cube_from_payload(payload))


@app.post("/cube/scramble")
def scramble_cube(payload: ScramblePayload) -> dict[str, Any]:
    moves = generate_scramble(payload.depth, payload.seed)
    cube = Cube.solved().apply_sequence(moves)
    response = _cube_response(cube)
    response["scramble"] = list(moves)
    return response


@app.post("/cube/apply-move")
def apply_move(payload: MovePayload) -> dict[str, Any]:
    cube = _cube_from_payload(payload).apply_move(payload.move)
    return _cube_response(cube)


@app.post("/solve/classical")
def solve_classical(payload: ReplayPackagePayload) -> dict[str, Any]:
    cube = _cube_from_payload(payload)
    result = solver.solve(cube)
    result_payload = result.to_dict()
    _record_solver_metrics(result_payload)
    package = build_replay_package(cube, result, session_id=payload.session_id)
    stored_session = _try_store_replay_package(package)
    if stored_session is not None:
        result_payload["session_id"] = stored_session["session_id"]
        result_payload["stored_session"] = stored_session
    return result_payload


@app.get("/solve/rl/status")
def rl_runtime_status(
    load: bool = Query(
        default=False,
        description="Attempt to load the configured RL policy before reporting status.",
    ),
) -> dict[str, Any]:
    return rl_solver.runtime_status(load_policy=load)


@app.post("/solve/rl")
def solve_rl(payload: ReplayPackagePayload) -> dict[str, Any]:
    cube = _cube_from_payload(payload)
    result = rl_solver.solve(cube)
    result_payload = result.to_dict()
    _record_solver_metrics(result_payload)
    package = build_replay_package(cube, result, session_id=payload.session_id)
    stored_session = _try_store_replay_package(package)
    if stored_session is not None:
        result_payload["session_id"] = stored_session["session_id"]
        result_payload["stored_session"] = stored_session
    return result_payload


@app.post("/solve/rl/replay-package")
def solve_rl_replay_package(payload: ReplayPackagePayload) -> dict[str, Any]:
    cube = _cube_from_payload(payload)
    result = rl_solver.solve(cube)
    _record_solver_metrics(result.to_dict())
    package = build_replay_package(cube, result, session_id=payload.session_id)
    _try_store_replay_package(package)
    return package


@app.websocket("/ws/solve/{session_id}")
async def solve_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    metrics.websocket_open(WS_SOLVE_ENDPOINT)
    try:
        payload = await websocket.receive_json()
        mode = _solver_mode_from_payload(payload)
        cube = _cube_from_payload(CubePayload.model_validate(payload))
        result = _stream_solver_for_mode(mode).solve(cube)
        result_payload = result.to_dict()
        _record_solver_metrics(result_payload)
        details = result_payload.get("details")
        details_dict = details if isinstance(details, dict) else {}
        decision_steps = _decision_steps_by_number(details)

        await websocket.send_json(
            {
                "session_id": session_id,
                "event": "started",
                "solver": result_payload.get("solver"),
                "status": result_payload.get("status"),
                "stickers": cube.to_string(),
            }
        )
        replay_cube = cube
        for index, move in enumerate(result.moves, start=1):
            step_details = decision_steps.get(index)
            if step_details is not None:
                await websocket.send_json(
                    _decision_event(
                        session_id=session_id,
                        result_payload=result_payload,
                        details=details_dict,
                        step=index,
                        step_details=step_details,
                        replay_cube=replay_cube,
                    )
                )
            replay_cube = replay_cube.apply_move(move)
            await websocket.send_json(
                {
                    "session_id": session_id,
                    "event": "move",
                    "solver": result_payload.get("solver"),
                    "step": index,
                    "move": move,
                    "stickers": replay_cube.to_string(),
                }
            )
        replay_package = build_replay_package(
            cube,
            result,
            session_id=session_id,
        )
        stored_session = _try_store_replay_package(replay_package)
        await websocket.send_json(
            {
                "session_id": session_id,
                "event": "completed",
                "solver": result_payload.get("solver"),
                "result": result_payload,
                "replay_package": replay_package,
                "stored_session": stored_session,
            }
        )
    except WebSocketDisconnect:
        return
    except Exception as error:
        metrics.websocket_error(WS_SOLVE_ENDPOINT)
        await websocket.send_json(
            {
                "session_id": session_id,
                "event": "error",
                "message": str(error),
            }
        )
    finally:
        metrics.websocket_close(WS_SOLVE_ENDPOINT)


@app.websocket("/ws/logs/{session_id}")
async def logs_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    metrics.websocket_open(WS_LOGS_ENDPOINT)
    try:
        await websocket.send_json(
            {
                "session_id": session_id,
                "event": "info",
                "message": "Structured solver logs will be added with RL integration",
            }
        )
    except WebSocketDisconnect:
        return
    except Exception:
        metrics.websocket_error(WS_LOGS_ENDPOINT)
        raise
    finally:
        metrics.websocket_close(WS_LOGS_ENDPOINT)

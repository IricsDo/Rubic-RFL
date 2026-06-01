from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
RL_DIR = ROOT / "rl"

for path in (str(BACKEND_DIR), str(RL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault(
    "RUBIC_SESSION_STORE_DIR",
    str(ROOT / ".tmp" / "performance-ws-sessions"),
)

from fastapi.testclient import TestClient  # noqa: E402

from app.cube import Cube  # noqa: E402
import app.main as main_module  # noqa: E402
from app.main import app  # noqa: E402
from app.solvers import RLSolver  # noqa: E402
from app.storage import ReplaySessionStore  # noqa: E402


DEFAULT_BUDGETS_MS = {
    "rl_websocket_first_event": 250.0,
    "rl_websocket_completed": 1000.0,
}


class ConstantPolicy:
    def __init__(self, action: int):
        self.action = action

    def predict(self, features):
        return [self.action for _ in range(features.shape[0])]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_ms": round(min(values), 3),
        "median_ms": round(_percentile(values, 0.50), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3),
    }


def _run_stream(client: TestClient, session_id: str) -> dict[str, Any]:
    scrambled = Cube.solved().apply_move("R")
    started_at = time.perf_counter()
    first_event_ms: float | None = None
    events: list[dict[str, Any]] = []

    with client.websocket_connect(f"/ws/solve/{session_id}") as websocket:
        websocket.send_json(
            {
                "solver": "rl",
                "stickers": scrambled.to_string(),
            }
        )

        while True:
            message = websocket.receive_json()
            if first_event_ms is None:
                first_event_ms = (time.perf_counter() - started_at) * 1000
            events.append(message)
            if message.get("event") == "completed":
                break
            if message.get("event") == "error":
                raise RuntimeError(f"WebSocket solver returned error: {message}")

    completed_ms = (time.perf_counter() - started_at) * 1000
    event_names = [str(event.get("event")) for event in events]
    expected_events = ["started", "decision", "move", "completed"]
    if event_names != expected_events:
        raise RuntimeError(f"Unexpected WebSocket event sequence: {event_names}")

    completed = events[-1]
    result = completed.get("result")
    if not isinstance(result, dict) or result.get("status") != "solved":
        raise RuntimeError(f"WebSocket solver did not solve: {completed}")
    if result.get("moves") != ["R'"]:
        raise RuntimeError(f"Unexpected WebSocket solution moves: {result}")

    return {
        "first_event_ms": first_event_ms or completed_ms,
        "completed_ms": completed_ms,
        "event_count": len(events),
    }


def _measure_stream(iterations: int, warmups: int) -> tuple[dict[str, Any], dict[str, Any]]:
    original_solver = main_module.rl_solver
    original_session_store = main_module.session_store
    main_module.rl_solver = RLSolver(
        policy=ConstantPolicy(9),
        model_path="in-memory-ws-benchmark-policy",
        max_steps=1,
        search_width=1,
        search_top_k=1,
        search_trace_limit=10,
    )
    main_module.session_store = ReplaySessionStore(ROOT / ".tmp" / "performance-ws-sessions")
    main_module.metrics.reset()

    client = TestClient(app)
    try:
        counter = 0
        for _ in range(warmups):
            counter += 1
            _run_stream(client, f"ws-benchmark-warmup-{counter}")

        first_event_ms: list[float] = []
        completed_ms: list[float] = []
        event_counts: list[int] = []

        for _ in range(iterations):
            counter += 1
            result = _run_stream(client, f"ws-benchmark-{counter}")
            first_event_ms.append(result["first_event_ms"])
            completed_ms.append(result["completed_ms"])
            event_counts.append(result["event_count"])
    finally:
        main_module.rl_solver = original_solver
        main_module.session_store = original_session_store
        main_module.metrics.reset()

    return (
        {
            "rl_websocket_first_event": _summary(first_event_ms),
            "rl_websocket_completed": _summary(completed_ms),
        },
        {
            "expected_event_count": 4,
            "min_event_count": min(event_counts),
            "max_event_count": max(event_counts),
        },
    )


def run_benchmarks(iterations: int, warmups: int) -> dict[str, Any]:
    benchmarks, stream = _measure_stream(iterations=iterations, warmups=warmups)
    failures = [
        {
            "benchmark": label,
            "budget_p95_ms": budget,
            "actual_p95_ms": benchmarks[label]["p95_ms"],
        }
        for label, budget in DEFAULT_BUDGETS_MS.items()
        if benchmarks[label]["p95_ms"] > budget
    ]

    return {
        "schema_version": "rubic-rfl-websocket-performance-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "iterations": iterations,
        "warmups": warmups,
        "budgets_ms": DEFAULT_BUDGETS_MS,
        "benchmarks": benchmarks,
        "stream": stream,
        "summary": {
            "status": "failed" if failures else "passed",
            "failures": failures,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a small in-process FastAPI WebSocket latency smoke benchmark."
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--enforce-budgets", action="store_true")
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")

    report = run_benchmarks(iterations=args.iterations, warmups=args.warmups)
    output = json.dumps(report, indent=2, sort_keys=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")

    print(output)

    if args.enforce_budgets and report["summary"]["failures"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
RL_DIR = ROOT / "rl"

for path in (str(BACKEND_DIR), str(RL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault(
    "RUBIC_SESSION_STORE_DIR",
    str(ROOT / ".tmp" / "performance-sessions"),
)

from fastapi.testclient import TestClient  # noqa: E402

from app.cube import Cube  # noqa: E402
from app.main import app  # noqa: E402


DEFAULT_BUDGETS_MS = {
    "health": 100.0,
    "validate_solved_cube": 150.0,
    "scramble_depth_4": 150.0,
    "classical_solve_depth_3": 2000.0,
    "analytics_solves": 250.0,
}


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


def _assert_ok(response: Any, label: str) -> dict[str, Any]:
    if response.status_code != 200:
        raise RuntimeError(f"{label} returned HTTP {response.status_code}: {response.text}")
    return response.json()


def _measure(
    label: str,
    operation: Callable[[], Any],
    *,
    iterations: int,
    warmups: int,
) -> dict[str, float | int]:
    for _ in range(warmups):
        operation()

    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        durations.append((time.perf_counter() - started) * 1000)

    return {
        "count": iterations,
        "min_ms": round(min(durations), 3),
        "median_ms": round(_percentile(durations, 0.50), 3),
        "p95_ms": round(_percentile(durations, 0.95), 3),
        "max_ms": round(max(durations), 3),
    }


def run_benchmarks(iterations: int, warmups: int) -> dict[str, Any]:
    client = TestClient(app)
    solved_stickers = Cube.solved().to_string()
    scramble = _assert_ok(
        client.post("/cube/scramble", json={"depth": 3, "seed": "performance-smoke"}),
        "initial scramble",
    )
    scrambled_stickers = scramble["stickers"]
    classical_run_id = f"perf-{int(time.time())}"
    classical_counter = 0

    def health() -> None:
        payload = _assert_ok(client.get("/health"), "health")
        if payload["status"] != "ok":
            raise RuntimeError(f"Unexpected health payload: {payload}")

    def validate_solved_cube() -> None:
        payload = _assert_ok(
            client.post("/cube/validate", json={"stickers": solved_stickers}),
            "validate_solved_cube",
        )
        if not payload["validation"]["valid"]:
            raise RuntimeError(f"Solved cube failed validation: {payload}")

    def scramble_depth_4() -> None:
        payload = _assert_ok(
            client.post("/cube/scramble", json={"depth": 4, "seed": "benchmark"}),
            "scramble_depth_4",
        )
        if len(payload["scramble"]) != 4:
            raise RuntimeError(f"Unexpected scramble payload: {payload}")

    def classical_solve_depth_3() -> None:
        nonlocal classical_counter
        classical_counter += 1
        payload = _assert_ok(
            client.post(
                "/solve/classical",
                json={
                    "stickers": scrambled_stickers,
                    "session_id": f"{classical_run_id}-{classical_counter}",
                },
            ),
            "classical_solve_depth_3",
        )
        if payload["status"] != "solved":
            raise RuntimeError(f"Classical solver did not solve: {payload}")

    def analytics_solves() -> None:
        payload = _assert_ok(client.get("/analytics/solves?limit=100"), "analytics_solves")
        if "totals" not in payload:
            raise RuntimeError(f"Unexpected analytics payload: {payload}")

    operations: dict[str, Callable[[], None]] = {
        "health": health,
        "validate_solved_cube": validate_solved_cube,
        "scramble_depth_4": scramble_depth_4,
        "classical_solve_depth_3": classical_solve_depth_3,
        "analytics_solves": analytics_solves,
    }

    benchmarks = {
        label: _measure(label, operation, iterations=iterations, warmups=warmups)
        for label, operation in operations.items()
    }

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
        "schema_version": "rubic-rfl-performance-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "iterations": iterations,
        "warmups": warmups,
        "budgets_ms": DEFAULT_BUDGETS_MS,
        "benchmarks": benchmarks,
        "summary": {
            "status": "failed" if failures else "passed",
            "failures": failures,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a small in-process FastAPI performance smoke benchmark."
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

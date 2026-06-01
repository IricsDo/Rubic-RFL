from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
RL_DIR = ROOT / "rl"

for path in (str(BACKEND_DIR), str(RL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

if os.environ.get("RUBIC_LOAD_TEST_USE_ENV_STORAGE") != "1":
    os.environ["RUBIC_SESSION_STORAGE_BACKEND"] = "files"
os.environ.setdefault(
    "RUBIC_SESSION_STORE_DIR",
    str(ROOT / ".tmp" / "backend-load-sessions"),
)

from fastapi.testclient import TestClient  # noqa: E402

from app.cube import Cube  # noqa: E402
import app.main as main_module  # noqa: E402
from app.main import app  # noqa: E402
from app.storage import ReplaySessionStore  # noqa: E402


DEFAULT_BUDGETS = {
    "max_error_rate": 0.0,
    "min_throughput_rps": 20.0,
    "max_overall_p95_ms": 2500.0,
    "scenario_p95_ms": {
        "health": 150.0,
        "metrics": 250.0,
        "validate_solved_cube": 250.0,
        "scramble_depth_8": 300.0,
        "apply_move": 300.0,
        "classical_solve_depth_3": 2500.0,
        "sessions_list": 600.0,
        "analytics_solves": 700.0,
    },
}

THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class Scenario:
    name: str
    weight: int
    run: Callable[[TestClient, int], None]


@dataclass(frozen=True)
class RequestSample:
    scenario: str
    duration_ms: float
    ok: bool
    error: str | None = None


def _client() -> TestClient:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = TestClient(app)
        THREAD_LOCAL.client = client
    return client


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
    if not values:
        return {
            "count": 0,
            "min_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "min_ms": round(min(values), 3),
        "median_ms": round(_percentile(values, 0.50), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "p99_ms": round(_percentile(values, 0.99), 3),
        "max_ms": round(max(values), 3),
    }


def _assert_ok(response: Any, label: str) -> dict[str, Any]:
    if response.status_code != 200:
        raise RuntimeError(f"{label} returned HTTP {response.status_code}: {response.text}")
    return response.json()


def _build_scenarios() -> list[Scenario]:
    solved_stickers = Cube.solved().to_string()
    seeded_client = TestClient(app)
    scramble = _assert_ok(
        seeded_client.post("/cube/scramble", json={"depth": 3, "seed": "backend-load"}),
        "initial scramble",
    )
    scrambled_stickers = str(scramble["stickers"])

    def health(client: TestClient, index: int) -> None:
        payload = _assert_ok(client.get("/health"), "health")
        if payload.get("status") != "ok":
            raise RuntimeError(f"Unexpected health payload: {payload}")

    def metrics(client: TestClient, index: int) -> None:
        response = client.get("/metrics")
        if response.status_code != 200:
            raise RuntimeError(f"metrics returned HTTP {response.status_code}: {response.text}")
        if "rubic_http_requests_total" not in response.text:
            raise RuntimeError("Metrics response did not include HTTP counters.")

    def validate_solved_cube(client: TestClient, index: int) -> None:
        payload = _assert_ok(
            client.post("/cube/validate", json={"stickers": solved_stickers}),
            "validate_solved_cube",
        )
        if not payload["validation"]["valid"]:
            raise RuntimeError(f"Solved cube failed validation: {payload}")

    def scramble_depth_8(client: TestClient, index: int) -> None:
        payload = _assert_ok(
            client.post("/cube/scramble", json={"depth": 8, "seed": f"load-{index}"}),
            "scramble_depth_8",
        )
        if len(payload["scramble"]) != 8:
            raise RuntimeError(f"Unexpected scramble payload: {payload}")

    def apply_move(client: TestClient, index: int) -> None:
        payload = _assert_ok(
            client.post(
                "/cube/apply-move",
                json={"stickers": solved_stickers, "history": [], "move": "R"},
            ),
            "apply_move",
        )
        if payload["history"] != ["R"]:
            raise RuntimeError(f"Unexpected apply-move payload: {payload}")

    def classical_solve_depth_3(client: TestClient, index: int) -> None:
        payload = _assert_ok(
            client.post(
                "/solve/classical",
                json={
                    "stickers": scrambled_stickers,
                    "session_id": f"backend-load-classical-{index}",
                },
            ),
            "classical_solve_depth_3",
        )
        if payload["status"] != "solved":
            raise RuntimeError(f"Classical solver did not solve: {payload}")

    def sessions_list(client: TestClient, index: int) -> None:
        payload = _assert_ok(client.get("/sessions?limit=25"), "sessions_list")
        if "sessions" not in payload or "storage" not in payload:
            raise RuntimeError(f"Unexpected sessions payload: {payload}")

    def analytics_solves(client: TestClient, index: int) -> None:
        payload = _assert_ok(client.get("/analytics/solves?limit=100"), "analytics_solves")
        if "totals" not in payload or "performance" not in payload:
            raise RuntimeError(f"Unexpected analytics payload: {payload}")

    return [
        Scenario("health", 8, health),
        Scenario("metrics", 2, metrics),
        Scenario("validate_solved_cube", 8, validate_solved_cube),
        Scenario("scramble_depth_8", 6, scramble_depth_8),
        Scenario("apply_move", 6, apply_move),
        Scenario("classical_solve_depth_3", 2, classical_solve_depth_3),
        Scenario("sessions_list", 3, sessions_list),
        Scenario("analytics_solves", 3, analytics_solves),
    ]


def _weighted_schedule(scenarios: list[Scenario], total_requests: int) -> list[Scenario]:
    weighted: list[Scenario] = []
    max_weight = max(scenario.weight for scenario in scenarios)
    for level in range(max_weight):
        for scenario in scenarios:
            if level < scenario.weight:
                weighted.append(scenario)
    return [weighted[index % len(weighted)] for index in range(total_requests)]


def _run_sample(scenario: Scenario, index: int) -> RequestSample:
    started_at = time.perf_counter()
    try:
        scenario.run(_client(), index)
    except Exception as error:  # noqa: BLE001 - load tests report request errors.
        return RequestSample(
            scenario=scenario.name,
            duration_ms=(time.perf_counter() - started_at) * 1000,
            ok=False,
            error=str(error),
        )
    return RequestSample(
        scenario=scenario.name,
        duration_ms=(time.perf_counter() - started_at) * 1000,
        ok=True,
    )


def _failure_summary(
    *,
    throughput_rps: float,
    error_rate: float,
    overall: dict[str, float | int],
    scenarios: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if error_rate > DEFAULT_BUDGETS["max_error_rate"]:
        failures.append(
            {
                "budget": "max_error_rate",
                "expected": DEFAULT_BUDGETS["max_error_rate"],
                "actual": round(error_rate, 4),
            }
        )
    if throughput_rps < DEFAULT_BUDGETS["min_throughput_rps"]:
        failures.append(
            {
                "budget": "min_throughput_rps",
                "expected": DEFAULT_BUDGETS["min_throughput_rps"],
                "actual": round(throughput_rps, 3),
            }
        )
    if float(overall["p95_ms"]) > DEFAULT_BUDGETS["max_overall_p95_ms"]:
        failures.append(
            {
                "budget": "max_overall_p95_ms",
                "expected": DEFAULT_BUDGETS["max_overall_p95_ms"],
                "actual": overall["p95_ms"],
            }
        )

    scenario_budgets = DEFAULT_BUDGETS["scenario_p95_ms"]
    for name, budget in scenario_budgets.items():
        actual = scenarios.get(name, {}).get("p95_ms", 0.0)
        if actual > budget:
            failures.append(
                {
                    "budget": "scenario_p95_ms",
                    "scenario": name,
                    "expected": budget,
                    "actual": actual,
                }
            )
    return failures


def run_load_test(total_requests: int, concurrency: int, warmups: int) -> dict[str, Any]:
    original_session_store = main_module.session_store
    main_module.session_store = ReplaySessionStore(ROOT / ".tmp" / "backend-load-sessions")
    main_module.metrics.reset()

    try:
        scenarios = _build_scenarios()
        for index, scenario in enumerate(_weighted_schedule(scenarios, warmups), start=1):
            result = _run_sample(scenario, index)
            if not result.ok:
                raise RuntimeError(f"Warmup request failed for {result.scenario}: {result.error}")

        schedule = _weighted_schedule(scenarios, total_requests)
        started_at = time.perf_counter()
        samples: list[RequestSample] = []
        max_workers = min(concurrency, total_requests)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_sample, scenario, index): scenario.name
                for index, scenario in enumerate(schedule, start=1)
            }
            for future in as_completed(futures):
                samples.append(future.result())
        elapsed_seconds = max(time.perf_counter() - started_at, 0.001)
    finally:
        main_module.session_store = original_session_store
        main_module.metrics.reset()

    durations = [sample.duration_ms for sample in samples]
    scenario_summaries: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        scenario_samples = [sample for sample in samples if sample.scenario == scenario.name]
        scenario_durations = [sample.duration_ms for sample in scenario_samples]
        scenario_errors = [sample for sample in scenario_samples if not sample.ok]
        scenario_summaries[scenario.name] = {
            **_summary(scenario_durations),
            "errors": len(scenario_errors),
            "error_rate": round(len(scenario_errors) / max(1, len(scenario_samples)), 4),
        }

    errors = [sample for sample in samples if not sample.ok]
    throughput_rps = len(samples) / elapsed_seconds
    error_rate = len(errors) / max(1, len(samples))
    overall = _summary(durations)
    failures = _failure_summary(
        throughput_rps=throughput_rps,
        error_rate=error_rate,
        overall=overall,
        scenarios=scenario_summaries,
    )

    return {
        "schema_version": "rubic-rfl-backend-load-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "requests": total_requests,
        "concurrency": concurrency,
        "warmups": warmups,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_rps": round(throughput_rps, 3),
        "error_rate": round(error_rate, 4),
        "budgets": DEFAULT_BUDGETS,
        "overall": overall,
        "scenarios": scenario_summaries,
        "errors": [
            {
                "scenario": sample.scenario,
                "duration_ms": round(sample.duration_ms, 3),
                "error": sample.error,
            }
            for sample in errors[:10]
        ],
        "summary": {
            "status": "failed" if failures else "passed",
            "failures": failures,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a concurrent in-process backend load test against mixed FastAPI routes."
    )
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--enforce-budgets", action="store_true")
    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")

    report = run_load_test(
        total_requests=args.requests,
        concurrency=args.concurrency,
        warmups=args.warmups,
    )
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

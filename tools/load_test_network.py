from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SOLVED_STICKERS = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

DEFAULT_BUDGETS = {
    "max_error_rate": 0.0,
    "min_throughput_rps": 5.0,
    "max_overall_p95_ms": 5000.0,
    "scenario_p95_ms": {
        "health": 300.0,
        "metrics": 500.0,
        "validate_solved_cube": 500.0,
        "scramble_depth_8": 800.0,
        "apply_move": 500.0,
        "classical_solve_depth_3": 5000.0,
        "sessions_list": 1000.0,
        "analytics_solves": 1000.0,
        "rl_status": 1000.0,
    },
}


@dataclass(frozen=True)
class Scenario:
    name: str
    weight: int
    run: Callable[[int], None]


@dataclass(frozen=True)
class RequestSample:
    scenario: str
    duration_ms: float
    ok: bool
    error: str | None = None


class NetworkClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get_json(self, path: str) -> dict[str, Any]:
        response, _headers = self.request("GET", path)
        return json.loads(response)

    def get_text(self, path: str) -> str:
        response, _headers = self.request("GET", path)
        return response

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response, _headers = self.request("POST", path, payload)
        return json.loads(response)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, str]]:
        data = None
        headers = {"User-Agent": "rubic-rfl-network-load/1.0"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                return body, response_headers
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} returned HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"{method} {path} failed: {error.reason}") from error


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


def _weighted_schedule(scenarios: list[Scenario], total_requests: int) -> list[Scenario]:
    weighted: list[Scenario] = []
    max_weight = max(scenario.weight for scenario in scenarios)
    for level in range(max_weight):
        for scenario in scenarios:
            if level < scenario.weight:
                weighted.append(scenario)
    return [weighted[index % len(weighted)] for index in range(total_requests)]


def _require_keys(payload: dict[str, Any], keys: set[str], label: str) -> None:
    missing = keys.difference(payload)
    if missing:
        raise RuntimeError(f"{label} missing keys: {sorted(missing)}")


def _build_scenarios(client: NetworkClient) -> list[Scenario]:
    initial = client.post_json("/cube/scramble", {"depth": 3, "seed": "network-load"})
    _require_keys(initial, {"stickers", "scramble"}, "initial scramble")
    scrambled_stickers = str(initial["stickers"])

    def health(index: int) -> None:
        payload = client.get_json("/health")
        if payload.get("status") != "ok":
            raise RuntimeError(f"Unexpected health payload: {payload}")

    def metrics(index: int) -> None:
        text = client.get_text("/metrics")
        if "rubic_http_requests_total" not in text:
            raise RuntimeError("Metrics response did not include HTTP counters.")

    def validate_solved_cube(index: int) -> None:
        payload = client.post_json("/cube/validate", {"stickers": SOLVED_STICKERS})
        if not payload.get("validation", {}).get("valid"):
            raise RuntimeError(f"Solved cube failed validation: {payload}")

    def scramble_depth_8(index: int) -> None:
        payload = client.post_json("/cube/scramble", {"depth": 8, "seed": f"network-load-{index}"})
        if len(payload.get("scramble", [])) != 8:
            raise RuntimeError(f"Unexpected scramble payload: {payload}")

    def apply_move(index: int) -> None:
        payload = client.post_json(
            "/cube/apply-move",
            {"stickers": SOLVED_STICKERS, "history": [], "move": "R"},
        )
        if payload.get("history") != ["R"]:
            raise RuntimeError(f"Unexpected apply-move payload: {payload}")

    def classical_solve_depth_3(index: int) -> None:
        payload = client.post_json(
            "/solve/classical",
            {
                "stickers": scrambled_stickers,
                "session_id": f"network-load-classical-{index}",
            },
        )
        if payload.get("status") != "solved":
            raise RuntimeError(f"Classical solver did not solve: {payload}")

    def sessions_list(index: int) -> None:
        payload = client.get_json("/sessions?limit=25")
        _require_keys(payload, {"sessions", "storage"}, "sessions_list")

    def analytics_solves(index: int) -> None:
        payload = client.get_json("/analytics/solves?limit=100")
        _require_keys(payload, {"totals", "performance"}, "analytics_solves")

    def rl_status(index: int) -> None:
        payload = client.get_json("/solve/rl/status")
        _require_keys(payload, {"solver", "status", "available"}, "rl_status")

    return [
        Scenario("health", 8, health),
        Scenario("metrics", 2, metrics),
        Scenario("validate_solved_cube", 8, validate_solved_cube),
        Scenario("scramble_depth_8", 6, scramble_depth_8),
        Scenario("apply_move", 6, apply_move),
        Scenario("classical_solve_depth_3", 2, classical_solve_depth_3),
        Scenario("sessions_list", 3, sessions_list),
        Scenario("analytics_solves", 3, analytics_solves),
        Scenario("rl_status", 2, rl_status),
    ]


def _run_sample(scenario: Scenario, index: int) -> RequestSample:
    started_at = time.perf_counter()
    try:
        scenario.run(index)
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

    for name, budget in DEFAULT_BUDGETS["scenario_p95_ms"].items():
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


def run_network_load_test(
    *,
    base_url: str,
    total_requests: int,
    concurrency: int,
    warmups: int,
    timeout_ms: int,
) -> dict[str, Any]:
    client = NetworkClient(base_url=base_url, timeout_seconds=timeout_ms / 1000)
    scenarios = _build_scenarios(client)

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
    durations = [sample.duration_ms for sample in samples]
    overall = _summary(durations)
    failures = _failure_summary(
        throughput_rps=throughput_rps,
        error_rate=error_rate,
        overall=overall,
        scenarios=scenario_summaries,
    )

    return {
        "schema_version": "rubic-rfl-network-load-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_url": base_url.rstrip("/"),
        "requests": total_requests,
        "concurrency": concurrency,
        "warmups": warmups,
        "timeout_ms": timeout_ms,
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
        description="Run an external network load test against a running Rubic RFL backend."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--enforce-budgets", action="store_true")
    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.timeout_ms < 1:
        parser.error("--timeout-ms must be at least 1")

    report = run_network_load_test(
        base_url=args.base_url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        warmups=args.warmups,
        timeout_ms=args.timeout_ms,
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

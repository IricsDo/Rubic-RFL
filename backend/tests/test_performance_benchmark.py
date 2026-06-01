import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_api import run_benchmarks


def test_api_benchmark_report_has_expected_metrics():
    report = run_benchmarks(iterations=1, warmups=0)

    assert report["schema_version"] == "rubic-rfl-performance-v1"
    assert report["summary"]["status"] == "passed"
    assert report["summary"]["failures"] == []

    expected = {
        "health",
        "validate_solved_cube",
        "scramble_depth_4",
        "classical_solve_depth_3",
        "analytics_solves",
    }
    assert set(report["benchmarks"]) == expected

    for metrics in report["benchmarks"].values():
        assert metrics["count"] == 1
        assert metrics["min_ms"] >= 0
        assert metrics["median_ms"] >= 0
        assert metrics["p95_ms"] >= 0
        assert metrics["max_ms"] >= 0

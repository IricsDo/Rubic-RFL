import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.load_test_backend import run_load_test


def test_backend_load_test_report_has_expected_mixed_scenarios():
    report = run_load_test(total_requests=8, concurrency=2, warmups=0)

    assert report["schema_version"] == "rubic-rfl-backend-load-v1"
    assert report["requests"] == 8
    assert report["concurrency"] == 2
    assert report["summary"]["status"] == "passed"
    assert report["summary"]["failures"] == []
    assert report["error_rate"] == 0
    assert report["throughput_rps"] > 0

    expected = {
        "health",
        "metrics",
        "validate_solved_cube",
        "scramble_depth_8",
        "apply_move",
        "classical_solve_depth_3",
        "sessions_list",
        "analytics_solves",
    }
    assert set(report["scenarios"]) == expected

    exercised = [metrics for metrics in report["scenarios"].values() if metrics["count"] > 0]
    assert len(exercised) >= 5
    for metrics in exercised:
        assert metrics["min_ms"] >= 0
        assert metrics["median_ms"] >= 0
        assert metrics["p95_ms"] >= 0
        assert metrics["p99_ms"] >= 0
        assert metrics["max_ms"] >= 0
        assert metrics["errors"] == 0

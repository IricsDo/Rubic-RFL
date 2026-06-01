import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_websocket import run_benchmarks


def test_websocket_benchmark_report_has_expected_metrics():
    report = run_benchmarks(iterations=1, warmups=0)

    assert report["schema_version"] == "rubic-rfl-websocket-performance-v1"
    assert report["summary"]["status"] == "passed"
    assert report["summary"]["failures"] == []

    expected = {
        "rl_websocket_first_event",
        "rl_websocket_completed",
    }
    assert set(report["benchmarks"]) == expected
    assert report["stream"]["expected_event_count"] == 4
    assert report["stream"]["min_event_count"] == 4
    assert report["stream"]["max_event_count"] == 4

    for metrics in report["benchmarks"].values():
        assert metrics["count"] == 1
        assert metrics["min_ms"] >= 0
        assert metrics["median_ms"] >= 0
        assert metrics["p95_ms"] >= 0
        assert metrics["max_ms"] >= 0

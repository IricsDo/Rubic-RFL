import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.load_test_network import run_network_load_test


SOLVED_STICKERS = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"


class LoadTestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text, status=200):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({"status": "ok"})
        elif path == "/metrics":
            self._send_text("# TYPE rubic_http_requests_total counter\nrubic_http_requests_total 1\n")
        elif path == "/sessions":
            self._send_json({"sessions": [], "storage": {"backend": "test"}})
        elif path == "/analytics/solves":
            self._send_json({"totals": {}, "performance": {}})
        elif path == "/solve/rl/status":
            self._send_json({"solver": "rl-policy", "status": "loaded", "available": True})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        payload = self._read_json()
        if path == "/cube/scramble":
            depth = int(payload.get("depth", 0))
            self._send_json({"stickers": SOLVED_STICKERS, "scramble": ["R"] * depth})
        elif path == "/cube/validate":
            self._send_json({"validation": {"valid": True}})
        elif path == "/cube/apply-move":
            history = list(payload.get("history", []))
            history.append(payload.get("move", "R"))
            self._send_json({"history": history})
        elif path == "/solve/classical":
            self._send_json({"status": "solved", "moves": [], "move_count": 0})
        else:
            self._send_json({"error": "not found"}, status=404)


def test_network_load_test_report_has_expected_mixed_scenarios():
    server = ThreadingHTTPServer(("127.0.0.1", 0), LoadTestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        report = run_network_load_test(
            base_url=f"http://127.0.0.1:{server.server_port}",
            total_requests=40,
            concurrency=4,
            warmups=2,
            timeout_ms=2000,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert report["schema_version"] == "rubic-rfl-network-load-v1"
    assert report["requests"] == 40
    assert report["concurrency"] == 4
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
        "rl_status",
    }
    assert set(report["scenarios"]) == expected

    for metrics in report["scenarios"].values():
        assert metrics["count"] > 0
        assert metrics["min_ms"] >= 0
        assert metrics["median_ms"] >= 0
        assert metrics["p95_ms"] >= 0
        assert metrics["p99_ms"] >= 0
        assert metrics["max_ms"] >= 0
        assert metrics["errors"] == 0

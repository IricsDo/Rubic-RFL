from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Mapping


LabelSet = tuple[tuple[str, str], ...]


class MetricsRegistry:
    """Dependency-free in-process metrics for local and container monitoring."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, defaultdict[LabelSet, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._gauges: dict[str, dict[LabelSet, float]] = defaultdict(dict)
        self._summary_counts: dict[str, defaultdict[LabelSet, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._summary_sums: dict[str, defaultdict[LabelSet, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._active_websockets: defaultdict[str, int] = defaultdict(int)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._summary_counts.clear()
            self._summary_sums.clear()
            self._active_websockets.clear()

    def record_http_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        labels = {
            "method": method.upper(),
            "path": path,
            "status": str(status_code),
        }
        with self._lock:
            self._increment_locked("rubic_http_requests_total", labels, 1)
            self._observe_locked(
                "rubic_http_request_duration_seconds",
                labels,
                max(duration_seconds, 0.0),
            )
            if status_code >= 500:
                self._increment_locked("rubic_http_request_errors_total", labels, 1)

    def record_solver_run(
        self,
        *,
        solver: str,
        status: str,
        duration_ms: float | int | None,
        move_count: float | int | None,
        model_version: str | None = None,
    ) -> None:
        labels = {
            "model_version": _safe_label(model_version or "none"),
            "solver": _safe_label(solver),
            "status": _safe_label(status),
        }
        with self._lock:
            self._increment_locked("rubic_solver_runs_total", labels, 1)
            if status == "solved":
                self._increment_locked(
                    "rubic_solver_success_total",
                    {
                        "model_version": labels["model_version"],
                        "solver": labels["solver"],
                    },
                    1,
                )
            if duration_ms is not None:
                self._observe_locked(
                    "rubic_solver_duration_seconds",
                    labels,
                    max(float(duration_ms) / 1000, 0.0),
                )
            if move_count is not None:
                self._increment_locked(
                    "rubic_solver_moves_total",
                    labels,
                    max(float(move_count), 0.0),
                )

    def websocket_open(self, endpoint: str) -> None:
        labels = {"endpoint": endpoint}
        with self._lock:
            self._increment_locked("rubic_websocket_connections_total", labels, 1)
            self._active_websockets[endpoint] += 1
            self._set_gauge_locked(
                "rubic_websocket_active_connections",
                labels,
                self._active_websockets[endpoint],
            )

    def websocket_close(self, endpoint: str) -> None:
        labels = {"endpoint": endpoint}
        with self._lock:
            self._active_websockets[endpoint] = max(
                self._active_websockets[endpoint] - 1,
                0,
            )
            self._set_gauge_locked(
                "rubic_websocket_active_connections",
                labels,
                self._active_websockets[endpoint],
            )

    def websocket_error(self, endpoint: str) -> None:
        with self._lock:
            self._increment_locked(
                "rubic_websocket_errors_total",
                {"endpoint": endpoint},
                1,
            )

    def render_prometheus(self) -> str:
        with self._lock:
            counters = {
                name: dict(values) for name, values in self._counters.items()
            }
            gauges = {name: dict(values) for name, values in self._gauges.items()}
            summary_counts = {
                name: dict(values) for name, values in self._summary_counts.items()
            }
            summary_sums = {
                name: dict(values) for name, values in self._summary_sums.items()
            }

        lines: list[str] = []
        self._append_metric(
            lines,
            "rubic_http_requests_total",
            "Total HTTP requests by method, route, and status.",
            "counter",
            counters,
        )
        self._append_metric(
            lines,
            "rubic_http_request_errors_total",
            "HTTP requests that returned a server error.",
            "counter",
            counters,
        )
        self._append_summary(
            lines,
            "rubic_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            summary_counts,
            summary_sums,
        )
        self._append_metric(
            lines,
            "rubic_websocket_connections_total",
            "Total accepted WebSocket connections.",
            "counter",
            counters,
        )
        self._append_metric(
            lines,
            "rubic_websocket_active_connections",
            "Currently active WebSocket connections.",
            "gauge",
            gauges,
        )
        self._append_metric(
            lines,
            "rubic_websocket_errors_total",
            "WebSocket handler errors.",
            "counter",
            counters,
        )
        self._append_metric(
            lines,
            "rubic_solver_runs_total",
            "Solver runs by solver, status, and model version.",
            "counter",
            counters,
        )
        self._append_metric(
            lines,
            "rubic_solver_success_total",
            "Solver runs that completed with solved status.",
            "counter",
            counters,
        )
        self._append_summary(
            lines,
            "rubic_solver_duration_seconds",
            "Solver duration in seconds.",
            summary_counts,
            summary_sums,
        )
        self._append_metric(
            lines,
            "rubic_solver_moves_total",
            "Total moves returned by solver runs.",
            "counter",
            counters,
        )
        return "\n".join(lines) + "\n"

    def _increment_locked(
        self,
        name: str,
        labels: Mapping[str, str],
        amount: float,
    ) -> None:
        self._counters[name][_label_set(labels)] += amount

    def _set_gauge_locked(
        self,
        name: str,
        labels: Mapping[str, str],
        value: float,
    ) -> None:
        self._gauges[name][_label_set(labels)] = value

    def _observe_locked(
        self,
        name: str,
        labels: Mapping[str, str],
        value: float,
    ) -> None:
        label_set = _label_set(labels)
        self._summary_counts[name][label_set] += 1
        self._summary_sums[name][label_set] += value

    @staticmethod
    def _append_metric(
        lines: list[str],
        name: str,
        help_text: str,
        metric_type: str,
        metrics: Mapping[str, Mapping[LabelSet, float]],
    ) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")
        for labels, value in sorted(metrics.get(name, {}).items()):
            lines.append(f"{name}{_format_labels(labels)} {_format_number(value)}")

    @staticmethod
    def _append_summary(
        lines: list[str],
        name: str,
        help_text: str,
        counts: Mapping[str, Mapping[LabelSet, float]],
        sums: Mapping[str, Mapping[LabelSet, float]],
    ) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} summary")
        label_sets = sorted(set(counts.get(name, {})) | set(sums.get(name, {})))
        for labels in label_sets:
            lines.append(
                f"{name}_count{_format_labels(labels)} "
                f"{_format_number(counts.get(name, {}).get(labels, 0))}"
            )
            lines.append(
                f"{name}_sum{_format_labels(labels)} "
                f"{_format_number(sums.get(name, {}).get(labels, 0))}"
            )


def _label_set(labels: Mapping[str, object]) -> LabelSet:
    return tuple(sorted((str(key), _safe_label(value)) for key, value in labels.items()))


def _safe_label(value: object) -> str:
    text = str("unknown" if value is None or value == "" else value)
    return text.replace("\r", " ").replace("\n", " ")


def _format_labels(labels: LabelSet) -> str:
    if not labels:
        return ""
    values = ",".join(f'{name}="{_escape_label_value(value)}"' for name, value in labels)
    return "{" + values + "}"


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")

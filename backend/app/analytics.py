from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable


def build_solve_analytics(
    sessions: Iterable[dict[str, Any]],
    *,
    recent_limit: int = 10,
) -> dict[str, Any]:
    ordered_sessions = sorted(
        list(sessions),
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )
    session_count = len(ordered_sessions)
    solved_count = sum(1 for session in ordered_sessions if _is_solved(session))

    return {
        "totals": {
            "session_count": session_count,
            "solved_count": solved_count,
            "unsolved_count": session_count - solved_count,
            "solve_rate": _rate(solved_count, session_count),
        },
        "performance": _performance(ordered_sessions),
        "by_status": _breakdown(ordered_sessions, "status"),
        "by_solver": _breakdown(ordered_sessions, "solver"),
        "by_model": _breakdown(ordered_sessions, "model_version"),
        "by_day": _daily_breakdown(ordered_sessions),
        "recent_sessions": ordered_sessions[: max(0, recent_limit)],
    }


def _performance(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    move_counts = [_number(session.get("move_count")) for session in sessions]
    durations = [_number(session.get("duration_ms")) for session in sessions]
    move_counts = [value for value in move_counts if value is not None]
    durations = [value for value in durations if value is not None]

    return {
        "average_move_count": _average(move_counts),
        "average_duration_ms": _average(durations),
        "best_move_count": min(move_counts) if move_counts else None,
        "best_duration_ms": min(durations) if durations else None,
        "max_move_count": max(move_counts) if move_counts else None,
        "max_duration_ms": max(durations) if durations else None,
    }


def _breakdown(sessions: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        raw_value = session.get(key)
        label = str(raw_value if raw_value not in {None, ""} else "unknown")
        groups[label].append(session)

    rows = []
    for label, group in groups.items():
        solved_count = sum(1 for session in group if _is_solved(session))
        move_counts = [
            value
            for value in (_number(session.get("move_count")) for session in group)
            if value is not None
        ]
        durations = [
            value
            for value in (_number(session.get("duration_ms")) for session in group)
            if value is not None
        ]
        rows.append(
            {
                key: label,
                "session_count": len(group),
                "solved_count": solved_count,
                "solve_rate": _rate(solved_count, len(group)),
                "average_move_count": _average(move_counts),
                "average_duration_ms": _average(durations),
            }
        )

    return sorted(rows, key=lambda item: (-int(item["session_count"]), str(item[key])))


def _daily_breakdown(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        groups[_date_bucket(session.get("created_at"))].append(session)

    rows = []
    for date, group in groups.items():
        solved_count = sum(1 for session in group if _is_solved(session))
        rows.append(
            {
                "date": date,
                "session_count": len(group),
                "solved_count": solved_count,
                "solve_rate": _rate(solved_count, len(group)),
            }
        )
    return sorted(rows, key=lambda item: item["date"])


def _date_bucket(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "unknown"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return "unknown"


def _is_solved(session: dict[str, Any]) -> bool:
    return str(session.get("status") or "").lower() == "solved"


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return _rounded(sum(values) / len(values))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _rounded(numerator / denominator)


def _rounded(value: float) -> float:
    rounded = round(value, 4)
    return int(rounded) if rounded.is_integer() else rounded

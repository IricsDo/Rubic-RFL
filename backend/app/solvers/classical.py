from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import kociemba

from app.cube import Cube, parse_moves, validate_stickers


@dataclass(frozen=True)
class SolverResult:
    solver: str
    status: str
    moves: tuple[str, ...]
    move_count: int
    duration_ms: int
    message: str
    details: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "solver": self.solver,
            "status": self.status,
            "moves": list(self.moves),
            "move_count": self.move_count,
            "duration_ms": self.duration_ms,
            "message": self.message,
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


class KociembaSolver:
    """Deterministic two-phase solver for arbitrary valid 3x3 cube states."""

    name = "classical-kociemba"

    def solve(self, cube: Cube) -> SolverResult:
        started_at = perf_counter()
        validation = validate_stickers(cube.stickers)
        if not validation.valid:
            duration = int((perf_counter() - started_at) * 1000)
            return SolverResult(
                self.name,
                "invalid",
                (),
                0,
                duration,
                "; ".join(validation.errors),
            )

        if cube.is_solved():
            duration = int((perf_counter() - started_at) * 1000)
            return SolverResult(self.name, "solved", (), 0, duration, "Cube is solved")

        try:
            solution_text = kociemba.solve(cube.to_string())
            moves = parse_moves(solution_text)
        except ValueError as error:
            duration = int((perf_counter() - started_at) * 1000)
            return SolverResult(
                self.name,
                "failed",
                (),
                0,
                duration,
                f"Kociemba solver could not solve this state: {error}",
            )

        solved_cube = cube.apply_sequence(moves, record_history=False)
        duration = int((perf_counter() - started_at) * 1000)
        if not solved_cube.is_solved():
            return SolverResult(
                self.name,
                "failed",
                moves,
                len(moves),
                duration,
                "Kociemba returned moves that did not solve the cube",
            )

        return SolverResult(
            self.name,
            "solved",
            moves,
            len(moves),
            duration,
            "Solved with Kociemba two-phase search",
        )

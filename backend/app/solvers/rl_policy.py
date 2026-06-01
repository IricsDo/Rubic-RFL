from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from app.cube import Cube, validate_stickers
from app.solvers.classical import SolverResult
from app.solvers.rl_search import (
    RL_ACTION_MOVES,
    PolicyGuidedSearch,
    PredictsActions,
    SearchConfig,
    SearchResult,
    predict_top_action,
)


class RLSolver:
    """Policy-guided search solver backed by a trained RL-track checkpoint."""

    name = "rl-policy"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        max_steps: int | None = None,
        search_width: int | None = None,
        search_top_k: int | None = None,
        search_trace_limit: int | None = None,
        policy: PredictsActions | None = None,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.max_steps = self._resolve_max_steps(max_steps)
        self.search_width = self._resolve_search_width(search_width)
        self.search_top_k = self._resolve_search_top_k(search_top_k)
        self.search_trace_limit = self._resolve_search_trace_limit(search_trace_limit)
        self._policy = policy
        self._load_error: str | None = None

    @classmethod
    def from_environment(cls) -> "RLSolver":
        return cls(
            model_path=os.environ.get("RUBIC_RL_MODEL_PATH"),
            max_steps=_int_from_env("RUBIC_RL_MAX_STEPS"),
            search_width=_int_from_env("RUBIC_RL_SEARCH_WIDTH"),
            search_top_k=_int_from_env("RUBIC_RL_SEARCH_TOP_K"),
            search_trace_limit=_int_from_env("RUBIC_RL_SEARCH_TRACE_LIMIT"),
        )

    def solve(self, cube: Cube) -> SolverResult:
        started_at = perf_counter()
        validation = validate_stickers(cube.stickers)
        if not validation.valid:
            return self._result(
                "invalid",
                (),
                started_at,
                "; ".join(validation.errors),
            )

        if cube.is_solved():
            return self._result("solved", (), started_at, "Cube is solved")

        policy = self._get_policy()
        if policy is None:
            message = self._load_error or "RL policy checkpoint is not configured"
            return self._result("unavailable", (), started_at, message)

        try:
            search_result = PolicyGuidedSearch(
                policy,
                config=SearchConfig(
                    max_depth=self.max_steps,
                    beam_width=self.search_width,
                    top_k=self.search_top_k,
                    trace_limit=self.search_trace_limit,
                ),
            ).solve(cube)
        except Exception as error:
            return self._result(
                "failed",
                (),
                started_at,
                f"RL policy search failed: {error}",
            )

        return self._result(
            search_result.status,
            search_result.moves,
            started_at,
            search_result.message,
            details=self._search_details(search_result),
        )

    def _get_policy(self) -> PredictsActions | None:
        if self._policy is not None:
            return self._policy

        if self.model_path is None:
            self._load_error = "RL policy checkpoint is not configured"
            return None
        if not self.model_path.exists():
            self._load_error = f"RL policy checkpoint not found: {self.model_path}"
            return None

        try:
            from rubic_rl.policies import load_policy

            self._policy = load_policy(self.model_path, policy_type="auto")
            self._load_error = None
            return self._policy
        except Exception as error:
            self._load_error = f"RL policy checkpoint could not be loaded: {error}"
            return None

    def _result(
        self,
        status: str,
        moves: tuple[str, ...],
        started_at: float,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> SolverResult:
        duration = int((perf_counter() - started_at) * 1000)
        return SolverResult(
            self.name,
            status,
            moves,
            len(moves),
            duration,
            message,
            details,
        )

    @staticmethod
    def _resolve_model_path(value: str | Path | None) -> Path | None:
        if value is None or str(value).strip() == "":
            default_path = (
                _project_root()
                / "checkpoints"
                / "mlp-baseline-depth-1-2-3.npz"
            )
            return default_path

        path = Path(value)
        if path.is_absolute():
            return path
        return _project_root() / path

    @staticmethod
    def _resolve_max_steps(value: int | None) -> int:
        steps = 30 if value is None else int(value)
        if steps <= 0:
            raise ValueError("max_steps must be positive")
        return steps

    @staticmethod
    def _resolve_search_width(value: int | None) -> int:
        width = 5 if value is None else int(value)
        if width <= 0:
            raise ValueError("search_width must be positive")
        return width

    @staticmethod
    def _resolve_search_top_k(value: int | None) -> int:
        top_k = 5 if value is None else int(value)
        if top_k <= 0:
            raise ValueError("search_top_k must be positive")
        return top_k

    @staticmethod
    def _resolve_search_trace_limit(value: int | None) -> int:
        limit = 200 if value is None else int(value)
        if limit < 0:
            raise ValueError("search_trace_limit cannot be negative")
        return limit

    def _search_details(self, result: SearchResult) -> dict[str, object]:
        return {
            "strategy": "policy-guided-beam-search",
            "model_version": self.model_path.stem if self.model_path else "in-memory-policy",
            "model_checkpoint": self.model_path.name if self.model_path else "in-memory-policy",
            "max_depth": self.max_steps,
            "beam_width": self.search_width,
            "top_k": self.search_top_k,
            "expanded_states": result.expanded_states,
            "visited_states": result.visited_states,
            "depth_reached": result.depth_reached,
            "search_trace": {
                "trace_limit": result.trace_limit,
                "truncated": result.trace_truncated,
                "records": [
                    {
                        "node_id": record.node_id,
                        "parent_id": record.parent_id,
                        "depth": record.depth,
                        "path": list(record.path),
                        "stickers": record.stickers,
                        "score": record.score,
                        "beam_rank": record.beam_rank,
                        "candidates": [
                            {
                                "action": candidate.action,
                                "move": candidate.move,
                                "probability": candidate.probability,
                                "outcome": candidate.outcome,
                                "child_id": candidate.child_id,
                                "child_score": candidate.child_score,
                                "reason": candidate.reason,
                            }
                            for candidate in record.candidates
                        ],
                    }
                    for record in result.search_trace
                ],
            },
            "steps": [
                {
                    "step": step.step,
                    "selected_move": step.selected_move,
                    "confidence": step.confidence,
                    "top_candidates": [
                        {
                            "action": candidate.action,
                            "move": candidate.move,
                            "probability": candidate.probability,
                        }
                        for candidate in step.top_candidates
                    ],
                }
                for step in result.steps
            ],
        }


def _predict_action(policy: PredictsActions, cube: Cube) -> int:
    return predict_top_action(policy, cube)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _int_from_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)

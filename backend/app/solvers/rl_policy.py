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
        policy_type: str | None = None,
        policy_device: str | None = None,
        policy: PredictsActions | None = None,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.max_steps = self._resolve_max_steps(max_steps)
        self.search_width = self._resolve_search_width(search_width)
        self.search_top_k = self._resolve_search_top_k(search_top_k)
        self.search_trace_limit = self._resolve_search_trace_limit(search_trace_limit)
        self.policy_type = self._resolve_policy_type(policy_type)
        self.policy_device = self._resolve_policy_device(policy_device)
        self._policy = policy
        self._loaded_policy_type: str | None = _policy_type_from_loaded_policy(policy)
        self._load_error: str | None = None

    @classmethod
    def from_environment(cls) -> "RLSolver":
        return cls(
            model_path=os.environ.get("RUBIC_RL_MODEL_PATH"),
            max_steps=_int_from_env("RUBIC_RL_MAX_STEPS"),
            search_width=_int_from_env("RUBIC_RL_SEARCH_WIDTH"),
            search_top_k=_int_from_env("RUBIC_RL_SEARCH_TOP_K"),
            search_trace_limit=_int_from_env("RUBIC_RL_SEARCH_TRACE_LIMIT"),
            policy_type=os.environ.get("RUBIC_RL_POLICY_TYPE"),
            policy_device=os.environ.get("RUBIC_RL_POLICY_DEVICE"),
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

            self._policy = load_policy(
                self.model_path,
                policy_type=self.policy_type,
                torch_map_location=self.policy_device,
            )
            self._loaded_policy_type = _policy_type_from_loaded_policy(self._policy)
            self._load_error = None
            return self._policy
        except Exception as error:
            self._load_error = f"RL policy checkpoint could not be loaded: {error}"
            return None

    def runtime_status(self, *, load_policy: bool = False) -> dict[str, object]:
        if load_policy:
            self._get_policy()

        checkpoint_exists = (
            self.model_path.exists() if self.model_path is not None else None
        )
        loaded = self._policy is not None
        runtime_state = self._runtime_state(checkpoint_exists)
        return {
            "solver": self.name,
            "status": runtime_state,
            "ready": loaded,
            "available": runtime_state in {"loaded", "checkpoint_available"},
            "configured": self.model_path is not None or loaded,
            "loaded": loaded,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_checkpoint": (
                self.model_path.name if self.model_path else "in-memory-policy"
            ),
            "checkpoint_exists": checkpoint_exists,
            "model_version": self._model_version(),
            "configured_policy_type": self.policy_type,
            "policy_type": self._effective_policy_type(),
            "policy_device": self.policy_device,
            "load_error": self._load_error,
            "search": {
                "strategy": "policy-guided-beam-search",
                "max_depth": self.max_steps,
                "beam_width": self.search_width,
                "top_k": self.search_top_k,
                "trace_limit": self.search_trace_limit,
            },
        }

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
                / "lightweight-linear-axis-samples-28-heldout"
                / "linear-policy.npz"
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

    @staticmethod
    def _resolve_policy_type(value: str | None) -> str:
        policy_type = (
            "auto"
            if value is None or str(value).strip() == ""
            else str(value).strip().lower()
        )
        if policy_type not in {"auto", "linear", "mlp", "torch"}:
            raise ValueError("policy_type must be one of auto, linear, mlp, or torch")
        return policy_type

    @staticmethod
    def _resolve_policy_device(value: str | None) -> str:
        device = (
            "cpu"
            if value is None or str(value).strip() == ""
            else str(value).strip()
        )
        if not device:
            raise ValueError("policy_device cannot be empty")
        return device

    def _search_details(self, result: SearchResult) -> dict[str, object]:
        return {
            "strategy": "policy-guided-beam-search",
            "model_version": self._model_version(),
            "model_checkpoint": (
                self.model_path.name if self.model_path else "in-memory-policy"
            ),
            "policy_type": self._effective_policy_type(),
            "policy_device": self.policy_device,
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

    def _model_version(self) -> str:
        policy_version = getattr(self._policy, "model_version", None)
        if policy_version not in {None, ""}:
            return str(policy_version)
        if self.model_path is not None:
            return self.model_path.stem
        return "in-memory-policy"

    def _effective_policy_type(self) -> str:
        if self._loaded_policy_type is not None:
            return self._loaded_policy_type
        if self.policy_type != "auto":
            return self.policy_type
        detected = _policy_type_from_path(self.model_path)
        return detected or "auto"

    def _runtime_state(self, checkpoint_exists: bool | None) -> str:
        if self._policy is not None:
            return "loaded"
        if self._load_error is not None:
            return "load_failed"
        if self.model_path is None:
            return "unconfigured"
        if checkpoint_exists:
            return "checkpoint_available"
        return "missing_checkpoint"


def _predict_action(policy: PredictsActions, cube: Cube) -> int:
    return predict_top_action(policy, cube)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _int_from_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _policy_type_from_loaded_policy(policy: object) -> str | None:
    if policy is None:
        return None
    raw_policy_type = getattr(policy, "policy_type", None)
    if raw_policy_type not in {None, ""}:
        return str(raw_policy_type)

    class_name = policy.__class__.__name__
    if class_name == "TorchPolicyValuePolicy":
        return "torch"
    if class_name == "MLPPolicy":
        return "mlp"
    if class_name == "LinearPolicy":
        return "linear"
    return "in-memory"


def _policy_type_from_path(path: Path | None) -> str | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return "torch"
    if suffix == ".npz":
        return "numpy"
    return None

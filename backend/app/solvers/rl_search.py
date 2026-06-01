from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Protocol

import numpy as np

from app.cube import Cube, FACE_ORDER, inverse_move


RL_ACTION_MOVES = (
    "U",
    "D",
    "L",
    "R",
    "F",
    "B",
    "U'",
    "D'",
    "L'",
    "R'",
    "F'",
    "B'",
    "U2",
    "D2",
    "L2",
    "R2",
    "F2",
    "B2",
)
STICKER_TO_INDEX = {face: index for index, face in enumerate(FACE_ORDER)}


class PredictsActions(Protocol):
    def predict(self, features):
        """Return integer action IDs for a batch of cube feature vectors."""


@dataclass(frozen=True)
class SearchConfig:
    max_depth: int = 30
    beam_width: int = 5
    top_k: int = 5
    trace_limit: int = 0

    def __post_init__(self) -> None:
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.trace_limit < 0:
            raise ValueError("trace_limit cannot be negative")


@dataclass(frozen=True)
class ActionCandidate:
    action: int
    move: str
    probability: float


@dataclass(frozen=True)
class SearchStep:
    step: int
    selected_move: str
    confidence: float
    top_candidates: tuple[ActionCandidate, ...]


@dataclass(frozen=True)
class SearchTraceCandidate:
    action: int
    move: str
    probability: float
    outcome: str
    child_id: int | None = None
    child_score: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SearchTraceRecord:
    node_id: int
    parent_id: int | None
    depth: int
    path: tuple[str, ...]
    stickers: str
    score: float
    beam_rank: int
    candidates: tuple[SearchTraceCandidate, ...]


@dataclass(frozen=True)
class SearchResult:
    status: str
    moves: tuple[str, ...]
    expanded_states: int
    visited_states: int
    depth_reached: int
    steps: tuple[SearchStep, ...]
    message: str
    search_trace: tuple[SearchTraceRecord, ...] = ()
    trace_limit: int = 0
    trace_truncated: bool = False


@dataclass(frozen=True)
class _SearchNode:
    node_id: int
    parent_id: int | None
    cube: Cube
    moves: tuple[str, ...]
    score: float
    steps: tuple[SearchStep, ...]


@dataclass(frozen=True)
class _PendingTraceCandidate:
    action: int
    move: str
    probability: float
    outcome: str
    child_id: int | None = None
    child_score: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _PendingTraceRecord:
    node: _SearchNode
    beam_rank: int
    candidates: tuple[_PendingTraceCandidate, ...]


class _SearchTraceRecorder:
    def __init__(self, limit: int):
        self.limit = limit
        self.records: list[SearchTraceRecord] = []
        self.truncated = False

    def append_pending(
        self,
        pending_records: tuple[_PendingTraceRecord, ...],
        *,
        kept_child_ids: set[int] | None = None,
    ) -> None:
        if not pending_records:
            return

        for pending in pending_records:
            candidates = tuple(
                _finalize_trace_candidate(candidate, kept_child_ids)
                for candidate in pending.candidates
            )
            self.append(
                SearchTraceRecord(
                    node_id=pending.node.node_id,
                    parent_id=pending.node.parent_id,
                    depth=len(pending.node.moves),
                    path=pending.node.moves,
                    stickers=pending.node.cube.to_string(),
                    score=pending.node.score,
                    beam_rank=pending.beam_rank,
                    candidates=candidates,
                )
            )

    def append(self, record: SearchTraceRecord) -> None:
        if self.limit <= 0:
            return
        if len(self.records) >= self.limit:
            self.truncated = True
            return
        self.records.append(record)


class PolicyGuidedSearch:
    """Beam search that uses policy probabilities to prioritize cube states."""

    def __init__(
        self,
        policy: PredictsActions,
        *,
        config: SearchConfig | None = None,
    ):
        self.policy = policy
        self.config = config or SearchConfig()

    def solve(self, cube: Cube) -> SearchResult:
        trace_recorder = _SearchTraceRecorder(self.config.trace_limit)
        if cube.is_solved():
            return SearchResult(
                "solved",
                (),
                0,
                1,
                0,
                (),
                "Cube is solved",
                (),
                self.config.trace_limit,
                False,
            )

        next_node_id = 1
        start = _SearchNode(
            node_id=next_node_id,
            parent_id=None,
            cube=cube,
            moves=(),
            score=0.0,
            steps=(),
        )
        beam = [start]
        visited = {cube.stickers}
        expanded_states = 0
        best_node = start
        depth_reached = 0

        for _ in range(self.config.max_depth):
            next_beam: list[_SearchNode] = []
            pending_trace_records: list[_PendingTraceRecord] = []
            for beam_rank, node in enumerate(beam, start=1):
                expanded_states += 1
                candidates = rank_action_candidates(
                    self.policy,
                    node.cube,
                    top_k=self.config.top_k,
                )
                trace_candidates: list[_PendingTraceCandidate] = []
                for candidate in candidates:
                    if _undoes_previous_move(node.moves, candidate.move):
                        trace_candidates.append(
                            _trace_candidate(
                                candidate,
                                "skipped_inverse",
                                reason="move immediately reverses the current path",
                            )
                        )
                        continue

                    next_cube = node.cube.apply_move(
                        candidate.move,
                        record_history=False,
                    )
                    if next_cube.stickers in visited:
                        trace_candidates.append(
                            _trace_candidate(
                                candidate,
                                "skipped_visited",
                                reason="cube state was already visited",
                            )
                        )
                        continue

                    visited.add(next_cube.stickers)
                    next_moves = node.moves + (candidate.move,)
                    next_steps = node.steps + (
                        SearchStep(
                            step=len(next_moves),
                            selected_move=candidate.move,
                            confidence=candidate.probability,
                            top_candidates=candidates,
                        ),
                    )
                    next_node_id += 1
                    next_score = node.score + log(max(candidate.probability, 1e-12))
                    next_node = _SearchNode(
                        node_id=next_node_id,
                        parent_id=node.node_id,
                        cube=next_cube,
                        moves=next_moves,
                        score=next_score,
                        steps=next_steps,
                    )
                    if next_cube.is_solved():
                        trace_candidates.append(
                            _trace_candidate(
                                candidate,
                                "solution",
                                child_id=next_node.node_id,
                                child_score=next_node.score,
                            )
                        )
                        pending_trace_records.append(
                            _PendingTraceRecord(
                                node=node,
                                beam_rank=beam_rank,
                                candidates=tuple(trace_candidates),
                            )
                        )
                        trace_recorder.append_pending(tuple(pending_trace_records))
                        return SearchResult(
                            "solved",
                            next_moves,
                            expanded_states,
                            len(visited),
                            len(next_moves),
                            next_steps,
                            "Solved with policy-guided beam search",
                            tuple(trace_recorder.records),
                            self.config.trace_limit,
                            trace_recorder.truncated,
                        )

                    trace_candidates.append(
                        _trace_candidate(
                            candidate,
                            "queued",
                            child_id=next_node.node_id,
                            child_score=next_node.score,
                        )
                    )
                    next_beam.append(next_node)

                pending_trace_records.append(
                    _PendingTraceRecord(
                        node=node,
                        beam_rank=beam_rank,
                        candidates=tuple(trace_candidates),
                    )
                )

            if not next_beam:
                trace_recorder.append_pending(tuple(pending_trace_records))
                break

            next_beam.sort(key=lambda item: (item.score, -len(item.moves)), reverse=True)
            beam = next_beam[: self.config.beam_width]
            kept_child_ids = {node.node_id for node in beam}
            trace_recorder.append_pending(
                tuple(pending_trace_records),
                kept_child_ids=kept_child_ids,
            )
            best_node = beam[0]
            depth_reached = len(best_node.moves)

        return SearchResult(
            "failed",
            best_node.moves,
            expanded_states,
            len(visited),
            depth_reached,
            best_node.steps,
            f"Policy-guided beam search did not solve within {self.config.max_depth} steps",
            tuple(trace_recorder.records),
            self.config.trace_limit,
            trace_recorder.truncated,
        )


def cube_features(cube: Cube) -> np.ndarray:
    indices = np.asarray([STICKER_TO_INDEX[value] for value in cube.stickers])
    features = np.zeros((54, len(FACE_ORDER)), dtype=np.float64)
    features[np.arange(54), indices] = 1.0
    return features.reshape(1, -1)


def predict_top_action(policy: PredictsActions, cube: Cube) -> int:
    return rank_action_candidates(policy, cube, top_k=1)[0].action


def rank_action_candidates(
    policy: PredictsActions,
    cube: Cube,
    *,
    top_k: int,
) -> tuple[ActionCandidate, ...]:
    probabilities = _action_probabilities(policy, cube)
    limit = min(top_k, len(RL_ACTION_MOVES))
    ordered_actions = np.argsort(probabilities)[::-1][:limit]
    return tuple(
        ActionCandidate(
            action=int(action),
            move=RL_ACTION_MOVES[int(action)],
            probability=float(probabilities[int(action)]),
        )
        for action in ordered_actions
    )


def _action_probabilities(policy: PredictsActions, cube: Cube) -> np.ndarray:
    features = cube_features(cube)
    predict_proba = getattr(policy, "predict_proba", None)
    if callable(predict_proba):
        raw_probabilities = np.asarray(predict_proba(features), dtype=np.float64)
        if raw_probabilities.ndim == 2:
            if raw_probabilities.shape[0] != 1:
                raise ValueError("predict_proba must return one row per cube state")
            raw_probabilities = raw_probabilities[0]
        if raw_probabilities.shape != (len(RL_ACTION_MOVES),):
            raise ValueError(
                f"predict_proba must return {len(RL_ACTION_MOVES)} action scores"
            )
        if np.any(~np.isfinite(raw_probabilities)):
            raise ValueError("predict_proba returned non-finite scores")
        if np.any(raw_probabilities < 0):
            raise ValueError("predict_proba returned negative scores")

        total = float(raw_probabilities.sum())
        if total <= 0:
            raise ValueError("predict_proba returned all-zero scores")
        return raw_probabilities / total

    action = int(policy.predict(features)[0])
    if action < 0 or action >= len(RL_ACTION_MOVES):
        raise ValueError(f"RL policy returned invalid action: {action}")
    probabilities = np.zeros(len(RL_ACTION_MOVES), dtype=np.float64)
    probabilities[action] = 1.0
    return probabilities


def _undoes_previous_move(moves: tuple[str, ...], next_move: str) -> bool:
    return bool(moves) and inverse_move(next_move) == moves[-1]


def _trace_candidate(
    candidate: ActionCandidate,
    outcome: str,
    *,
    child_id: int | None = None,
    child_score: float | None = None,
    reason: str | None = None,
) -> _PendingTraceCandidate:
    return _PendingTraceCandidate(
        action=candidate.action,
        move=candidate.move,
        probability=candidate.probability,
        outcome=outcome,
        child_id=child_id,
        child_score=child_score,
        reason=reason,
    )


def _finalize_trace_candidate(
    candidate: _PendingTraceCandidate,
    kept_child_ids: set[int] | None,
) -> SearchTraceCandidate:
    outcome = candidate.outcome
    if outcome == "queued" and kept_child_ids is not None:
        outcome = (
            "kept_in_beam"
            if candidate.child_id in kept_child_ids
            else "pruned_by_beam"
        )

    return SearchTraceCandidate(
        action=candidate.action,
        move=candidate.move,
        probability=candidate.probability,
        outcome=outcome,
        child_id=candidate.child_id,
        child_score=candidate.child_score,
        reason=candidate.reason,
    )

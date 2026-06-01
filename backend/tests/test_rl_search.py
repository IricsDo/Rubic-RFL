import unittest

import numpy as np
from fastapi.testclient import TestClient

from app.cube import Cube
from app.main import app
import app.main as main_module
from app.solvers import RLSolver
from app.solvers.rl_policy import RL_ACTION_MOVES
from app.solvers.rl_search import PolicyGuidedSearch, SearchConfig


class ConstantRankedPolicy:
    def __init__(self, ranked_actions: tuple[int, ...]):
        self.ranked_actions = ranked_actions

    def predict_proba(self, features):
        probabilities = np.full((features.shape[0], len(RL_ACTION_MOVES)), 1e-6)
        for rank, action in enumerate(self.ranked_actions):
            probabilities[:, action] = len(self.ranked_actions) - rank
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, features):
        return np.asarray([self.ranked_actions[0] for _ in range(features.shape[0])])


class RlSearchTests(unittest.TestCase):
    def test_beam_search_finds_lower_ranked_solution_path(self):
        policy = ConstantRankedPolicy(
            (
                RL_ACTION_MOVES.index("F"),
                RL_ACTION_MOVES.index("U'"),
                RL_ACTION_MOVES.index("R'"),
            )
        )
        cube = Cube.solved().apply_sequence(("R", "U"), record_history=False)

        result = PolicyGuidedSearch(
            policy,
            config=SearchConfig(max_depth=2, beam_width=3, top_k=3, trace_limit=20),
        ).solve(cube)

        self.assertEqual(result.status, "solved")
        self.assertEqual(result.moves, ("U'", "R'"))
        self.assertGreater(result.expanded_states, 0)
        self.assertEqual(result.depth_reached, 2)
        self.assertEqual(result.steps[0].selected_move, "U'")
        self.assertEqual(result.steps[0].top_candidates[0].move, "F")
        self.assertIn("U'", [candidate.move for candidate in result.steps[0].top_candidates])
        self.assertFalse(result.trace_truncated)
        self.assertGreater(len(result.search_trace), 0)
        self.assertEqual(result.search_trace[0].node_id, 1)
        self.assertEqual(result.search_trace[0].depth, 0)
        solution_candidates = [
            candidate
            for record in result.search_trace
            for candidate in record.candidates
            if candidate.outcome == "solution"
        ]
        self.assertEqual(solution_candidates[0].move, "R'")

    def test_search_trace_marks_beam_pruned_branches(self):
        policy = ConstantRankedPolicy(
            (
                RL_ACTION_MOVES.index("F"),
                RL_ACTION_MOVES.index("U'"),
                RL_ACTION_MOVES.index("R'"),
            )
        )
        cube = Cube.solved().apply_sequence(("R", "U"), record_history=False)

        result = PolicyGuidedSearch(
            policy,
            config=SearchConfig(max_depth=1, beam_width=1, top_k=3, trace_limit=2),
        ).solve(cube)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.trace_limit, 2)
        outcomes = [candidate.outcome for candidate in result.search_trace[0].candidates]
        self.assertIn("kept_in_beam", outcomes)
        self.assertIn("pruned_by_beam", outcomes)

    def test_rl_endpoint_uses_search_and_returns_metadata(self):
        original_solver = main_module.rl_solver
        main_module.rl_solver = RLSolver(
            policy=ConstantRankedPolicy(
                (
                    RL_ACTION_MOVES.index("F"),
                    RL_ACTION_MOVES.index("U'"),
                    RL_ACTION_MOVES.index("R'"),
                )
            ),
            model_path="in-memory-search-policy",
            max_steps=2,
            search_width=3,
            search_top_k=3,
        )
        cube = Cube.solved().apply_sequence(("R", "U"), record_history=False)
        try:
            response = TestClient(app).post(
                "/solve/rl", json={"stickers": cube.to_string()}
            )
        finally:
            main_module.rl_solver = original_solver

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "solved")
        self.assertEqual(payload["moves"], ["U'", "R'"])
        self.assertEqual(payload["details"]["strategy"], "policy-guided-beam-search")
        self.assertEqual(payload["details"]["model_version"], "in-memory-search-policy")
        self.assertEqual(payload["details"]["model_checkpoint"], "in-memory-search-policy")
        self.assertGreater(payload["details"]["expanded_states"], 0)
        self.assertEqual(payload["details"]["beam_width"], 3)
        self.assertEqual(payload["details"]["top_k"], 3)


if __name__ == "__main__":
    unittest.main()

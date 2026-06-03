"""Unit tests for the vectorized cube engine (rubic_rl.cube_ops).

These guard the move permutation tables against the canonical engine
(app.cube.Cube), since a single wrong permutation entry would silently corrupt
all DAVI training and search.
"""

from __future__ import annotations

import unittest

import numpy as np

from app.cube import Cube
from rubic_rl import cube_ops as C
from rubic_rl.policies.linear_policy import featurize_indices


class CubeOpsTests(unittest.TestCase):
    def test_move_permutations_match_engine_for_random_sequences(self) -> None:
        rng = np.random.default_rng(20260603)
        for _ in range(100):
            length = int(rng.integers(1, 25))
            actions = rng.integers(0, C.NUM_ACTIONS, size=length)

            cube = Cube.solved()
            for action in actions:
                cube = cube.apply_move(C.MOVES[int(action)], record_history=False)
            engine_state = C.cube_to_state(cube)

            state = C.SOLVED_STATE.copy()[None, :]
            for action in actions:
                state = C.apply_actions(state, np.array([action]))

            np.testing.assert_array_equal(state[0], engine_state)

    def test_solved_detection(self) -> None:
        self.assertTrue(bool(C.is_solved(C.solved_batch(8)).all()))
        scrambled, _ = C.scramble_batch(8, 6, np.random.default_rng(0))
        self.assertTrue(bool((~C.is_solved(scrambled)).all()))

    def test_expand_all_actions_matches_apply_actions(self) -> None:
        states, _ = C.scramble_batch(5, 8, np.random.default_rng(1))
        children = C.expand_all_actions(states)
        self.assertEqual(children.shape, (5, C.NUM_ACTIONS, C.NUM_STICKERS))
        for action in range(C.NUM_ACTIONS):
            np.testing.assert_array_equal(
                children[:, action, :], C.apply_actions(states, np.full(5, action))
            )

    def test_scramble_then_inverse_returns_to_solved(self) -> None:
        rng = np.random.default_rng(7)
        # Build one scramble via the engine, invert it, and confirm solved.
        from app.cube.model import inverse_move

        for _ in range(20):
            length = int(rng.integers(1, 15))
            moves = [C.MOVES[int(a)] for a in rng.integers(0, C.NUM_ACTIONS, size=length)]
            cube = Cube.solved().apply_sequence(moves, record_history=False)
            for move in reversed(moves):
                cube = cube.apply_move(inverse_move(move), record_history=False)
            self.assertTrue(cube.is_solved())

    def test_one_hot_matches_featurize_indices(self) -> None:
        states, _ = C.scramble_batch(12, 10, np.random.default_rng(2))
        actual = C.one_hot_features(states)
        expected = np.vstack(
            [featurize_indices(tuple(int(v) for v in row)) for row in states]
        ).astype(np.float32)
        self.assertEqual(actual.shape, (12, 324))
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()

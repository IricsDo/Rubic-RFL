import unittest

import gymnasium as gym
import numpy as np

from app.cube import Cube
from rubic_rl import ENV_ID, ACTION_MOVES, MOVE_TO_ACTION, RubiksCubeEnv


class RubiksCubeEnvTests(unittest.TestCase):
    def test_action_mapping_matches_project_plan(self):
        self.assertEqual(
            ACTION_MOVES,
            (
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
            ),
        )
        self.assertEqual(len(ACTION_MOVES), 18)
        self.assertEqual(len(set(ACTION_MOVES)), 18)
        self.assertEqual(MOVE_TO_ACTION["R"], 3)

    def test_reset_returns_one_hot_observation_and_info(self):
        env = RubiksCubeEnv(scramble_depth=4, max_steps=20)
        observation, info = env.reset(seed=123)

        self.assertEqual(observation.shape, (54, 6))
        self.assertEqual(observation.dtype, np.int8)
        np.testing.assert_array_equal(observation.sum(axis=1), np.ones(54))
        self.assertEqual(info["steps"], 0)
        self.assertEqual(len(info["scramble"]), 4)
        self.assertEqual(info["history"], info["scramble"])

    def test_reset_is_deterministic_for_same_seed(self):
        first = RubiksCubeEnv(scramble_depth=8)
        second = RubiksCubeEnv(scramble_depth=8)

        first_observation, first_info = first.reset(seed=20260531)
        second_observation, second_info = second.reset(seed=20260531)

        self.assertEqual(first_info["scramble"], second_info["scramble"])
        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(first.cube.to_string(), second.cube.to_string())

    def test_step_applies_action_and_reports_move(self):
        env = RubiksCubeEnv(scramble_depth=0, max_steps=5)
        env.reset(seed=1)

        observation, reward, terminated, truncated, info = env.step(MOVE_TO_ACTION["R"])

        self.assertEqual(observation.shape, (54, 6))
        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["move"], "R")
        self.assertEqual(
            env.cube.to_string(),
            Cube.solved().apply_move("R").to_string(),
        )

    def test_inverse_scramble_actions_solve_environment(self):
        env = RubiksCubeEnv(scramble_depth=7, max_steps=20)
        env.reset(seed=44)

        final_reward = None
        terminated = False
        for action in env.inverse_scramble_actions():
            _, final_reward, terminated, truncated, _ = env.step(action)
            self.assertFalse(truncated)

        self.assertTrue(terminated)
        self.assertEqual(final_reward, 1.0)
        self.assertTrue(env.cube.is_solved())

    def test_indices_observation_mode(self):
        env = RubiksCubeEnv(scramble_depth=0, observation_mode="indices")
        observation, info = env.reset(seed=7)

        self.assertEqual(observation.shape, (54,))
        self.assertEqual(observation.dtype, np.int64)
        self.assertTrue(info["is_solved"])
        self.assertEqual(set(observation.tolist()), set(range(6)))

    def test_truncates_at_max_steps_when_unsolved(self):
        env = RubiksCubeEnv(scramble_depth=0, max_steps=1)
        env.reset(seed=1)

        _, _, terminated, truncated, _ = env.step(MOVE_TO_ACTION["R"])

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_accepts_imported_stickers(self):
        cube = Cube.solved().apply_sequence(["R", "U"])
        env = RubiksCubeEnv(scramble_depth=0)

        _, info = env.reset(options={"stickers": cube.to_string()})

        self.assertFalse(info["is_solved"])
        self.assertEqual(info["scramble"], [])
        self.assertEqual(env.cube.to_string(), cube.to_string())

    def test_accepts_imported_sticker_sequence(self):
        cube = Cube.solved().apply_move("F")
        env = RubiksCubeEnv(scramble_depth=0)

        _, info = env.reset(options={"stickers": cube.stickers})

        self.assertFalse(info["is_solved"])
        self.assertEqual(env.cube.to_string(), cube.to_string())

    def test_rejects_invalid_action(self):
        env = RubiksCubeEnv(scramble_depth=0)

        with self.assertRaises(ValueError):
            env.action_to_move(-1)

        with self.assertRaises(ValueError):
            env.action_to_move(18)

    def test_registered_gymnasium_id_can_create_environment(self):
        env = gym.make(ENV_ID, scramble_depth=0)

        observation, info = env.reset(seed=5)

        self.assertEqual(observation.shape, (54, 6))
        self.assertTrue(info["is_solved"])
        env.close()


if __name__ == "__main__":
    unittest.main()

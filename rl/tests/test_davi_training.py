import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from app.cube import Cube
    from rubic_rl import cube_ops as C
    from rubic_rl.evaluation.davi_eval import _summarize_cases
    from rubic_rl.search import weighted_astar_solve
    from rubic_rl.training.davi import DAVIConfig, _loss, _sample_training_states


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DAVITrainingTests(unittest.TestCase):
    def test_hard_depth_fraction_samples_only_current_depth_when_full(self):
        config = DAVIConfig(
            batch_size=12,
            min_scramble_depth=1,
            max_scramble_depth=30,
            hard_depth_fraction=1.0,
        )

        _states, depths = _sample_training_states(
            config, depth=17, rng=np.random.default_rng(123)
        )

        self.assertEqual(depths.tolist(), [17] * 12)

    def test_smooth_l1_loss_option_is_available(self):
        config = DAVIConfig(loss_type="smooth_l1", huber_delta=1.0)
        predicted = torch.tensor([0.0, 2.0, 4.0])
        target = torch.tensor([0.0, 1.0, 7.0])

        actual = _loss(predicted, target, config)
        expected = torch.nn.functional.smooth_l1_loss(predicted, target, beta=1.0)

        torch.testing.assert_close(actual, expected)

    def test_rejects_invalid_hard_depth_fraction(self):
        with self.assertRaises(ValueError):
            DAVIConfig(hard_depth_fraction=1.5)

    def test_policy_weight_guides_weighted_astar_when_values_are_tied(self):
        state = C.cube_to_state(Cube.solved().apply_sequence(("R", "U")))

        def zero_value(states):
            return np.zeros(len(states), dtype=np.float32)

        def policy(states):
            probabilities = np.full((len(states), C.NUM_ACTIONS), 1e-4, dtype=np.float32)
            for index, row in enumerate(states):
                cube = C.state_to_cube(row)
                if cube.to_string() == Cube.solved().apply_sequence(("R", "U")).to_string():
                    probabilities[index, C.MOVES.index("U'")] = 1.0
                else:
                    probabilities[index, C.MOVES.index("R'")] = 1.0
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return probabilities

        result = weighted_astar_solve(
            state,
            zero_value,
            action_policy_fn=policy,
            weight=0.0,
            policy_weight=1.0,
            batch_expansion=1,
            max_nodes=20,
        )

        self.assertTrue(result["solved"])
        self.assertEqual(result["moves"], ("U'", "R'"))
        self.assertLessEqual(result["expanded"], 2)

    def test_davi_eval_summary_uses_completed_cases(self):
        cases = [
            {"depth": 15, "index": 0, "solved": True, "length": 10, "expanded": 100, "ms": 1000.0},
            {"depth": 15, "index": 1, "solved": False, "length": 0, "expanded": 200, "ms": 2000.0},
            {"depth": 16, "index": 0, "solved": True, "length": 12, "expanded": 300, "ms": 3000.0},
        ]

        summary = _summarize_cases(cases, depths=(15, 16))

        self.assertEqual(summary["overall"]["samples"], 3)
        self.assertAlmostEqual(summary["overall"]["solve_rate"], 2 / 3)
        self.assertEqual(summary["by_depth"]["15"]["samples"], 2)
        self.assertAlmostEqual(summary["by_depth"]["15"]["solve_rate"], 0.5)
        self.assertEqual(summary["by_depth"]["15"]["avg_len_solved"], 10.0)


if __name__ == "__main__":
    unittest.main()

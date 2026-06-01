import unittest
from pathlib import Path

import numpy as np

from app.cube import Cube, inverse_sequence
from rubic_rl import ACTION_MOVES, MOVE_TO_ACTION
from rubic_rl.encoding import stickers_to_indices
from rubic_rl.evaluation import PolicyEvaluationConfig
from rubic_rl.evaluation.compare_policies import (
    NamedPolicy,
    compare_loaded_policies,
    parse_model_specs,
)


class SingleMoveOraclePolicy:
    def __init__(self):
        self._actions_by_state = {}
        for move in ACTION_MOVES:
            cube = Cube.solved().apply_move(move)
            state = tuple(int(value) for value in stickers_to_indices(cube.stickers))
            inverse_move = inverse_sequence((move,))[0]
            self._actions_by_state[state] = MOVE_TO_ACTION[inverse_move]

    def predict(self, features: np.ndarray) -> np.ndarray:
        states = features.reshape((-1, 54, 6)).argmax(axis=2)
        return np.asarray(
            [
                self._actions_by_state.get(tuple(int(v) for v in row), 0)
                for row in states
            ],
            dtype=np.int64,
        )


class ConstantPolicy:
    def __init__(self, action: int):
        self._action = action

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self._action, dtype=np.int64)


class PolicyComparisonTests(unittest.TestCase):
    def test_parse_model_specs_accepts_labeled_paths(self):
        specs = parse_model_specs(
            [
                "linear=..\\checkpoints\\linear-policy-smoke.npz",
                "mlp=..\\checkpoints\\mlp-policy-smoke.npz",
            ]
        )

        self.assertEqual([spec.label for spec in specs], ["linear", "mlp"])
        self.assertEqual(specs[0].path, Path("..\\checkpoints\\linear-policy-smoke.npz"))
        self.assertEqual(specs[0].policy_type, "auto")

    def test_parse_model_specs_rejects_missing_or_duplicate_labels(self):
        with self.assertRaisesRegex(ValueError, "label=path"):
            parse_model_specs(["checkpoint.npz"])

        with self.assertRaisesRegex(ValueError, "model label must not be empty"):
            parse_model_specs(["=checkpoint.npz"])

        with self.assertRaisesRegex(ValueError, "duplicate model label"):
            parse_model_specs(["baseline=a.npz", "baseline=b.npz"])

    def test_compare_loaded_policies_ranks_by_solve_rate(self):
        config = PolicyEvaluationConfig(
            depths=(1,),
            samples_per_depth=24,
            max_steps=1,
            seed=20260531,
        )

        report = compare_loaded_policies(
            [
                NamedPolicy("constant", ConstantPolicy(MOVE_TO_ACTION["U"])),
                NamedPolicy("oracle", SingleMoveOraclePolicy()),
            ],
            config,
        )

        self.assertEqual(report["config"], config.to_json_dict())
        self.assertEqual(
            [entry["label"] for entry in report["ranking"]],
            ["oracle", "constant"],
        )
        self.assertEqual(report["ranking"][0]["rank"], 1)
        self.assertEqual(report["policies"][1]["summary"]["overall"]["solve_rate"], 1.0)
        self.assertLess(report["policies"][0]["summary"]["overall"]["solve_rate"], 1.0)
        self.assertNotIn("episodes", report["policies"][0])

    def test_compare_loaded_policies_can_include_episodes(self):
        config = PolicyEvaluationConfig(
            depths=(1,),
            samples_per_depth=2,
            max_steps=1,
            seed=17,
        )

        report = compare_loaded_policies(
            [NamedPolicy("oracle", SingleMoveOraclePolicy())],
            config,
            include_episodes=True,
        )

        episodes = report["policies"][0]["episodes"]
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["steps"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from app.cube import Cube, inverse_sequence
from rubic_rl import ACTION_MOVES, MOVE_TO_ACTION
from rubic_rl.datasets import DatasetRecord
from rubic_rl.encoding import stickers_to_indices
from rubic_rl.evaluation import (
    PolicyEvaluationConfig,
    evaluate_policy_on_env,
    evaluate_policy_on_scrambles,
    summarize_results,
)
from rubic_rl.policies import TrainingConfig, train_policy


def canonical_single_move_records(repeats: int = 1) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for repeat in range(repeats):
        for index, move in enumerate(ACTION_MOVES):
            cube = Cube.solved().apply_move(move)
            target_moves = inverse_sequence((move,))
            records.append(
                DatasetRecord(
                    record_id=f"eval-single-{repeat}-{index}",
                    stickers=cube.to_string(),
                    sticker_indices=tuple(
                        int(value) for value in stickers_to_indices(cube.stickers)
                    ),
                    scramble=(move,),
                    depth=1,
                    target_moves=target_moves,
                    target_action=MOVE_TO_ACTION[target_moves[0]],
                    is_solved=False,
                    seed=index,
                    sample_index=index,
                )
            )
    return records


def trained_single_move_policy():
    model, _ = train_policy(
        canonical_single_move_records(repeats=4),
        TrainingConfig(
            epochs=250,
            learning_rate=0.4,
            batch_size=18,
            validation_split=0.0,
            seed=31,
        ),
    )
    return model


class PolicyEvaluationTests(unittest.TestCase):
    def test_evaluates_policy_on_explicit_scrambles(self):
        model = trained_single_move_policy()

        results = evaluate_policy_on_scrambles(
            model,
            ([move] for move in ACTION_MOVES),
            max_steps=1,
        )

        self.assertEqual(len(results), 18)
        self.assertTrue(all(result.solved for result in results))
        self.assertEqual({result.steps for result in results}, {1})
        self.assertEqual(results[0].scramble, ("U",))
        self.assertEqual(results[0].moves, ("U'",))

    def test_summarizes_results_by_depth_and_overall(self):
        model = trained_single_move_policy()
        results = evaluate_policy_on_scrambles(
            model,
            [["R"], ["U"], ["R", "U"]],
            max_steps=1,
        )

        summary = summarize_results(results)

        self.assertEqual(summary["overall"]["attempts"], 3)
        self.assertEqual(summary["by_depth"]["1"]["attempts"], 2)
        self.assertEqual(summary["by_depth"]["1"]["solve_rate"], 1.0)
        self.assertEqual(summary["by_depth"]["2"]["attempts"], 1)
        self.assertEqual(summary["by_depth"]["2"]["solve_rate"], 0.0)
        self.assertIsNone(summary["by_depth"]["2"]["avg_steps_solved"])

    def test_evaluates_seeded_environment_rollouts(self):
        model = trained_single_move_policy()

        first = evaluate_policy_on_env(
            model,
            PolicyEvaluationConfig(
                depths=(1,),
                samples_per_depth=6,
                max_steps=1,
                seed=20260531,
            ),
        )
        second = evaluate_policy_on_env(
            model,
            PolicyEvaluationConfig(
                depths=(1,),
                samples_per_depth=6,
                max_steps=1,
                seed=20260531,
            ),
        )

        self.assertEqual(first, second)
        self.assertTrue(all(result.solved for result in first))
        np.testing.assert_array_equal(
            [result.sample_index for result in first],
            np.arange(6),
        )

    def test_invalid_evaluation_config_is_rejected(self):
        with self.assertRaises(ValueError):
            PolicyEvaluationConfig(depths=(), samples_per_depth=1)

        with self.assertRaises(ValueError):
            PolicyEvaluationConfig(depths=(-1,), samples_per_depth=1)

        with self.assertRaises(ValueError):
            PolicyEvaluationConfig(depths=(1,), samples_per_depth=0)

        with self.assertRaises(ValueError):
            PolicyEvaluationConfig(depths=(1,), samples_per_depth=1, max_steps=0)


if __name__ == "__main__":
    unittest.main()

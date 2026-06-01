from io import BytesIO
import unittest

import numpy as np

from app.cube import Cube, inverse_sequence
from rubic_rl import ACTION_MOVES, MOVE_TO_ACTION
from rubic_rl.datasets import DatasetRecord
from rubic_rl.encoding import stickers_to_indices
from rubic_rl.evaluation import evaluate_policy_on_scrambles
from rubic_rl.policies import (
    MLPPolicy,
    MLPTrainingConfig,
    load_policy,
    records_to_arrays,
    train_mlp_policy,
)


def canonical_single_move_records(repeats: int = 1) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for repeat in range(repeats):
        for index, move in enumerate(ACTION_MOVES):
            cube = Cube.solved().apply_move(move)
            target_moves = inverse_sequence((move,))
            records.append(
                DatasetRecord(
                    record_id=f"mlp-single-{repeat}-{index}",
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


class MLPPolicyTests(unittest.TestCase):
    def test_mlp_policy_learns_canonical_single_move_targets(self):
        records = canonical_single_move_records(repeats=4)

        model, result = train_mlp_policy(
            records,
            MLPTrainingConfig(
                hidden_units=48,
                epochs=250,
                learning_rate=0.2,
                batch_size=18,
                validation_split=0.0,
                seed=11,
            ),
        )
        features, labels = records_to_arrays(canonical_single_move_records())

        predictions = model.predict(features)

        self.assertGreater(result.train_accuracy, 0.95)
        np.testing.assert_array_equal(predictions, labels)

    def test_mlp_model_roundtrip_preserves_predictions(self):
        model, _ = train_mlp_policy(
            canonical_single_move_records(repeats=3),
            MLPTrainingConfig(
                hidden_units=32,
                epochs=180,
                learning_rate=0.2,
                batch_size=18,
                validation_split=0.0,
                seed=17,
            ),
        )
        features, _ = records_to_arrays(canonical_single_move_records())
        expected = model.predict(features)
        buffer = BytesIO()

        model.save(buffer)
        buffer.seek(0)
        loaded = MLPPolicy.load(buffer)

        np.testing.assert_array_equal(loaded.predict(features), expected)
        self.assertEqual(loaded.hidden_units, 32)
        self.assertEqual(loaded.action_count, 18)

    def test_auto_loader_detects_mlp_checkpoint(self):
        model = MLPPolicy.initialize(hidden_units=24, seed=29)
        buffer = BytesIO()

        model.save(buffer)
        buffer.seek(0)
        loaded = load_policy(buffer)

        self.assertIsInstance(loaded, MLPPolicy)
        self.assertEqual(loaded.hidden_units, 24)

    def test_mlp_policy_solves_explicit_one_move_scrambles(self):
        model, _ = train_mlp_policy(
            canonical_single_move_records(repeats=4),
            MLPTrainingConfig(
                hidden_units=48,
                epochs=250,
                learning_rate=0.2,
                batch_size=18,
                validation_split=0.0,
                seed=23,
            ),
        )

        results = evaluate_policy_on_scrambles(
            model,
            [(move,) for move in ACTION_MOVES],
            max_steps=1,
        )

        self.assertEqual(len(results), 18)
        self.assertTrue(all(result.solved for result in results))

    def test_invalid_mlp_training_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hidden_units must be positive"):
            MLPTrainingConfig(hidden_units=0)

        with self.assertRaisesRegex(ValueError, "dropout is not supported"):
            MLPTrainingConfig(dropout=0.1)


if __name__ == "__main__":
    unittest.main()

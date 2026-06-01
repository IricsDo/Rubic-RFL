from io import BytesIO
import unittest

import numpy as np

from app.cube import Cube, inverse_sequence
from rubic_rl import ACTION_MOVES, MOVE_TO_ACTION
from rubic_rl.datasets import DatasetRecord
from rubic_rl.encoding import stickers_to_indices
from rubic_rl.policies import (
    LinearPolicy,
    TrainingConfig,
    records_to_arrays,
    train_policy,
)


def canonical_single_move_records(repeats: int = 1) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for repeat in range(repeats):
        for index, move in enumerate(ACTION_MOVES):
            cube = Cube.solved().apply_move(move)
            target_moves = inverse_sequence((move,))
            records.append(
                DatasetRecord(
                    record_id=f"single-{repeat}-{index}",
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


class SupervisedPolicyTests(unittest.TestCase):
    def test_records_to_arrays_excludes_solved_records(self):
        records = canonical_single_move_records()
        solved = DatasetRecord(
            record_id="solved",
            stickers=Cube.solved().to_string(),
            sticker_indices=tuple(
                int(value) for value in stickers_to_indices(Cube.solved().stickers)
            ),
            scramble=(),
            depth=0,
            target_moves=(),
            target_action=None,
            is_solved=True,
            seed=None,
            sample_index=0,
        )

        features, labels = records_to_arrays([solved, *records])

        self.assertEqual(features.shape, (18, 324))
        self.assertEqual(labels.shape, (18,))
        np.testing.assert_array_equal(features.sum(axis=1), np.full(18, 54.0))
        self.assertEqual(set(labels.tolist()), set(range(18)))

    def test_policy_learns_canonical_single_move_targets(self):
        records = canonical_single_move_records(repeats=4)

        model, result = train_policy(
            records,
            TrainingConfig(
                epochs=250,
                learning_rate=0.4,
                batch_size=18,
                validation_split=0.0,
                seed=7,
            ),
        )
        features, labels = records_to_arrays(canonical_single_move_records())

        predictions = model.predict(features)

        self.assertGreater(result.train_accuracy, 0.95)
        np.testing.assert_array_equal(predictions, labels)

    def test_model_roundtrip_preserves_predictions(self):
        model, _ = train_policy(
            canonical_single_move_records(repeats=2),
            TrainingConfig(epochs=120, learning_rate=0.4, validation_split=0.0),
        )
        features, _ = records_to_arrays(canonical_single_move_records())
        expected = model.predict(features)
        buffer = BytesIO()

        model.save(buffer)
        buffer.seek(0)
        loaded = LinearPolicy.load(buffer)

        np.testing.assert_array_equal(loaded.predict(features), expected)
        self.assertEqual(loaded.action_count, 18)


if __name__ == "__main__":
    unittest.main()

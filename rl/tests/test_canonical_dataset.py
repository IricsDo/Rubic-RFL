import unittest
from collections import Counter

from app.cube import Cube
from rubic_rl import MOVE_TO_ACTION
from rubic_rl.datasets import CanonicalDatasetConfig, generate_canonical_dataset


class CanonicalDatasetTests(unittest.TestCase):
    def test_depth_one_contains_every_action_once(self):
        records = generate_canonical_dataset(CanonicalDatasetConfig(depths=(1,)))

        self.assertEqual(len(records), 18)
        self.assertEqual({record.depth for record in records}, {1})
        self.assertEqual(
            {record.scramble for record in records},
            {(move,) for move in MOVE_TO_ACTION},
        )
        self.assertEqual(len({record.stickers for record in records}), len(records))

    def test_depth_two_is_axis_pruned_and_deduplicated(self):
        records = generate_canonical_dataset(CanonicalDatasetConfig(depths=(1, 2)))
        counts = Counter(record.depth for record in records)

        self.assertEqual(counts[1], 18)
        self.assertEqual(counts[2], 216)
        self.assertEqual(len({record.stickers for record in records}), len(records))

    def test_targets_solve_all_canonical_states(self):
        records = generate_canonical_dataset(
            CanonicalDatasetConfig(depths=(0, 1, 2), include_solved=True)
        )

        solved_records = [record for record in records if record.depth == 0]
        self.assertEqual(len(solved_records), 1)
        self.assertIsNone(solved_records[0].target_action)

        for record in records:
            cube = Cube.from_string(record.stickers)
            self.assertTrue(cube.apply_sequence(record.target_moves).is_solved())
            if record.target_action is not None:
                self.assertEqual(
                    record.target_action,
                    MOVE_TO_ACTION[record.target_moves[0]],
                )

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            CanonicalDatasetConfig(depths=())

        with self.assertRaises(ValueError):
            CanonicalDatasetConfig(depths=(-1,))


if __name__ == "__main__":
    unittest.main()

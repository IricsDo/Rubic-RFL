import unittest
from io import StringIO

from app.cube import Cube
from rubic_rl import MOVE_TO_ACTION
from rubic_rl.datasets import (
    DatasetConfig,
    generate_dataset,
    read_jsonl,
    write_jsonl,
)


class DatasetGeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic_for_same_config(self):
        config = DatasetConfig(depths=(1, 2, 3), samples_per_depth=2, seed=20260531)

        first = generate_dataset(config)
        second = generate_dataset(config)

        self.assertEqual(first, second)

    def test_generated_targets_solve_source_state(self):
        records = generate_dataset(
            DatasetConfig(depths=(1, 2, 4), samples_per_depth=3, seed=44)
        )

        for record in records:
            cube = Cube.from_string(record.stickers)
            solved = cube.apply_sequence(record.target_moves)

            self.assertTrue(solved.is_solved())
            self.assertEqual(len(record.scramble), record.depth)
            self.assertEqual(len(record.target_moves), record.depth)
            self.assertEqual(record.target_action, MOVE_TO_ACTION[record.target_moves[0]])

    def test_depth_counts_and_solved_record(self):
        records = generate_dataset(
            DatasetConfig(
                depths=(1, 3),
                samples_per_depth=4,
                seed=7,
                include_solved=True,
            )
        )

        self.assertEqual(len(records), 9)
        self.assertEqual(sum(1 for record in records if record.depth == 1), 4)
        self.assertEqual(sum(1 for record in records if record.depth == 3), 4)

        solved = records[0]
        self.assertTrue(solved.is_solved)
        self.assertEqual(solved.depth, 0)
        self.assertEqual(solved.target_moves, ())
        self.assertIsNone(solved.target_action)

    def test_jsonl_roundtrip_preserves_records(self):
        records = generate_dataset(
            DatasetConfig(depths=(2,), samples_per_depth=2, seed=123)
        )

        output = StringIO()

        write_jsonl(records, output)
        output.seek(0)
        loaded = read_jsonl(output)

        self.assertEqual(loaded, records)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            DatasetConfig(depths=(), samples_per_depth=1)

        with self.assertRaises(ValueError):
            DatasetConfig(depths=(-1,), samples_per_depth=1)

        with self.assertRaises(ValueError):
            DatasetConfig(depths=(1,), samples_per_depth=0)


if __name__ == "__main__":
    unittest.main()

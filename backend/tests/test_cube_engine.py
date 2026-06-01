import unittest

from app.cube import ALL_MOVES, Cube, generate_scramble, inverse_sequence, validate_stickers
from app.cube.model import INDEX_TO_LOCATION


def indices_at(position):
    return [
        index
        for index, location in enumerate(INDEX_TO_LOCATION)
        if location.position == position
    ]


class CubeEngineTests(unittest.TestCase):
    def test_solved_cube_validates(self):
        validation = validate_stickers(Cube.solved().stickers)
        self.assertTrue(validation.valid, validation.errors)

    def test_move_then_inverse_returns_to_original(self):
        for move in ALL_MOVES:
            with self.subTest(move=move):
                cube = Cube.solved().apply_move(move).apply_sequence(inverse_sequence([move]))
                self.assertEqual(cube.to_string(), Cube.solved().to_string())

    def test_quarter_turn_four_times_returns_to_original(self):
        for move in ("U", "D", "L", "R", "F", "B"):
            with self.subTest(move=move):
                cube = Cube.solved().apply_sequence([move, move, move, move])
                self.assertEqual(cube.to_string(), Cube.solved().to_string())

    def test_scramble_then_inverse_returns_solved(self):
        moves = generate_scramble(depth=25, seed=20260531)
        cube = Cube.solved().apply_sequence(moves).apply_sequence(inverse_sequence(moves))
        self.assertTrue(cube.is_solved())
        self.assertTrue(validate_stickers(cube.stickers).valid)

    def test_invalid_sticker_count_is_rejected(self):
        stickers = list(Cube.solved().stickers)
        stickers[0] = "R"
        validation = validate_stickers(tuple(stickers))
        self.assertFalse(validation.valid)
        self.assertIn("Sticker U appears 8 times instead of 9", validation.errors)

    def test_single_edge_flip_is_rejected(self):
        stickers = list(Cube.solved().stickers)
        edge = indices_at((0, 1, 1))
        stickers[edge[0]], stickers[edge[1]] = stickers[edge[1]], stickers[edge[0]]
        validation = validate_stickers(tuple(stickers))
        self.assertFalse(validation.valid)
        self.assertIn("Edge orientation sum is impossible", validation.errors)

    def test_single_edge_swap_is_rejected_by_parity(self):
        stickers = list(Cube.solved().stickers)
        first = indices_at((0, 1, 1))
        second = indices_at((1, 1, 0))
        for left, right in zip(first, second):
            stickers[left], stickers[right] = stickers[right], stickers[left]
        validation = validate_stickers(tuple(stickers))
        self.assertFalse(validation.valid)
        self.assertIn("Corner and edge permutation parity do not match", validation.errors)


if __name__ == "__main__":
    unittest.main()


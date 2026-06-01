from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from .constants import ALL_MOVES, AXIS_INDEX, FACE_ORDER, FACE_TO_AXIS

Vector = tuple[int, int, int]
LocationKey = tuple[Vector, Vector]


@dataclass(frozen=True)
class StickerLocation:
    position: Vector
    normal: Vector


def _face_location(face: str, row: int, col: int) -> StickerLocation:
    if face == "U":
        return StickerLocation((col - 1, 1, row - 1), (0, 1, 0))
    if face == "D":
        return StickerLocation((col - 1, -1, 1 - row), (0, -1, 0))
    if face == "R":
        return StickerLocation((1, 1 - row, 1 - col), (1, 0, 0))
    if face == "L":
        return StickerLocation((-1, 1 - row, col - 1), (-1, 0, 0))
    if face == "F":
        return StickerLocation((col - 1, 1 - row, 1), (0, 0, 1))
    if face == "B":
        return StickerLocation((1 - col, 1 - row, -1), (0, 0, -1))
    raise ValueError(f"Unknown face {face!r}")


def _build_location_maps() -> tuple[tuple[StickerLocation, ...], dict[LocationKey, int]]:
    locations: list[StickerLocation] = []
    reverse: dict[LocationKey, int] = {}
    for face in FACE_ORDER:
        for row in range(3):
            for col in range(3):
                location = _face_location(face, row, col)
                reverse[(location.position, location.normal)] = len(locations)
                locations.append(location)
    return tuple(locations), reverse


INDEX_TO_LOCATION, LOCATION_TO_INDEX = _build_location_maps()


def solved_stickers() -> tuple[str, ...]:
    return tuple(face for face in FACE_ORDER for _ in range(9))


def _rotate_vector(vector: Vector, axis: str, quarter_turns: int) -> Vector:
    x, y, z = vector
    for _ in range(quarter_turns % 4):
        if axis == "x":
            x, y, z = x, -z, y
        elif axis == "y":
            x, y, z = z, y, -x
        elif axis == "z":
            x, y, z = -y, x, z
        else:
            raise ValueError(f"Unknown axis {axis!r}")
    return x, y, z


def _normalize_move(move: str) -> str:
    move = move.strip()
    if not move:
        raise ValueError("Move cannot be empty")
    if move not in ALL_MOVES:
        raise ValueError(f"Unsupported move {move!r}")
    return move


MOVE_PATTERN = re.compile(r"([URFDLB])(['2]?)")


def parse_moves(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if re.search(r"[\s,]+", text):
            raw_moves = [part for part in re.split(r"[\s,]+", text) if part]
            return tuple(_normalize_move(move) for move in raw_moves)

        moves: list[str] = []
        cursor = 0
        while cursor < len(text):
            match = MOVE_PATTERN.match(text, cursor)
            if not match:
                raise ValueError(f"Cannot parse move sequence near {text[cursor:]!r}")
            moves.append(_normalize_move(match.group(0)))
            cursor = match.end()
        return tuple(moves)

    return tuple(_normalize_move(move) for move in value)


def inverse_move(move: str) -> str:
    move = _normalize_move(move)
    if move.endswith("'"):
        return move[0]
    if move.endswith("2"):
        return move
    return f"{move}'"


def inverse_sequence(moves: Iterable[str]) -> tuple[str, ...]:
    return tuple(inverse_move(move) for move in reversed(parse_moves(moves)))


@dataclass(frozen=True)
class Cube:
    stickers: tuple[str, ...] = field(default_factory=solved_stickers)
    history: tuple[str, ...] = ()

    @classmethod
    def solved(cls) -> "Cube":
        return cls()

    @classmethod
    def from_string(cls, value: str, history: Iterable[str] = ()) -> "Cube":
        stickers = tuple(value.strip())
        if len(stickers) != 54:
            raise ValueError("Cube string must contain exactly 54 stickers")
        return cls(stickers=stickers, history=parse_moves(history))

    @classmethod
    def from_faces(
        cls, faces: dict[str, Iterable[str]], history: Iterable[str] = ()
    ) -> "Cube":
        stickers: list[str] = []
        for face in FACE_ORDER:
            values = tuple(faces.get(face, ()))
            if len(values) != 9:
                raise ValueError(f"Face {face} must contain exactly 9 stickers")
            stickers.extend(values)
        return cls(stickers=tuple(stickers), history=parse_moves(history))

    def to_string(self) -> str:
        return "".join(self.stickers)

    def to_faces(self) -> dict[str, list[str]]:
        return {
            face: list(self.stickers[index * 9 : index * 9 + 9])
            for index, face in enumerate(FACE_ORDER)
        }

    def is_solved(self) -> bool:
        return self.stickers == solved_stickers()

    def without_history(self) -> "Cube":
        return Cube(self.stickers, ())

    def apply_move(self, move: str, record_history: bool = True) -> "Cube":
        move = _normalize_move(move)
        face = move[0]
        suffix = move[1:]
        axis, layer_sign = FACE_TO_AXIS[face]
        turns = 2 if suffix == "2" else -1 if suffix == "'" else 1
        quarter_turns = -layer_sign * turns
        axis_index = AXIS_INDEX[axis]

        next_stickers = list(self.stickers)
        for source_index, location in enumerate(INDEX_TO_LOCATION):
            if location.position[axis_index] != layer_sign:
                continue

            next_position = _rotate_vector(location.position, axis, quarter_turns)
            next_normal = _rotate_vector(location.normal, axis, quarter_turns)
            target_index = LOCATION_TO_INDEX[(next_position, next_normal)]
            next_stickers[target_index] = self.stickers[source_index]

        next_history = self.history + (move,) if record_history else self.history
        return Cube(tuple(next_stickers), next_history)

    def apply_sequence(
        self, moves: str | Iterable[str], record_history: bool = True
    ) -> "Cube":
        cube = self
        for move in parse_moves(moves):
            cube = cube.apply_move(move, record_history=record_history)
        return cube

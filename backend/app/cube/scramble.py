from __future__ import annotations

import random

from .constants import FACE_TO_AXIS, VALID_SUFFIXES


def generate_scramble(depth: int = 20, seed: int | str | None = None) -> tuple[str, ...]:
    if depth < 0:
        raise ValueError("Scramble depth cannot be negative")

    rng = random.Random(seed)
    faces = tuple(FACE_TO_AXIS.keys())
    moves: list[str] = []
    previous_axis: str | None = None

    for _ in range(depth):
        candidates = [
            face for face in faces if FACE_TO_AXIS[face][0] != previous_axis
        ]
        face = rng.choice(candidates)
        suffix = rng.choice(VALID_SUFFIXES)
        moves.append(face + suffix)
        previous_axis = FACE_TO_AXIS[face][0]

    return tuple(moves)


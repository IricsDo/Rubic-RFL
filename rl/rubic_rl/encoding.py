from __future__ import annotations

import numpy as np

from app.cube import Cube, FACE_ORDER

STICKER_TO_INDEX = {face: index for index, face in enumerate(FACE_ORDER)}
INDEX_TO_STICKER = {index: face for face, index in STICKER_TO_INDEX.items()}


def stickers_to_indices(stickers: tuple[str, ...] | list[str] | str) -> np.ndarray:
    values = tuple(stickers)
    if len(values) != 54:
        raise ValueError("Cube observation must contain exactly 54 stickers")

    try:
        return np.asarray([STICKER_TO_INDEX[value] for value in values], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown sticker value: {error.args[0]!r}") from error


def stickers_to_one_hot(stickers: tuple[str, ...] | list[str] | str) -> np.ndarray:
    indices = stickers_to_indices(stickers)
    observation = np.zeros((54, len(FACE_ORDER)), dtype=np.int8)
    observation[np.arange(54), indices] = 1
    return observation


def encode_cube(cube: Cube, mode: str = "one_hot") -> np.ndarray:
    if mode == "one_hot":
        return stickers_to_one_hot(cube.stickers)
    if mode == "indices":
        return stickers_to_indices(cube.stickers)
    raise ValueError(f"Unsupported observation mode: {mode!r}")


def decode_sticker_indices(indices: np.ndarray) -> tuple[str, ...]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.shape != (54,):
        raise ValueError("Sticker index observation must have shape (54,)")
    try:
        return tuple(INDEX_TO_STICKER[int(value)] for value in values)
    except KeyError as error:
        raise ValueError(f"Unknown sticker index: {error.args[0]!r}") from error

"""Vectorized Rubik's Cube operations for value-iteration training and search.

The cube is represented as an array of 54 color indices in ``[0, 5]`` matching
``rubic_rl.encoding.stickers_to_indices`` (sticker order = ``FACE_ORDER``). A
batch of cubes is an ``(N, 54)`` int array.

Move permutations are extracted directly from the canonical engine
(``app.cube.Cube``) so this module never re-implements move logic: applying a
move to a cube whose stickers are the integers ``0..53`` makes the engine return,
at each destination index, the source index that moved there — i.e. exactly the
gather permutation ``new_state = old_state[perm]``.
"""

from __future__ import annotations

import numpy as np

from app.cube import Cube, FACE_ORDER
from app.cube.model import inverse_move
from rubic_rl import ACTION_MOVES

# Canonical 18-move action set, in the same order the policy/value network uses.
MOVES: tuple[str, ...] = tuple(ACTION_MOVES)
NUM_ACTIONS: int = len(MOVES)
NUM_STICKERS: int = 54
NUM_COLORS: int = len(FACE_ORDER)


def _build_move_permutations() -> np.ndarray:
    """Return an ``(NUM_ACTIONS, 54)`` gather table: ``new = state[perm[a]]``."""
    identity = tuple(range(NUM_STICKERS))
    table = np.empty((NUM_ACTIONS, NUM_STICKERS), dtype=np.int64)
    for action, move in enumerate(MOVES):
        permuted = Cube(stickers=identity).apply_move(move, record_history=False)
        table[action] = np.asarray(permuted.stickers, dtype=np.int64)
    return table


# perm[a, i] = source index whose value lands at position i after move a.
MOVE_PERMUTATIONS: np.ndarray = _build_move_permutations()

# Inverse action index for each action (used to avoid trivial move cancellation).
INVERSE_ACTION: np.ndarray = np.asarray(
    [MOVES.index(inverse_move(move)) for move in MOVES], dtype=np.int64
)

# Solved state as color indices: face f repeated 9 times for f in FACE_ORDER.
SOLVED_STATE: np.ndarray = np.repeat(np.arange(NUM_COLORS, dtype=np.int64), 9)


def solved_batch(n: int) -> np.ndarray:
    """Return ``(n, 54)`` array of solved cubes."""
    return np.tile(SOLVED_STATE, (n, 1))


def apply_actions(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Apply one (possibly different) action per cube. ``states`` is ``(N, 54)``."""
    gather = MOVE_PERMUTATIONS[np.asarray(actions, dtype=np.int64)]  # (N, 54)
    return np.take_along_axis(states, gather, axis=1)


def expand_all_actions(states: np.ndarray) -> np.ndarray:
    """Return ``(N, NUM_ACTIONS, 54)`` children for every action of every cube."""
    # states[:, perm] broadcast over the action axis.
    return states[:, MOVE_PERMUTATIONS]  # (N, NUM_ACTIONS, 54)


def is_solved(states: np.ndarray) -> np.ndarray:
    """Boolean mask of solved cubes for an ``(N, 54)`` batch."""
    return np.all(states == SOLVED_STATE, axis=-1)


def one_hot_features(states: np.ndarray) -> np.ndarray:
    """Encode ``(N, 54)`` color indices as ``(N, 324)`` float32 one-hot features.

    Matches ``rubic_rl.policies.linear_policy.featurize_indices`` exactly.
    """
    states = np.asarray(states, dtype=np.int64)
    n = states.shape[0]
    features = np.zeros((n, NUM_STICKERS, NUM_COLORS), dtype=np.float32)
    rows = np.arange(n)[:, None]
    cols = np.arange(NUM_STICKERS)[None, :]
    features[rows, cols, states] = 1.0
    return features.reshape(n, NUM_STICKERS * NUM_COLORS)


def scramble_batch(
    n: int,
    max_depth: int,
    rng: np.random.Generator,
    *,
    min_depth: int = 1,
    avoid_inverse: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` scrambled cubes with per-cube depth in ``[min_depth, max_depth]``.

    Returns ``(states, depths)`` where ``states`` is ``(n, 54)`` and ``depths`` is
    ``(n,)``. Vectorized: all cubes advance together; a cube freezes once it has
    received its target number of moves.
    """
    if min_depth < 0 or max_depth < min_depth:
        raise ValueError("require 0 <= min_depth <= max_depth")

    states = solved_batch(n)
    depths = rng.integers(min_depth, max_depth + 1, size=n)
    prev_actions = np.full(n, -1, dtype=np.int64)

    for step in range(max_depth):
        active = step < depths
        if not np.any(active):
            break
        actions = rng.integers(0, NUM_ACTIONS, size=n)
        if avoid_inverse:
            # Resample actions that would immediately undo the previous move.
            clashes = (prev_actions >= 0) & (actions == INVERSE_ACTION[prev_actions])
            while np.any(clashes):
                actions[clashes] = rng.integers(0, NUM_ACTIONS, size=int(clashes.sum()))
                clashes = (prev_actions >= 0) & (
                    actions == INVERSE_ACTION[prev_actions]
                )
        moved = apply_actions(states, actions)
        states = np.where(active[:, None], moved, states)
        prev_actions = np.where(active, actions, prev_actions)

    return states, depths


def state_to_cube(state: np.ndarray) -> Cube:
    """Convert a single ``(54,)`` color-index state back to an ``app.cube.Cube``."""
    stickers = tuple(FACE_ORDER[int(index)] for index in np.asarray(state).reshape(-1))
    return Cube(stickers=stickers)


def cube_to_state(cube: Cube) -> np.ndarray:
    """Convert an ``app.cube.Cube`` to a ``(54,)`` color-index state."""
    lookup = {face: index for index, face in enumerate(FACE_ORDER)}
    return np.asarray([lookup[value] for value in cube.stickers], dtype=np.int64)

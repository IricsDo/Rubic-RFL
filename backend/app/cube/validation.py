from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .constants import FACE_ORDER
from .model import LOCATION_TO_INDEX, solved_stickers

Vector = tuple[int, int, int]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    details: dict[str, object]


CORNER_SLOTS: tuple[tuple[Vector, tuple[Vector, Vector, Vector]], ...] = (
    ((1, 1, 1), ((0, 1, 0), (1, 0, 0), (0, 0, 1))),  # URF
    ((-1, 1, 1), ((0, 1, 0), (0, 0, 1), (-1, 0, 0))),  # UFL
    ((-1, 1, -1), ((0, 1, 0), (-1, 0, 0), (0, 0, -1))),  # ULB
    ((1, 1, -1), ((0, 1, 0), (0, 0, -1), (1, 0, 0))),  # UBR
    ((1, -1, 1), ((0, -1, 0), (0, 0, 1), (1, 0, 0))),  # DFR
    ((-1, -1, 1), ((0, -1, 0), (-1, 0, 0), (0, 0, 1))),  # DLF
    ((-1, -1, -1), ((0, -1, 0), (0, 0, -1), (-1, 0, 0))),  # DBL
    ((1, -1, -1), ((0, -1, 0), (1, 0, 0), (0, 0, -1))),  # DRB
)
EDGE_SLOTS: tuple[tuple[Vector, tuple[Vector, Vector]], ...] = (
    ((1, 1, 0), ((0, 1, 0), (1, 0, 0))),  # UR
    ((0, 1, 1), ((0, 1, 0), (0, 0, 1))),  # UF
    ((-1, 1, 0), ((0, 1, 0), (-1, 0, 0))),  # UL
    ((0, 1, -1), ((0, 1, 0), (0, 0, -1))),  # UB
    ((1, -1, 0), ((0, -1, 0), (1, 0, 0))),  # DR
    ((0, -1, 1), ((0, -1, 0), (0, 0, 1))),  # DF
    ((-1, -1, 0), ((0, -1, 0), (-1, 0, 0))),  # DL
    ((0, -1, -1), ((0, -1, 0), (0, 0, -1))),  # DB
    ((1, 0, 1), ((0, 0, 1), (1, 0, 0))),  # FR
    ((-1, 0, 1), ((0, 0, 1), (-1, 0, 0))),  # FL
    ((-1, 0, -1), ((0, 0, -1), (-1, 0, 0))),  # BL
    ((1, 0, -1), ((0, 0, -1), (1, 0, 0))),  # BR
)
SOLVED = solved_stickers()


def _colors_at_slot(
    stickers: tuple[str, ...], slot: tuple[Vector, tuple[Vector, ...]]
) -> tuple[str, ...]:
    position, normals = slot
    return tuple(
        stickers[LOCATION_TO_INDEX[(position, normal)]] for normal in normals
    )


CORNER_ID_BY_COLORS = {
    frozenset(_colors_at_slot(SOLVED, slot)): index
    for index, slot in enumerate(CORNER_SLOTS)
}
EDGE_ID_BY_COLORS = {
    frozenset(_colors_at_slot(SOLVED, slot)): index
    for index, slot in enumerate(EDGE_SLOTS)
}
EDGE_COLORS_BY_ID = tuple(_colors_at_slot(SOLVED, slot) for slot in EDGE_SLOTS)


def _permutation_parity(permutation: list[int]) -> int:
    inversions = 0
    for index, value in enumerate(permutation):
        for later in permutation[index + 1 :]:
            if value > later:
                inversions += 1
    return inversions % 2


def _corner_orientation(piece: tuple[tuple[Vector, str], ...]) -> int:
    for index, (_, color) in enumerate(piece):
        if color in {"U", "D"}:
            return index
    raise ValueError("Corner is missing a U/D sticker")


def _edge_orientation(colors: tuple[str, ...], piece_id: int) -> int:
    return 0 if colors == EDGE_COLORS_BY_ID[piece_id] else 1


def _extract_permutation(
    stickers: tuple[str, ...],
    slots: tuple[tuple[Vector, tuple[Vector, ...]], ...],
    piece_ids: dict[frozenset[str], int],
) -> tuple[list[int], list[str]]:
    permutation: list[int] = []
    errors: list[str] = []
    seen: set[int] = set()

    for position, normals in slots:
        colors = frozenset(_colors_at_slot(stickers, (position, normals)))
        piece_id = piece_ids.get(colors)
        if piece_id is None:
            errors.append(f"Unknown piece colors at position {position}: {sorted(colors)}")
            continue
        if piece_id in seen:
            errors.append(f"Duplicate piece detected for colors {sorted(colors)}")
            continue
        seen.add(piece_id)
        permutation.append(piece_id)

    if len(permutation) != len(slots):
        errors.append("Piece permutation is incomplete")

    return permutation, errors


def validate_stickers(stickers: tuple[str, ...] | list[str] | str) -> ValidationResult:
    values = tuple(stickers)
    errors: list[str] = []
    details: dict[str, object] = {}

    if len(values) != 54:
        return ValidationResult(False, ("Cube must contain exactly 54 stickers",), {})

    counts = Counter(values)
    details["sticker_counts"] = dict(counts)
    unknown = sorted(set(values) - set(FACE_ORDER))
    if unknown:
        errors.append(f"Unknown stickers: {', '.join(unknown)}")

    for face in FACE_ORDER:
        if counts[face] != 9:
            errors.append(f"Sticker {face} appears {counts[face]} times instead of 9")

    if errors:
        return ValidationResult(False, tuple(errors), details)

    corner_permutation, corner_errors = _extract_permutation(
        values, CORNER_SLOTS, CORNER_ID_BY_COLORS
    )
    edge_permutation, edge_errors = _extract_permutation(
        values, EDGE_SLOTS, EDGE_ID_BY_COLORS
    )
    errors.extend(corner_errors)
    errors.extend(edge_errors)

    if not errors:
        corner_orientation_sum = sum(
            _corner_orientation(tuple(zip(slot[1], _colors_at_slot(values, slot))))
            for slot in CORNER_SLOTS
        )
        edge_orientation_sum = sum(
            _edge_orientation(_colors_at_slot(values, slot), piece_id)
            for slot, piece_id in zip(EDGE_SLOTS, edge_permutation)
        )
        corner_parity = _permutation_parity(corner_permutation)
        edge_parity = _permutation_parity(edge_permutation)

        details.update(
            {
                "corner_orientation_sum_mod_3": corner_orientation_sum % 3,
                "edge_orientation_sum_mod_2": edge_orientation_sum % 2,
                "corner_parity": corner_parity,
                "edge_parity": edge_parity,
            }
        )

        if corner_orientation_sum % 3 != 0:
            errors.append("Corner orientation sum is impossible")
        if edge_orientation_sum % 2 != 0:
            errors.append("Edge orientation sum is impossible")
        if corner_parity != edge_parity:
            errors.append("Corner and edge permutation parity do not match")

    return ValidationResult(not errors, tuple(errors), details)

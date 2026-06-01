from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.cube import Cube
from app.cube.constants import FACE_TO_AXIS
from rubic_rl.envs.rubiks_cube import ACTION_MOVES

from .generator import build_dataset_record, write_jsonl, DatasetRecord


@dataclass(frozen=True)
class CanonicalDatasetConfig:
    depths: tuple[int, ...] = (1, 2)
    include_solved: bool = False
    max_records_per_depth: int | None = None

    def __post_init__(self) -> None:
        depths = tuple(int(depth) for depth in self.depths)
        if not depths:
            raise ValueError("depths must contain at least one value")
        if any(depth < 0 for depth in depths):
            raise ValueError("depths cannot contain negative values")
        if (
            self.max_records_per_depth is not None
            and self.max_records_per_depth <= 0
        ):
            raise ValueError("max_records_per_depth must be positive when set")

        object.__setattr__(self, "depths", depths)


def generate_canonical_dataset(
    config: CanonicalDatasetConfig | None = None,
) -> list[DatasetRecord]:
    config = config or CanonicalDatasetConfig()
    records: list[DatasetRecord] = []
    seen_stickers: set[str] = set()

    if config.include_solved or 0 in config.depths:
        _append_record(
            records=records,
            seen_stickers=seen_stickers,
            record_id="canonical-solved-000000",
            scramble=(),
            sample_index=0,
        )

    for depth in config.depths:
        if depth == 0:
            continue

        records_for_depth = 0
        for sequence in _iter_axis_pruned_sequences(depth):
            added = _append_record(
                records=records,
                seen_stickers=seen_stickers,
                record_id=f"canonical-d{depth:03d}-{records_for_depth:06d}",
                scramble=sequence,
                sample_index=records_for_depth,
            )
            if not added:
                continue
            records_for_depth += 1
            if (
                config.max_records_per_depth is not None
                and records_for_depth >= config.max_records_per_depth
            ):
                break

    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic canonical Rubik's Cube records as JSONL."
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument(
        "--depths",
        nargs="+",
        default=("1", "2"),
        help="Canonical depths, for example: --depths 1 2 or --depths 1,2.",
    )
    parser.add_argument(
        "--include-solved",
        action="store_true",
        help="Add one solved-state record before scrambled records.",
    )
    parser.add_argument(
        "--max-records-per-depth",
        type=int,
        default=None,
        help="Optional cap for smoke datasets.",
    )
    args = parser.parse_args(argv)

    records = generate_canonical_dataset(
        CanonicalDatasetConfig(
            depths=_parse_depth_args(args.depths),
            include_solved=args.include_solved,
            max_records_per_depth=args.max_records_per_depth,
        )
    )
    output_path = write_jsonl(records, args.out)
    print(f"Wrote {len(records)} records to {output_path}")
    return 0


def _append_record(
    *,
    records: list[DatasetRecord],
    seen_stickers: set[str],
    record_id: str,
    scramble: tuple[str, ...],
    sample_index: int,
) -> bool:
    cube = Cube.solved().apply_sequence(scramble)
    stickers = cube.to_string()
    if stickers in seen_stickers:
        return False

    seen_stickers.add(stickers)
    records.append(
        build_dataset_record(
            record_id=record_id,
            cube=cube,
            scramble=scramble,
            seed="canonical",
            sample_index=sample_index,
        )
    )
    return True


def _iter_axis_pruned_sequences(depth: int) -> Iterator[tuple[str, ...]]:
    if depth == 0:
        yield ()
        return

    yield from _extend_sequence(prefix=(), previous_axis=None, remaining=depth)


def _extend_sequence(
    *,
    prefix: tuple[str, ...],
    previous_axis: str | None,
    remaining: int,
) -> Iterator[tuple[str, ...]]:
    if remaining == 0:
        yield prefix
        return

    for move in ACTION_MOVES:
        axis = FACE_TO_AXIS[move[0]][0]
        if previous_axis is not None and axis == previous_axis:
            continue
        yield from _extend_sequence(
            prefix=prefix + (move,),
            previous_axis=axis,
            remaining=remaining - 1,
        )


def _parse_depth_args(values: Sequence[str]) -> tuple[int, ...]:
    raw_depths: list[str] = []
    for value in values:
        raw_depths.extend(part for part in value.split(",") if part)
    return tuple(int(depth) for depth in raw_depths)


if __name__ == "__main__":
    raise SystemExit(main())

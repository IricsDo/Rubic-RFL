from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, TextIO

from app.cube import Cube, generate_scramble, inverse_sequence, parse_moves
from rubic_rl.encoding import stickers_to_indices
from rubic_rl.envs.rubiks_cube import MOVE_TO_ACTION


JsonValue = str | int | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class DatasetConfig:
    depths: tuple[int, ...] = (1, 2, 3)
    samples_per_depth: int = 100
    seed: int | str | None = None
    include_solved: bool = False

    def __post_init__(self) -> None:
        depths = tuple(int(depth) for depth in self.depths)
        if not depths:
            raise ValueError("depths must contain at least one value")
        if any(depth < 0 for depth in depths):
            raise ValueError("depths cannot contain negative values")
        if self.samples_per_depth <= 0:
            raise ValueError("samples_per_depth must be positive")

        object.__setattr__(self, "depths", depths)


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    stickers: str
    sticker_indices: tuple[int, ...]
    scramble: tuple[str, ...]
    depth: int
    target_moves: tuple[str, ...]
    target_action: int | None
    is_solved: bool
    seed: int | str | None
    sample_index: int

    def to_json_dict(self) -> dict[str, JsonValue]:
        return {
            "record_id": self.record_id,
            "stickers": self.stickers,
            "sticker_indices": list(self.sticker_indices),
            "scramble": list(self.scramble),
            "depth": self.depth,
            "target_moves": list(self.target_moves),
            "target_action": self.target_action,
            "is_solved": self.is_solved,
            "seed": self.seed,
            "sample_index": self.sample_index,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "DatasetRecord":
        return cls(
            record_id=str(data["record_id"]),
            stickers=str(data["stickers"]),
            sticker_indices=tuple(int(value) for value in data["sticker_indices"]),
            scramble=parse_moves(data["scramble"]),
            depth=int(data["depth"]),
            target_moves=parse_moves(data["target_moves"]),
            target_action=(
                None if data["target_action"] is None else int(data["target_action"])
            ),
            is_solved=bool(data["is_solved"]),
            seed=data["seed"],
            sample_index=int(data["sample_index"]),
        )


def generate_dataset(config: DatasetConfig) -> list[DatasetRecord]:
    rng = random.Random(config.seed)
    records: list[DatasetRecord] = []

    if config.include_solved:
        records.append(
            build_dataset_record(
                record_id="solved-000000",
                cube=Cube.solved(),
                scramble=(),
                seed=config.seed,
                sample_index=0,
            )
        )

    for depth_index, depth in enumerate(config.depths):
        for sample_index in range(config.samples_per_depth):
            sample_seed = rng.randrange(0, 2**32)
            scramble = generate_scramble(depth=depth, seed=sample_seed)
            cube = Cube.solved().apply_sequence(scramble)
            records.append(
                build_dataset_record(
                    record_id=f"d{depth:03d}-{depth_index:02d}-{sample_index:06d}",
                    cube=cube,
                    scramble=scramble,
                    seed=sample_seed,
                    sample_index=sample_index,
                )
            )

    return records


def deduplicate_records(records: Iterable[DatasetRecord]) -> list[DatasetRecord]:
    deduplicated: list[DatasetRecord] = []
    seen_stickers: set[str] = set()
    for record in records:
        if record.stickers in seen_stickers:
            continue
        seen_stickers.add(record.stickers)
        deduplicated.append(record)
    return deduplicated


def write_jsonl(
    records: Iterable[DatasetRecord], output_path: str | Path | TextIO
) -> Path | None:
    if hasattr(output_path, "write"):
        _write_jsonl_lines(records, output_path)
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        _write_jsonl_lines(records, output)
    return path


def read_jsonl(input_path: str | Path | TextIO) -> list[DatasetRecord]:
    if hasattr(input_path, "read"):
        return _read_jsonl_lines(input_path)

    path = Path(input_path)
    with path.open("r", encoding="utf-8") as input_file:
        return _read_jsonl_lines(input_file)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate supervised Rubik's Cube training data as JSONL."
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument(
        "--depths",
        nargs="+",
        default=("1", "2", "3"),
        help="Scramble depths, for example: --depths 1 2 3 or --depths 1,2,3.",
    )
    parser.add_argument(
        "--samples-per-depth",
        type=int,
        default=100,
        help="Number of generated records for each depth.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Dataset seed. Numeric values are parsed as integers.",
    )
    parser.add_argument(
        "--include-solved",
        action="store_true",
        help="Add one solved-state record before scrambled records.",
    )
    args = parser.parse_args(argv)

    config = DatasetConfig(
        depths=_parse_depth_args(args.depths),
        samples_per_depth=args.samples_per_depth,
        seed=_parse_seed(args.seed),
        include_solved=args.include_solved,
    )
    records = generate_dataset(config)
    output_path = write_jsonl(records, args.out)
    print(f"Wrote {len(records)} records to {output_path}")
    return 0


def build_dataset_record(
    *,
    record_id: str,
    cube: Cube,
    scramble: tuple[str, ...],
    seed: int | str | None,
    sample_index: int,
) -> DatasetRecord:
    target_moves = inverse_sequence(scramble)
    return DatasetRecord(
        record_id=record_id,
        stickers=cube.to_string(),
        sticker_indices=tuple(int(value) for value in stickers_to_indices(cube.stickers)),
        scramble=scramble,
        depth=len(scramble),
        target_moves=target_moves,
        target_action=MOVE_TO_ACTION[target_moves[0]] if target_moves else None,
        is_solved=cube.is_solved(),
        seed=seed,
        sample_index=sample_index,
    )


_build_record = build_dataset_record


def _write_jsonl_lines(records: Iterable[DatasetRecord], output: TextIO) -> None:
    for record in records:
        output.write(json.dumps(record.to_json_dict(), separators=(",", ":")))
        output.write("\n")


def _read_jsonl_lines(input_file: TextIO) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for line_number, line in enumerate(input_file, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}") from error
        records.append(DatasetRecord.from_json_dict(data))
    return records


def _parse_depth_args(values: Sequence[str]) -> tuple[int, ...]:
    raw_depths: list[str] = []
    for value in values:
        raw_depths.extend(part for part in value.split(",") if part)
    return tuple(int(depth) for depth in raw_depths)


def _parse_seed(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())

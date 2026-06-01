from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .canonical import CanonicalDatasetConfig, generate_canonical_dataset
    from .generator import (
        build_dataset_record,
        DatasetConfig,
        DatasetRecord,
        deduplicate_records,
        generate_dataset,
        read_jsonl,
        write_jsonl,
    )

__all__ = [
    "build_dataset_record",
    "CanonicalDatasetConfig",
    "DatasetConfig",
    "DatasetRecord",
    "deduplicate_records",
    "generate_dataset",
    "generate_canonical_dataset",
    "read_jsonl",
    "write_jsonl",
]


def __getattr__(name: str):
    if name in {"CanonicalDatasetConfig", "generate_canonical_dataset"}:
        from . import canonical

        return getattr(canonical, name)
    if name in __all__:
        from . import generator

        return getattr(generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

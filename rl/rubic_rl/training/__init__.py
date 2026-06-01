from __future__ import annotations

from typing import TYPE_CHECKING

_BASELINE_EXPERIMENT_EXPORTS = {
    "BaselineExperimentConfig",
    "run_baseline_experiment",
}

if TYPE_CHECKING:
    from .baseline_experiment import BaselineExperimentConfig, run_baseline_experiment

__all__ = [
    "BaselineExperimentConfig",
    "run_baseline_experiment",
]


def __getattr__(name: str):
    if name in _BASELINE_EXPERIMENT_EXPORTS:
        from . import baseline_experiment

        return getattr(baseline_experiment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

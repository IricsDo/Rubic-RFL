from __future__ import annotations

from typing import TYPE_CHECKING

_BASELINE_EXPERIMENT_EXPORTS = {
    "BaselineExperimentConfig",
    "run_baseline_experiment",
}
_ADI_EXPORTS = {
    "ADITrainingConfig",
    "run_adi_training",
}

if TYPE_CHECKING:
    from .adi import ADITrainingConfig, run_adi_training
    from .baseline_experiment import BaselineExperimentConfig, run_baseline_experiment

__all__ = [
    "ADITrainingConfig",
    "BaselineExperimentConfig",
    "run_adi_training",
    "run_baseline_experiment",
]


def __getattr__(name: str):
    if name in _BASELINE_EXPERIMENT_EXPORTS:
        from . import baseline_experiment

        return getattr(baseline_experiment, name)
    if name in _ADI_EXPORTS:
        from . import adi

        return getattr(adi, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

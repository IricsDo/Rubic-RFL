from __future__ import annotations

from typing import TYPE_CHECKING

_TORCH_EXPORTS = {
    "CheckpointMetadata",
    "ResidualBlock",
    "RubiksPolicyValueNet",
    "TorchModelConfig",
    "load_policy_value_checkpoint",
    "save_policy_value_checkpoint",
}

if TYPE_CHECKING:
    from .policy_value import (
        CheckpointMetadata,
        ResidualBlock,
        RubiksPolicyValueNet,
        TorchModelConfig,
        load_policy_value_checkpoint,
        save_policy_value_checkpoint,
    )

__all__ = sorted(_TORCH_EXPORTS)


def __getattr__(name: str):
    if name in _TORCH_EXPORTS:
        from . import policy_value

        return getattr(policy_value, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

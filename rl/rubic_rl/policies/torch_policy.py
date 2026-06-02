from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from rubic_rl.models.policy_value import (
    CheckpointMetadata,
    RubiksPolicyValueNet,
    load_policy_value_checkpoint,
)
from rubic_rl.policies.linear_policy import _as_feature_matrix

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - exercised by install path
    raise ModuleNotFoundError(
        "PyTorch is required to load Torch policy/value checkpoints. "
        'Install the optional dependency with: python -m pip install -e ".[torch]"'
    ) from error


@dataclass(frozen=True)
class TorchPolicyValuePolicy:
    """Policy adapter for PyTorch policy/value checkpoints."""

    model: RubiksPolicyValueNet
    metadata: CheckpointMetadata

    @classmethod
    def load(
        cls,
        input_path: str | Path | BinaryIO,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "TorchPolicyValuePolicy":
        model, metadata = load_policy_value_checkpoint(
            input_path,
            map_location=map_location,
        )
        return cls(model=model, metadata=metadata)

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def predict_value(self, features: np.ndarray) -> np.ndarray:
        matrix = _as_feature_matrix(features).astype(np.float32)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            _, values = self.model(
                torch.as_tensor(matrix, dtype=torch.float32, device=device)
            )
        return values.detach().cpu().numpy() * self.metadata.value_scale

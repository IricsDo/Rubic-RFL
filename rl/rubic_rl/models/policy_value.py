from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rubic_rl import ACTION_MOVES
from rubic_rl.policies.linear_policy import FEATURE_SIZE, _as_feature_matrix

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - exercised by install path
    raise ModuleNotFoundError(
        "PyTorch is required for rubic_rl.models.policy_value. "
        'Install the optional dependency with: python -m pip install -e ".[torch]"'
    ) from error


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class TorchModelConfig:
    input_size: int = FEATURE_SIZE
    hidden_dim: int = 256
    residual_blocks: int = 2
    action_count: int = len(ACTION_MOVES)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.residual_blocks < 0:
            raise ValueError("residual_blocks cannot be negative")
        if self.action_count <= 0:
            raise ValueError("action_count must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_json_dict(self) -> dict[str, JsonValue]:
        return {
            "input_size": self.input_size,
            "hidden_dim": self.hidden_dim,
            "residual_blocks": self.residual_blocks,
            "action_count": self.action_count,
            "dropout": self.dropout,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "TorchModelConfig":
        return cls(
            input_size=int(data.get("input_size", FEATURE_SIZE)),
            hidden_dim=int(data.get("hidden_dim", 256)),
            residual_blocks=int(data.get("residual_blocks", 2)),
            action_count=int(data.get("action_count", len(ACTION_MOVES))),
            dropout=float(data.get("dropout", 0.0)),
        )


@dataclass(frozen=True)
class CheckpointMetadata:
    model_version: str = "torch-policy-value-v0.1"
    value_scale: float = 1.0
    action_moves: tuple[str, ...] = ACTION_MOVES
    training: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if self.value_scale <= 0:
            raise ValueError("value_scale must be positive")
        object.__setattr__(self, "action_moves", tuple(self.action_moves))

    def to_json_dict(self) -> dict[str, JsonValue]:
        return {
            "model_type": "torch_policy_value",
            "model_version": self.model_version,
            "value_scale": self.value_scale,
            "action_moves": list(self.action_moves),
            "training": {} if self.training is None else self.training,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        return cls(
            model_version=str(data.get("model_version", "torch-policy-value-v0.1")),
            value_scale=float(data.get("value_scale", 1.0)),
            action_moves=tuple(str(move) for move in data.get("action_moves", ACTION_MOVES)),
            training=dict(data.get("training", {})),
        )


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class RubiksPolicyValueNet(nn.Module):
    """Residual policy/value network for one-hot Rubik sticker features."""

    def __init__(self, config: TorchModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or TorchModelConfig()
        self.input_layer = nn.Sequential(
            nn.Linear(self.config.input_size, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        self.residual_layers = nn.ModuleList(
            ResidualBlock(self.config.hidden_dim, self.config.dropout)
            for _ in range(self.config.residual_blocks)
        )
        self.policy_head = nn.Linear(self.config.hidden_dim, self.config.action_count)
        self.value_head = nn.Linear(self.config.hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[1] != self.config.input_size:
            raise ValueError(
                f"Expected feature tensor with shape (batch, {self.config.input_size})"
            )

        hidden = self.input_layer(features.float())
        for layer in self.residual_layers:
            hidden = layer(hidden)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = _as_feature_matrix(features).astype(np.float32)
        device = next(self.parameters()).device
        with torch.no_grad():
            logits, _ = self(torch.as_tensor(matrix, dtype=torch.float32, device=device))
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1).astype(np.int64)


def save_policy_value_checkpoint(
    output_path: str | Path,
    *,
    model: RubiksPolicyValueNet,
    metadata: CheckpointMetadata,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "format_version": 1,
            "model_config": model.config.to_json_dict(),
            "metadata": metadata.to_json_dict(),
            "state_dict": state_dict,
        },
        path,
    )
    return path


def load_policy_value_checkpoint(
    input_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RubiksPolicyValueNet, CheckpointMetadata]:
    payload = torch.load(input_path, map_location=map_location)
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("Unsupported Torch policy/value checkpoint format")

    config = TorchModelConfig.from_json_dict(dict(payload["model_config"]))
    metadata = CheckpointMetadata.from_json_dict(dict(payload["metadata"]))
    model = RubiksPolicyValueNet(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata

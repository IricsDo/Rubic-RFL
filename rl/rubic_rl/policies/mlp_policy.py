from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from rubic_rl import ACTION_MOVES
from rubic_rl.datasets import DatasetRecord
from rubic_rl.policies.linear_policy import (
    FEATURE_SIZE,
    TrainingResult,
    _as_feature_matrix,
    _softmax,
    _split_train_validation,
    records_to_arrays,
)


@dataclass(frozen=True)
class MLPTrainingConfig:
    hidden_units: int = 64
    epochs: int = 300
    learning_rate: float = 0.05
    batch_size: int = 128
    l2: float = 1e-4
    validation_split: float = 0.2
    seed: int | None = 0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.l2 < 0:
            raise ValueError("l2 cannot be negative")
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be in [0, 1)")
        if self.dropout != 0.0:
            raise ValueError("dropout is not supported by this NumPy baseline")


@dataclass(frozen=True)
class MLPPolicy:
    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray

    @classmethod
    def initialize(
        cls,
        *,
        feature_count: int = FEATURE_SIZE,
        hidden_units: int = 64,
        action_count: int = len(ACTION_MOVES),
        seed: int | None = 0,
    ) -> "MLPPolicy":
        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        rng = np.random.default_rng(seed)
        input_scale = np.sqrt(2.0 / feature_count)
        output_scale = np.sqrt(2.0 / hidden_units)
        return cls(
            input_weights=rng.normal(
                0.0,
                input_scale,
                size=(feature_count, hidden_units),
            ).astype(np.float64),
            hidden_bias=np.zeros(hidden_units, dtype=np.float64),
            output_weights=rng.normal(
                0.0,
                output_scale,
                size=(hidden_units, action_count),
            ).astype(np.float64),
            output_bias=np.zeros(action_count, dtype=np.float64),
        )

    @property
    def feature_count(self) -> int:
        return int(self.input_weights.shape[0])

    @property
    def hidden_units(self) -> int:
        return int(self.input_weights.shape[1])

    @property
    def action_count(self) -> int:
        return int(self.output_weights.shape[1])

    def hidden_activations(self, features: np.ndarray) -> np.ndarray:
        matrix = _as_feature_matrix(features)
        if matrix.shape[1] != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, got {matrix.shape[1]}"
            )
        return np.maximum(0.0, matrix @ self.input_weights + self.hidden_bias)

    def logits(self, features: np.ndarray) -> np.ndarray:
        hidden = self.hidden_activations(features)
        return hidden @ self.output_weights + self.output_bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(features), axis=1).astype(np.int64)

    def save(self, output: str | Path | BinaryIO) -> None:
        np.savez_compressed(
            output,
            policy_type=np.asarray("mlp"),
            input_weights=self.input_weights,
            hidden_bias=self.hidden_bias,
            output_weights=self.output_weights,
            output_bias=self.output_bias,
            action_moves=np.asarray(ACTION_MOVES),
        )

    @classmethod
    def load(cls, input_path: str | Path | BinaryIO) -> "MLPPolicy":
        with np.load(input_path, allow_pickle=False) as data:
            return cls(
                input_weights=np.asarray(data["input_weights"], dtype=np.float64),
                hidden_bias=np.asarray(data["hidden_bias"], dtype=np.float64),
                output_weights=np.asarray(data["output_weights"], dtype=np.float64),
                output_bias=np.asarray(data["output_bias"], dtype=np.float64),
            )


def train_mlp_policy(
    records: list[DatasetRecord],
    config: MLPTrainingConfig | None = None,
) -> tuple[MLPPolicy, TrainingResult]:
    config = config or MLPTrainingConfig()
    features, labels = records_to_arrays(records)
    train_x, train_y, validation_x, validation_y = _split_train_validation(
        features,
        labels,
        validation_split=config.validation_split,
        seed=config.seed,
    )
    model = MLPPolicy.initialize(
        hidden_units=config.hidden_units,
        seed=config.seed,
    )
    rng = np.random.default_rng(config.seed)

    for _ in range(config.epochs):
        order = rng.permutation(len(train_y))
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]

            hidden_pre = batch_x @ model.input_weights + model.hidden_bias
            hidden = np.maximum(0.0, hidden_pre)
            logits = hidden @ model.output_weights + model.output_bias
            probabilities = _softmax(logits)
            gradient_logits = probabilities.copy()
            gradient_logits[np.arange(len(batch_y)), batch_y] -= 1.0
            gradient_logits /= len(batch_y)

            output_weight_gradient = (
                hidden.T @ gradient_logits + config.l2 * model.output_weights
            )
            output_bias_gradient = gradient_logits.sum(axis=0)
            hidden_gradient = gradient_logits @ model.output_weights.T
            hidden_gradient[hidden_pre <= 0.0] = 0.0
            input_weight_gradient = (
                batch_x.T @ hidden_gradient + config.l2 * model.input_weights
            )
            hidden_bias_gradient = hidden_gradient.sum(axis=0)

            model = MLPPolicy(
                input_weights=(
                    model.input_weights
                    - config.learning_rate * input_weight_gradient
                ),
                hidden_bias=(
                    model.hidden_bias - config.learning_rate * hidden_bias_gradient
                ),
                output_weights=(
                    model.output_weights
                    - config.learning_rate * output_weight_gradient
                ),
                output_bias=(
                    model.output_bias - config.learning_rate * output_bias_gradient
                ),
            )

    train_metrics = evaluate_mlp_policy(model, train_x, train_y, l2=config.l2)
    validation_metrics: dict[str, float] | None = None
    if validation_x is not None and validation_y is not None:
        validation_metrics = evaluate_mlp_policy(
            model,
            validation_x,
            validation_y,
            l2=config.l2,
        )

    return (
        model,
        TrainingResult(
            epochs=config.epochs,
            samples=len(train_y),
            validation_samples=0 if validation_y is None else len(validation_y),
            train_loss=train_metrics["loss"],
            train_accuracy=train_metrics["accuracy"],
            validation_loss=(
                None if validation_metrics is None else validation_metrics["loss"]
            ),
            validation_accuracy=(
                None if validation_metrics is None else validation_metrics["accuracy"]
            ),
        ),
    )


def evaluate_mlp_policy(
    model: MLPPolicy,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 0.0,
) -> dict[str, float]:
    matrix = _as_feature_matrix(features)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(matrix) != len(targets):
        raise ValueError("features and labels must contain the same number of rows")
    if len(targets) == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    probabilities = model.predict_proba(matrix)
    clipped = np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)
    penalty = 0.5 * l2 * (
        np.sum(model.input_weights**2) + np.sum(model.output_weights**2)
    )
    loss = float(-np.log(clipped).mean() + penalty)
    accuracy = float((np.argmax(probabilities, axis=1) == targets).mean())
    return {"loss": loss, "accuracy": accuracy}

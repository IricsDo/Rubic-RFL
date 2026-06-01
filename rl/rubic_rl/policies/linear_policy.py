from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from rubic_rl import ACTION_MOVES
from rubic_rl.datasets import DatasetRecord


FEATURE_SIZE = 54 * 6


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    learning_rate: float = 0.2
    batch_size: int = 128
    l2: float = 1e-4
    validation_split: float = 0.2
    seed: int | None = 0

    def __post_init__(self) -> None:
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


@dataclass(frozen=True)
class TrainingResult:
    epochs: int
    samples: int
    validation_samples: int
    train_loss: float
    train_accuracy: float
    validation_loss: float | None
    validation_accuracy: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "epochs": self.epochs,
            "samples": self.samples,
            "validation_samples": self.validation_samples,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
        }


@dataclass(frozen=True)
class LinearPolicy:
    weights: np.ndarray
    bias: np.ndarray

    @classmethod
    def initialize(
        cls,
        *,
        feature_count: int = FEATURE_SIZE,
        action_count: int = len(ACTION_MOVES),
        seed: int | None = 0,
    ) -> "LinearPolicy":
        rng = np.random.default_rng(seed)
        weights = rng.normal(0.0, 0.01, size=(feature_count, action_count))
        bias = np.zeros(action_count, dtype=np.float64)
        return cls(weights=weights.astype(np.float64), bias=bias)

    @property
    def feature_count(self) -> int:
        return int(self.weights.shape[0])

    @property
    def action_count(self) -> int:
        return int(self.weights.shape[1])

    def logits(self, features: np.ndarray) -> np.ndarray:
        matrix = _as_feature_matrix(features)
        if matrix.shape[1] != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, got {matrix.shape[1]}"
            )
        return matrix @ self.weights + self.bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(features), axis=1).astype(np.int64)

    def save(self, output: str | Path | BinaryIO) -> None:
        np.savez_compressed(
            output,
            policy_type=np.asarray("linear"),
            weights=self.weights,
            bias=self.bias,
            action_moves=np.asarray(ACTION_MOVES),
        )

    @classmethod
    def load(cls, input_path: str | Path | BinaryIO) -> "LinearPolicy":
        with np.load(input_path, allow_pickle=False) as data:
            return cls(
                weights=np.asarray(data["weights"], dtype=np.float64),
                bias=np.asarray(data["bias"], dtype=np.float64),
            )


def records_to_arrays(records: list[DatasetRecord]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for record in records:
        if record.target_action is None:
            continue
        features.append(featurize_indices(record.sticker_indices))
        labels.append(record.target_action)

    if not features:
        raise ValueError("No trainable records found; target_action is missing")

    return (
        np.vstack(features).astype(np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def featurize_indices(sticker_indices: tuple[int, ...] | list[int]) -> np.ndarray:
    values = np.asarray(sticker_indices, dtype=np.int64).reshape(-1)
    if values.shape != (54,):
        raise ValueError("sticker_indices must contain exactly 54 values")
    if np.any(values < 0) or np.any(values > 5):
        raise ValueError("sticker_indices values must be in [0, 5]")

    features = np.zeros((54, 6), dtype=np.float64)
    features[np.arange(54), values] = 1.0
    return features.reshape(FEATURE_SIZE)


def train_policy(
    records: list[DatasetRecord],
    config: TrainingConfig | None = None,
) -> tuple[LinearPolicy, TrainingResult]:
    config = config or TrainingConfig()
    features, labels = records_to_arrays(records)
    train_x, train_y, validation_x, validation_y = _split_train_validation(
        features,
        labels,
        validation_split=config.validation_split,
        seed=config.seed,
    )
    model = LinearPolicy.initialize(seed=config.seed)
    rng = np.random.default_rng(config.seed)

    for _ in range(config.epochs):
        order = rng.permutation(len(train_y))
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]
            probabilities = model.predict_proba(batch_x)
            gradient_logits = probabilities
            gradient_logits[np.arange(len(batch_y)), batch_y] -= 1.0
            gradient_logits /= len(batch_y)

            weight_gradient = batch_x.T @ gradient_logits + config.l2 * model.weights
            bias_gradient = gradient_logits.sum(axis=0)
            model = LinearPolicy(
                weights=model.weights - config.learning_rate * weight_gradient,
                bias=model.bias - config.learning_rate * bias_gradient,
            )

    train_metrics = evaluate_policy(model, train_x, train_y, l2=config.l2)
    validation_metrics: dict[str, float] | None = None
    if validation_x is not None and validation_y is not None:
        validation_metrics = evaluate_policy(
            model,
            validation_x,
            validation_y,
            l2=config.l2,
        )

    result = TrainingResult(
        epochs=config.epochs,
        samples=len(train_y),
        validation_samples=0 if validation_y is None else len(validation_y),
        train_loss=train_metrics["loss"],
        train_accuracy=train_metrics["accuracy"],
        validation_loss=None if validation_metrics is None else validation_metrics["loss"],
        validation_accuracy=(
            None if validation_metrics is None else validation_metrics["accuracy"]
        ),
    )
    return model, result


def evaluate_policy(
    model: LinearPolicy,
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
    loss = float(-np.log(clipped).mean() + 0.5 * l2 * np.sum(model.weights**2))
    accuracy = float((np.argmax(probabilities, axis=1) == targets).mean())
    return {"loss": loss, "accuracy": accuracy}


def _split_train_validation(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    validation_split: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if validation_split == 0.0 or len(labels) < 2:
        return features, labels, None, None

    validation_count = int(round(len(labels) * validation_split))
    validation_count = max(1, min(validation_count, len(labels) - 1))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    return (
        features[train_indices],
        labels[train_indices],
        features[validation_indices],
        labels[validation_indices],
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("features must be a 1D or 2D array")
    return matrix

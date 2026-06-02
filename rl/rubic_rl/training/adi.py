from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.cube import Cube
from rubic_rl.datasets import (
    CanonicalDatasetConfig,
    DatasetConfig,
    DatasetRecord,
    build_dataset_record,
    deduplicate_records,
    generate_canonical_dataset,
    generate_dataset,
    write_jsonl,
)
from rubic_rl.models.policy_value import (
    CheckpointMetadata,
    RubiksPolicyValueNet,
    TorchModelConfig,
    save_policy_value_checkpoint,
)
from rubic_rl.policies.linear_policy import featurize_indices

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError as error:  # pragma: no cover - exercised by install path
    raise ModuleNotFoundError(
        "PyTorch is required for ADI training. "
        'Install the optional dependency with: python -m pip install -e ".[torch]"'
    ) from error


DEFAULT_DATASET_OUT = Path("../datasets/adi-policy-value.jsonl")
DEFAULT_CHECKPOINT_OUT = Path("../checkpoints/torch-policy-value-adi.pt")
DEFAULT_REPORT_OUT = Path("../reports/adi-training.json")


@dataclass(frozen=True)
class ADITrainingConfig:
    dataset_out: Path | None = DEFAULT_DATASET_OUT
    checkpoint_out: Path = DEFAULT_CHECKPOINT_OUT
    report_out: Path | None = DEFAULT_REPORT_OUT
    depths: tuple[int, ...] = (1, 2, 3)
    samples_per_depth: int = 100
    canonical_depths: tuple[int, ...] = (1, 2)
    include_solved: bool = True
    iterations: int = 1
    epochs_per_iteration: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 0.25
    validation_split: float = 0.2
    hidden_dim: int = 256
    residual_blocks: int = 2
    dropout: float = 0.0
    seed: int | None = 20260531
    model_version: str = "torch-policy-value-v0.1"
    device: str = "cpu"
    train_on_cumulative_records: bool = True

    def __post_init__(self) -> None:
        depths = tuple(int(depth) for depth in self.depths)
        if not depths:
            raise ValueError("depths must contain at least one value")
        if any(depth < 0 for depth in depths):
            raise ValueError("depths cannot contain negative values")
        canonical_depths = tuple(int(depth) for depth in self.canonical_depths)
        if any(depth < 0 for depth in canonical_depths):
            raise ValueError("canonical_depths cannot contain negative values")
        if self.samples_per_depth <= 0:
            raise ValueError("samples_per_depth must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.epochs_per_iteration <= 0:
            raise ValueError("epochs_per_iteration must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.value_loss_weight < 0:
            raise ValueError("value_loss_weight cannot be negative")
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be in [0, 1)")

        object.__setattr__(self, "depths", depths)
        object.__setattr__(self, "canonical_depths", canonical_depths)
        object.__setattr__(
            self,
            "dataset_out",
            None if self.dataset_out is None else Path(self.dataset_out),
        )
        object.__setattr__(self, "checkpoint_out", Path(self.checkpoint_out))
        object.__setattr__(
            self,
            "report_out",
            None if self.report_out is None else Path(self.report_out),
        )

    @property
    def value_scale(self) -> float:
        return float(max(1, *self.depths, *self.canonical_depths))

    @property
    def model_config(self) -> TorchModelConfig:
        return TorchModelConfig(
            hidden_dim=self.hidden_dim,
            residual_blocks=self.residual_blocks,
            dropout=self.dropout,
        )

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset_out"] = None if self.dataset_out is None else str(self.dataset_out)
        data["checkpoint_out"] = str(self.checkpoint_out)
        data["report_out"] = None if self.report_out is None else str(self.report_out)
        data["depths"] = list(self.depths)
        data["canonical_depths"] = list(self.canonical_depths)
        return data


def run_adi_training(config: ADITrainingConfig) -> dict[str, Any]:
    _seed_everything(config.seed)
    device = torch.device(config.device)
    model = RubiksPolicyValueNet(config.model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    all_records: list[DatasetRecord] = []
    iterations: list[dict[str, Any]] = []

    for iteration in range(config.iterations):
        records = _generate_iteration_records(config, iteration)
        all_records = deduplicate_records([*all_records, *records])
        training_records = all_records if config.train_on_cumulative_records else records
        tensors = _records_to_tensors(
            training_records,
            value_scale=config.value_scale,
            device=device,
        )
        split = _split_indices(
            len(training_records),
            validation_split=config.validation_split,
            rng=rng,
        )
        metrics = _fit_iteration(
            model=model,
            optimizer=optimizer,
            tensors=tensors,
            train_indices=split["train"],
            validation_indices=split["validation"],
            config=config,
            rng=rng,
        )
        iterations.append(
            {
                "iteration": iteration + 1,
                "records": len(training_records),
                "generated_records": len(records),
                "cumulative_records": len(all_records),
                "training_records": len(training_records),
                "train_on_cumulative_records": config.train_on_cumulative_records,
                "policy_records": int(tensors["policy_mask"].sum().item()),
                **metrics,
            }
        )

    if config.dataset_out is not None:
        write_jsonl(all_records, config.dataset_out)

    metadata = CheckpointMetadata(
        model_version=config.model_version,
        value_scale=config.value_scale,
        training=config.to_json_dict(),
    )
    checkpoint_path = save_policy_value_checkpoint(
        config.checkpoint_out,
        model=model,
        metadata=metadata,
    )
    report = _build_report(
        config=config,
        records=all_records,
        checkpoint_path=checkpoint_path,
        iterations=iterations,
        metadata=metadata,
    )
    if config.report_out is not None:
        config.report_out.parent.mkdir(parents=True, exist_ok=True)
        config.report_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PyTorch residual policy/value baseline on reverse-scrambled "
            "Rubik states."
        )
    )
    parser.add_argument("--dataset-out", type=Path, default=DEFAULT_DATASET_OUT)
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--depths", nargs="+", default=("1", "2", "3"))
    parser.add_argument("--samples-per-depth", type=int, default=100)
    parser.add_argument("--canonical-depths", nargs="*", default=("1", "2"))
    parser.add_argument("--include-solved", action="store_true", default=True)
    parser.add_argument("--exclude-solved", action="store_false", dest="include_solved")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs-per-iteration", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", default="20260531")
    parser.add_argument("--model-version", default="torch-policy-value-v0.1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--train-on-latest-records",
        action="store_false",
        dest="train_on_cumulative_records",
        help=(
            "Train each ADI iteration only on that iteration's generated records. "
            "The default trains on all cumulative deduplicated records."
        ),
    )
    args = parser.parse_args(argv)

    report = run_adi_training(
        ADITrainingConfig(
            dataset_out=args.dataset_out,
            checkpoint_out=args.checkpoint_out,
            report_out=args.report_out,
            depths=_parse_depth_args(args.depths),
            samples_per_depth=args.samples_per_depth,
            canonical_depths=_parse_optional_depth_args(args.canonical_depths),
            include_solved=args.include_solved,
            iterations=args.iterations,
            epochs_per_iteration=args.epochs_per_iteration,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            value_loss_weight=args.value_loss_weight,
            validation_split=args.validation_split,
            hidden_dim=args.hidden_dim,
            residual_blocks=args.residual_blocks,
            dropout=args.dropout,
            seed=_parse_seed(args.seed),
            model_version=args.model_version,
            device=args.device,
            train_on_cumulative_records=args.train_on_cumulative_records,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _generate_iteration_records(
    config: ADITrainingConfig,
    iteration: int,
) -> list[DatasetRecord]:
    solved_records = (
        [
            build_dataset_record(
                record_id=f"adi-iter{iteration + 1:03d}-solved",
                cube=Cube.solved(),
                scramble=(),
                seed=config.seed,
                sample_index=0,
            )
        ]
        if config.include_solved
        else []
    )
    canonical_records = (
        generate_canonical_dataset(
            CanonicalDatasetConfig(depths=config.canonical_depths)
        )
        if config.canonical_depths
        else []
    )
    random_seed = None if config.seed is None else config.seed + iteration
    random_records = generate_dataset(
        DatasetConfig(
            depths=config.depths,
            samples_per_depth=config.samples_per_depth,
            seed=random_seed,
            include_solved=False,
        )
    )
    return deduplicate_records([*solved_records, *canonical_records, *random_records])


def _records_to_tensors(
    records: Sequence[DatasetRecord],
    *,
    value_scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    features = np.vstack(
        [featurize_indices(record.sticker_indices) for record in records]
    ).astype(np.float32)
    policy_labels = np.asarray(
        [-1 if record.target_action is None else record.target_action for record in records],
        dtype=np.int64,
    )
    value_targets = np.asarray(
        [len(record.target_moves) / value_scale for record in records],
        dtype=np.float32,
    )
    return {
        "features": torch.as_tensor(features, dtype=torch.float32, device=device),
        "policy_labels": torch.as_tensor(policy_labels, dtype=torch.long, device=device),
        "policy_mask": torch.as_tensor(policy_labels >= 0, dtype=torch.bool, device=device),
        "value_targets": torch.as_tensor(value_targets, dtype=torch.float32, device=device),
    }


def _fit_iteration(
    *,
    model: RubiksPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: ADITrainingConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    train_tensor_indices = torch.as_tensor(
        train_indices,
        dtype=torch.long,
        device=tensors["features"].device,
    )
    for _ in range(config.epochs_per_iteration):
        model.train()
        order = rng.permutation(len(train_indices))
        for start in range(0, len(order), config.batch_size):
            batch = train_tensor_indices[
                torch.as_tensor(
                    order[start : start + config.batch_size],
                    dtype=torch.long,
                    device=tensors["features"].device,
                )
            ]
            losses = _losses_for_indices(
                model=model,
                tensors=tensors,
                indices=batch,
                value_loss_weight=config.value_loss_weight,
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

    train_metrics = _evaluate_indices(
        model=model,
        tensors=tensors,
        indices=train_indices,
        value_scale=config.value_scale,
    )
    validation_metrics = (
        None
        if len(validation_indices) == 0
        else _evaluate_indices(
            model=model,
            tensors=tensors,
            indices=validation_indices,
            value_scale=config.value_scale,
        )
    )
    return {
        "epochs": config.epochs_per_iteration,
        "train": train_metrics,
        "validation": validation_metrics,
    }


def _losses_for_indices(
    *,
    model: RubiksPolicyValueNet,
    tensors: dict[str, torch.Tensor],
    indices: torch.Tensor,
    value_loss_weight: float,
) -> dict[str, torch.Tensor]:
    policy_logits, value_predictions = model(tensors["features"][indices])
    labels = tensors["policy_labels"][indices]
    mask = tensors["policy_mask"][indices]
    if bool(mask.any()):
        policy_loss = functional.cross_entropy(policy_logits[mask], labels[mask])
    else:
        policy_loss = policy_logits.sum() * 0.0
    value_loss = functional.mse_loss(
        value_predictions,
        tensors["value_targets"][indices],
    )
    return {
        "loss": policy_loss + value_loss_weight * value_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }


def _evaluate_indices(
    *,
    model: RubiksPolicyValueNet,
    tensors: dict[str, torch.Tensor],
    indices: np.ndarray,
    value_scale: float,
) -> dict[str, float | int | None]:
    model.eval()
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=tensors["features"].device)
    with torch.no_grad():
        losses = _losses_for_indices(
            model=model,
            tensors=tensors,
            indices=index_tensor,
            value_loss_weight=1.0,
        )
        policy_logits, value_predictions = model(tensors["features"][index_tensor])
        labels = tensors["policy_labels"][index_tensor]
        mask = tensors["policy_mask"][index_tensor]
        if bool(mask.any()):
            predictions = torch.argmax(policy_logits[mask], dim=1)
            accuracy = float((predictions == labels[mask]).float().mean().item())
            policy_samples = int(mask.sum().item())
        else:
            accuracy = None
            policy_samples = 0
        value_targets = tensors["value_targets"][index_tensor]
        value_mae_moves = float(
            (torch.abs(value_predictions - value_targets).mean() * value_scale).item()
        )
    return {
        "samples": int(len(indices)),
        "policy_samples": policy_samples,
        "loss": float(losses["loss"].item()),
        "policy_loss": float(losses["policy_loss"].item()),
        "policy_accuracy": accuracy,
        "value_loss": float(losses["value_loss"].item()),
        "value_mae_moves": value_mae_moves,
    }


def _split_indices(
    record_count: int,
    *,
    validation_split: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if record_count <= 0:
        raise ValueError("Cannot train on an empty record set")
    if validation_split == 0.0 or record_count < 2:
        return {
            "train": np.arange(record_count, dtype=np.int64),
            "validation": np.asarray([], dtype=np.int64),
        }
    validation_count = int(round(record_count * validation_split))
    validation_count = max(1, min(validation_count, record_count - 1))
    order = rng.permutation(record_count).astype(np.int64)
    return {
        "train": order[validation_count:],
        "validation": order[:validation_count],
    }


def _build_report(
    *,
    config: ADITrainingConfig,
    records: Sequence[DatasetRecord],
    checkpoint_path: Path,
    iterations: list[dict[str, Any]],
    metadata: CheckpointMetadata,
) -> dict[str, Any]:
    return {
        "experiment": "adi-policy-value",
        "dataset": {
            "path": None if config.dataset_out is None else str(config.dataset_out),
            "records": len(records),
            "trainable_policy_records": sum(
                1 for record in records if record.target_action is not None
            ),
            "depths": list(config.depths),
            "samples_per_depth": config.samples_per_depth,
            "canonical_depths": list(config.canonical_depths),
            "include_solved": config.include_solved,
            "value_scale": config.value_scale,
            "seed": config.seed,
        },
        "model": {
            "checkpoint_out": str(checkpoint_path),
            "config": config.model_config.to_json_dict(),
            "metadata": metadata.to_json_dict(),
        },
        "training": {
            "config": config.to_json_dict(),
            "iterations": iterations,
            "final": iterations[-1],
        },
    }


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_depth_args(values: Sequence[str]) -> tuple[int, ...]:
    raw_depths: list[str] = []
    for value in values:
        raw_depths.extend(part for part in value.split(",") if part)
    return tuple(int(depth) for depth in raw_depths)


def _parse_optional_depth_args(values: Sequence[str]) -> tuple[int, ...]:
    if not values:
        return ()
    if len(values) == 1 and values[0].lower() in {"none", "off", "false"}:
        return ()
    return _parse_depth_args(values)


def _parse_seed(value: str | None) -> int | None:
    if value is None or value.lower() == "none":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())

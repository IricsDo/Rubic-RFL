from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from app.cube import Cube
from rubic_rl.datasets import (
    CanonicalDatasetConfig,
    DatasetConfig,
    build_dataset_record,
    deduplicate_records,
    generate_canonical_dataset,
    generate_dataset,
    write_jsonl,
)
from rubic_rl.evaluation import PolicyEvaluationConfig
from rubic_rl.evaluation.compare_policies import NamedPolicy, compare_loaded_policies
from rubic_rl.policies import (
    MLPTrainingConfig,
    TrainingConfig,
    train_mlp_policy,
    train_policy,
)


DEFAULT_DATASET_OUT = Path("../datasets/baseline-depth-1-2-3.jsonl")
DEFAULT_LINEAR_MODEL_OUT = Path("../checkpoints/linear-baseline-depth-1-2-3.npz")
DEFAULT_MLP_MODEL_OUT = Path("../checkpoints/mlp-baseline-depth-1-2-3.npz")
DEFAULT_REPORT_OUT = Path("../reports/baseline-experiment.json")


@dataclass(frozen=True)
class BaselineExperimentConfig:
    dataset_out: Path = DEFAULT_DATASET_OUT
    linear_model_out: Path = DEFAULT_LINEAR_MODEL_OUT
    mlp_model_out: Path = DEFAULT_MLP_MODEL_OUT
    report_out: Path | None = DEFAULT_REPORT_OUT
    depths: tuple[int, ...] = (1, 2, 3)
    samples_per_depth: int = 100
    canonical_depths: tuple[int, ...] = (1, 2)
    include_solved: bool = False
    seed: int | None = 20260531
    evaluation_samples_per_depth: int = 100
    evaluation_max_steps: int = 30
    linear_config: TrainingConfig = field(default_factory=TrainingConfig)
    mlp_config: MLPTrainingConfig = field(default_factory=MLPTrainingConfig)

    def __post_init__(self) -> None:
        depths = tuple(int(depth) for depth in self.depths)
        if not depths:
            raise ValueError("depths must contain at least one value")
        if any(depth < 0 for depth in depths):
            raise ValueError("depths cannot contain negative values")
        if self.samples_per_depth <= 0:
            raise ValueError("samples_per_depth must be positive")
        canonical_depths = tuple(int(depth) for depth in self.canonical_depths)
        if any(depth < 0 for depth in canonical_depths):
            raise ValueError("canonical_depths cannot contain negative values")
        if self.evaluation_samples_per_depth <= 0:
            raise ValueError("evaluation_samples_per_depth must be positive")
        if self.evaluation_max_steps <= 0:
            raise ValueError("evaluation_max_steps must be positive")

        object.__setattr__(self, "dataset_out", Path(self.dataset_out))
        object.__setattr__(self, "linear_model_out", Path(self.linear_model_out))
        object.__setattr__(self, "mlp_model_out", Path(self.mlp_model_out))
        object.__setattr__(
            self,
            "report_out",
            None if self.report_out is None else Path(self.report_out),
        )
        object.__setattr__(self, "depths", depths)
        object.__setattr__(self, "canonical_depths", canonical_depths)


def run_baseline_experiment(config: BaselineExperimentConfig) -> dict[str, Any]:
    solved_records = (
        [
            build_dataset_record(
                record_id="solved-000000",
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
    random_dataset_config = DatasetConfig(
        depths=config.depths,
        samples_per_depth=config.samples_per_depth,
        seed=config.seed,
        include_solved=False,
    )
    random_records = generate_dataset(random_dataset_config)
    raw_record_count = len(solved_records) + len(canonical_records) + len(random_records)
    records = deduplicate_records([*solved_records, *canonical_records, *random_records])
    write_jsonl(records, config.dataset_out)

    linear_model, linear_result = train_policy(records, config.linear_config)
    _save_model(linear_model, config.linear_model_out)

    mlp_model, mlp_result = train_mlp_policy(records, config.mlp_config)
    _save_model(mlp_model, config.mlp_model_out)

    evaluation_config = PolicyEvaluationConfig(
        depths=config.depths,
        samples_per_depth=config.evaluation_samples_per_depth,
        max_steps=config.evaluation_max_steps,
        seed=config.seed,
    )
    comparison = compare_loaded_policies(
        [
            NamedPolicy(
                "linear",
                linear_model,
                model_path=str(config.linear_model_out),
                policy_type="linear",
            ),
            NamedPolicy(
                "mlp",
                mlp_model,
                model_path=str(config.mlp_model_out),
                policy_type="mlp",
            ),
        ],
        evaluation_config,
    )
    model_paths = {
        "linear": config.linear_model_out,
        "mlp": config.mlp_model_out,
    }
    recommended_label = str(comparison["ranking"][0]["label"])
    recommended_model_path = _path_for_backend(model_paths[recommended_label])

    report = {
        "experiment": "supervised-baseline",
        "dataset": {
            "path": str(config.dataset_out),
            "records": len(records),
            "trainable_records": sum(
                1 for record in records if record.target_action is not None
            ),
            "depths": list(config.depths),
            "samples_per_depth": config.samples_per_depth,
            "canonical_depths": list(config.canonical_depths),
            "solved_records": len(solved_records),
            "canonical_records": len(canonical_records),
            "random_records": len(random_records),
            "deduplicated_records": raw_record_count - len(records),
            "include_solved": config.include_solved,
            "seed": config.seed,
        },
        "training": {
            "linear": {
                "model_out": str(config.linear_model_out),
                "config": _config_to_json_dict(config.linear_config),
                "result": linear_result.to_dict(),
            },
            "mlp": {
                "model_out": str(config.mlp_model_out),
                "config": _config_to_json_dict(config.mlp_config),
                "result": mlp_result.to_dict(),
            },
        },
        "comparison": comparison,
        "backend": {
            "recommended_label": recommended_label,
            "recommended_model_path": recommended_model_path,
            "environment": {
                "RUBIC_RL_MODEL_PATH": recommended_model_path,
                "RUBIC_RL_MAX_STEPS": str(config.evaluation_max_steps),
            },
        },
    }

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
            "Generate data, train linear and MLP Rubik baselines, compare them, "
            "and write a reproducible experiment report."
        )
    )
    parser.add_argument("--dataset-out", type=Path, default=DEFAULT_DATASET_OUT)
    parser.add_argument(
        "--linear-model-out",
        type=Path,
        default=DEFAULT_LINEAR_MODEL_OUT,
    )
    parser.add_argument("--mlp-model-out", type=Path, default=DEFAULT_MLP_MODEL_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument(
        "--depths",
        nargs="+",
        default=("1", "2", "3"),
        help="Scramble depths, for example: --depths 1 2 3 or --depths 1,2,3.",
    )
    parser.add_argument("--samples-per-depth", type=int, default=100)
    parser.add_argument(
        "--canonical-depths",
        nargs="*",
        default=("1", "2"),
        help=(
            "Deterministic canonical depths to prepend and deduplicate. "
            "Use --canonical-depths none to disable."
        ),
    )
    parser.add_argument("--include-solved", action="store_true")
    parser.add_argument(
        "--seed",
        default="20260531",
        help="Shared dataset/training/evaluation seed. Use 'none' for no seed.",
    )
    parser.add_argument("--evaluation-samples-per-depth", type=int, default=100)
    parser.add_argument("--evaluation-max-steps", type=int, default=30)
    parser.add_argument("--linear-epochs", type=int, default=200)
    parser.add_argument("--linear-learning-rate", type=float, default=0.2)
    parser.add_argument("--linear-batch-size", type=int, default=128)
    parser.add_argument("--linear-l2", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-units", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.05)
    parser.add_argument("--mlp-batch-size", type=int, default=128)
    parser.add_argument("--mlp-l2", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    args = parser.parse_args(argv)

    seed = _parse_seed(args.seed)
    config = BaselineExperimentConfig(
        dataset_out=args.dataset_out,
        linear_model_out=args.linear_model_out,
        mlp_model_out=args.mlp_model_out,
        report_out=args.report_out,
        depths=_parse_depth_args(args.depths),
        samples_per_depth=args.samples_per_depth,
        canonical_depths=_parse_optional_depth_args(args.canonical_depths),
        include_solved=args.include_solved,
        seed=seed,
        evaluation_samples_per_depth=args.evaluation_samples_per_depth,
        evaluation_max_steps=args.evaluation_max_steps,
        linear_config=TrainingConfig(
            epochs=args.linear_epochs,
            learning_rate=args.linear_learning_rate,
            batch_size=args.linear_batch_size,
            l2=args.linear_l2,
            validation_split=args.validation_split,
            seed=seed,
        ),
        mlp_config=MLPTrainingConfig(
            hidden_units=args.mlp_hidden_units,
            epochs=args.mlp_epochs,
            learning_rate=args.mlp_learning_rate,
            batch_size=args.mlp_batch_size,
            l2=args.mlp_l2,
            validation_split=args.validation_split,
            seed=seed,
        ),
    )
    report = run_baseline_experiment(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _save_model(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        model.save(output)  # type: ignore[attr-defined]


def _config_to_json_dict(config: TrainingConfig | MLPTrainingConfig) -> dict[str, Any]:
    return dict(asdict(config))


def _path_for_backend(path: Path) -> str:
    resolved = path if path.is_absolute() else Path.cwd() / path
    try:
        return str(resolved.resolve().relative_to(_project_root()))
    except ValueError:
        return str(resolved)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


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

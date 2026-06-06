from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

from rubic_rl.evaluation.benchmark_suite import (
    WeightedAStarBenchmarkConfig,
    generate_suite_cases,
    load_benchmark_cases,
    preset_config,
    run_benchmark_suite,
    write_benchmark_cases,
)
from rubic_rl.evaluation.search_eval import SearchBenchmarkConfig
from rubic_rl.policies import MLPTrainingConfig, TrainingConfig
from rubic_rl.training.adi import ADITrainingConfig, run_adi_training
from rubic_rl.training.baseline_experiment import (
    BaselineExperimentConfig,
    run_baseline_experiment,
)
from rubic_rl.training.davi import DAVIConfig, run_davi_training


EXPERIMENT_TYPES = {"baseline", "linear", "adi", "davi"}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    experiment_type: str
    config: dict[str, Any]
    description: str | None = None
    benchmark: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        normalized_type = str(self.experiment_type).strip().lower()
        if not self.name or not str(self.name).strip():
            raise ValueError("Experiment name is required")
        if normalized_type not in EXPERIMENT_TYPES:
            raise ValueError(
                f"Unsupported experiment type: {self.experiment_type!r}. "
                f"Expected one of {sorted(EXPERIMENT_TYPES)!r}."
            )
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "experiment_type", normalized_type)
        object.__setattr__(self, "config", dict(self.config))
        object.__setattr__(
            self,
            "benchmark",
            None if self.benchmark is None else dict(self.benchmark),
        )

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "ExperimentSpec":
        experiment_type = payload.get("type", payload.get("experiment_type"))
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("Experiment config must be a JSON object")
        return cls(
            name=str(payload.get("name", "")).strip(),
            experiment_type=str(experiment_type or "").strip(),
            description=(
                None
                if payload.get("description") in {None, ""}
                else str(payload["description"])
            ),
            config=config,
            benchmark=(
                None
                if payload.get("benchmark") is None
                else dict(payload["benchmark"])
            ),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.experiment_type,
            "description": self.description,
            "config": self.config,
            "benchmark": self.benchmark,
        }


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment file must contain a top-level JSON object")
    return ExperimentSpec.from_json_dict(payload)


def build_default_output_paths(
    experiment_type: str,
    *,
    run_name: str,
    project_root: str | Path | None = None,
) -> dict[str, str]:
    normalized_type = str(experiment_type).strip().lower()
    if normalized_type not in EXPERIMENT_TYPES:
        raise ValueError(f"Unsupported experiment type: {experiment_type!r}")

    root = Path(project_root) if project_root is not None else _project_root()
    dataset_dir = root / "datasets" / run_name
    checkpoint_dir = root / "checkpoints" / run_name
    report_dir = root / "reports" / run_name
    paths: dict[str, str] = {
        "report_out": str(report_dir / "report.json"),
        "manifest_out": str(report_dir / "manifest.json"),
        "benchmark_report_out": str(report_dir / "benchmark-report.json"),
        "benchmark_cases_out": str(report_dir / "benchmark-cases.json"),
    }

    if normalized_type in {"baseline", "linear"}:
        paths.update(
            {
                "dataset_out": str(dataset_dir / "dataset.jsonl"),
                "linear_model_out": str(checkpoint_dir / "linear-policy.npz"),
            }
        )
        if normalized_type == "baseline":
            paths["mlp_model_out"] = str(checkpoint_dir / "mlp-policy.npz")
    elif normalized_type == "adi":
        paths.update(
            {
                "dataset_out": str(dataset_dir / "dataset.jsonl"),
                "checkpoint_out": str(checkpoint_dir / "torch-policy-value.pt"),
            }
        )
    elif normalized_type == "davi":
        checkpoint_out = checkpoint_dir / "torch-value.pt"
        paths.update(
            {
                "checkpoint_out": str(checkpoint_out),
                "resume_from": str(Path(str(checkpoint_out) + ".resume")),
            }
        )
    return paths


def run_experiment(
    spec: ExperimentSpec,
    *,
    spec_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    project_root_path = Path(project_root) if project_root is not None else _project_root()
    default_paths = build_default_output_paths(
        spec.experiment_type,
        run_name=spec.name,
        project_root=project_root_path,
    )
    merged_config = {**default_paths, **spec.config}
    manifest_out = Path(merged_config.pop("manifest_out", default_paths["manifest_out"]))
    merged_config.pop("benchmark_report_out", None)
    merged_config.pop("benchmark_cases_out", None)
    benchmark_report_out = Path(
        default_paths["benchmark_report_out"]
        if spec.benchmark is None
        else spec.benchmark.get("out", default_paths["benchmark_report_out"])
    )
    benchmark_cases_out = Path(
        default_paths["benchmark_cases_out"]
        if spec.benchmark is None
        else spec.benchmark.get("cases_out", default_paths["benchmark_cases_out"])
    )

    if spec.experiment_type in {"baseline", "linear"}:
        if "linear_config" in merged_config and isinstance(merged_config["linear_config"], dict):
            merged_config["linear_config"] = TrainingConfig(**merged_config["linear_config"])
        if "mlp_config" in merged_config and isinstance(merged_config["mlp_config"], dict):
            merged_config["mlp_config"] = MLPTrainingConfig(**merged_config["mlp_config"])
        if spec.experiment_type == "linear":
            merged_config["train_mlp"] = False
            merged_config.setdefault("mlp_model_out", None)
        config = BaselineExperimentConfig(**merged_config)
        report = run_baseline_experiment(config)
        outputs = {
            "dataset_out": str(config.dataset_out),
            "linear_model_out": str(config.linear_model_out),
            "mlp_model_out": None if config.mlp_model_out is None else str(config.mlp_model_out),
            "report_out": None if config.report_out is None else str(config.report_out),
        }
    elif spec.experiment_type == "adi":
        config = ADITrainingConfig(**merged_config)
        report = run_adi_training(config)
        outputs = {
            "dataset_out": None if config.dataset_out is None else str(config.dataset_out),
            "checkpoint_out": str(config.checkpoint_out),
            "report_out": None if config.report_out is None else str(config.report_out),
        }
    else:
        config = DAVIConfig(**merged_config)
        report = run_davi_training(config)
        outputs = {
            "checkpoint_out": str(config.checkpoint_out),
            "resume_from": None if config.resume_from is None else str(config.resume_from),
            "report_out": None if config.report_out is None else str(config.report_out),
        }

    benchmark_result = None
    if spec.benchmark is not None:
        benchmark_result = _run_post_training_benchmark(
            spec,
            project_root=project_root_path,
            outputs=outputs,
            benchmark_config=spec.benchmark,
            benchmark_report_out=benchmark_report_out,
            benchmark_cases_out=benchmark_cases_out,
        )
        outputs["benchmark_report_out"] = str(benchmark_report_out)
        outputs["benchmark_cases_out"] = str(benchmark_cases_out)

    manifest = _build_manifest(
        spec,
        spec_path=spec_path,
        project_root=project_root_path,
        outputs=outputs,
        report=report,
        benchmark_result=benchmark_result,
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "experiment": spec.to_json_dict(),
        "outputs": outputs,
        "manifest_path": str(manifest_out),
        "report": report,
        **({"benchmark": benchmark_result} if benchmark_result is not None else {}),
    }


def _build_manifest(
    spec: ExperimentSpec,
    *,
    spec_path: str | Path | None,
    project_root: Path,
    outputs: dict[str, str | None],
    report: dict[str, Any],
    benchmark_result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "name": spec.name,
        "type": spec.experiment_type,
        "description": spec.description,
        "spec_path": None if spec_path is None else str(Path(spec_path)),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(project_root),
        "config": spec.config,
        "benchmark": spec.benchmark,
        "outputs": outputs,
        "headline_metrics": _headline_metrics(spec.experiment_type, report),
        **(
            {"benchmark_summary": _benchmark_headline_metrics(benchmark_result)}
            if benchmark_result is not None
            else {}
        ),
        **(
            {"promotion_gate": _promotion_gate_summary(spec, benchmark_result)}
            if benchmark_result is not None
            else {}
        ),
    }


def _headline_metrics(experiment_type: str, report: dict[str, Any]) -> dict[str, Any]:
    if experiment_type in {"baseline", "linear"}:
        comparison = report.get("comparison", {})
        ranking = comparison.get("ranking", [])
        return {
            "recommended_label": report.get("backend", {}).get("recommended_label"),
            "top_solve_rate": ranking[0].get("solve_rate") if ranking else None,
        }
    if experiment_type == "adi":
        iterations = report.get("training", {}).get("iterations", [])
        last_iteration = iterations[-1] if iterations else {}
        return {
            "records": report.get("dataset", {}).get("records"),
            "policy_records": report.get("dataset", {}).get("policy_records"),
            "final_validation_policy_accuracy": last_iteration.get(
                "validation_policy_accuracy"
            ),
        }
    if experiment_type == "davi":
        history = report.get("history", [])
        last_step = history[-1] if history else {}
        return {
            "completed_steps": report.get("completed_steps"),
            "final_curriculum_depth": last_step.get("curriculum_depth"),
            "final_mae": last_step.get("mae"),
        }
    return {}


def _benchmark_headline_metrics(benchmark_result: dict[str, Any]) -> dict[str, Any]:
    ranking = benchmark_result.get("ranking", [])
    top = ranking[0] if ranking else {}
    return {
        "top_label": top.get("label"),
        "top_solve_rate": top.get("solve_rate"),
        "top_avg_duration_ms": top.get("avg_duration_ms"),
    }


def _promotion_gate_summary(
    spec: ExperimentSpec,
    benchmark_result: dict[str, Any],
) -> dict[str, Any]:
    ranking = benchmark_result.get("ranking", [])
    if not isinstance(ranking, list):
        return {
            "eligible": False,
            "reason": "benchmark ranking missing",
        }

    candidate_prefixes = tuple(
        prefix.strip()
        for prefix in _promotion_prefixes(spec.experiment_type)
        if prefix.strip()
    )
    reference_prefixes = _reference_prefixes(spec.benchmark)
    candidate = _select_best_ranked_entry(
        ranking,
        include_prefixes=candidate_prefixes,
        exclude_prefixes=reference_prefixes,
    )
    reference = _select_best_ranked_entry(
        ranking,
        include_prefixes=reference_prefixes,
    )

    if candidate is None:
        return {
            "eligible": False,
            "reason": "no candidate entrant found for current run",
        }
    if reference is None:
        return {
            "eligible": False,
            "candidate_label": candidate.get("label"),
            "candidate_rank": candidate.get("rank"),
            "candidate_solve_rate": candidate.get("solve_rate"),
            "reason": "no reference entrant found",
        }

    candidate_solve_rate = _float_or_none(candidate.get("solve_rate"))
    reference_solve_rate = _float_or_none(reference.get("solve_rate"))
    candidate_duration_ms = _float_or_none(candidate.get("avg_duration_ms"))
    reference_duration_ms = _float_or_none(reference.get("avg_duration_ms"))
    solve_rate_delta = (
        None
        if candidate_solve_rate is None or reference_solve_rate is None
        else candidate_solve_rate - reference_solve_rate
    )
    duration_delta_ms = (
        None
        if candidate_duration_ms is None or reference_duration_ms is None
        else candidate_duration_ms - reference_duration_ms
    )

    beats_reference = False
    if solve_rate_delta is not None:
        if solve_rate_delta > 0:
            beats_reference = True
        elif solve_rate_delta == 0 and duration_delta_ms is not None and duration_delta_ms < 0:
            beats_reference = True

    return {
        "eligible": True,
        "candidate_label": candidate.get("label"),
        "candidate_rank": candidate.get("rank"),
        "candidate_solve_rate": candidate_solve_rate,
        "reference_label": reference.get("label"),
        "reference_rank": reference.get("rank"),
        "reference_solve_rate": reference_solve_rate,
        "solve_rate_delta": solve_rate_delta,
        "duration_delta_ms": duration_delta_ms,
        "beats_reference": beats_reference,
    }


def _run_post_training_benchmark(
    spec: ExperimentSpec,
    *,
    project_root: Path,
    outputs: dict[str, str | None],
    benchmark_config: dict[str, Any],
    benchmark_report_out: Path,
    benchmark_cases_out: Path,
) -> dict[str, Any]:
    suite_name = str(benchmark_config.get("suite", "custom"))
    config = _benchmark_suite_config(benchmark_config)
    cases_in = benchmark_config.get("cases_in")
    if cases_in in {None, ""}:
        cases = generate_suite_cases(config)
    else:
        cases = load_benchmark_cases(_resolve_benchmark_cases_path(cases_in, project_root))
    write_benchmark_cases(
        benchmark_cases_out,
        suite_name=suite_name,
        config=config,
        cases=cases,
    )

    policy_type = str(benchmark_config.get("policy_type", "auto"))
    policy_models = _default_policy_models(spec.experiment_type, outputs)
    davi_models = _default_davi_models(spec.experiment_type, outputs)
    reference_models = _resolve_reference_models(
        benchmark_config.get("compare_runs", []),
        project_root=project_root,
        require_existing=bool(benchmark_config.get("require_compare_runs", False)),
    )
    policy_models.extend(reference_models["policy"])
    davi_models.extend(reference_models["davi"])
    policy_models.extend(
        (*_parse_labelled_path(raw), policy_type)
        for raw in benchmark_config.get("policy_models", [])
    )
    davi_models.extend(
        _parse_labelled_path(raw) for raw in benchmark_config.get("davi_models", [])
    )

    return run_benchmark_suite(
        suite_name=suite_name,
        config=config,
        cases=cases,
        policy_models=policy_models,
        davi_models=davi_models,
        weighted_astar_config=WeightedAStarBenchmarkConfig(
            weight=float(benchmark_config.get("weight", 0.6)),
            policy_weight=float(benchmark_config.get("policy_weight", 0.0)),
            batch_expansion=int(benchmark_config.get("batch_expansion", 100)),
            max_nodes=int(benchmark_config.get("max_nodes", 1_000_000)),
        ),
        include_episodes=bool(benchmark_config.get("include_episodes", False)),
        output_path=benchmark_report_out,
    )


def _benchmark_suite_config(benchmark_config: dict[str, Any]) -> SearchBenchmarkConfig:
    raw_config = benchmark_config.get("config")
    if raw_config is not None:
        return SearchBenchmarkConfig(
            depths=tuple(int(depth) for depth in raw_config.get("depths", (1, 2, 3))),
            samples_per_depth=int(raw_config.get("samples_per_depth", 100)),
            max_depth=int(raw_config.get("max_depth", 30)),
            beam_width=int(raw_config.get("beam_width", 5)),
            top_k=int(raw_config.get("top_k", 5)),
            seed=raw_config.get("seed", 0),
        )
    suite_name = str(benchmark_config.get("suite", "shallow"))
    return preset_config(suite_name)


def _resolve_benchmark_cases_path(raw_path: str | Path, project_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _default_policy_models(
    experiment_type: str,
    outputs: dict[str, str | None],
) -> list[tuple[str, Path, str]]:
    models: list[tuple[str, Path, str]] = []
    if experiment_type in {"baseline", "linear"}:
        linear_model = outputs.get("linear_model_out")
        if linear_model:
            models.append(("linear", Path(linear_model), "linear"))
        mlp_model = outputs.get("mlp_model_out")
        if experiment_type == "baseline" and mlp_model:
            models.append(("mlp", Path(mlp_model), "mlp"))
    elif experiment_type == "adi":
        checkpoint = outputs.get("checkpoint_out")
        if checkpoint:
            models.append(("adi", Path(checkpoint), "torch"))
    return models


def _default_davi_models(
    experiment_type: str,
    outputs: dict[str, str | None],
) -> list[tuple[str, Path]]:
    checkpoint = outputs.get("checkpoint_out")
    if experiment_type == "davi" and checkpoint:
        return [("davi", Path(checkpoint))]
    return []


def _parse_labelled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Expected label=path format")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError("Expected non-empty label=path format")
    return label, Path(raw_path)


def _promotion_prefixes(experiment_type: str) -> tuple[str, ...]:
    if experiment_type == "baseline":
        return ("linear:", "mlp:")
    if experiment_type == "linear":
        return ("linear:",)
    if experiment_type == "adi":
        return ("adi:",)
    if experiment_type == "davi":
        return ("davi",)
    return ()


def _reference_prefixes(benchmark_config: dict[str, Any] | None) -> tuple[str, ...]:
    if benchmark_config is None:
        return ()
    compare_runs = benchmark_config.get("compare_runs", [])
    prefixes: list[str] = []
    if isinstance(compare_runs, list):
        for entry in compare_runs:
            if not isinstance(entry, dict):
                continue
            prefix = str(entry.get("label_prefix", "")).strip()
            if prefix:
                prefixes.append(f"{prefix}:")
            else:
                run_name = str(entry.get("run", "")).strip()
                if run_name:
                    prefixes.append(f"{run_name}:")
    return tuple(prefixes)


def _select_best_ranked_entry(
    ranking: list[Any],
    *,
    include_prefixes: tuple[str, ...],
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_rank = None
    for entry in ranking:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", ""))
        if include_prefixes and not any(label.startswith(prefix) for prefix in include_prefixes):
            continue
        if exclude_prefixes and any(label.startswith(prefix) for prefix in exclude_prefixes):
            continue
        rank = entry.get("rank")
        if not isinstance(rank, int):
            continue
        if best is None or best_rank is None or rank < best_rank:
            best = entry
            best_rank = rank
    return best


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _resolve_reference_models(
    compare_runs: Any,
    *,
    project_root: Path,
    require_existing: bool,
) -> dict[str, list[tuple[Any, ...]]]:
    policy_models: list[tuple[str, Path, str]] = []
    davi_models: list[tuple[str, Path]] = []
    if not compare_runs:
        return {"policy": policy_models, "davi": davi_models}
    if not isinstance(compare_runs, list):
        raise ValueError("benchmark.compare_runs must be a JSON array")

    for entry in compare_runs:
        if not isinstance(entry, dict):
            raise ValueError("Each benchmark.compare_runs entry must be a JSON object")
        run_name = str(entry.get("run", "")).strip()
        run_type = str(entry.get("type", "baseline")).strip().lower()
        label_prefix = str(entry.get("label_prefix", run_name)).strip() or run_name
        if not run_name:
            raise ValueError("benchmark.compare_runs entries require a non-empty run")
        if run_type not in EXPERIMENT_TYPES:
            raise ValueError(
                f"Unsupported compare_runs type: {run_type!r}. "
                f"Expected one of {sorted(EXPERIMENT_TYPES)!r}."
            )

        default_paths = build_default_output_paths(
            run_type,
            run_name=run_name,
            project_root=project_root,
        )
        if run_type in {"baseline", "linear"}:
            models = [("linear", Path(default_paths["linear_model_out"]), "linear")]
            if run_type == "baseline":
                models.append(("mlp", Path(default_paths["mlp_model_out"]), "mlp"))
            selected_labels = _normalize_reference_labels(
                entry.get("models"),
                allowed={"linear"} if run_type == "linear" else {"linear", "mlp"},
                default=("linear",) if run_type == "linear" else ("linear", "mlp"),
            )
            for model_label, model_path, policy_type in models:
                if model_label not in selected_labels:
                    continue
                if not model_path.exists():
                    if require_existing:
                        raise FileNotFoundError(
                            f"Reference baseline model not found: {model_path}"
                        )
                    continue
                policy_models.append(
                    (f"{label_prefix}:{model_label}", model_path, policy_type)
                )
        elif run_type == "adi":
            model_path = Path(default_paths["checkpoint_out"])
            if not model_path.exists():
                if require_existing:
                    raise FileNotFoundError(
                        f"Reference ADI checkpoint not found: {model_path}"
                    )
            else:
                policy_models.append((label_prefix, model_path, "torch"))
        else:
            model_path = Path(default_paths["checkpoint_out"])
            if not model_path.exists():
                if require_existing:
                    raise FileNotFoundError(
                        f"Reference DAVI checkpoint not found: {model_path}"
                    )
            else:
                davi_models.append((label_prefix, model_path))
    return {"policy": policy_models, "davi": davi_models}


def _normalize_reference_labels(
    value: Any,
    *,
    allowed: set[str],
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError("benchmark.compare_runs[].models must be a JSON array")
    labels = tuple(str(item).strip().lower() for item in value)
    invalid = [label for label in labels if label not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported reference model labels: {invalid!r}. Expected subset of {sorted(allowed)!r}."
        )
    return labels


def _git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]

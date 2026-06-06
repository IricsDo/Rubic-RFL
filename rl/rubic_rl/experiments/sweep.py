from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .runner import load_experiment_spec, run_experiment


def run_experiment_sweep(
    config_paths: Sequence[str | Path],
    *,
    sweep_name: str,
    project_root: str | Path | None = None,
    summary_out: str | Path | None = None,
) -> dict[str, Any]:
    if not config_paths:
        raise ValueError("config_paths must contain at least one experiment config")

    root = Path(project_root) if project_root is not None else _project_root()
    normalized_paths = [Path(path) for path in config_paths]
    summary_path = (
        Path(summary_out)
        if summary_out is not None
        else root / "reports" / sweep_name / "summary.json"
    )

    runs: list[dict[str, Any]] = []
    for config_path in normalized_paths:
        spec = load_experiment_spec(config_path)
        result = run_experiment(spec, spec_path=config_path, project_root=root)
        manifest_path = Path(result["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs.append(_summarize_run(config_path, result, manifest))

    ranking = sorted(runs, key=_ranking_key)
    for index, run in enumerate(ranking, start=1):
        run["sweep_rank"] = index

    summary = {
        "name": sweep_name,
        "config_paths": [str(path) for path in normalized_paths],
        "summary_out": str(summary_path),
        "runs": ranking,
        "best_run": ranking[0] if ranking else None,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _summarize_run(
    config_path: Path,
    result: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    benchmark_summary = manifest.get("benchmark_summary", {})
    promotion_gate = manifest.get("promotion_gate", {})
    return {
        "name": manifest.get("name"),
        "type": manifest.get("type"),
        "config_path": str(config_path),
        "manifest_path": result.get("manifest_path"),
        "report_out": result.get("outputs", {}).get("report_out"),
        "benchmark_report_out": result.get("outputs", {}).get("benchmark_report_out"),
        "benchmark_cases_out": result.get("outputs", {}).get("benchmark_cases_out"),
        "recommended_label": manifest.get("headline_metrics", {}).get("recommended_label"),
        "benchmark_top_label": benchmark_summary.get("top_label"),
        "benchmark_top_solve_rate": benchmark_summary.get("top_solve_rate"),
        "eligible": bool(promotion_gate.get("eligible", False)),
        "beats_reference": bool(promotion_gate.get("beats_reference", False)),
        "candidate_label": promotion_gate.get("candidate_label"),
        "candidate_solve_rate": promotion_gate.get("candidate_solve_rate"),
        "reference_label": promotion_gate.get("reference_label"),
        "reference_solve_rate": promotion_gate.get("reference_solve_rate"),
        "solve_rate_delta": promotion_gate.get("solve_rate_delta"),
        "duration_delta_ms": promotion_gate.get("duration_delta_ms"),
    }


def _ranking_key(run: dict[str, Any]) -> tuple[float, float, float, float, str]:
    eligible = 1.0 if run.get("eligible") else 0.0
    beats_reference = 1.0 if run.get("beats_reference") else 0.0
    solve_rate_delta = _float_or_default(run.get("solve_rate_delta"), -1.0)
    candidate_solve_rate = _float_or_default(run.get("candidate_solve_rate"), -1.0)
    duration_delta_ms = _float_or_default(run.get("duration_delta_ms"), float("inf"))
    return (
        -beats_reference,
        -eligible,
        -solve_rate_delta,
        -candidate_solve_rate,
        duration_delta_ms,
        str(run.get("name", "")),
    )


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]

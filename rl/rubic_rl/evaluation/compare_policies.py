from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from rubic_rl.evaluation.policy_eval import (
    PolicyEvaluationConfig,
    PredictsActions,
    _parse_depth_args,
    _parse_seed,
    evaluate_policy_on_env,
    summarize_results,
)
from rubic_rl.policies import PolicyType, load_policy


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path
    policy_type: PolicyType = "auto"

    def __post_init__(self) -> None:
        label = self.label.strip()
        if not label:
            raise ValueError("model label must not be empty")
        if not str(self.path):
            raise ValueError("model path must not be empty")

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "path", Path(self.path))

    def to_json_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "path": str(self.path),
            "policy_type": self.policy_type,
        }


@dataclass(frozen=True)
class NamedPolicy:
    label: str
    policy: PredictsActions
    model_path: str | None = None
    policy_type: str = "in_memory"

    def __post_init__(self) -> None:
        label = self.label.strip()
        if not label:
            raise ValueError("policy label must not be empty")

        object.__setattr__(self, "label", label)


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in value:
        raise ValueError("model spec must use label=path")

    label, path_text = value.split("=", 1)
    label = label.strip()
    path_text = path_text.strip()
    if not label:
        raise ValueError("model label must not be empty")
    if not path_text:
        raise ValueError("model path must not be empty")

    return ModelSpec(label=label, path=Path(path_text))


def parse_model_specs(values: Sequence[str]) -> tuple[ModelSpec, ...]:
    if not values:
        raise ValueError("at least one model spec is required")

    specs = tuple(parse_model_spec(value) for value in values)
    labels: set[str] = set()
    for spec in specs:
        if spec.label in labels:
            raise ValueError(f"duplicate model label: {spec.label}")
        labels.add(spec.label)
    return specs


def load_named_policies(specs: Sequence[ModelSpec]) -> tuple[NamedPolicy, ...]:
    return tuple(
        NamedPolicy(
            label=spec.label,
            policy=load_policy(spec.path, policy_type=spec.policy_type),
            model_path=str(spec.path),
            policy_type=spec.policy_type,
        )
        for spec in specs
    )


def compare_loaded_policies(
    policies: Sequence[NamedPolicy],
    config: PolicyEvaluationConfig | None = None,
    *,
    include_episodes: bool = False,
) -> dict[str, Any]:
    if not policies:
        raise ValueError("at least one policy is required")

    _validate_unique_policy_labels(policies)
    config = config or PolicyEvaluationConfig()
    policy_reports = [
        _evaluate_named_policy(policy, config, include_episodes=include_episodes)
        for policy in policies
    ]
    return {
        "config": config.to_json_dict(),
        "policies": policy_reports,
        "ranking": rank_policy_reports(policy_reports),
    }


def rank_policy_reports(
    policy_reports: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        policy_reports,
        key=lambda report: (
            -_overall_float(report, "solve_rate", default=0.0),
            _overall_float(report, "avg_steps_solved", default=float("inf")),
            _overall_float(report, "avg_steps_all", default=float("inf")),
            -_overall_float(report, "avg_reward", default=float("-inf")),
            str(report["label"]),
        ),
    )

    ranking: list[dict[str, Any]] = []
    for rank, report in enumerate(ranked, start=1):
        overall = report["summary"]["overall"]
        ranking.append(
            {
                "rank": rank,
                "label": report["label"],
                "attempts": overall["attempts"],
                "solved": overall["solved"],
                "solve_rate": overall["solve_rate"],
                "avg_steps_solved": overall["avg_steps_solved"],
                "avg_steps_all": overall["avg_steps_all"],
                "avg_reward": overall["avg_reward"],
            }
        )
    return ranking


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare trained Rubik policy checkpoints with seeded rollouts."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Policy checkpoint as label=path. Repeat for each policy.",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        default=("1", "2", "3"),
        help="Scramble depths, for example: --depths 1 2 3 or --depths 1,2,3.",
    )
    parser.add_argument("--samples-per-depth", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument(
        "--seed",
        default="0",
        help="Evaluation seed. Use 'none' for non-deterministic rollouts.",
    )
    parser.add_argument(
        "--include-episodes",
        action="store_true",
        help="Include per-episode details in the JSON output.",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON report output path.")
    args = parser.parse_args(argv)

    specs = parse_model_specs(args.model)
    config = PolicyEvaluationConfig(
        depths=_parse_depth_args(args.depths),
        samples_per_depth=args.samples_per_depth,
        max_steps=args.max_steps,
        seed=_parse_seed(args.seed),
    )
    report = compare_loaded_policies(
        load_named_policies(specs),
        config,
        include_episodes=args.include_episodes,
    )
    output = json.dumps(report, indent=2, sort_keys=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


def _evaluate_named_policy(
    policy: NamedPolicy,
    config: PolicyEvaluationConfig,
    *,
    include_episodes: bool,
) -> dict[str, Any]:
    results = evaluate_policy_on_env(policy.policy, config)
    report: dict[str, Any] = {
        "label": policy.label,
        "policy_type": policy.policy_type,
        "summary": summarize_results(results),
    }
    if policy.model_path is not None:
        report["model_path"] = policy.model_path
    if include_episodes:
        report["episodes"] = [result.to_json_dict() for result in results]
    return report


def _validate_unique_policy_labels(policies: Sequence[NamedPolicy]) -> None:
    labels: set[str] = set()
    for policy in policies:
        if policy.label in labels:
            raise ValueError(f"duplicate policy label: {policy.label}")
        labels.add(policy.label)


def _overall_float(report: dict[str, Any], field: str, *, default: float) -> float:
    value = report["summary"]["overall"][field]
    if value is None:
        return default
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())

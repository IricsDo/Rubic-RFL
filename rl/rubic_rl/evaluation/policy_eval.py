from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from app.cube import parse_moves
from rubic_rl import ACTION_MOVES, RubiksCubeEnv
from rubic_rl.policies import load_policy
from rubic_rl.policies.linear_policy import featurize_indices


class PredictsActions(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return integer action IDs for a batch of feature vectors."""


@dataclass(frozen=True)
class PolicyEvaluationConfig:
    depths: tuple[int, ...] = (1, 2, 3)
    samples_per_depth: int = 100
    max_steps: int = 30
    seed: int | None = 0

    def __post_init__(self) -> None:
        depths = tuple(int(depth) for depth in self.depths)
        if not depths:
            raise ValueError("depths must contain at least one value")
        if any(depth < 0 for depth in depths):
            raise ValueError("depths cannot contain negative values")
        if self.samples_per_depth <= 0:
            raise ValueError("samples_per_depth must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        object.__setattr__(self, "depths", depths)

    def to_json_dict(self) -> dict[str, int | list[int] | None]:
        return {
            "depths": list(self.depths),
            "samples_per_depth": self.samples_per_depth,
            "max_steps": self.max_steps,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class EpisodeResult:
    depth: int
    sample_index: int
    seed: int | None
    scramble: tuple[str, ...]
    solved: bool
    truncated: bool
    steps: int
    total_reward: float
    moves: tuple[str, ...]

    def to_json_dict(self) -> dict[str, int | float | bool | None | list[str]]:
        return {
            "depth": self.depth,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "scramble": list(self.scramble),
            "solved": self.solved,
            "truncated": self.truncated,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "moves": list(self.moves),
        }


@dataclass(frozen=True)
class DepthSummary:
    attempts: int
    solved: int
    solve_rate: float
    avg_steps_solved: float | None
    avg_steps_all: float | None
    avg_reward: float | None

    def to_json_dict(self) -> dict[str, int | float | None]:
        return {
            "attempts": self.attempts,
            "solved": self.solved,
            "solve_rate": self.solve_rate,
            "avg_steps_solved": self.avg_steps_solved,
            "avg_steps_all": self.avg_steps_all,
            "avg_reward": self.avg_reward,
        }


def evaluate_policy_on_env(
    policy: PredictsActions,
    config: PolicyEvaluationConfig | None = None,
) -> list[EpisodeResult]:
    config = config or PolicyEvaluationConfig()
    rng = np.random.default_rng(config.seed)
    results: list[EpisodeResult] = []

    for depth in config.depths:
        env = RubiksCubeEnv(
            scramble_depth=depth,
            max_steps=config.max_steps,
            observation_mode="indices",
        )
        try:
            for sample_index in range(config.samples_per_depth):
                episode_seed = _next_seed(rng, config.seed)
                observation, info = env.reset(seed=episode_seed)
                results.append(
                    _rollout(
                        policy,
                        env,
                        observation,
                        info,
                        depth=depth,
                        sample_index=sample_index,
                        seed=episode_seed,
                    )
                )
        finally:
            env.close()

    return results


def evaluate_policy_on_scrambles(
    policy: PredictsActions,
    scrambles: Iterable[Sequence[str]],
    *,
    max_steps: int = 30,
) -> list[EpisodeResult]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    env = RubiksCubeEnv(max_steps=max_steps, observation_mode="indices")
    results: list[EpisodeResult] = []
    try:
        for sample_index, scramble in enumerate(scrambles):
            moves = parse_moves(scramble)
            observation, info = env.reset(options={"scramble": moves})
            results.append(
                _rollout(
                    policy,
                    env,
                    observation,
                    info,
                    depth=len(moves),
                    sample_index=sample_index,
                    seed=None,
                )
            )
    finally:
        env.close()

    return results


def summarize_results(
    results: Sequence[EpisodeResult],
) -> dict[str, dict[str, int | float | None] | dict[str, dict[str, int | float | None]]]:
    by_depth: dict[str, dict[str, int | float | None]] = {}
    for depth in sorted({result.depth for result in results}):
        depth_results = [result for result in results if result.depth == depth]
        by_depth[str(depth)] = _summarize_group(depth_results).to_json_dict()

    return {
        "overall": _summarize_group(results).to_json_dict(),
        "by_depth": by_depth,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Rubik policy with Gymnasium rollouts."
    )
    parser.add_argument("--model", required=True, type=Path, help="Input .npz model.")
    parser.add_argument(
        "--policy-type",
        choices=("auto", "linear", "mlp"),
        default="auto",
        help="Checkpoint policy type. 'auto' detects linear and MLP checkpoints.",
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
    args = parser.parse_args(argv)

    model = load_policy(args.model, policy_type=args.policy_type)
    config = PolicyEvaluationConfig(
        depths=_parse_depth_args(args.depths),
        samples_per_depth=args.samples_per_depth,
        max_steps=args.max_steps,
        seed=_parse_seed(args.seed),
    )
    results = evaluate_policy_on_env(model, config)
    output: dict[str, object] = {
        "config": config.to_json_dict(),
        "summary": summarize_results(results),
    }
    if args.include_episodes:
        output["episodes"] = [result.to_json_dict() for result in results]

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _rollout(
    policy: PredictsActions,
    env: RubiksCubeEnv,
    observation: np.ndarray,
    info: dict[str, object],
    *,
    depth: int,
    sample_index: int,
    seed: int | None,
) -> EpisodeResult:
    moves: list[str] = []
    total_reward = 0.0
    terminated = bool(info["is_solved"])
    truncated = False

    while not terminated and not truncated:
        action = _predict_action(policy, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        moves.append(str(info["move"]))

    return EpisodeResult(
        depth=depth,
        sample_index=sample_index,
        seed=seed,
        scramble=tuple(str(move) for move in info["scramble"]),
        solved=bool(info["is_solved"]),
        truncated=truncated,
        steps=int(info["steps"]),
        total_reward=total_reward,
        moves=tuple(moves),
    )


def _predict_action(policy: PredictsActions, observation: np.ndarray) -> int:
    features = featurize_indices(observation).reshape(1, -1)
    action = int(policy.predict(features)[0])
    if action < 0 or action >= len(ACTION_MOVES):
        raise ValueError(f"Policy returned invalid action: {action}")
    return action


def _summarize_group(results: Sequence[EpisodeResult]) -> DepthSummary:
    attempts = len(results)
    solved_results = [result for result in results if result.solved]
    solved = len(solved_results)
    return DepthSummary(
        attempts=attempts,
        solved=solved,
        solve_rate=0.0 if attempts == 0 else solved / attempts,
        avg_steps_solved=_mean_or_none(result.steps for result in solved_results),
        avg_steps_all=_mean_or_none(result.steps for result in results),
        avg_reward=_mean_or_none(result.total_reward for result in results),
    )


def _mean_or_none(values: Iterable[int | float]) -> float | None:
    collected = [float(value) for value in values]
    if not collected:
        return None
    return sum(collected) / len(collected)


def _next_seed(rng: np.random.Generator, root_seed: int | None) -> int | None:
    if root_seed is None:
        return None
    return int(rng.integers(0, np.iinfo(np.int32).max))


def _parse_depth_args(values: Sequence[str]) -> tuple[int, ...]:
    raw_depths: list[str] = []
    for value in values:
        raw_depths.extend(part for part in value.split(",") if part)
    return tuple(int(depth) for depth in raw_depths)


def _parse_seed(value: str | None) -> int | None:
    if value is None or value.lower() == "none":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())

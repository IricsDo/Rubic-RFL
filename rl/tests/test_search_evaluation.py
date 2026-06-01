import json
from pathlib import Path
import unittest

import numpy as np

from app.solvers.rl_search import RL_ACTION_MOVES
from rubic_rl.evaluation.search_eval import (
    BenchmarkCase,
    SearchBenchmarkConfig,
    generate_benchmark_cases,
    run_search_benchmark,
    summarize_benchmark_results,
)


class RankedPolicy:
    def __init__(self, rankings: dict[str, float]):
        self._probabilities = np.full(len(RL_ACTION_MOVES), 0.001, dtype=np.float64)
        for move, score in rankings.items():
            self._probabilities[RL_ACTION_MOVES.index(move)] = score

    def predict(self, features: np.ndarray) -> np.ndarray:
        action = int(np.argmax(self._probabilities))
        return np.full(features.shape[0], action, dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(self._probabilities, (features.shape[0], 1))


class SearchEvaluationTests(unittest.TestCase):
    def test_benchmark_compares_greedy_and_beam_search_on_same_cases(self):
        policy = RankedPolicy({"F": 0.8, "U'": 0.7, "R'": 0.6})
        config = SearchBenchmarkConfig(
            depths=(2,),
            samples_per_depth=1,
            max_depth=2,
            beam_width=3,
            top_k=3,
            seed=20260531,
        )
        cases = (
            BenchmarkCase(
                depth=2,
                sample_index=0,
                seed=17,
                scramble=("R", "U"),
            ),
        )

        report = run_search_benchmark(
            policy,
            config,
            cases=cases,
            include_episodes=True,
            timer=_fixed_timer(),
        )

        self.assertEqual(report["config"], config.to_json_dict())
        self.assertEqual(report["cases"][0]["scramble"], ["R", "U"])
        self.assertEqual(
            [strategy["label"] for strategy in report["strategies"]],
            ["greedy", "beam_search"],
        )
        greedy, beam_search = report["strategies"]
        self.assertEqual(greedy["summary"]["overall"]["solve_rate"], 0.0)
        self.assertEqual(beam_search["summary"]["overall"]["solve_rate"], 1.0)
        self.assertEqual(beam_search["episodes"][0]["moves"], ["U'", "R'"])
        self.assertGreater(beam_search["episodes"][0]["expanded_states"], 0)
        self.assertEqual(report["ranking"][0]["label"], "beam_search")

    def test_generates_seeded_cases_deterministically(self):
        config = SearchBenchmarkConfig(
            depths=(1, 2),
            samples_per_depth=3,
            seed=20260531,
        )

        first = generate_benchmark_cases(config)
        second = generate_benchmark_cases(config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual([case.sample_index for case in first[:3]], [0, 1, 2])
        self.assertEqual({case.depth for case in first}, {1, 2})
        self.assertTrue(all(len(case.scramble) == case.depth for case in first))

    def test_summarizes_failures_and_search_costs_by_depth(self):
        results = [
            _episode(depth=1, solved=True, moves=("U",), duration_ms=2.0),
            _episode(depth=1, solved=False, moves=("U", "U"), duration_ms=4.0),
            _episode(depth=2, solved=True, moves=("R", "U"), duration_ms=6.0),
        ]

        summary = summarize_benchmark_results(results)

        self.assertEqual(summary["overall"]["attempts"], 3)
        self.assertEqual(summary["overall"]["solved"], 2)
        self.assertEqual(summary["by_depth"]["1"]["attempts"], 2)
        self.assertEqual(summary["by_depth"]["1"]["solve_rate"], 0.5)
        self.assertEqual(summary["by_depth"]["1"]["avg_move_count_solved"], 1.0)
        self.assertEqual(summary["by_depth"]["1"]["avg_duration_ms"], 3.0)
        self.assertEqual(summary["by_depth"]["2"]["avg_expanded_states"], 2.0)

    def test_cli_writes_json_report(self):
        policy = RankedPolicy({"U'": 1.0})
        config = SearchBenchmarkConfig(depths=(1,), samples_per_depth=1, max_depth=1)
        output = Path("tests/generated/search-evaluation/report.json")

        report = run_search_benchmark(
            policy,
            config,
            cases=(BenchmarkCase(1, 0, 1, ("U",)),),
            output_path=output,
            timer=_fixed_timer(),
        )

        self.assertTrue(output.exists())
        with output.open("r", encoding="utf-8") as report_file:
            self.assertEqual(json.load(report_file), report)


def _episode(
    *,
    depth: int,
    solved: bool,
    moves: tuple[str, ...],
    duration_ms: float,
):
    from rubic_rl.evaluation.search_eval import BenchmarkEpisodeResult

    return BenchmarkEpisodeResult(
        depth=depth,
        sample_index=0,
        seed=1,
        scramble=("U",) * depth,
        status="solved" if solved else "failed",
        solved=solved,
        moves=moves,
        move_count=len(moves),
        duration_ms=duration_ms,
        expanded_states=2,
        visited_states=3,
        depth_reached=len(moves),
    )


def _fixed_timer():
    current = -0.001

    def timer() -> float:
        nonlocal current
        current += 0.001
        return current

    return timer


if __name__ == "__main__":
    unittest.main()

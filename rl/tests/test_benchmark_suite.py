import json
from pathlib import Path
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from rubic_rl.evaluation.benchmark_suite import (
    WeightedAStarBenchmarkConfig,
    load_benchmark_cases,
    run_benchmark_suite,
    write_benchmark_cases,
)
from rubic_rl.evaluation.search_eval import BenchmarkCase, SearchBenchmarkConfig

if torch is not None:
    from rubic_rl.models.policy_value import (
        CheckpointMetadata,
        RubiksPolicyValueNet,
        TorchModelConfig,
        save_policy_value_checkpoint,
    )


class SolvedPolicy:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(features.shape[0], dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((features.shape[0], 18), dtype=np.float32)
        probabilities[:, 0] = 1.0
        return probabilities


class BenchmarkSuitePersistenceTests(unittest.TestCase):
    def test_writes_and_loads_benchmark_cases(self):
        path = Path("tests/generated/benchmark-suite/cases.json")
        config = SearchBenchmarkConfig(depths=(1,), samples_per_depth=1, seed=123)
        cases = (BenchmarkCase(depth=1, sample_index=0, seed=123, scramble=("U",)),)

        write_benchmark_cases(path, suite_name="smoke", config=config, cases=cases)
        loaded = load_benchmark_cases(path)

        self.assertEqual(loaded, cases)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["suite"], "smoke")
        self.assertEqual(payload["config"]["depths"], [1])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BenchmarkSuiteRunnerTests(unittest.TestCase):
    def test_runs_policy_and_davi_entrants_on_shared_cases(self):
        root = Path("tests/generated/benchmark-suite")
        checkpoint = root / "solved-davi.pt"
        torch.manual_seed(31)
        model = RubiksPolicyValueNet(TorchModelConfig(hidden_dim=16, residual_blocks=1))
        save_policy_value_checkpoint(
            checkpoint,
            model=model,
            metadata=CheckpointMetadata(model_version="suite-test", value_scale=1.0),
        )

        config = SearchBenchmarkConfig(
            depths=(0,),
            samples_per_depth=1,
            max_depth=1,
            beam_width=1,
            top_k=1,
            seed=123,
        )
        cases = (BenchmarkCase(depth=0, sample_index=0, seed=123, scramble=()),)

        report = run_benchmark_suite(
            suite_name="smoke",
            config=config,
            cases=cases,
            policy_models=(("policy", checkpoint, "torch"),),
            davi_models=(("davi", checkpoint),),
            weighted_astar_config=WeightedAStarBenchmarkConfig(
                weight=0.6,
                policy_weight=0.0,
                batch_expansion=1,
                max_nodes=10,
            ),
            include_episodes=True,
        )

        self.assertEqual(report["benchmark"], "rl-benchmark-suite")
        self.assertEqual(len(report["cases"]), 1)
        self.assertEqual(
            {entry["label"] for entry in report["entrants"]},
            {"policy:greedy", "policy:beam_search", "davi"},
        )
        self.assertEqual(len(report["ranking"]), 3)
        for entry in report["entrants"]:
            self.assertEqual(entry["summary"]["overall"]["solve_rate"], 1.0)
            self.assertEqual(entry["episodes"][0]["scramble"], [])


if __name__ == "__main__":
    unittest.main()

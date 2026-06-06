import json
from pathlib import Path
import unittest

from rubic_rl.experiments import run_experiment_sweep


class ExperimentSweepTests(unittest.TestCase):
    def test_runs_sweep_and_ranks_results(self):
        root = Path("tests/generated/experiment-sweep")
        config_dir = root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        incumbent_config = {
            "name": "incumbent-linear",
            "type": "linear",
            "config": {
                "depths": [1],
                "samples_per_depth": 2,
                "canonical_depths": [1],
                "include_solved": True,
                "seed": 101,
                "evaluation_samples_per_depth": 1,
                "evaluation_max_steps": 1,
                "linear_config": {
                    "epochs": 1,
                    "learning_rate": 0.1,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 101,
                },
            },
            "benchmark": {
                "suite": "tiny",
                "config": {
                    "depths": [1],
                    "samples_per_depth": 1,
                    "max_depth": 1,
                    "beam_width": 1,
                    "top_k": 1,
                    "seed": 101,
                },
            },
        }
        incumbent_path = config_dir / "incumbent-linear.json"
        incumbent_path.write_text(json.dumps(incumbent_config, indent=2), encoding="utf-8")
        run_experiment_sweep(
            [incumbent_path],
            sweep_name="seed-incumbent",
            project_root=root,
            summary_out=root / "reports" / "seed-incumbent.json",
        )

        first_config = {
            "name": "sweep-linear-a",
            "type": "linear",
            "config": {
                "depths": [1],
                "samples_per_depth": 2,
                "canonical_depths": [1],
                "include_solved": True,
                "seed": 111,
                "evaluation_samples_per_depth": 1,
                "evaluation_max_steps": 1,
                "linear_config": {
                    "epochs": 1,
                    "learning_rate": 0.1,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 111,
                },
            },
            "benchmark": {
                "suite": "tiny",
                "config": {
                    "depths": [1],
                    "samples_per_depth": 1,
                    "max_depth": 1,
                    "beam_width": 1,
                    "top_k": 1,
                    "seed": 111,
                },
                "compare_runs": [
                    {
                        "run": "incumbent-linear",
                        "type": "linear",
                        "label_prefix": "incumbent",
                    }
                ],
            },
        }
        second_config = {
            "name": "sweep-linear-b",
            "type": "linear",
            "config": {
                "depths": [1],
                "samples_per_depth": 2,
                "canonical_depths": [1],
                "include_solved": True,
                "seed": 112,
                "evaluation_samples_per_depth": 1,
                "evaluation_max_steps": 1,
                "linear_config": {
                    "epochs": 1,
                    "learning_rate": 0.1,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 112,
                },
            },
            "benchmark": {
                "suite": "tiny",
                "config": {
                    "depths": [1],
                    "samples_per_depth": 1,
                    "max_depth": 1,
                    "beam_width": 1,
                    "top_k": 1,
                    "seed": 112,
                },
            },
        }
        first_path = config_dir / "sweep-linear-a.json"
        second_path = config_dir / "sweep-linear-b.json"
        first_path.write_text(json.dumps(first_config, indent=2), encoding="utf-8")
        second_path.write_text(json.dumps(second_config, indent=2), encoding="utf-8")

        summary = run_experiment_sweep(
            [first_path, second_path],
            sweep_name="linear-sweep-test",
            project_root=root,
        )

        summary_path = root / "reports" / "linear-sweep-test" / "summary.json"
        self.assertTrue(summary_path.exists())
        self.assertEqual(summary["name"], "linear-sweep-test")
        self.assertEqual(len(summary["runs"]), 2)
        self.assertEqual(summary["best_run"]["sweep_rank"], 1)
        self.assertEqual(summary["runs"][0]["name"], summary["best_run"]["name"])
        self.assertEqual(summary["runs"][0]["manifest_path"], summary["best_run"]["manifest_path"])


if __name__ == "__main__":
    unittest.main()

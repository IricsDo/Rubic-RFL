import json
from pathlib import Path
import unittest

from rubic_rl.policies import MLPTrainingConfig, TrainingConfig, load_policy
from rubic_rl.training.baseline_experiment import (
    BaselineExperimentConfig,
    run_baseline_experiment,
)


class BaselineExperimentTests(unittest.TestCase):
    def test_run_baseline_experiment_writes_outputs_and_report(self):
        root = Path("tests/generated/baseline-experiment")
        root.mkdir(parents=True, exist_ok=True)
        config = BaselineExperimentConfig(
            dataset_out=root / "dataset.jsonl",
            linear_model_out=root / "linear.npz",
            mlp_model_out=root / "mlp.npz",
            report_out=root / "report.json",
            depths=(1, 2),
            samples_per_depth=4,
            canonical_depths=(1,),
            include_solved=True,
            seed=123,
            evaluation_samples_per_depth=2,
            evaluation_max_steps=1,
            linear_config=TrainingConfig(
                epochs=2,
                learning_rate=0.1,
                batch_size=2,
                validation_split=0.0,
                seed=123,
            ),
            mlp_config=MLPTrainingConfig(
                hidden_units=8,
                epochs=2,
                learning_rate=0.05,
                batch_size=2,
                validation_split=0.0,
                seed=123,
            ),
        )

        report = run_baseline_experiment(config)

        self.assertTrue(config.dataset_out.exists())
        self.assertTrue(config.linear_model_out.exists())
        self.assertTrue(config.mlp_model_out.exists())
        self.assertTrue(config.report_out.exists())
        self.assertEqual(report["dataset"]["records"], 23)
        self.assertEqual(report["dataset"]["canonical_records"], 18)
        self.assertEqual(report["dataset"]["random_records"], 8)
        self.assertEqual(report["dataset"]["deduplicated_records"], 4)
        self.assertEqual(report["dataset"]["trainable_records"], 22)
        self.assertEqual(report["dataset"]["canonical_depths"], [1])
        self.assertEqual(report["comparison"]["config"]["depths"], [1, 2])
        self.assertEqual(
            {entry["label"] for entry in report["comparison"]["ranking"]},
            {"linear", "mlp"},
        )
        self.assertFalse(
            report["backend"]["environment"]["RUBIC_RL_MODEL_PATH"].startswith("..")
        )
        self.assertTrue(
            report["backend"]["environment"]["RUBIC_RL_MODEL_PATH"].endswith(".npz")
        )
        self.assertEqual(
            report["backend"]["recommended_label"],
            report["comparison"]["ranking"][0]["label"],
        )

        with config.report_out.open("r", encoding="utf-8") as report_file:
            self.assertEqual(json.load(report_file), report)
        self.assertEqual(load_policy(config.linear_model_out).action_count, 18)
        self.assertEqual(load_policy(config.mlp_model_out).action_count, 18)

    def test_run_baseline_experiment_can_skip_mlp_training(self):
        root = Path("tests/generated/linear-experiment")
        root.mkdir(parents=True, exist_ok=True)
        config = BaselineExperimentConfig(
            dataset_out=root / "dataset.jsonl",
            linear_model_out=root / "linear.npz",
            mlp_model_out=None,
            report_out=root / "report.json",
            depths=(1,),
            samples_per_depth=2,
            canonical_depths=(1,),
            include_solved=True,
            seed=321,
            evaluation_samples_per_depth=1,
            evaluation_max_steps=1,
            linear_config=TrainingConfig(
                epochs=1,
                learning_rate=0.1,
                batch_size=2,
                validation_split=0.0,
                seed=321,
            ),
            train_mlp=False,
        )

        report = run_baseline_experiment(config)

        self.assertEqual(report["experiment"], "supervised-linear")
        self.assertEqual(set(report["training"]), {"linear"})
        self.assertEqual(
            {entry["label"] for entry in report["comparison"]["ranking"]},
            {"linear"},
        )
        self.assertEqual(report["backend"]["recommended_label"], "linear")
        self.assertFalse((root / "mlp.npz").exists())


if __name__ == "__main__":
    unittest.main()

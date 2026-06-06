import json
from pathlib import Path
import unittest

from rubic_rl.experiments import (
    build_default_output_paths,
    load_experiment_spec,
    run_experiment,
)
from rubic_rl.experiments.runner import ExperimentSpec
from rubic_rl.evaluation.benchmark_suite import (
    generate_suite_cases,
    write_benchmark_cases,
)
from rubic_rl.evaluation.search_eval import SearchBenchmarkConfig


class ExperimentRunnerTests(unittest.TestCase):
    def test_loads_json_experiment_spec(self):
        spec = load_experiment_spec(
            Path(__file__).resolve().parents[1] / "experiments" / "baseline-smoke.json"
        )

        self.assertEqual(spec.name, "baseline-smoke")
        self.assertEqual(spec.experiment_type, "baseline")
        self.assertIn("linear_config", spec.config)
        self.assertIn("mlp_config", spec.config)
        self.assertEqual(spec.benchmark["suite"], "smoke")

    def test_builds_default_output_paths_for_each_experiment_type(self):
        root = Path("tests/generated/experiment-paths")

        baseline = build_default_output_paths("baseline", run_name="demo", project_root=root)
        linear = build_default_output_paths("linear", run_name="demo", project_root=root)
        adi = build_default_output_paths("adi", run_name="demo", project_root=root)
        davi = build_default_output_paths("davi", run_name="demo", project_root=root)

        self.assertTrue(baseline["dataset_out"].endswith("datasets\\demo\\dataset.jsonl"))
        self.assertTrue(baseline["linear_model_out"].endswith("checkpoints\\demo\\linear-policy.npz"))
        self.assertTrue(linear["linear_model_out"].endswith("checkpoints\\demo\\linear-policy.npz"))
        self.assertNotIn("mlp_model_out", linear)
        self.assertTrue(adi["checkpoint_out"].endswith("checkpoints\\demo\\torch-policy-value.pt"))
        self.assertTrue(davi["checkpoint_out"].endswith("checkpoints\\demo\\torch-value.pt"))
        self.assertTrue(davi["resume_from"].endswith("torch-value.pt.resume"))
        self.assertTrue(baseline["manifest_out"].endswith("reports\\demo\\manifest.json"))

    def test_runs_small_baseline_experiment_and_writes_manifest(self):
        root = Path("tests/generated/experiment-runner")
        spec = ExperimentSpec(
            name="baseline-runner-test",
            experiment_type="baseline",
            description="tiny baseline smoke run",
            config={
                "depths": [1, 2],
                "samples_per_depth": 4,
                "canonical_depths": [1],
                "include_solved": True,
                "seed": 123,
                "evaluation_samples_per_depth": 2,
                "evaluation_max_steps": 1,
                "linear_config": {
                    "epochs": 2,
                    "learning_rate": 0.1,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 123,
                },
                "mlp_config": {
                    "hidden_units": 8,
                    "epochs": 2,
                    "learning_rate": 0.05,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 123,
                },
            },
            benchmark={
                "suite": "tiny",
                "config": {
                    "depths": [1],
                    "samples_per_depth": 1,
                    "max_depth": 1,
                    "beam_width": 1,
                    "top_k": 1,
                    "seed": 123,
                },
            },
        )

        result = run_experiment(spec, project_root=root)

        manifest_path = Path(result["manifest_path"])
        report = result["report"]
        benchmark = result["benchmark"]

        self.assertTrue(manifest_path.exists())
        self.assertEqual(report["experiment"], "supervised-baseline")
        self.assertEqual(result["outputs"]["dataset_out"], str(root / "datasets" / spec.name / "dataset.jsonl"))
        self.assertEqual(
            result["outputs"]["linear_model_out"],
            str(root / "checkpoints" / spec.name / "linear-policy.npz"),
        )
        self.assertTrue(Path(result["outputs"]["benchmark_report_out"]).exists())
        self.assertTrue(Path(result["outputs"]["benchmark_cases_out"]).exists())
        self.assertEqual(benchmark["suite"], "tiny")
        self.assertEqual(len(benchmark["entrants"]), 4)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], spec.name)
        self.assertEqual(manifest["type"], "baseline")
        self.assertEqual(
            manifest["headline_metrics"]["recommended_label"],
            report["backend"]["recommended_label"],
        )
        self.assertEqual(
            manifest["outputs"]["mlp_model_out"],
            str(root / "checkpoints" / spec.name / "mlp-policy.npz"),
        )
        self.assertIn("benchmark_summary", manifest)
        self.assertIn("promotion_gate", manifest)
        self.assertEqual(manifest["benchmark"]["suite"], "tiny")

    def test_benchmark_compare_runs_include_existing_references_and_skip_missing(self):
        root = Path("tests/generated/experiment-runner-references")

        incumbent_spec = ExperimentSpec(
            name="incumbent-baseline",
            experiment_type="baseline",
            config={
                "depths": [1],
                "samples_per_depth": 2,
                "canonical_depths": [1],
                "include_solved": True,
                "seed": 321,
                "evaluation_samples_per_depth": 1,
                "evaluation_max_steps": 1,
                "linear_config": {
                    "epochs": 1,
                    "learning_rate": 0.1,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 321,
                },
                "mlp_config": {
                    "hidden_units": 4,
                    "epochs": 1,
                    "learning_rate": 0.05,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 321,
                },
            },
        )
        run_experiment(incumbent_spec, project_root=root)

        spec = ExperimentSpec(
            name="lightweight-reference-test",
            experiment_type="baseline",
            config={
                "depths": [1],
                "samples_per_depth": 2,
                "canonical_depths": [1],
                "include_solved": True,
                "seed": 123,
                "evaluation_samples_per_depth": 1,
                "evaluation_max_steps": 1,
                "linear_config": {
                    "epochs": 1,
                    "learning_rate": 0.1,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 123,
                },
                "mlp_config": {
                    "hidden_units": 4,
                    "epochs": 1,
                    "learning_rate": 0.05,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 123,
                },
            },
            benchmark={
                "suite": "tiny",
                "config": {
                    "depths": [1],
                    "samples_per_depth": 1,
                    "max_depth": 1,
                    "beam_width": 1,
                    "top_k": 1,
                    "seed": 123,
                },
                "compare_runs": [
                    {
                        "run": "incumbent-baseline",
                        "type": "baseline",
                        "label_prefix": "incumbent",
                    },
                    {
                        "run": "missing-baseline",
                        "type": "baseline",
                        "label_prefix": "missing",
                    },
                ],
            },
        )

        result = run_experiment(spec, project_root=root)
        labels = {entrant["label"] for entrant in result["benchmark"]["entrants"]}
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

        self.assertIn("linear:greedy", labels)
        self.assertIn("mlp:beam_search", labels)
        self.assertIn("incumbent:linear:greedy", labels)
        self.assertIn("incumbent:mlp:beam_search", labels)
        self.assertFalse(any(label.startswith("missing:") for label in labels))
        self.assertTrue(manifest["promotion_gate"]["eligible"])
        self.assertTrue(
            manifest["promotion_gate"]["candidate_label"].startswith(("linear:", "mlp:"))
        )
        self.assertTrue(
            manifest["promotion_gate"]["reference_label"].startswith("incumbent:")
        )
        self.assertTrue(manifest["promotion_gate"]["beats_reference"])

    def test_runs_small_linear_experiment_without_mlp_outputs(self):
        root = Path("tests/generated/experiment-runner-linear")
        spec = ExperimentSpec(
            name="linear-runner-test",
            experiment_type="linear",
            description="tiny linear-only smoke run",
            config={
                "depths": [1],
                "samples_per_depth": 2,
                "canonical_depths": [1],
                "include_solved": True,
                "seed": 456,
                "evaluation_samples_per_depth": 1,
                "evaluation_max_steps": 1,
                "linear_config": {
                    "epochs": 1,
                    "learning_rate": 0.1,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 456,
                },
            },
            benchmark={
                "suite": "tiny",
                "config": {
                    "depths": [1],
                    "samples_per_depth": 1,
                    "max_depth": 1,
                    "beam_width": 1,
                    "top_k": 1,
                    "seed": 456,
                },
            },
        )

        result = run_experiment(spec, project_root=root)
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

        self.assertEqual(result["report"]["experiment"], "supervised-linear")
        self.assertEqual(set(result["report"]["training"]), {"linear"})
        self.assertIsNone(result["outputs"]["mlp_model_out"])
        self.assertEqual(manifest["type"], "linear")
        self.assertEqual(manifest["headline_metrics"]["recommended_label"], "linear")

    def test_benchmark_can_load_held_out_cases_from_file(self):
        root = Path("tests/generated/experiment-runner-heldout")
        cases_path = root / "benchmarks" / "heldout.json"
        heldout_config = SearchBenchmarkConfig(
            depths=(1,),
            samples_per_depth=1,
            max_depth=1,
            beam_width=1,
            top_k=1,
            seed=987,
        )
        write_benchmark_cases(
            cases_path,
            suite_name="heldout-tiny",
            config=heldout_config,
            cases=generate_suite_cases(heldout_config),
        )
        spec = ExperimentSpec(
            name="linear-heldout-test",
            experiment_type="linear",
            config={
                "depths": [1],
                "samples_per_depth": 2,
                "canonical_depths": [1],
                "include_solved": True,
                "seed": 789,
                "evaluation_samples_per_depth": 1,
                "evaluation_max_steps": 1,
                "linear_config": {
                    "epochs": 1,
                    "learning_rate": 0.1,
                    "batch_size": 2,
                    "validation_split": 0.0,
                    "seed": 789,
                },
            },
            benchmark={
                "config": {
                    "depths": [1],
                    "samples_per_depth": 1,
                    "max_depth": 1,
                    "beam_width": 1,
                    "top_k": 1,
                    "seed": 987,
                },
                "cases_in": "benchmarks/heldout.json",
            },
        )

        expected_cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
        result = run_experiment(spec, project_root=root)
        benchmark = result["benchmark"]

        self.assertEqual(benchmark["suite"], "custom")
        self.assertEqual(len(benchmark["cases"]), 1)
        self.assertEqual(benchmark["cases"], expected_cases)


if __name__ == "__main__":
    unittest.main()

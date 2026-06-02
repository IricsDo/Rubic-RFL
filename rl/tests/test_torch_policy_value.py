from pathlib import Path
import unittest
from unittest.mock import patch

from rubic_rl.datasets import DatasetRecord

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from rubic_rl.models.policy_value import (
        CheckpointMetadata,
        RubiksPolicyValueNet,
        TorchModelConfig,
        load_policy_value_checkpoint,
        save_policy_value_checkpoint,
    )
    from rubic_rl.training.adi import ADITrainingConfig, run_adi_training


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchPolicyValueTests(unittest.TestCase):
    def test_forward_returns_policy_logits_and_value(self):
        config = TorchModelConfig(hidden_dim=16, residual_blocks=1)
        model = RubiksPolicyValueNet(config)

        policy_logits, values = model(torch.zeros((4, config.input_size)))

        self.assertEqual(tuple(policy_logits.shape), (4, 18))
        self.assertEqual(tuple(values.shape), (4,))

    def test_checkpoint_roundtrip_preserves_metadata_and_predictions(self):
        root = Path("tests/generated/adi-training")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "roundtrip.pt"
        torch.manual_seed(123)
        config = TorchModelConfig(hidden_dim=16, residual_blocks=1)
        model = RubiksPolicyValueNet(config)
        features = torch.zeros((2, config.input_size))
        expected_logits, expected_values = model(features)

        save_policy_value_checkpoint(
            path,
            model=model,
            metadata=CheckpointMetadata(
                model_version="test-v1",
                value_scale=3.0,
                training={"seed": 123},
            ),
        )
        loaded, metadata = load_policy_value_checkpoint(path)
        actual_logits, actual_values = loaded(features)

        self.assertEqual(metadata.model_version, "test-v1")
        self.assertEqual(metadata.value_scale, 3.0)
        torch.testing.assert_close(actual_logits, expected_logits)
        torch.testing.assert_close(actual_values, expected_values)

    def test_run_adi_training_writes_checkpoint_and_report(self):
        root = Path("tests/generated/adi-training")
        root.mkdir(parents=True, exist_ok=True)
        config = ADITrainingConfig(
            dataset_out=root / "dataset.jsonl",
            checkpoint_out=root / "policy-value.pt",
            report_out=root / "report.json",
            depths=(1,),
            samples_per_depth=2,
            canonical_depths=(1,),
            include_solved=True,
            iterations=1,
            epochs_per_iteration=1,
            batch_size=8,
            validation_split=0.0,
            hidden_dim=16,
            residual_blocks=1,
            seed=123,
        )

        report = run_adi_training(config)

        self.assertTrue(config.dataset_out.exists())
        self.assertTrue(config.checkpoint_out.exists())
        self.assertTrue(config.report_out.exists())
        self.assertEqual(report["experiment"], "adi-policy-value")
        self.assertEqual(report["dataset"]["value_scale"], 1.0)
        self.assertGreater(report["dataset"]["records"], 0)
        self.assertEqual(len(report["training"]["iterations"]), 1)
        loaded, metadata = load_policy_value_checkpoint(config.checkpoint_out)
        self.assertEqual(loaded.config.hidden_dim, 16)
        self.assertEqual(metadata.model_version, "torch-policy-value-v0.1")

    def test_run_adi_training_uses_cumulative_records_by_default(self):
        root = Path("tests/generated/adi-training")
        root.mkdir(parents=True, exist_ok=True)
        batches = [
            [_dataset_record("iter1-a", "stickers-a", sample_index=0)],
            [_dataset_record("iter2-b", "stickers-b", sample_index=1)],
        ]
        tensor_record_counts: list[int] = []

        def fake_records_to_tensors(records, *, value_scale, device):
            tensor_record_counts.append(len(records))
            return {
                "policy_mask": torch.ones(
                    len(records),
                    dtype=torch.bool,
                    device=device,
                )
            }

        def fake_fit_iteration(
            *,
            model,
            optimizer,
            tensors,
            train_indices,
            validation_indices,
            config,
            rng,
        ):
            return {
                "epochs": config.epochs_per_iteration,
                "train": {"samples": int(len(train_indices))},
                "validation": None,
            }

        config = ADITrainingConfig(
            dataset_out=None,
            checkpoint_out=root / "cumulative.pt",
            report_out=None,
            depths=(1,),
            samples_per_depth=1,
            canonical_depths=(),
            include_solved=False,
            iterations=2,
            epochs_per_iteration=1,
            batch_size=8,
            validation_split=0.0,
            hidden_dim=16,
            residual_blocks=1,
            seed=123,
        )

        with (
            patch(
                "rubic_rl.training.adi._generate_iteration_records",
                side_effect=lambda _config, iteration: batches[iteration],
            ),
            patch(
                "rubic_rl.training.adi._records_to_tensors",
                side_effect=fake_records_to_tensors,
            ),
            patch(
                "rubic_rl.training.adi._fit_iteration",
                side_effect=fake_fit_iteration,
            ),
        ):
            report = run_adi_training(config)

        self.assertEqual(tensor_record_counts, [1, 2])
        self.assertTrue(report["training"]["config"]["train_on_cumulative_records"])
        iterations = report["training"]["iterations"]
        self.assertEqual(iterations[0]["generated_records"], 1)
        self.assertEqual(iterations[0]["training_records"], 1)
        self.assertEqual(iterations[1]["generated_records"], 1)
        self.assertEqual(iterations[1]["training_records"], 2)
        self.assertEqual(iterations[1]["cumulative_records"], 2)


def _dataset_record(
    record_id: str,
    stickers: str,
    *,
    sample_index: int,
) -> DatasetRecord:
    return DatasetRecord(
        record_id=record_id,
        stickers=stickers,
        sticker_indices=tuple(0 for _ in range(54)),
        scramble=("U",),
        depth=1,
        target_moves=("U'",),
        target_action=1,
        is_solved=False,
        seed=123,
        sample_index=sample_index,
    )


if __name__ == "__main__":
    unittest.main()

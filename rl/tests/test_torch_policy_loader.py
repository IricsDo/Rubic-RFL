from pathlib import Path
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from rubic_rl.models.policy_value import (
        CheckpointMetadata,
        RubiksPolicyValueNet,
        TorchModelConfig,
        save_policy_value_checkpoint,
    )
    from rubic_rl.policies import load_policy
    from rubic_rl.policies.torch_policy import TorchPolicyValuePolicy

from rubic_rl.policies.loaders import _detect_policy_type


class TorchPolicyDetectionTests(unittest.TestCase):
    def test_auto_detection_identifies_torch_checkpoint_extension(self):
        self.assertEqual(_detect_policy_type(Path("checkpoint.pt")), "torch")
        self.assertEqual(_detect_policy_type(Path("checkpoint.pth")), "torch")


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchPolicyLoaderTests(unittest.TestCase):
    def test_load_policy_auto_detects_torch_checkpoint(self):
        root = Path("tests/generated/adi-training")
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "loader-auto.pt"
        torch.manual_seed(31)
        model = RubiksPolicyValueNet(TorchModelConfig(hidden_dim=16, residual_blocks=1))
        save_policy_value_checkpoint(
            checkpoint,
            model=model,
            metadata=CheckpointMetadata(model_version="loader-test", value_scale=4.0),
        )

        policy = load_policy(checkpoint)
        probabilities = policy.predict_proba(np.zeros((2, model.config.input_size)))
        actions = policy.predict(np.zeros((2, model.config.input_size)))
        values = policy.predict_value(np.zeros((2, model.config.input_size)))

        self.assertIsInstance(policy, TorchPolicyValuePolicy)
        self.assertEqual(policy.model_version, "loader-test")
        self.assertEqual(probabilities.shape, (2, 18))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(values.shape, (2,))


if __name__ == "__main__":
    unittest.main()

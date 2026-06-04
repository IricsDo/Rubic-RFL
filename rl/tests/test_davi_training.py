import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from rubic_rl.training.davi import DAVIConfig, _loss, _sample_training_states


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DAVITrainingTests(unittest.TestCase):
    def test_hard_depth_fraction_samples_only_current_depth_when_full(self):
        config = DAVIConfig(
            batch_size=12,
            min_scramble_depth=1,
            max_scramble_depth=30,
            hard_depth_fraction=1.0,
        )

        _states, depths = _sample_training_states(
            config, depth=17, rng=np.random.default_rng(123)
        )

        self.assertEqual(depths.tolist(), [17] * 12)

    def test_smooth_l1_loss_option_is_available(self):
        config = DAVIConfig(loss_type="smooth_l1", huber_delta=1.0)
        predicted = torch.tensor([0.0, 2.0, 4.0])
        target = torch.tensor([0.0, 1.0, 7.0])

        actual = _loss(predicted, target, config)
        expected = torch.nn.functional.smooth_l1_loss(predicted, target, beta=1.0)

        torch.testing.assert_close(actual, expected)

    def test_rejects_invalid_hard_depth_fraction(self):
        with self.assertRaises(ValueError):
            DAVIConfig(hard_depth_fraction=1.5)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
from torch import nn

from calibration_net.policy.baseline_finetune import (
    RidgeCorrectionBaseline,
    adapt_model,
    configure_baseline_model,
)


class RidgeBaselineTests(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def test_identity_prior_and_unpenalized_bias(self):
        model = RidgeCorrectionBaseline(nn.Identity(), 2, alpha=10.0)
        offset = torch.tensor([0.25, -0.75])
        target = self.inputs + offset
        losses = model.fit_closed_form(self.inputs, target)
        torch.testing.assert_close(model.matrix, torch.eye(2))
        torch.testing.assert_close(model.bias, offset)
        torch.testing.assert_close(model.predict(self.inputs), target)
        self.assertLess(losses[-1], losses[0])

    def test_factory_and_closed_form_adaptation(self):
        base = nn.Linear(2, 2)
        model = configure_baseline_model(
            base,
            "ridge",
            target_modules=[],
            input_shape=(2,),
            output_size=2,
            ridge_alpha=1e-3,
        )
        self.assertIsInstance(model, RidgeCorrectionBaseline)
        self.assertFalse(
            any(parameter.requires_grad for parameter in base.parameters())
        )
        target = model.predict(self.inputs).detach() + 0.5
        losses, elapsed = adapt_model(
            model, self.inputs, target, steps=200, lr=1e-4, weight_decay=0.0
        )
        self.assertEqual(len(losses), 2)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertFalse(model.training)
        torch.testing.assert_close(model.predict(self.inputs), target)

    def test_rejects_negative_regularization(self):
        with self.assertRaises(ValueError):
            RidgeCorrectionBaseline(nn.Identity(), 2, alpha=-1.0)

    def test_rejects_mismatched_output_dimensions(self):
        model = RidgeCorrectionBaseline(nn.Identity(), 2, alpha=1e-3)
        with self.assertRaises(ValueError):
            model.fit_closed_form(self.inputs, torch.zeros(4, 3))


if __name__ == "__main__":
    unittest.main()

"""Regression tests for optional bounded AdaLN modulation."""

import os
import sys
import unittest

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model.model_modules import AddAdaZeroLayerNorm


BOUNDS = {"gamma": 0.1, "beta": 0.1, "alpha": 1.0}


class AdaLNModulationBoundsTests(unittest.TestCase):
    def test_none_preserves_legacy_unbounded_forward(self):
        layer = AddAdaZeroLayerNorm(
            d_model=2,
            p_drop=0.0,
            adaptive_dim=1,
            adaln_modulation_bounds=None,
        )
        with torch.no_grad():
            layer.adaLN_modulation[1].weight.zero_()
            layer.adaLN_modulation[1].bias.copy_(
                torch.tensor([0.5, 0.5, 0.25, 0.25, 2.0, 2.0])
            )

        inputs = torch.tensor([[[1.0, 3.0], [2.0, 6.0]]])
        style = torch.zeros(1, 1)
        normalized = layer.LN(inputs)
        expected = inputs + 2.0 * (1.5 * normalized + 0.25)
        actual = layer(inputs, lambda value: value, style)

        self.assertTrue(torch.allclose(actual, expected))

    def test_configured_bounds_limit_all_modulation_components(self):
        layer = AddAdaZeroLayerNorm(
            d_model=4,
            p_drop=0.0,
            adaptive_dim=2,
            adaln_modulation_bounds=BOUNDS,
        )
        values = torch.tensor([-100.0, -0.01, 0.0, 0.01, 100.0])

        gamma = layer._smooth_bound(values, layer.adaln_modulation_bounds[0])
        beta = layer._smooth_bound(values, layer.adaln_modulation_bounds[1])
        alpha = layer._smooth_bound(values, layer.adaln_modulation_bounds[2])

        self.assertLessEqual(torch.max(torch.abs(gamma)), BOUNDS["gamma"])
        self.assertLessEqual(torch.max(torch.abs(beta)), BOUNDS["beta"])
        self.assertLessEqual(torch.max(torch.abs(alpha)), BOUNDS["alpha"])

    def test_bounds_do_not_change_checkpoint_parameter_keys(self):
        legacy = AddAdaZeroLayerNorm(4, 0.0, 2, adaln_modulation_bounds=None)
        bounded = AddAdaZeroLayerNorm(4, 0.0, 2, adaln_modulation_bounds=BOUNDS)

        self.assertEqual(legacy.state_dict().keys(), bounded.state_dict().keys())
        bounded.load_state_dict(legacy.state_dict(), strict=True)

    def test_invalid_bounds_are_rejected(self):
        invalid_bounds = [
            {"gamma": 0.1, "beta": 0.1},
            {"gamma": 0.1, "beta": 0.1, "alpha": 0.0},
            {"gamma": 0.1, "beta": 0.1, "alpha": 1.0, "extra": 1.0},
        ]
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds):
                with self.assertRaises((TypeError, ValueError)):
                    AddAdaZeroLayerNorm(
                        4, 0.0, 2, adaln_modulation_bounds=bounds
                    )


if __name__ == "__main__":
    unittest.main()

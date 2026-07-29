"""Regression tests for sequence-only LN and convolutional ablation models."""

import os
import sys
import unittest

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model.prediction_heads import PsiteDensityHead
from model.translation_base_model_LN import BaseModelLN
from model.translation_base_model_conv import BaseModelConv


def _models():
    return (
        BaseModelLN(
            d_seq=4,
            d_model=12,
            n_heads=3,
            number_of_layers=2,
            d_ff=24,
            p_drop=0.0,
        ),
        BaseModelConv(
            d_seq=4,
            d_model=12,
            number_of_layers=2,
            d_ff=12,
            kernel_size=7,
            p_drop=0.0,
        ),
    )


class SequenceOnlyAblationModelTests(unittest.TestCase):
    def test_density_head_initializes_relu_output_in_positive_region(self):
        torch.manual_seed(19)
        head = PsiteDensityHead(
            d_model=64,
            d_count=1,
            d_pred_h=256,
            p_drop=0.0,
        )
        output_layer = head.net[-2]

        self.assertAlmostEqual(float(output_layer.bias.item()), 0.1, places=6)
        self.assertAlmostEqual(
            float(output_layer.weight.std(unbiased=False)), 1e-3, delta=2e-4
        )

        head.eval()
        representations = torch.randn(4, 128, 64)
        predictions = head(representations)
        self.assertTrue(torch.all(predictions > 0))
        self.assertGreater(float(predictions.std()), 0.0)

    def test_state_dict_has_no_environment_parameters_or_buffers(self):
        forbidden_fragments = ("expr", "species", "adaln", "mean_expr")
        for model in _models():
            with self.subTest(model=type(model).__name__):
                keys = tuple(key.lower() for key in model.state_dict())
                for fragment in forbidden_fragments:
                    self.assertFalse(any(fragment in key for key in keys))

    def test_forward_is_invariant_to_environment_arguments(self):
        torch.manual_seed(11)
        sequence = torch.randn(2, 9, 4)
        mask = torch.ones(2, 9, dtype=torch.bool)
        first_expression = torch.zeros(2, 16840)
        second_expression = torch.randn(2, 16840)

        for model in _models():
            model.eval()
            with self.subTest(model=type(model).__name__):
                first = model(
                    sequence,
                    species=["human", "human"],
                    cell_type=["liver", "brain_cerebrum"],
                    expr_vector=first_expression,
                    src_mask=mask,
                )
                second = model(
                    sequence,
                    species=["mouse", "macaque"],
                    cell_type=["HeLa", "HEK293T"],
                    expr_vector=second_expression,
                    src_mask=mask,
                )
                self.assertTrue(torch.equal(first, second))

    def test_prediction_head_and_predict_api_are_compatible(self):
        sequence = torch.randn(2, 8, 4)
        mask = torch.tensor(
            [[True] * 8, [True] * 5 + [False] * 3], dtype=torch.bool
        )

        for model in _models():
            model.add_head(
                "count",
                PsiteDensityHead.create_from_model(
                    model, d_pred_h=8, p_drop=0.0
                ),
            )
            model.eval()
            with self.subTest(model=type(model).__name__):
                result = model.predict(
                    sequence,
                    species=["human", "human"],
                    expr_vector=torch.randn(2, 16840),
                    src_mask=mask,
                    head_names=["count"],
                )
                self.assertEqual(result["count"].shape, (2, 8, 1))
                self.assertTrue(
                    torch.equal(
                        result["count"][1, 5:],
                        torch.zeros_like(result["count"][1, 5:]),
                    )
                )

    def test_base_model_environment_config_fields_are_safely_ignored(self):
        shared_config = {
            "d_seq": 4,
            "d_model": 12,
            "d_expr": 16840,
            "d_cell_env": 64,
            "d_species": 16,
            "adaptive_dim": 16,
            "adaln_modulation_bounds": {
                "gamma": 0.5,
                "beta": 0.5,
                "alpha": 1.0,
            },
            "number_of_layers": 2,
            "d_ff": 24,
            "n_heads": 3,
            "p_drop": 0.0,
        }
        ln_model = BaseModelLN.from_config(shared_config)
        conv_model = BaseModelConv.from_config(
            {**shared_config, "d_ff": 12, "kernel_size": 7}
        )

        self.assertFalse(hasattr(ln_model, "expr_projector"))
        self.assertFalse(hasattr(conv_model, "expr_projector"))


if __name__ == "__main__":
    unittest.main()

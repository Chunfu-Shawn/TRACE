"""Regression tests for the 7-layer Pre-LN plus 5-layer Pre-AdaLN model."""

import os
import sys
import unittest

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model.base_model_hybrid import BaseModelHybrid
from model.model_modules import AdaZeroEncoderLayer, EncoderLayer
from model.prediction_heads import PsiteDensityHead


def create_model():
    return BaseModelHybrid(
        d_seq=4,
        d_model=12,
        d_expr=10,
        d_cell_env=8,
        all_species=["human"],
        d_species=4,
        n_heads=3,
        number_of_layers=12,
        d_ff=24,
        adaptive_dim=6,
        p_drop=0.0,
        adaln_modulation_bounds={"gamma": 0.5, "beta": 0.5, "alpha": 1.0},
    )


class HybridBaseModelTests(unittest.TestCase):
    def test_encoder_has_seven_pre_ln_and_five_pre_adaln_layers(self):
        model = create_model()
        layers = model.encoder.encoder_layers

        self.assertEqual(len(layers), 12)
        self.assertTrue(all(isinstance(layer, EncoderLayer) for layer in layers[:7]))
        self.assertTrue(
            all(isinstance(layer, AdaZeroEncoderLayer) for layer in layers[7:])
        )

    def test_adaln_zero_gates_survive_hybrid_initialization(self):
        model = create_model()
        for layer in model.encoder.encoder_layers[7:]:
            for sublayer in layer.sublayers:
                projection = sublayer.adaLN_modulation[1]
                self.assertTrue(torch.equal(projection.weight, torch.zeros_like(projection.weight)))
                self.assertTrue(torch.equal(projection.bias, torch.zeros_like(projection.bias)))

    def test_forward_and_prediction_head_are_compatible(self):
        model = create_model()
        model.add_head(
            "count",
            PsiteDensityHead.create_from_model(model, d_pred_h=8, p_drop=0.0),
        )
        model.eval()
        sequence = torch.randn(2, 9, 4)
        expression = torch.randn(2, 10)
        mask = torch.tensor(
            [[True] * 9, [True] * 6 + [False] * 3], dtype=torch.bool
        )

        result = model.predict(
            sequence,
            species=["human", "human"],
            expr_vector=expression,
            src_mask=mask,
            head_names=["count"],
        )

        self.assertEqual(result["count"].shape, (2, 9, 1))
        self.assertTrue(
            torch.equal(
                result["count"][1, 6:],
                torch.zeros_like(result["count"][1, 6:]),
            )
        )

    def test_invalid_layer_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "number_of_adaln_layers"):
            BaseModelHybrid(
                d_seq=4,
                d_model=12,
                d_expr=10,
                n_heads=3,
                number_of_layers=10,
                number_of_adaln_layers=11,
            )

    def test_config_controls_number_of_adaln_layers(self):
        model = BaseModelHybrid.from_config(
            {
                "d_seq": 4,
                "d_model": 12,
                "d_expr": 10,
                "d_cell_env": 8,
                "d_species": 4,
                "all_species": ["human"],
                "n_heads": 3,
                "number_of_layers": 12,
                "number_of_adaln_layers": 3,
                "d_ff": 24,
                "adaptive_dim": 6,
                "p_drop": 0.0,
                "adaln_modulation_bounds": {
                    "gamma": 0.5,
                    "beta": 0.5,
                    "alpha": 1.0,
                },
                "model_name": "hybrid_test",
            }
        )

        self.assertEqual(model.pre_ln_layers, 9)
        self.assertEqual(model.pre_adaln_layers, 3)
        self.assertTrue(
            all(
                isinstance(layer, EncoderLayer)
                for layer in model.encoder.encoder_layers[:9]
            )
        )
        self.assertTrue(
            all(
                isinstance(layer, AdaZeroEncoderLayer)
                for layer in model.encoder.encoder_layers[9:]
            )
        )


if __name__ == "__main__":
    unittest.main()

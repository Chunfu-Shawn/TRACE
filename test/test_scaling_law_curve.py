"""Tests for multi-model Trainer loss-curve parsing and plotting."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from plot.scaling_law_curve import (
    ComputeEstimate,
    estimate_flops_from_lengths,
    load_run_history,
    parse_epoch_json,
    parse_text_log,
    plot_model_loss_curves,
    validate_comparison,
)


class ScalingLawCurveTests(unittest.TestCase):
    def test_transformer_flops_use_linear_and_quadratic_length_terms(self):
        flops = estimate_flops_from_lengths(
            [10, 20],
            {
                "d_seq": 4,
                "d_model": 8,
                "d_ff": 16,
                "number_of_layers": 2,
                "model_name": "base_model_ln",
            },
            head_hidden_dim=8,
        )

        self.assertEqual(flops, 299040)

    def test_current_json_structure_and_resume_duplicates(self):
        payload = [
            {
                "epoch": 1,
                "alpha": 0.2,
                "train_loss": 0.30,
                "valid_loss": 0.28,
                "profile_spearman": 0.20,
                "scale_spearman": 0.50,
            },
            {"epoch": 2, "train_loss": 0.25, "valid_loss": 0.24},
            {"epoch": 2, "train_loss": 0.22, "valid_loss": 0.21},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.epoch_data.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records, duplicates = parse_epoch_json(path)

        self.assertEqual(duplicates, 1)
        self.assertEqual([record["epoch"] for record in records], [1, 2])
        self.assertEqual(records[-1]["valid_loss"], 0.21)

    def test_current_console_log_structure(self):
        text = """
Epoch 4 training time: 120.0 seconds, mean loss: tensor([0.1730], device='cuda:0')
Epoch 4 evaluating time: 10.0 seconds, mean loss: tensor([0.2088], device='cuda:0')
Epoch 4 validation metrics: profile Spearman=0.492527 (100/100 RNAs), CDS-mean scale Spearman=0.719662
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer.log"
            path.write_text(text, encoding="utf-8")
            records, duplicates = parse_text_log(path)

        self.assertEqual(duplicates, 0)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["train_loss"], 0.1730)
        self.assertAlmostEqual(records[0]["valid_loss"], 0.2088)
        self.assertAlmostEqual(records[0]["profile_spearman"], 0.492527)
        self.assertAlmostEqual(records[0]["scale_spearman"], 0.719662)

    def test_multiple_inline_histories_plot_together(self):
        first = load_run_history(
            {
                "method": "TRACE-Zero",
                "dataset": "same-validation-set",
                "color": "#777777",
                "loss_data": [
                    {"epoch": 1, "train_loss": 0.30, "valid_loss": 0.28},
                    {"epoch": 2, "train_loss": 0.25, "valid_loss": 0.24},
                ],
            }
        )
        second = load_run_history(
            {
                "label": "TRACE-Mask+Interpolation",
                "dataset": "same-validation-set",
                "color": "#166A9A",
                "loss_data": [
                    {"epoch": 1, "train_loss": [0.27], "valid_loss": [0.25]},
                    {"epoch": 2, "train_loss": [0.21], "valid_loss": [0.19]},
                ],
            }
        )
        validate_comparison([first, second])
        figures = plot_model_loss_curves([first, second], x_axis="epoch")

        self.assertEqual(set(figures), {"train", "valid"})
        self.assertEqual(len(figures["train"].axes), 1)
        self.assertEqual(len(figures["valid"].axes), 1)
        self.assertEqual(second.best_validation(), (2, 0.19))
        self.assertTrue(np.isfinite(second.valid_loss).all())
        for figure in figures.values():
            plt.close(figure)

    def test_flop_axis_uses_distinct_compute_per_run(self):
        histories = []
        for label, lengths in (("22c", [10, 20]), ("40c", [10, 20, 30])):
            history = load_run_history(
                {
                    "label": label,
                    "dataset": "same-validation-set",
                    "loss_data": [
                        {"epoch": 1, "valid_loss": 0.3},
                        {"epoch": 2, "valid_loss": 0.2},
                    ],
                }
            )
            flops = estimate_flops_from_lengths(
                lengths,
                {"d_model": 8, "d_ff": 16, "number_of_layers": 2},
                head_hidden_dim=8,
            )
            history.compute = ComputeEstimate(
                training_dataset=f"{label}.h5",
                n_transcripts=len(lengths),
                total_length=float(sum(lengths)),
                total_length_squared=float(sum(length**2 for length in lengths)),
                flops_per_epoch=flops,
            )
            histories.append(history)

        figures = plot_model_loss_curves(histories, x_axis="flops", x_log=True)
        valid_axis = figures["valid"].axes[0]
        first_x = valid_axis.lines[0].get_xdata()
        second_x = valid_axis.lines[1].get_xdata()
        self.assertGreater(second_x[-1], first_x[-1])
        self.assertEqual(valid_axis.get_xscale(), "log")
        self.assertIn("EFLOPs", valid_axis.get_xlabel())
        for figure in figures.values():
            plt.close(figure)

    def test_mixed_validation_datasets_are_rejected(self):
        histories = []
        for label, dataset in (("first", "dataset-a"), ("second", "dataset-b")):
            histories.append(
                load_run_history(
                    {
                        "label": label,
                        "dataset": dataset,
                        "loss_data": [{"epoch": 1, "valid_loss": 0.2}],
                    }
                )
            )
        with self.assertRaises(ValueError):
            validate_comparison(histories)


if __name__ == "__main__":
    unittest.main()

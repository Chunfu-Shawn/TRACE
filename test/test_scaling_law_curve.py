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
    load_run_history,
    parse_epoch_json,
    parse_text_log,
    plot_model_loss_curves,
    validate_comparison,
)


class ScalingLawCurveTests(unittest.TestCase):
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
        figure = plot_model_loss_curves([first, second], show_training_panel=True)

        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(second.best_validation(), (2, 0.19))
        self.assertTrue(np.isfinite(second.valid_loss).all())
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

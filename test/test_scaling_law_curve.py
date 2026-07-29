"""Tests for configurable epoch-metric Trainer plots."""

import csv
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
    plot_model_metric_curves,
    validate_comparison,
    write_source_data,
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
                "cds_mean_mae": 0.14,
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
        self.assertAlmostEqual(records[0]["cds_mean_mae"], 0.14)

    def test_current_console_log_structure(self):
        text = """
Epoch 4 training time: 120.0 seconds, mean loss: tensor([0.1730], device='cuda:0')
Epoch 4 evaluating time: 10.0 seconds, mean loss: tensor([0.2088], device='cuda:0')
Epoch 4 validation metrics: profile Spearman=0.492527 (100/100 RNAs), CDS-mean scale Spearman=0.719662, CDS-mean MAE=0.096166, calibration target=0.024713 + 0.944779*prediction (100 RNAs)
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
        self.assertAlmostEqual(records[0]["cds_mean_mae"], 0.096166)
        self.assertAlmostEqual(records[0]["calibration_intercept"], 0.024713)
        self.assertAlmostEqual(records[0]["calibration_slope"], 0.944779)

    def test_custom_metric_figures_and_axis_settings(self):
        histories = [
            load_run_history(
                {
                    "label": "TRACE-Zero",
                    "dataset": "same-validation-set",
                    "color": "#777777",
                    "loss_data": [
                        {
                            "epoch": 1,
                            "train_loss": 0.30,
                            "valid_loss": 0.28,
                            "profile_spearman": 0.32,
                            "scale_spearman": 0.50,
                            "cds_mean_mae": 0.12,
                        },
                        {
                            "epoch": 2,
                            "train_loss": 0.25,
                            "valid_loss": 0.24,
                            "profile_spearman": 0.35,
                            "scale_spearman": 0.54,
                            "cds_mean_mae": 0.10,
                        },
                    ],
                }
            ),
            load_run_history(
                {
                    "label": "TRACE-Mask+Interpolation",
                    "dataset": "same-validation-set",
                    "color": "#166A9A",
                    "loss_data": [
                        {
                            "epoch": 1,
                            "train_loss": [0.27],
                            "valid_loss": [0.25],
                            "profile_spearman": 0.36,
                            "scale_spearman": 0.56,
                            "cds_mean_mae": 0.11,
                        },
                        {
                            "epoch": 2,
                            "train_loss": [0.21],
                            "valid_loss": [0.19],
                            "profile_spearman": 0.40,
                            "scale_spearman": 0.61,
                            "cds_mean_mae": 0.08,
                        },
                    ],
                }
            ),
        ]
        metric_configs = [
            {
                "key": "valid_loss",
                "title": "Validation loss",
                "y_label": "Loss",
                "filename": "valid_loss",
                "y_log": True,
                "best": "min",
            },
            {
                "key": "cds_mean_mae",
                "title": "CDS-mean MAE",
                "y_label": "Mean absolute error",
                "filename": "cds_mean_mae",
                "y_log": False,
                "best": "min",
            },
            {
                "key": "scale_spearman",
                "title": "CDS-mean scale Spearman",
                "y_label": "Spearman correlation",
                "filename": "scale_spearman",
                "y_log": False,
                "best": "max",
            },
        ]

        validate_comparison(histories)
        figures = plot_model_metric_curves(histories, metric_configs, x_log=True)

        self.assertEqual(
            set(figures), {"valid_loss", "cds_mean_mae", "scale_spearman"}
        )
        valid_axis = figures["valid_loss"].axes[0]
        self.assertEqual(valid_axis.get_xscale(), "log")
        self.assertEqual(valid_axis.get_yscale(), "log")
        self.assertEqual(valid_axis.get_xlabel(), "Epoch")
        self.assertEqual(valid_axis.get_ylabel(), "Loss")
        self.assertEqual(histories[1].best_metric("valid_loss", "min"), (2, 0.19))
        self.assertEqual(
            histories[1].best_metric("scale_spearman", "max"), (2, 0.61)
        )
        self.assertTrue(np.isfinite(histories[1].cds_mean_mae).all())
        for figure in figures.values():
            plt.close(figure)

    def test_source_data_contains_all_supported_metrics(self):
        history = load_run_history(
            {
                "label": "TRACE",
                "dataset": "same-validation-set",
                "loss_data": [
                    {
                        "epoch": 1,
                        "train_loss": 0.3,
                        "valid_loss": 0.2,
                        "profile_spearman": 0.4,
                        "scale_spearman": 0.5,
                        "cds_mean_mae": 0.1,
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "metric_comparison"
            write_source_data([history], prefix)
            with prefix.with_name(prefix.name + ".source_data.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["model"], "TRACE")
        self.assertAlmostEqual(float(row["cds_mean_mae"]), 0.1)
        self.assertAlmostEqual(float(row["scale_spearman"]), 0.5)

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

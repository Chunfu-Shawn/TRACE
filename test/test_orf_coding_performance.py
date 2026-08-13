"""Tests for cell-aware ORF Precision@K and Recall@K evaluation."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from eval.orf_coding_performance import (
    calculate_top_k_precision,
    plot_top_k_metric,
    plot_top_k_precision,
    plot_top_k_recall,
)


class OrfTopKTests(unittest.TestCase):
    def test_cell_aware_unique_matching_and_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gt_path = directory / "gt.csv"
            pred_path = directory / "pred.csv"

            pd.DataFrame(
                {
                    "Tid": ["ENST1.1", "ENST1.1"],
                    "Cell_Type": ["cell_a", "cell_b"],
                    "CDS_Start_0based": [0, 300],
                    "CDS_End_0based": [90, 390],
                }
            ).to_csv(gt_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1.2"] * 4,
                    "Cell_Type": ["cell_a", "cell_a", "cell_b", "cell_a"],
                    "start": [0, 0, 300, 300],
                    "stop": [90, 87, 390, 390],
                    "score": [0.9, 0.8, 0.7, 0.6],
                }
            ).to_csv(pred_path, index=False)

            result = calculate_top_k_precision(
                str(pred_path),
                str(gt_path),
                target_score_col="score",
            )

        self.assertEqual(result["Is_TP"].tolist(), [1, 0, 1, 0])
        self.assertEqual(result["TP_Count"].tolist(), [1, 1, 2, 2])
        np.testing.assert_allclose(result["Precision"], [1.0, 0.5, 2 / 3, 0.5])
        np.testing.assert_allclose(result["Recall"], [0.5, 0.5, 1.0, 1.0])
        self.assertTrue((result["Total_GT_ORFs"] == 2).all())

    def test_missing_gt_cell_type_requires_explicit_cell_for_multicell_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gt_path = directory / "gt.csv"
            pred_path = directory / "pred.csv"
            pd.DataFrame(
                {
                    "Tid": ["ENST1"],
                    "CDS_Start_0based": [0],
                    "CDS_End_0based": [90],
                }
            ).to_csv(gt_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1", "ENST1"],
                    "Cell_Type": ["cell_a", "cell_b"],
                    "start": [0, 0],
                    "stop": [90, 90],
                    "score": [0.9, 0.8],
                }
            ).to_csv(pred_path, index=False)

            with self.assertRaisesRegex(ValueError, "Supply cell_type explicitly"):
                calculate_top_k_precision(
                    str(pred_path),
                    str(gt_path),
                    target_score_col="score",
                )

    def test_precision_and_recall_plotters_save_pdf(self):
        top_k = pd.DataFrame(
            {
                "K": [1, 2, 3],
                "Precision": [1.0, 0.5, 2 / 3],
                "Recall": [0.5, 0.5, 1.0],
                "Score_Type": ["score"] * 3,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            precision_path = plot_top_k_precision(top_k, directory)
            recall_path = plot_top_k_recall(top_k, directory)
            generic_recall_path = plot_top_k_metric(
                top_k, "Recall", directory, max_k=2
            )
            self.assertTrue(Path(precision_path).is_file())
            self.assertTrue(Path(recall_path).is_file())
            self.assertTrue(Path(generic_recall_path).is_file())
            self.assertEqual(Path(precision_path).suffix, ".pdf")
            self.assertEqual(Path(recall_path).suffix, ".pdf")


if __name__ == "__main__":
    unittest.main()

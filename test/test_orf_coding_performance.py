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
    normalize_transcript_id,
    plot_top_k_metric,
    plot_top_k_precision,
    plot_top_k_recall,
)


class OrfTopKTests(unittest.TestCase):
    def test_only_enst_ids_have_version_suffix_removed(self):
        self.assertEqual(normalize_transcript_id("ENST00000381348.7"), "ENST00000381348")
        self.assertEqual(normalize_transcript_id("PB.123.4"), "PB.123.4")
        self.assertEqual(normalize_transcript_id("ENSG000001234.5"), "ENSG000001234.5")
        self.assertEqual(normalize_transcript_id("XM_001.2"), "XM_001.2")

    def test_multiple_prediction_files_and_cell_type_gt_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gt_a_path = directory / "gt_a.csv"
            gt_b_path = directory / "gt_b.csv"
            pred_a_path = directory / "pred_a.csv"
            pred_b_path = directory / "pred_b.csv"

            pd.DataFrame(
                {
                    "Tid": ["ENST1"],
                    "CDS_Start_0based": [0],
                    "CDS_End_0based": [90],
                }
            ).to_csv(gt_a_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1"],
                    "CDS_Start_0based": [300],
                    "CDS_End_0based": [390],
                }
            ).to_csv(gt_b_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1"],
                    "Cell_Type": ["cell_a"],
                    "start": [0],
                    "stop": [90],
                    "score": [0.9],
                }
            ).to_csv(pred_a_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1"],
                    "Cell_Type": ["cell_b"],
                    "start": [300],
                    "stop": [390],
                    "score": [0.8],
                }
            ).to_csv(pred_b_path, index=False)

            result = calculate_top_k_precision(
                pred_csv_paths=[str(pred_a_path), str(pred_b_path)],
                gt_csv_paths={
                    "cell_a": str(gt_a_path),
                    "cell_b": str(gt_b_path),
                },
                target_score_col="score",
            )

        self.assertEqual(result["Cell_Type"].tolist(), ["cell_a", "cell_b"])
        self.assertEqual(result["Is_TP"].tolist(), [1, 1])
        np.testing.assert_allclose(result["Recall"], [0.5, 1.0])
        self.assertEqual(result["Cell_Type_K"].tolist(), [1, 1])
        np.testing.assert_allclose(result["Cell_Type_Recall"], [1.0, 1.0])
        self.assertEqual(result["Prediction_Source"].nunique(), 2)
        self.assertEqual(result["Matched_GT_Source"].nunique(), 2)

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
        self.assertEqual(result["Cell_Type_K"].tolist(), [1, 2, 1, 3])
        np.testing.assert_allclose(
            result["Cell_Type_Precision"], [1.0, 0.5, 1.0, 1 / 3]
        )
        np.testing.assert_allclose(result["Cell_Type_Recall"], [1.0] * 4)
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
                "Cell_Type": ["cell_a", "cell_a", "cell_b"],
                "Cell_Type_K": [1, 2, 1],
                "Cell_Type_Precision": [1.0, 0.5, 1.0],
                "Cell_Type_Recall": [0.5, 0.5, 1.0],
                "Score_Type": ["score"] * 3,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            precision_path = plot_top_k_precision(top_k, directory)
            recall_path = plot_top_k_recall(top_k, directory)
            generic_recall_path = plot_top_k_metric(
                top_k, "Recall", directory, max_k=2
            )
            cell_type_precision_path = plot_top_k_precision(
                top_k, directory, rank_scope="cell_type"
            )
            self.assertTrue(Path(precision_path).is_file())
            self.assertTrue(Path(recall_path).is_file())
            self.assertTrue(Path(generic_recall_path).is_file())
            self.assertTrue(Path(cell_type_precision_path).is_file())
            self.assertEqual(Path(precision_path).suffix, ".pdf")
            self.assertEqual(Path(recall_path).suffix, ".pdf")


if __name__ == "__main__":
    unittest.main()

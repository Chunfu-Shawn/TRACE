"""Tests for multi-model ORF Recall@K benchmark preparation."""

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

from plot.benchmark_orf_ident import (
    _extract_top_k_recall_curve,
    plot_multi_model_top_k_recall,
)


class OrfRecallBenchmarkTests(unittest.TestCase):
    def test_recall_counts_each_gt_only_once(self):
        evaluation = pd.DataFrame(
            {
                "Record_Type": [
                    "Prediction", "Prediction", "Prediction", "Prediction",
                    "Missed_GT",
                ],
                "Matched_GT_Index": [0, 0, 1, np.nan, 2],
                "score": [0.9, 0.8, 0.7, 0.6, -1.0],
            }
        )

        curve, total_gt = _extract_top_k_recall_curve(
            evaluation, score_col="score"
        )

        self.assertEqual(total_gt, 3)
        np.testing.assert_allclose(
            curve["Recall"], [1 / 3, 1 / 3, 2 / 3, 2 / 3]
        )

    def test_precomputed_recall_table_is_supported(self):
        top_k = pd.DataFrame(
            {
                "K": [1, 2, 3],
                "Recall_at_K": [0.1, 0.2, 0.3],
                "Total_GT_ORFs": [10, 10, 10],
            }
        )

        curve, total_gt = _extract_top_k_recall_curve(
            top_k, score_col="unused"
        )

        self.assertEqual(total_gt, 10)
        np.testing.assert_allclose(curve["Recall"], [0.1, 0.2, 0.3])

    def test_different_gt_denominators_raise_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path_a = directory / "model_a.csv"
            path_b = directory / "model_b.csv"
            pd.DataFrame(
                {
                    "K": [1, 2],
                    "Recall": [0.1, 0.2],
                    "Total_GT_ORFs": [10, 10],
                }
            ).to_csv(path_a, index=False)
            pd.DataFrame(
                {
                    "K": [1, 2],
                    "Recall": [0.1, 0.2],
                    "Total_GT_ORFs": [20, 20],
                }
            ).to_csv(path_b, index=False)

            with self.assertRaisesRegex(ValueError, "different callable GT"):
                plot_multi_model_top_k_recall(
                    manifest=[
                        {"model": "TRACE", "path": str(path_a)},
                        {"model": "RiboTIE", "path": str(path_b)},
                    ],
                    out_dir=str(directory),
                )


if __name__ == "__main__":
    unittest.main()

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
    _extract_incomplete_curve_endpoints,
    _extract_top_k_recall_curve,
    compare_multi_model_roc_auc,
    plot_multi_model_top_k_precision,
    plot_multi_model_top_k_recall,
)


class OrfRecallBenchmarkTests(unittest.TestCase):
    def test_incomplete_curve_endpoints_mark_only_short_models(self):
        curves = pd.DataFrame(
            {
                "Model": ["TRACE"] * 3 + ["RiboTIE"] * 5,
                "K": [1, 2, 3, 1, 2, 3, 4, 5],
                "Precision_Smooth": [0.8] * 8,
            }
        )

        endpoints = _extract_incomplete_curve_endpoints(curves, max_k=5)

        self.assertEqual(endpoints["Model"].tolist(), ["TRACE"])
        self.assertEqual(endpoints["K"].tolist(), [3])
        self.assertTrue(
            _extract_incomplete_curve_endpoints(curves, max_k=None).empty
        )

    def test_multi_model_roc_accepts_trace_feature_combination(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trace_path = directory / "trace.csv"
            other_path = directory / "other.csv"
            shared_columns = {
                "Record_Type": ["Prediction"] * 4,
                "Cell_Type": ["brain"] * 4,
                "y_true": [1, 0, 1, 0],
                "Total_GT_ORFs": [2] * 4,
            }
            pd.DataFrame(
                {
                    **shared_columns,
                    "base_expr_score": [0.9, 0.8, 0.7, 0.6],
                    "uniformity_of_signal": [0.9, 0.1, 0.8, 0.2],
                }
            ).to_csv(trace_path, index=False)
            pd.DataFrame(
                {
                    **shared_columns,
                    "score": [0.9, 0.2, 0.8, 0.1],
                }
            ).to_csv(other_path, index=False)

            summary, curves, save_path = compare_multi_model_roc_auc(
                manifest=[
                    {"model": "TRACE", "path": str(trace_path)},
                    {
                        "model": "RiboTIE",
                        "path": str(other_path),
                        "score_col": "score",
                    },
                ],
                out_dir=str(directory),
                trace_combined_score={
                    "Base_Score": "base_expr_score",
                    "Features": "uniformity_of_signal",
                    "Method": "product",
                },
            )

            self.assertTrue(Path(save_path).is_file())
            self.assertEqual(set(summary["Model"]), {"TRACE", "RiboTIE"})
            self.assertEqual(set(curves["Model"]), {"TRACE", "RiboTIE"})
            trace_label = summary.loc[
                summary["Model"] == "TRACE", "Score_Type"
            ].iloc[0]
            self.assertEqual(
                trace_label,
                "base_expr_score | product(uniformity_of_signal)",
            )

    def test_precomputed_precision_rejects_new_trace_combination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "top_k.csv"
            pd.DataFrame(
                {"K": [1, 2], "Precision": [1.0, 0.5]}
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "cannot be reranked"):
                plot_multi_model_top_k_precision(
                    manifest=[{"model": "TRACE", "path": str(path)}],
                    out_dir=directory,
                    trace_combined_score={
                        "Base_Score": "base_expr_score",
                        "Features": "uniformity_of_signal",
                        "Method": "product",
                    },
                )

    def test_precomputed_recall_rejects_new_trace_combination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "top_k.csv"
            pd.DataFrame(
                {
                    "K": [1, 2],
                    "Recall": [0.1, 0.2],
                    "Total_GT_ORFs": [10, 10],
                }
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "cannot be reranked"):
                plot_multi_model_top_k_recall(
                    manifest=[{"model": "TRACE", "path": str(path)}],
                    out_dir=directory,
                    trace_combined_score={
                        "Base_Score": "base_expr_score",
                        "Features": "uniformity_of_signal",
                        "Method": "product",
                    },
                )

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

"""Tests for cell-aware ORF Precision@K and Recall@K evaluation."""

import os
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from eval.orf_coding_performance import (
    add_feature_combination_scores,
    calculate_top_k_from_evaluation,
    calculate_top_k_precision,
    evaluate_orf_level_predictions,
    load_and_filter_data,
    match_and_build_eval_df,
    normalize_transcript_id,
    plot_top_k_metric,
    plot_top_k_precision,
    plot_top_k_recall,
    summarize_top_k_values,
)
from model.orf_caller import FastSignalDrivenORFCaller, TranslationSignalORFCaller


class OrfTopKTests(unittest.TestCase):
    def test_top_k_score_col_alias_selects_existing_column(self):
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
                    "Tid": ["ENST1", "ENST2"],
                    "Cell_Type": ["brain", "brain"],
                    "start": [0, 300],
                    "stop": [90, 390],
                    "score": [0.1, 0.9],
                    "mean_intensity": [0.8, 0.2],
                }
            ).to_csv(pred_path, index=False)

            result = calculate_top_k_precision(
                pred_csv_paths=[str(pred_path)],
                gt_csv_paths={"brain": str(gt_path)},
                score_col="mean_intensity",
            )

        self.assertEqual(result["Tid"].tolist(), ["ENST1", "ENST2"])
        self.assertEqual(result["Score_Type"].unique().tolist(), ["mean_intensity"])
        self.assertEqual(result["Precision"].iloc[0], 1.0)

    def test_top_k_recomputes_combined_score_from_evaluation_definition(self):
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
                    "Tid": ["ENST1", "ENST2"],
                    "Cell_Type": ["brain", "brain"],
                    "start": [0, 300],
                    "stop": [90, 390],
                    "score": [0.1, 0.9],
                    "base_expr_score": [0.5, 0.8],
                    "tri_nucleotide_periodicity": [0.9, 0.1],
                    "uniformity_of_signal": [0.8, 0.5],
                }
            ).to_csv(pred_path, index=False)

            result = calculate_top_k_precision(
                pred_csv_paths=[str(pred_path)],
                gt_csv_paths={"brain": str(gt_path)},
                combined_score=pd.Series({
                    "Base_Score": "base_expr_score",
                    "Features": (
                        "tri_nucleotide_periodicity+uniformity_of_signal"
                    ),
                    "Method": "product",
                }),
            )

        self.assertEqual(result["Tid"].tolist(), ["ENST1", "ENST2"])
        np.testing.assert_allclose(result["Score"], [0.36, 0.04])
        self.assertEqual(result["Precision"].iloc[0], 1.0)
        self.assertEqual(
            result["Score_Type"].unique().tolist(),
            [
                "base_expr_score | product("
                "tri_nucleotide_periodicity+uniformity_of_signal)"
            ],
        )

    def test_top_k_filters_global_targets_and_callable_start_codons(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gt_path = directory / "gt.csv"
            pred_path = directory / "pred.csv"
            pd.DataFrame(
                {
                    "Tid": ["ENST1.1", "ENST2.1", "ENST3.1"],
                    "CDS_Start_0based": [0, 300, 600],
                    "CDS_End_0based": [90, 390, 690],
                    "Start_Codon": ["ATG", "CTG", "ATG"],
                }
            ).to_csv(gt_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1.2", "ENST2.2", "ENST3.2"],
                    "Cell_Type": ["brain", "brain", "brain"],
                    "start": [0, 300, 600],
                    "stop": [90, 390, 690],
                    "start_codon": ["ATG", "CTG", "ATG"],
                    "score": [0.9, 0.8, 0.7],
                }
            ).to_csv(pred_path, index=False)

            result = calculate_top_k_precision(
                pred_csv_paths=[str(pred_path)],
                gt_csv_paths={"brain": str(gt_path)},
                target_transcript_ids=np.array(["ENST1.9", "ENST2.9"]),
                callable_start_codons=["ATG"],
                target_score_col="score",
            )

        self.assertEqual(result["Tid"].tolist(), ["ENST1.2"])
        self.assertEqual(result["Total_GT_ORFs"].tolist(), [1])
        self.assertEqual(result["Is_TP"].tolist(), [1])
        self.assertEqual(result["Precision"].tolist(), [1.0])

    def test_top_k_filters_cell_specific_target_transcript_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            brain_gt_path = directory / "brain_gt.csv"
            liver_gt_path = directory / "liver_gt.csv"
            pred_path = directory / "pred.csv"
            gt_frame = pd.DataFrame(
                {
                    "Tid": ["ENST1.1", "ENST2.1"],
                    "CDS_Start_0based": [0, 300],
                    "CDS_End_0based": [90, 390],
                    "Start_Codon": ["ATG", "ATG"],
                }
            )
            gt_frame.to_csv(brain_gt_path, index=False)
            gt_frame.to_csv(liver_gt_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1.2", "ENST2.2", "ENST1.3", "ENST2.3"],
                    "Cell_Type": ["brain", "brain", "liver", "liver"],
                    "start": [0, 300, 0, 300],
                    "stop": [90, 390, 90, 390],
                    "start_codon": ["ATG", "ATG", "ATG", "ATG"],
                    "score": [0.9, 0.8, 0.7, 0.6],
                }
            ).to_csv(pred_path, index=False)

            result = calculate_top_k_precision(
                pred_csv_paths=[str(pred_path)],
                gt_csv_paths={
                    "brain": str(brain_gt_path),
                    "liver": str(liver_gt_path),
                },
                target_transcript_ids={
                    "brain": ["ENST1.8"],
                    "liver": ["ENST2.8"],
                },
                callable_start_codons=["ATG"],
                target_score_col="score",
            )

        self.assertEqual(
            list(zip(result["Cell_Type"], result["Tid"])),
            [("brain", "ENST1.2"), ("liver", "ENST2.3")],
        )
        self.assertEqual(result["Is_TP"].tolist(), [1, 1])
        self.assertEqual(result["Total_GT_ORFs"].tolist(), [2, 2])

    def test_max_len_removes_long_candidates_before_collapse_and_nms(self):
        sequence = "ATG" + "AAA" * 19 + "ATG" + "AAA" * 12 + "TAA"
        caller = FastSignalDrivenORFCaller(min_len=30, max_len=60)

        candidates = caller.extract_all_candidates(sequence)

        self.assertEqual(
            [(candidate["start"], candidate["stop"], candidate["length"])
             for candidate in candidates],
            [(60, 99, 42)],
        )
        self.assertIn(
            "max_len",
            inspect.signature(TranslationSignalORFCaller.run).parameters,
        )

    def test_max_len_cannot_be_shorter_than_min_len(self):
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            FastSignalDrivenORFCaller(min_len=60, max_len=30)

    def test_long_mode_uses_sequence_length_score_without_signal_filter(self):
        sequence = "ATG" + "AAA" * 19 + "ATG" + "AAA" * 12 + "TAA"
        signal = np.zeros(len(sequence), dtype=np.float32)
        caller = FastSignalDrivenORFCaller(min_len=30, mode="long")

        candidates = caller.extract_features(
            sequence,
            signal,
            intensity_threshold=0.01,
        )

        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            expected = np.log10(candidate["length"] + 1)
            self.assertAlmostEqual(candidate["sequence_length_score"], expected)
            self.assertAlmostEqual(candidate["score"], expected)

        collapsed = caller.collapse_and_nms(candidates, iou_threshold=0.3)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["start"], 0)
        self.assertEqual(collapsed[0]["length"], len(sequence))

    def test_long_mode_applies_baseline_start_codon_weights(self):
        sequence = "CTG" + "AAA" * 19 + "TAA"
        signal = np.zeros(len(sequence), dtype=np.float32)
        caller = FastSignalDrivenORFCaller(
            start_codons=["CTG"],
            min_len=30,
            mode="long",
        )

        candidate = caller.extract_features(sequence, signal)[0]

        expected = 0.8 * np.log10(candidate["length"] + 1)
        self.assertAlmostEqual(candidate["sequence_length_score"], expected)
        self.assertAlmostEqual(candidate["score"], expected)

    def test_long_mode_uses_baseline_start_codons_by_default(self):
        caller = FastSignalDrivenORFCaller(mode="long")
        balanced_caller = FastSignalDrivenORFCaller(mode="balanced")

        self.assertEqual(
            caller.start_codons,
            ["ATG", "CTG", "GTG", "TTG", "ACG"],
        )
        self.assertEqual(
            balanced_caller.start_codons,
            ["ATG", "CTG", "GTG"],
        )

    def test_legacy_long_mode_can_still_require_translation_signal(self):
        sequence = "ATG" + "AAA" * 19 + "TAA"
        signal = np.zeros(len(sequence), dtype=np.float32)
        caller = FastSignalDrivenORFCaller(
            min_len=30,
            mode="long",
            long_mode_length_only=False,
        )

        candidates = caller.extract_features(
            sequence,
            signal,
            intensity_threshold=0.01,
        )

        self.assertEqual(candidates, [])

    def test_cell_specific_transcript_targets_filter_gt_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            brain_gt_path = directory / "brain_gt.csv"
            liver_gt_path = directory / "liver_gt.csv"
            pred_path = directory / "pred.csv"

            gt_frame = pd.DataFrame(
                {
                    "Tid": ["ENST1.1", "ENST2.1"],
                    "CDS_Start_0based": [0, 300],
                    "CDS_End_0based": [90, 390],
                }
            )
            gt_frame.to_csv(brain_gt_path, index=False)
            gt_frame.to_csv(liver_gt_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1.2", "ENST2.2", "ENST1.3", "ENST2.3"],
                    "Cell_Type": ["brain", "brain", "liver", "liver"],
                    "start": [0, 300, 0, 300],
                    "stop": [90, 390, 90, 390],
                    "score": [0.9, 0.8, 0.7, 0.6],
                }
            ).to_csv(pred_path, index=False)

            pred_df, gt_df, _ = load_and_filter_data(
                pred_csv_paths=[str(pred_path)],
                gt_csv_paths={
                    "brain": str(brain_gt_path),
                    "liver": str(liver_gt_path),
                },
                target_transcript_ids={
                    "brain": np.array(["ENST1.9"]),
                    "liver": np.array(["ENST2.9"]),
                },
                target_score_col="score",
            )

        self.assertEqual(
            list(zip(pred_df["Cell_Type"], pred_df["Tid_clean"])),
            [("brain", "ENST1"), ("liver", "ENST2")],
        )
        self.assertEqual(
            list(zip(gt_df["Cell_Type"], gt_df["Tid_clean"])),
            [("brain", "ENST1"), ("liver", "ENST2")],
        )

    def test_global_transcript_array_still_supports_single_cell_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gt_path = directory / "gt.csv"
            pred_path = directory / "pred.csv"
            pd.DataFrame(
                {
                    "Tid": ["ENST1.1", "ENST2.1"],
                    "CDS_Start_0based": [0, 300],
                    "CDS_End_0based": [90, 390],
                }
            ).to_csv(gt_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1.2", "ENST2.2"],
                    "Cell_Type": ["brain", "brain"],
                    "start": [0, 300],
                    "stop": [90, 390],
                    "score": [0.9, 0.8],
                }
            ).to_csv(pred_path, index=False)

            pred_df, gt_df, _ = load_and_filter_data(
                pred_csv_paths=[str(pred_path)],
                gt_csv_paths={"brain": str(gt_path)},
                target_transcript_ids=np.array(["ENST2.8"]),
                target_score_col="score",
            )

        self.assertEqual(pred_df["Tid_clean"].tolist(), ["ENST2"])
        self.assertEqual(gt_df["Tid_clean"].tolist(), ["ENST2"])

    def test_nms_only_suppresses_candidates_in_the_same_frame(self):
        caller = FastSignalDrivenORFCaller()
        candidates = [
            {"start": 0, "stop": 99, "length": 102, "score": 1.0},
            {"start": 1, "stop": 100, "length": 102, "score": 0.9},
            {"start": 3, "stop": 99, "length": 99, "score": 0.8},
        ]

        result = caller.fast_nms(candidates, iou_threshold=0.7)

        self.assertEqual([(row["start"], row["stop"]) for row in result], [(0, 99), (1, 100)])

    def test_nms_can_suppress_candidates_across_frames(self):
        caller = FastSignalDrivenORFCaller()
        candidates = [
            {"start": 0, "stop": 99, "length": 102, "score": 1.0},
            {"start": 1, "stop": 100, "length": 102, "score": 0.9},
            {"start": 3, "stop": 99, "length": 99, "score": 0.8},
        ]

        result = caller.fast_nms(
            candidates,
            iou_threshold=0.7,
            nms_respect_frame=False,
        )

        self.assertEqual(
            [(row["start"], row["stop"]) for row in result],
            [(0, 99)],
        )
        self.assertTrue(
            all("suppressed" not in candidate for candidate in candidates)
        )

    def test_run_exposes_nms_respect_frame(self):
        self.assertIn(
            "nms_respect_frame",
            inspect.signature(TranslationSignalORFCaller.run).parameters,
        )

    def test_feature_combination_scores_are_enumerated(self):
        pred_df = pd.DataFrame(
            {
                "translation_score": [2.0],
                "tri_nucleotide_periodicity": [0.5],
                "uniformity_of_signal": [0.25],
            }
        )

        result, metadata = add_feature_combination_scores(
            pred_df,
            base_score_columns=["translation_score"],
            feature_columns=[
                "tri_nucleotide_periodicity",
                "uniformity_of_signal",
            ],
            method="product",
        )

        combo_columns = metadata.loc[
            metadata["Method"] == "product", "Score_Column"
        ].tolist()
        self.assertEqual(len(combo_columns), 3)
        np.testing.assert_allclose(
            result[combo_columns].iloc[0].sort_values().to_numpy(),
            [0.25, 0.5, 1.0],
        )

    def test_gt_cell_types_without_prediction_files_are_not_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pred_path = directory / "brain_pred.csv"
            brain_gt_path = directory / "brain_gt.csv"
            liver_gt_path = directory / "liver_gt.csv"
            pd.DataFrame(
                {
                    "Tid": ["ENST1"],
                    "Cell_Type": ["brain"],
                    "start": [0],
                    "stop": [90],
                    "length": [93],
                    "score": [1.0],
                }
            ).to_csv(pred_path, index=False)
            for path in (brain_gt_path, liver_gt_path):
                pd.DataFrame(
                    {
                        "Tid": ["ENST1"],
                        "CDS_Start_0based": [0],
                        "CDS_End_0based": [90],
                    }
                ).to_csv(path, index=False)

            _, gt_df, _ = load_and_filter_data(
                pred_csv_paths=[str(pred_path)],
                gt_csv_paths={
                    "brain": str(brain_gt_path),
                    "liver": str(liver_gt_path),
                },
                target_score_col="score",
            )

        self.assertEqual(gt_df["Cell_Type"].unique().tolist(), ["brain"])
        self.assertEqual(gt_df["length"].tolist(), [93])

    def test_callable_start_codons_filter_gt_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pred_path = directory / "pred.csv"
            gt_path = directory / "gt.csv"
            pd.DataFrame(
                {
                    "Tid": ["ENST1", "ENST2"],
                    "Cell_Type": ["brain", "brain"],
                    "start": [0, 0],
                    "stop": [90, 90],
                    "start_codon": ["ATG", "TTG"],
                    "score": [1.0, 0.9],
                }
            ).to_csv(pred_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1", "ENST2"],
                    "CDS_Start_0based": [0, 0],
                    "CDS_End_0based": [90, 90],
                    "Start_Codon": ["ATG", "TTG"],
                }
            ).to_csv(gt_path, index=False)

            pred_df, gt_df, _ = load_and_filter_data(
                pred_csv_paths=[str(pred_path)],
                gt_csv_paths={"brain": str(gt_path)},
                target_score_col="score",
                callable_start_codons=["ATG"],
            )

        self.assertEqual(pred_df["Tid_clean"].tolist(), ["ENST1"])
        self.assertEqual(gt_df["Tid_clean"].tolist(), ["ENST1"])

    def test_main_evaluation_allows_multiple_predictions_per_gt(self):
        gt_df = pd.DataFrame(
            {
                "Cell_Type": ["cell_a"],
                "Tid_clean": ["ENST1"],
                "gt_idx": [0],
                "start_gt": [0],
                "stop_gt": [90],
                "length": [90],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Cell_Type": ["cell_a", "cell_a"],
                "Tid_clean": ["ENST1", "ENST1"],
                "pred_idx": [0, 1],
                "start": [0, 0],
                "stop": [90, 87],
                "length": [90, 87],
                "score": [0.9, 0.8],
            }
        )

        result = match_and_build_eval_df(
            pred_df, gt_df, ["score"], overlap_threshold=0.7
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result["y_true"].tolist(), [1, 1])

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

    def test_cell_aware_many_to_one_matching_and_unique_gt_recall(self):
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

        self.assertEqual(result["Is_TP"].tolist(), [1, 1, 1, 0])
        self.assertEqual(result["TP_Count"].tolist(), [1, 2, 3, 3])
        self.assertEqual(result["Unique_GT_Hit_Count"].tolist(), [1, 1, 2, 2])
        self.assertEqual(result["Is_New_GT_Hit"].tolist(), [1, 0, 1, 0])
        np.testing.assert_allclose(result["Precision"], [1.0, 1.0, 1.0, 0.75])
        np.testing.assert_allclose(result["Recall"], [0.5, 0.5, 1.0, 1.0])
        self.assertEqual(result["Cell_Type_K"].tolist(), [1, 2, 1, 3])
        np.testing.assert_allclose(
            result["Cell_Type_Precision"], [1.0, 1.0, 1.0, 2 / 3]
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

    def test_comprehensive_evaluation_and_top_k_share_one_match_table(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gt_path = directory / "gt.csv"
            pred_path = directory / "pred.csv"
            output_dir = directory / "output"
            pd.DataFrame(
                {
                    "Tid": ["ENST1.1", "ENST2.1"],
                    "CDS_Start_0based": [0, 300],
                    "CDS_End_0based": [90, 390],
                    "Start_Codon": ["ATG", "ATG"],
                }
            ).to_csv(gt_path, index=False)
            pd.DataFrame(
                {
                    "Tid": ["ENST1.2", "ENST1.2", "ENST2.2", "ENST3.1"],
                    "Cell_Type": ["brain"] * 4,
                    "start": [0, 0, 300, 600],
                    "stop": [90, 90, 390, 690],
                    "start_codon": ["ATG"] * 4,
                    "score": [0.9, 0.8, 0.7, 0.6],
                }
            ).to_csv(pred_path, index=False)

            with patch(
                "eval.orf_coding_performance.evaluate_and_plot_global"
            ):
                results = evaluate_orf_level_predictions(
                    pred_csv_paths=[str(pred_path)],
                    gt_csv_paths={"brain": str(gt_path)},
                    out_dir=str(output_dir),
                    target_score_col="score",
                    combination_top_k_values=[1, 2, 3],
                )

            recalculated = calculate_top_k_precision(
                evaluation_df=results["evaluation"],
                score_col="score",
            )
            output_files_exist = all([
                (output_dir / "unified_evaluation_table.csv").is_file(),
                (output_dir / "Precision_at_K_data.csv").is_file(),
                (output_dir / "top_k_metrics_summary.csv").is_file(),
            ])

        self.assertEqual(len(results["evaluation"].query(
            "Record_Type == 'Prediction'"
        )), 3)
        pd.testing.assert_frame_equal(
            results["top_k"].reset_index(drop=True),
            recalculated.reset_index(drop=True),
            check_dtype=False,
        )
        self.assertEqual(
            results["top_k_summary"]["Precision"].tolist(),
            [1.0, 1.0, 2 / 3],
        )
        self.assertTrue(output_files_exist)

    def test_top_k_helper_uses_exact_requested_k_values(self):
        evaluation = pd.DataFrame(
            {
                "Record_Type": ["Prediction", "Prediction", "Missed_GT"],
                "Cell_Type": ["brain", "brain", "brain"],
                "Tid": ["ENST1", "ENST2", "ENST3"],
                "Matched_GT_Index": [0, np.nan, 1],
                "y_true": [1, 0, 1],
                "score": [0.9, 0.8, -1.0],
                "Total_GT_ORFs": [2, 2, 2],
                "Cell_Type_Total_GT_ORFs": [2, 2, 2],
            }
        )

        top_k = calculate_top_k_from_evaluation(evaluation, "score")
        summary = summarize_top_k_values(top_k, [1, 2, 100])

        self.assertEqual(summary["Effective_K"].tolist(), [1, 2, 2])
        np.testing.assert_allclose(summary["Precision"], [1.0, 0.5, 0.5])
        np.testing.assert_allclose(summary["Recall"], [0.5, 0.5, 0.5])

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

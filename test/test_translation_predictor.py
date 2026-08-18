"""Tests for cell-aware expressed-transcript selection."""

import os
import sys
import tempfile
import unittest
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model.translation_predictor import (
    TranslationProfilePredictor,
    _resolve_prediction_expression_vector,
    get_active_transcripts,
)


class ActiveTranscriptTests(unittest.TestCase):
    def test_returns_array_or_cell_type_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tpm_path = directory / "expression.csv"
            mapping_path = directory / "mapping.tsv"

            pd.DataFrame(
                {
                    "brain": [2.0, 0.1],
                    "liver": [0.1, 3.0],
                },
                index=["ENSG1", "ENSG2"],
            ).to_csv(tpm_path)
            pd.DataFrame(
                {
                    "Gene stable ID": ["ENSG1", "ENSG1", "ENSG2"],
                    "Transcript stable ID": ["ENST1", "ENST1B", "ENST2"],
                }
            ).to_csv(mapping_path, sep="\t", index=False)

            single_result = get_active_transcripts(
                str(tpm_path),
                str(mapping_path),
                cell_type="brain",
                min_tpm=1.0,
            )
            multi_result = get_active_transcripts(
                str(tpm_path),
                str(mapping_path),
                cell_type=["brain", "liver"],
                min_tpm=1.0,
            )

        self.assertIsInstance(single_result, np.ndarray)
        self.assertEqual(single_result.tolist(), ["ENST1", "ENST1B"])
        self.assertEqual(list(multi_result), ["brain", "liver"])
        self.assertEqual(multi_result["brain"].tolist(), ["ENST1", "ENST1B"])
        self.assertEqual(multi_result["liver"].tolist(), ["ENST2"])


class PredictionExpressionVectorTests(unittest.TestCase):
    def test_run_allows_expression_vector_to_be_omitted(self):
        parameter = signature(TranslationProfilePredictor.run).parameters[
            "cell_expr_vector"
        ]
        self.assertIsNone(parameter.default)

    def test_missing_vector_becomes_model_sized_zeros(self):
        model = SimpleNamespace(d_expr=4)

        result = _resolve_prediction_expression_vector(model, None)

        np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))
        self.assertEqual(result.dtype, np.float32)

    def test_mean_buffer_is_used_only_to_infer_dimension(self):
        model = SimpleNamespace(mean_expr_vector=torch.ones(3))

        result = _resolve_prediction_expression_vector(model, None)

        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))

    def test_explicit_vector_is_preserved_and_validated(self):
        model = SimpleNamespace(d_expr=3)
        supplied = np.array([0.1, 0.2, 0.3], dtype=np.float64)

        result = _resolve_prediction_expression_vector(model, supplied)

        np.testing.assert_allclose(result, supplied)
        self.assertEqual(result.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "wrong length"):
            _resolve_prediction_expression_vector(model, np.zeros(2))


if __name__ == "__main__":
    unittest.main()

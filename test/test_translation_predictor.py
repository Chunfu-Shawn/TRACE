"""Tests for cell-aware expressed-transcript selection."""

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

from model.translation_predictor import get_active_transcripts


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


if __name__ == "__main__":
    unittest.main()

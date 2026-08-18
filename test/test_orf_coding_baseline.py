"""Tests for transcript-restricted sequence ORF baselines."""

import tempfile
import unittest
import sys
import types
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable: iterable
    sys.modules["tqdm"] = tqdm_module

from eval.orf_coding_baseline import BaselineORFIdentifier


class BaselineORFIdentifierTests(unittest.TestCase):
    def _write_fasta(self, directory: Path) -> Path:
        fasta_path = directory / "transcripts.fa"
        fasta_path.write_text(
            ">ENST000001.3\nATGAAATAA\n"
            ">ENST000002.7\nATGCCCTAG\n"
            ">PB.100.4\nATGGGGTGA\n"
        )
        return fasta_path

    def test_shared_target_transcripts_are_applied_to_all_cell_types(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            identifier = BaselineORFIdentifier(
                fasta_file=str(self._write_fasta(directory)),
                cell_types=["brain", "liver"],
            )

            result = identifier.run(
                out_dir=str(directory / "output"),
                target_transcript_ids=["ENST000001.99"],
                min_len=9,
            )

            self.assertEqual(set(result["Tid"]), {"ENST000001"})
            self.assertEqual(set(result["Cell_Type"]), {"brain", "liver"})

    def test_cell_type_mapping_does_not_expand_missing_cells_to_all(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            identifier = BaselineORFIdentifier(
                fasta_file=str(self._write_fasta(directory)),
                cell_types=["brain", "liver"],
            )

            result = identifier.run(
                out_dir=str(directory / "output"),
                target_transcript_ids={"brain": ["PB.100.4"]},
                min_len=9,
            )

            self.assertEqual(set(result["Tid"]), {"PB.100.4"})
            self.assertEqual(set(result["Cell_Type"]), {"brain"})

    def test_new_and_legacy_target_arguments_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            identifier = BaselineORFIdentifier(
                fasta_file=str(self._write_fasta(directory)),
                cell_types=["brain"],
            )

            with self.assertRaisesRegex(ValueError, "Specify only one"):
                identifier.run(
                    out_dir=str(directory / "output"),
                    target_transcript_ids=["ENST000001"],
                    target_tids=["ENST000002"],
                )

    def test_outputs_kozak_and_start_codon_prior_scores(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            fasta_path = directory / "kozak.fa"
            fasta_path.write_text(
                ">ENST_KOZAK.1\nGCCACCATGGGGTAA\n"
                ">PB.KOZAK.1\nGCCACCCTGGGGTAA\n"
            )
            identifier = BaselineORFIdentifier(
                fasta_file=str(fasta_path),
                cell_types=["brain"],
            )

            result = identifier.run(
                out_dir=str(directory / "output"),
                min_len=9,
            ).set_index("Tid")

            self.assertIn("kozak_score", result.columns)
            self.assertIn("start_codon_score", result.columns)
            self.assertAlmostEqual(
                result.loc["ENST_KOZAK", "kozak_score"],
                1.58,
            )
            self.assertAlmostEqual(
                result.loc["ENST_KOZAK", "start_codon_score"],
                1.0,
            )
            self.assertAlmostEqual(
                result.loc["PB.KOZAK.1", "kozak_score"],
                0.88,
            )
            self.assertAlmostEqual(
                result.loc["PB.KOZAK.1", "start_codon_score"],
                0.3,
            )

    def test_kozak_score_is_missing_without_complete_context(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            identifier = BaselineORFIdentifier(
                fasta_file=str(self._write_fasta(directory)),
                cell_types=["brain"],
            )

            result = identifier.run(
                out_dir=str(directory / "output"),
                target_transcript_ids=["ENST000001"],
                min_len=9,
            )

            self.assertTrue(np.isnan(result.iloc[0]["kozak_score"]))
            self.assertEqual(result.iloc[0]["start_codon_score"], 1.0)


if __name__ == "__main__":
    unittest.main()

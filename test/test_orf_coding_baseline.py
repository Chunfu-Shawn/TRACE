"""Tests for transcript-restricted sequence ORF baselines."""

import tempfile
import unittest
import sys
import types
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()

"""Tests for transcript-level translation-amplitude evaluation."""

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

from eval.cds_mean_corr import (
    aggregate_by_transcript,
    classify_gene_type,
    decode_one_hot_sequence,
    find_longest_complete_orf,
    load_housekeeping_transcripts,
    load_transcript_biotypes,
    select_amplitude_region,
    summarize_by_biotype,
)


def encode_sequence(sequence):
    """Encode an A/C/G/T/N sequence using the project channel order."""
    base_index = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    indices = [base_index[base] for base in sequence]
    return np.eye(5, dtype=np.float32)[indices, :4]


class CdsMeanCorrelationTests(unittest.TestCase):
    def test_sequence_decoding_and_longest_complete_orf(self):
        sequence = "CCCATGAAATAACCC"
        self.assertEqual(decode_one_hot_sequence(encode_sequence(sequence)), sequence)
        self.assertEqual(
            find_longest_complete_orf(sequence, min_orf_codons=3),
            (3, 12),
        )
        self.assertIsNone(
            find_longest_complete_orf("CCCCCCCCCCCC", min_orf_codons=3)
        )

    def test_region_selection_priority(self):
        sequence = "CCCATGAAATAACCC"
        self.assertEqual(
            select_amplitude_region(
                {"cds_start_pos": 2, "cds_end_pos": 10},
                sequence,
                len(sequence),
                min_orf_codons=3,
            ),
            (1, 10, "annotated_CDS"),
        )
        self.assertEqual(
            select_amplitude_region(
                {"cds_start_pos": -1, "cds_end_pos": -1},
                sequence,
                len(sequence),
                min_orf_codons=3,
            ),
            (3, 12, "longest_complete_ORF"),
        )
        no_orf = "CCCCCCCCCCCC"
        self.assertEqual(
            select_amplitude_region(
                {"cds_start_pos": -1, "cds_end_pos": -1},
                no_orf,
                len(no_orf),
                min_orf_codons=3,
            ),
            (0, len(no_orf), "transcript_wide"),
        )

    def test_transcript_aggregation_equalizes_cell_type_replicates(self):
        sample_df = pd.DataFrame(
            [
                {
                    "Tid": "TX1.1",
                    "Tid_Clean": "TX1",
                    "Biotype": "protein_coding",
                    "Gene_Type": "Housekeeping",
                    "Region_Source": "annotated_CDS",
                    "Cell_Type": "cell_a",
                    "Region_Length": 90,
                    "Observed_Mean_Linear": 1.0,
                    "Predicted_Mean_Linear": 1.5,
                },
                {
                    "Tid": "TX1.1",
                    "Tid_Clean": "TX1",
                    "Biotype": "protein_coding",
                    "Gene_Type": "Housekeeping",
                    "Region_Source": "annotated_CDS",
                    "Cell_Type": "cell_b",
                    "Region_Length": 90,
                    "Observed_Mean_Linear": 3.0,
                    "Predicted_Mean_Linear": 3.5,
                },
            ]
        )
        result = aggregate_by_transcript(sample_df).iloc[0]
        self.assertEqual(result["Cell_Type_Count"], 2)
        self.assertAlmostEqual(result["Observed_Mean_Linear"], 2.0)
        self.assertAlmostEqual(result["Predicted_Mean_Linear"], 2.5)

    def test_gtf_biotype_parser_removes_transcript_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.gtf"
            path.write_text(
                "chr1\ttest\ttranscript\t1\t100\t.\t+\t.\t"
                'gene_id "G1"; transcript_id "TX1.2"; '
                'transcript_type "lncRNA";\n',
                encoding="utf-8",
            )
            result = load_transcript_biotypes(path)
        self.assertEqual(result, {"TX1": "lncRNA"})

    def test_housekeeping_parser_removes_transcript_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "housekeeping.tsv"
            path.write_text("Transcript ID\nTX1.2\nTX2.1\n", encoding="utf-8")
            result = load_housekeeping_transcripts(path)
        self.assertEqual(result, {"TX1", "TX2"})

    def test_gene_type_classification_prioritizes_housekeeping(self):
        housekeeping = {"TX1"}
        self.assertEqual(
            classify_gene_type("TX1.2", "lncRNA", housekeeping),
            "Housekeeping",
        )
        self.assertEqual(classify_gene_type("TX2", "lncRNA", housekeeping), "lncRNA")
        self.assertEqual(
            classify_gene_type("TX3", "protein_coding", housekeeping),
            "Other",
        )

    def test_summary_uses_fixed_three_type_order(self):
        transcript_df = pd.DataFrame(
            {
                "Gene_Type": ["Housekeeping", "Other", "lncRNA"] * 2,
                "Observed_Mean_Log1p": [3.0, 1.5, 0.3, 3.5, 1.8, 0.5],
                "Predicted_Mean_Log1p": [2.9, 1.6, 0.4, 3.4, 1.7, 0.6],
            }
        )
        _, summary = summarize_by_biotype(transcript_df, n_bootstrap=0)
        self.assertEqual(
            summary["Gene_Type"].tolist(),
            ["Other", "lncRNA", "Housekeeping"],
        )


if __name__ == "__main__":
    unittest.main()

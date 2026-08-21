"""Tests for matched RBP-motif translation-effect analysis."""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda value: value
    yaml_module.safe_dump = lambda value, stream: None
    sys.modules["yaml"] = yaml_module

from eval.rbp_translation_effect import (
    RBPMotifMutagenesisEvaluator,
    build_motif_position_profiles,
    collect_rbp_motif_hits,
    collect_unique_transcript_samples,
    discover_de_novo_translation_motifs,
    disrupt_pwm_hit,
    extract_signed_translation_attribution_windows,
    load_known_motif_scan_cache,
    save_known_motif_scan_cache,
    scan_pwm_hits,
    summarize_rbp_motif_effects,
    validate_rbp_pwm_library,
)
from model.base_model import BaseModel


class TestCountHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(d_model, 1)

    def forward(self, representations, pad_mask=None):
        output = torch.nn.functional.softplus(self.projection(representations))
        if pad_mask is not None:
            output = output * pad_mask.unsqueeze(-1).to(output.dtype)
        return output


class RBPTranslationEffectTests(unittest.TestCase):
    def _build_model(self):
        model = BaseModel(
            d_seq=4,
            d_model=8,
            d_expr=2,
            d_cell_env=4,
            all_species=["human"],
            d_species=2,
            n_heads=2,
            number_of_layers=1,
            d_ff=16,
            adaptive_dim=4,
            p_drop=0.0,
        )
        model.add_head("count", TestCountHead(model.d_model))
        return model

    def test_pwm_scanning_recovers_exact_consensus(self):
        pwm = np.array([
            [10, 0, 0, 0],
            [0, 10, 0, 0],
            [0, 0, 10, 0],
        ], dtype=float)
        hits = scan_pwm_hits("TTACGTT", pwm, score_threshold=0.95)

        self.assertEqual(hits["Start"].tolist(), [2])
        self.assertEqual(hits["Sequence"].tolist(), ["ACG"])

    def test_pwm_validation_skips_invalid_and_missing_matrices(self):
        metadata = pd.DataFrame({
            "Matrix_id": ["M_VALID", "M_NAN", "M_NEG", "M_MISSING"],
            "Gene_name": ["RBP1", "RBP2", "RBP3", "RBP4"],
        })
        library = {
            "M_VALID": np.ones((3, 4)),
            "M_NAN": np.array([[1, 0, np.nan, 0]], dtype=float),
            "M_NEG": np.array([[1, 0, -1, 0]], dtype=float),
        }

        valid, audit = validate_rbp_pwm_library(library, metadata=metadata)
        status = audit.set_index("Matrix_ID")["Status"].to_dict()

        self.assertEqual(set(valid), {"M_VALID"})
        self.assertEqual(status["M_VALID"], "Valid")
        self.assertEqual(status["M_NAN"], "Invalid")
        self.assertEqual(status["M_NEG"], "Invalid")
        self.assertEqual(status["M_MISSING"], "Missing")

    def test_parallel_motif_scan_matches_sequential_scan(self):
        sequence = "TTACGTTACGTT"
        embedding = np.zeros((len(sequence), 4), dtype=np.float32)
        for position, base in enumerate(sequence):
            embedding[position, "ACGT".index(base)] = 1
        sample = {
            "Sequence": sequence,
            "Seq_Emb": embedding,
            "Transcript_Length": len(sequence),
            "CDS_Start_0based": 2,
            "CDS_End_exclusive": 10,
        }
        samples = {"TX1": sample, "TX2": dict(sample)}
        library = {
            "M1": np.array([
                [10, 0, 0, 0],
                [0, 10, 0, 0],
                [0, 0, 10, 0],
            ], dtype=float),
        }
        metadata = pd.DataFrame({
            "Matrix_id": ["M1"],
            "Gene_name": ["RBP1"],
        })

        sequential = collect_rbp_motif_hits(
            samples,
            library,
            metadata,
            score_threshold=0.95,
            num_workers=1,
        )
        parallel = collect_rbp_motif_hits(
            samples,
            library,
            metadata,
            score_threshold=0.95,
            num_workers=2,
        )
        process_parallel = collect_rbp_motif_hits(
            samples,
            library,
            metadata,
            score_threshold=0.95,
            num_workers=2,
            scan_backend="process",
            scan_chunk_size=1,
        )

        pd.testing.assert_frame_equal(sequential, parallel)
        pd.testing.assert_frame_equal(sequential, process_parallel)

    def test_known_motif_scan_cache_is_signature_aware(self):
        hits = pd.DataFrame({
            "Tid": ["TX1"],
            "Start": [3],
            "End": [9],
        })
        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / "known_rbp_motif_hits.pkl")
            save_known_motif_scan_cache(hits, cache_path, "signature-a")

            restored = load_known_motif_scan_cache(
                cache_path,
                expected_signature="signature-a",
            )
            stale = load_known_motif_scan_cache(
                cache_path,
                expected_signature="signature-b",
            )

        pd.testing.assert_frame_equal(restored, hits)
        self.assertIsNone(stale)

    def test_position_profiles_cover_known_and_de_novo_motifs(self):
        sequence = "AAACCCGGGTTTAAACCCGGGTTT"
        samples = {
            "T1": {
                "Sequence": sequence,
                "CDS_Start_0based": 6,
                "CDS_End_exclusive": 18,
                "Transcript_Length": len(sequence),
            }
        }
        hits = pd.DataFrame([{
            "Tid": "T1",
            "RBP_Name": "RBP1",
            "Start": 1,
            "End": 4,
            "Region": "5UTR",
            "PWM_Length": 3,
        }])
        de_novo = pd.DataFrame([{
            "Direction": "Positive",
            "Kmer": "AAA",
        }])

        profiles = build_motif_position_profiles(
            samples,
            known_hits=hits,
            de_novo_motifs=de_novo,
            bins_per_region=5,
        )

        known = profiles["known_rbp"]
        discovered = profiles["de_novo"]
        self.assertEqual(len(known), 15)
        self.assertEqual(known["Total_Hits"].max(), 1)
        self.assertGreater(discovered["Total_Hits"].max(), 0)
        self.assertAlmostEqual(known["Spatial_Probability"].sum(), 1.0)
        self.assertEqual(known["Bin_Size"].unique().tolist(), [20])
        self.assertEqual(known["Fixed_CDS_Length"].unique().tolist(), [100])
        self.assertTrue(np.isfinite(
            known["Log2_Positional_Enrichment"]
        ).all())

    def test_position_profiles_accept_fixed_lengths_and_bin_width(self):
        sequence = "A" * 60
        samples = {
            "T1": {
                "Sequence": sequence,
                "CDS_Start_0based": 10,
                "CDS_End_exclusive": 50,
                "Transcript_Length": len(sequence),
            }
        }
        hits = pd.DataFrame([{
            "Tid": "T1",
            "RBP_Name": "RBP1",
            "Start": 20,
            "End": 24,
            "Region": "CDS",
            "PWM_Length": 4,
        }])

        profile = build_motif_position_profiles(
            samples,
            known_hits=hits,
            bin_size=10,
            utr5_length=20,
            cds_length=40,
            utr3_length=20,
        )["known_rbp"]

        self.assertEqual(len(profile), 8)
        self.assertEqual(profile["Metagene_Position"].min(), -15)
        self.assertEqual(profile["Metagene_Position"].max(), 55)

    def test_unique_sample_collection_ignores_repeated_cell_types(self):
        sequence = np.eye(4, dtype=np.float32)[[0, 1, 2, 3, 0, 1, 2, 3, 0]]
        dataset = [
            (
                f"ENST000001.2-{cell_type}-0",
                "human",
                cell_type,
                np.zeros(2, dtype=np.float32),
                {"cds_start_pos": 4, "cds_end_pos": 9},
                sequence,
            )
            for cell_type in ("brain", "liver", "kidney")
        ]
        samples = collect_unique_transcript_samples(dataset, random_state=3)

        self.assertEqual(list(samples), ["ENST000001"])

    def test_pwm_disruption_changes_every_native_position(self):
        sequence = np.eye(4, dtype=np.float32)[[3, 0, 1, 2, 3]]
        pwm = np.eye(4, dtype=float)[[0, 1, 2]]
        disrupted, changes = disrupt_pwm_hit(sequence, 1, pwm)

        self.assertEqual(changes, 3)
        self.assertFalse(np.array_equal(disrupted[1:4], sequence[1:4]))

    def test_summary_uses_transcript_level_effects(self):
        effects = pd.DataFrame({
            "RBP_Name": ["RBP1"] * 6 + ["RBP2"] * 6,
            "Region": ["5UTR"] * 12,
            "Tid": [f"T{i}" for i in range(6)] * 2,
            "Delta_Log2_TE": [0.3, 0.2, 0.4, 0.1, 0.2, 0.3]
            + [-0.3, -0.2, -0.4, -0.1, -0.2, -0.3],
        })
        summary = summarize_rbp_motif_effects(
            effects,
            min_transcripts=5,
            bootstrap_iterations=200,
            random_state=1,
        ).set_index("RBP_Name")

        self.assertEqual(summary.loc["RBP1", "Direction"], "Positive")
        self.assertEqual(summary.loc["RBP2", "Direction"], "Negative")

    def test_evaluator_runs_paired_disruption_with_basemodel(self):
        sequence = np.eye(4, dtype=np.float32)[
            [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0]
        ]
        samples = {
            "ENST1": {
                "Seq_Emb": sequence,
                "Expr_Vector": np.zeros(2, dtype=np.float32),
                "Species": "human",
                "Cell_Type": "brain",
                "CDS_Start_0based": 3,
                "CDS_End_exclusive": 12,
                "Transcript_Length": 12,
                "Sequence": "ACGACGTACGTA",
            }
        }
        hits = pd.DataFrame([{
            "Hit_ID": "H1",
            "Tid": "ENST1",
            "RBP_Name": "RBP1",
            "Matrix_ID": "M1",
            "Region": "5UTR",
            "Start": 0,
            "End": 3,
            "PWM_Score": 1.0,
            "Motif_Sequence": "ACG",
            "Context_Sequence": "ACGACG",
        }])
        pwm = np.eye(4, dtype=float)[[0, 1, 2]]
        evaluator = RBPMotifMutagenesisEvaluator(
            self._build_model(),
            {"M1": pwm},
            prediction_scale="linear",
        )
        result = evaluator.evaluate_hits(
            hits,
            samples,
            batch_size=2,
            cds_skip_codons=0,
        )

        self.assertEqual(len(result), 1)
        self.assertTrue(np.isfinite(result.loc[0, "Delta_Log2_TE"]))

    def test_signed_attribution_returns_window_schema(self):
        sequence = np.eye(4, dtype=np.float32)[
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        ]
        samples = {
            "ENST1": {
                "Seq_Emb": sequence,
                "Expr_Vector": np.zeros(2, dtype=np.float32),
                "Species": "human",
                "Cell_Type": "brain",
                "CDS_Start_0based": 3,
                "CDS_End_exclusive": 12,
                "Transcript_Length": 12,
                "Sequence": "ACGTACGTACGT",
            }
        }
        windows = extract_signed_translation_attribution_windows(
            self._build_model(),
            samples,
            prediction_scale="linear",
            num_transcripts=1,
            peaks_per_direction=1,
            window_radius=2,
            cds_skip_codons=0,
        )

        self.assertIn("Signed_Attribution", windows.columns)
        self.assertIn("Context_Sequence", windows.columns)

    def test_de_novo_discovery_finds_enriched_kmer(self):
        rows = []
        for index in range(8):
            rows.append({
                "Tid": f"P{index}",
                "Context_Sequence": f"CCCAAAT{index % 4}".replace("0", "A")
                .replace("1", "C").replace("2", "G").replace("3", "T"),
                "Delta_Log2_TE": 0.8,
            })
        for index in range(10):
            rows.append({
                "Tid": f"B{index}",
                "Context_Sequence": "CCCGGGT",
                "Delta_Log2_TE": 0.01,
            })
        results, alignments = discover_de_novo_translation_motifs(
            pd.DataFrame(rows),
            k_values=(3,),
            min_foreground_occurrences=5,
            top_n_per_direction=5,
            logo_flank=1,
        )

        self.assertIn("AAA", set(results["Kmer"]))
        self.assertIn("Positive|AAA", alignments)


if __name__ == "__main__":
    unittest.main()

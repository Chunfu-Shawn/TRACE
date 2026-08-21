"""Tests for matched RBP-motif translation-effect analysis."""

import sys
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
    collect_unique_transcript_samples,
    discover_de_novo_translation_motifs,
    disrupt_pwm_hit,
    extract_signed_translation_attribution_windows,
    scan_pwm_hits,
    summarize_rbp_motif_effects,
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

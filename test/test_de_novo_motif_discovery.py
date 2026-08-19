"""Tests for BaseModel-compatible de novo motif discovery."""

import sys
import types
import unittest
from pathlib import Path

import numpy as np
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

try:
    import logomaker  # noqa: F401
except ModuleNotFoundError:
    logomaker_module = types.ModuleType("logomaker")
    sys.modules["logomaker"] = logomaker_module

from eval.de_novo_motif_discovery import (
    _extract_sample,
    _sequence_mask,
    compute_saliency_profile,
    extract_attention_positional_importance,
)
from model.base_model import BaseModel


class TestCountHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(d_model, 1)

    def forward(self, representations, pad_mask=None):
        output = torch.nn.functional.softplus(
            self.projection(representations)
        )
        if pad_mask is not None:
            output = output * pad_mask.unsqueeze(-1).to(output.dtype)
        return output


class DeNovoMotifBaseModelTests(unittest.TestCase):
    def _build_model(self):
        model = BaseModel(
            d_seq=4,
            d_model=8,
            d_expr=3,
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

    def _build_dataset(self):
        sequence_indices = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0])
        seq_emb = np.eye(4, dtype=np.float32)[sequence_indices]
        return [(
            "ENST1-brain-0",
            "human",
            "brain",
            np.zeros(3, dtype=np.float32),
            {"cds_start_pos": 4, "cds_end_pos": 9},
            seq_emb,
            object(),
        )]

    def test_sample_extraction_ignores_observed_count_supervision(self):
        sample = _extract_sample(self._build_dataset(), 0)

        self.assertNotIn("ce", sample)
        self.assertEqual(sample["se"].shape, (9, 4))
        self.assertEqual(sample["cds_start_0"], 3)

    def test_unpadded_sequence_mask_keeps_every_nucleotide(self):
        sequence = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0],
                                  [0.0, 0.0, 0.0, 0.0]]])

        mask = _sequence_mask(sequence)

        self.assertTrue(mask.all())

    def test_saliency_and_attention_run_with_basemodel_only(self):
        model = self._build_model()
        dataset = self._build_dataset()

        saliency = compute_saliency_profile(
            model,
            dataset,
            n_samples=1,
            max_len=20,
        )
        attention = extract_attention_positional_importance(
            model,
            dataset,
            n_samples=1,
            min_len=0,
            max_len=20,
        )

        self.assertIn("mean_saliency", saliency.columns)
        self.assertIn("mean_attn", attention.columns)

if __name__ == "__main__":
    unittest.main()

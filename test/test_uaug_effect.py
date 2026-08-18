"""Tests for BaseModel-compatible upstream start-codon evaluation."""

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch


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

from eval.uaug_effect import (
    MutantDataset,
    collate_fn_mutants,
    get_samples_5utr_clean_starts,
    uStartCodonEvaluatorEmb,
)
from model.base_model import BaseModel


class RecordingBaseModel(BaseModel):
    def __init__(self):
        super().__init__(
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
        self.last_predict_kwargs = None

    def predict(self, **kwargs):
        self.last_predict_kwargs = kwargs
        seq_batch = kwargs["seq_batch"]
        return {
            "count": torch.full(
                (seq_batch.shape[0], seq_batch.shape[1], 1),
                np.log(2.0),
                device=seq_batch.device,
            )
        }


class UaugBaseModelTests(unittest.TestCase):
    def test_prediction_uses_basemodel_inputs_without_count_batch(self):
        model = RecordingBaseModel()
        evaluator = uStartCodonEvaluatorEmb(model)
        seq_batch = torch.zeros(2, 7, 4)
        expr_batch = torch.zeros(2, 3)

        result = evaluator._predict_batch(
            seq_batch,
            expr_batch,
            ["human", "human"],
        )

        self.assertEqual(result.shape, (2, 7))
        np.testing.assert_allclose(result, 1.0)
        self.assertNotIn("count_batch", model.last_predict_kwargs)
        self.assertEqual(
            model.last_predict_kwargs["species"],
            ["human", "human"],
        )

    def test_sample_selection_ignores_trailing_count_supervision(self):
        seq_emb = np.eye(4, dtype=np.float32)[
            np.array(([0, 1, 2, 3] * 10)[:40])
        ]
        dataset = [
            (
                "sample-1",
                "human",
                "brain",
                np.zeros(3, dtype=np.float32),
                {"cds_start_pos": 21, "cds_end_pos": 35},
                seq_emb,
                np.ones((40, 1), dtype=np.float32),
            )
        ]

        samples = get_samples_5utr_clean_starts(
            dataset,
            top_n=1,
            utr_len_range=[20, 20],
            check_region_len=10,
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["species"], "human")
        self.assertNotIn("count_emb", samples[0])

    def test_mutant_collation_preserves_species(self):
        records = [{
            "uuid": "sample-1",
            "species": "human",
            "distance": 3,
            "codon": "ATG",
            "frame": "In-frame",
            "te_wt": 1.0,
            "m_start": 10,
            "m_end": 20,
            "seq_emb": np.zeros((25, 4), dtype=np.float32),
            "expr_vector": np.zeros(3, dtype=np.float32),
        }]

        batch = collate_fn_mutants([MutantDataset(records)[0]])

        self.assertEqual(batch[1], ["human"])


if __name__ == "__main__":
    unittest.main()

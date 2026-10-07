"""Regression tests for portable FASTA demo data loading."""

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.save_prediction_results import _prepare_prediction_dataloader
from model.translation_predictor import (
    DeNovoSequenceDataset,
    TranslationProfilePredictor,
    collate_fn_denovo,
)


class PredictionDataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.dataset = DeNovoSequenceDataset(
            {"demo": "ATG" * 100}, "human", "liver", np.zeros(16840, dtype=np.float32)
        )

    def test_fasta_defaults_to_zero_workers(self):
        parameter = inspect.signature(TranslationProfilePredictor.run).parameters["num_workers"]
        self.assertEqual(parameter.default, 0)

    def test_single_process_loader(self):
        loader, rank, world_size = _prepare_prediction_dataloader(
            self.dataset, collate_fn_denovo, None, 1, num_workers=0
        )
        self.assertEqual((rank, world_size), (0, 1))
        self.assertEqual(loader.num_workers, 0)
        self.assertIsNone(loader.prefetch_factor)
        self.assertFalse(loader.persistent_workers)
        batch = next(iter(loader))
        self.assertEqual(batch[4].shape, (1, 300, 4))

    def test_other_prediction_helpers_keep_four_workers(self):
        loader, _, _ = _prepare_prediction_dataloader(
            self.dataset, collate_fn_denovo, None, 1
        )
        self.assertEqual(loader.num_workers, 4)
        self.assertEqual(loader.prefetch_factor, 4)
        self.assertTrue(loader.persistent_workers)

    def test_negative_workers_raise(self):
        with self.assertRaisesRegex(ValueError, "num_workers"):
            _prepare_prediction_dataloader(
                self.dataset, collate_fn_denovo, None, 1, num_workers=-1
            )


if __name__ == "__main__":
    unittest.main()

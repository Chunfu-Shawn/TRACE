"""Regression tests for strict zero-expression Trainer mode."""

import os
import sys

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train.model_trainer import Trainer as LegacyTrainer
from train.model_trainer_seq import Trainer as SequenceTrainer


def _batch():
    """Build a minimal batch matching TranslationDataset.__getitem__."""
    expression = torch.tensor([1.0, -2.0, 3.0, 0.5])
    sequence = torch.zeros(6, 4)
    target = torch.ones(6, 1)
    metadata = {
        "cds_start_pos": 1,
        "cds_end_pos": 6,
        "motif_occ": [],
    }
    return [("sample-1", "human", "liver", expression, metadata, sequence, target)]


def _bare_trainer(trainer_class, force_zero_expression):
    """Create only the state required by the collate method."""
    trainer = object.__new__(trainer_class)
    trainer.mask_perc = {"species": 0.0, "cell": 0.0}
    trainer.force_zero_expression = force_zero_expression
    trainer.cell_mean_expr = {}
    return trainer


def test_sequence_trainer_forces_zero_during_training_and_validation():
    trainer = _bare_trainer(SequenceTrainer, True)

    for is_eval in (False, True):
        result = trainer.collate_mask_pad_batch_to_cuda(_batch(), is_eval=is_eval)
        cell_mask, expression = result[3], result[4]
        assert torch.all(cell_mask)
        assert torch.equal(expression, torch.zeros_like(expression))


def test_sequence_trainer_preserves_real_validation_expression_when_disabled():
    trainer = _bare_trainer(SequenceTrainer, False)
    result = trainer.collate_mask_pad_batch_to_cuda(_batch(), is_eval=True)

    cell_mask, expression = result[3], result[4]
    assert not torch.any(cell_mask)
    assert torch.equal(expression[0], _batch()[0][3])


def test_legacy_trainer_forces_zero_during_training_and_validation():
    trainer = _bare_trainer(LegacyTrainer, True)
    trainer.current_mask_range = (1.0, 1.0)
    trainer.current_replacement_probs = {"mask": 1.0, "random": 0.0, "keep": 0.0}

    for is_eval in (False, True):
        result = trainer.collate_mask_pad_batch_to_cuda(_batch(), is_eval=is_eval)
        cell_mask, expression = result[2], result[3]
        assert torch.all(cell_mask)
        assert torch.equal(expression, torch.zeros_like(expression))

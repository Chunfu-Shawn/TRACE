"""Tests for continuous cell-expression interpolation."""

import os
import sys

import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train.expression_augmentation import augment_expression_batch


def test_strict_zero_mask_remains_exact_with_noise():
    expression = torch.ones(4, 8)
    zero_mask = torch.tensor([True, False, True, False])
    augmented, strengths = augment_expression_batch(
        expression,
        zero_mask,
        interpolation_probability=1.0,
        noise_std=0.1,
    )

    assert torch.equal(augmented[zero_mask], torch.zeros_like(augmented[zero_mask]))
    assert torch.equal(strengths[zero_mask], torch.zeros_like(strengths[zero_mask]))


def test_all_eligible_samples_receive_continuous_strengths():
    torch.manual_seed(7)
    expression = torch.ones(32, 4)
    zero_mask = torch.zeros(32, dtype=torch.bool)
    augmented, strengths = augment_expression_batch(
        expression,
        zero_mask,
        interpolation_probability=1.0,
        noise_std=0.0,
    )

    assert torch.all((strengths >= 0.0) & (strengths <= 1.0))
    assert torch.any((strengths > 0.0) & (strengths < 1.0))
    assert torch.allclose(augmented, strengths.unsqueeze(1).expand_as(expression))


def test_disabled_interpolation_keeps_unmasked_expression():
    expression = torch.randn(6, 5)
    zero_mask = torch.tensor([False, True, False, False, True, False])
    augmented, strengths = augment_expression_batch(
        expression,
        zero_mask,
        interpolation_probability=0.0,
        noise_std=0.0,
    )

    assert torch.equal(augmented[~zero_mask], expression[~zero_mask])
    assert torch.equal(augmented[zero_mask], torch.zeros_like(augmented[zero_mask]))
    assert torch.equal(strengths[~zero_mask], torch.ones_like(strengths[~zero_mask]))


def test_noise_is_added_after_expression_strength():
    torch.manual_seed(11)
    expression = torch.ones(8, 3)
    zero_mask = torch.zeros(8, dtype=torch.bool)
    augmented, strengths = augment_expression_batch(
        expression,
        zero_mask,
        interpolation_probability=1.0,
        noise_std=0.2,
    )

    torch.manual_seed(11)
    torch.rand(8)
    expected_strengths = torch.rand(8)
    expected_noise = torch.randn_like(expression) * 0.2
    expected = expression * expected_strengths.unsqueeze(1) + expected_noise

    assert torch.equal(strengths, expected_strengths)
    assert torch.allclose(augmented, expected)

"""Training-time augmentation for continuous cell-expression vectors."""

from __future__ import annotations

from typing import Tuple

import torch


def augment_expression_batch(
    expression_batch: torch.Tensor,
    strict_zero_mask: torch.Tensor,
    interpolation_probability: float,
    noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mix selected non-masked vectors continuously between zero and real expression.

    Strictly masked samples remain exactly zero. Other samples retain their full
    expression vector or, with ``interpolation_probability``, receive a scalar
    strength sampled uniformly from [0, 1]. Gaussian noise is applied before
    scaling so the zero endpoint remains exact.
    """
    if expression_batch.ndim != 2:
        raise ValueError(
            f"expression_batch must be 2D, got shape {tuple(expression_batch.shape)}"
        )
    probability = float(interpolation_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("interpolation_probability must be between 0 and 1")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    zero_mask = torch.as_tensor(
        strict_zero_mask, dtype=torch.bool, device=expression_batch.device
    ).reshape(-1)
    if zero_mask.numel() != expression_batch.shape[0]:
        raise ValueError(
            "strict_zero_mask length must equal the expression batch size"
        )

    strengths = torch.ones(
        expression_batch.shape[0], dtype=torch.float32, device=expression_batch.device
    )
    eligible = ~zero_mask
    if probability > 0.0:
        interpolation_mask = (
            torch.rand(expression_batch.shape[0], device=expression_batch.device)
            < probability
        ) & eligible
        sampled_strengths = torch.rand(
            expression_batch.shape[0], device=expression_batch.device
        )
        strengths = torch.where(interpolation_mask, sampled_strengths, strengths)
    strengths[zero_mask] = 0.0

    augmented = expression_batch
    if noise_std > 0:
        noise = (
            torch.randn_like(expression_batch, dtype=torch.float32) * float(noise_std)
        )
        augmented = augmented + noise.to(dtype=expression_batch.dtype)
    augmented = augmented * strengths.to(dtype=expression_batch.dtype).unsqueeze(1)
    return augmented, strengths

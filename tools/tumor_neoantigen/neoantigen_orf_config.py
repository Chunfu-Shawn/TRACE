"""Shared ORF-calling configuration for tumor-antigen workflows."""

from __future__ import annotations

from typing import Any


NEOANTIGEN_START_CODONS = ("ATG", "CTG", "GTG", "TTG")


def build_neoantigen_orf_kwargs(mode: str = "balanced") -> dict[str, Any]:
    """Return an independent copy of the validated tumor-antigen ORF settings."""
    return {
        "start_codons": list(NEOANTIGEN_START_CODONS),
        "min_len": 30,
        "mode": mode,
        "use_mane_filter": False,
        "plot_density": False,
        "long_mode_length_only": False,
        "hard_thresh_intensity": 0.01,
        "hard_thresh_periodicity": 0.5,
        "hard_thresh_uniformity": 0.8,
        "hard_thresh_step_up": 0.5,
        "hard_thresh_drop_off": 0.8,
        "ranking_strategy": "occupancy_expression",
        "score_features": ["step_up_contrast", "drop_off"],
        "tpm_exponent": 1.0,
        "collapse_boundary_weight": 0.5,
        "start_codon_prior_strength": 0.25,
        "nms_iou_threshold": 0.7,
        "nms_respect_frame": False,
    }

#!/usr/bin/env python3
"""Run test inference and orchestrate the existing TRACE evaluation APIs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.translation_dataset import TranslationDataset
from eval.metagene_profile import evaluate_metagene_TIS_TTS_profile
from eval.periodicity_corr import (
    evaluate_periodicity_correlation,
    summarize_periodicity_results,
)
from eval.psite_pos_wise_corr_depth import (
    calculate_correlations_multitissue,
    plot_correlation_by_cell_type,
)
from eval.save_prediction_results import save_count_predictions
from eval.utr_cds_region_difference import evaluate_region_specificity
from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead
from plot.psite_profile_pred import PredictionVisualizer


# -----------------------------------------------------------------------------
# Evaluation configuration: edit this section before running the script.
# -----------------------------------------------------------------------------
DATASET_PATH = (
    PROJECT_ROOT.parent
    / "dataset"
    / "human_7c_6k_depth0.1_cov0.1_rpm1.test.h5"
)
MODEL_CONFIG_PATH = SRC_DIR / "config/base_model_384d_16h_12l_64env_32ad.yaml"
CHECKPOINT_PATH = (
    PROJECT_ROOT.parent
    / "checkpoint/train"
    / (
        "base_model_384d_16h_12l_64env_32ad-PsiteDensityHead."
        "hs_22c_18c_26c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1_e50_a2_b02.100_0.001."
        "best_profile.pt"
    )
)

OUTPUT_DIR = PROJECT_ROOT.parent / "results/test_evaluation"
PREDICTION_SUFFIX = "best_profile"
HEAD_HIDDEN_DIM = 384
BATCH_SIZE = 1
REUSE_EXISTING_PREDICTIONS = False
METAGENE_WINDOW = 60

# Set these paths to enable transcript-class labels in the periodicity plot.
HOUSEKEEPING_GENES_PATH = None
GTF_PATH = None

CASE_TRANSCRIPTS = [
    ("ENST00000332859.11", "brain_cerebrum"),
    ("ENST00000654422.1", "testis"),
    ("ENST00000789734.1", "prostate"),
]


def load_model(device: torch.device):
    """Create BaseModel with its count head and restore one training checkpoint."""
    if not MODEL_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Model config was not found: {MODEL_CONFIG_PATH}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {CHECKPOINT_PATH}\n"
            "Edit CHECKPOINT_PATH at the top of this script."
        )

    model = BaseModel.from_config(str(MODEL_CONFIG_PATH))
    model.add_head(
        "count",
        PsiteDensityHead.create_from_model(model, d_pred_h=HEAD_HIDDEN_DIM),
        overwrite=True,
        move_to_model_device=False,
    )
    model.to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        metadata = checkpoint
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint
    elif isinstance(checkpoint, dict) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
        metadata = {}
    else:
        raise ValueError(f"Unsupported checkpoint format: {CHECKPOINT_PATH}")

    state_dict = model._strip_head_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(
        f"Loaded checkpoint: {CHECKPOINT_PATH}\n"
        f"  epoch={metadata.get('epoch', 'unknown')}\n"
        f"  best profile Spearman={metadata.get('best_profile_spearman', 'unknown')}\n"
        f"  best scale Spearman={metadata.get('best_scale_spearman', 'unknown')}"
    )
    return model, metadata


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Test dataset was not found: {DATASET_PATH}\n"
            "Edit DATASET_PATH at the top of this script."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = TranslationDataset.from_h5(str(DATASET_PATH), lazy=True)
    print(f"Test samples: {len(dataset):,}; cell types: {dataset.cell_type_counts}")

    model, checkpoint_metadata = load_model(device)
    prediction_name = (
        f"predictions_count.{model.model_name}.{PREDICTION_SUFFIX}.pkl"
    )
    prediction_path = OUTPUT_DIR / prediction_name
    if REUSE_EXISTING_PREDICTIONS and prediction_path.is_file():
        print(f"Reusing prediction file: {prediction_path}")
    else:
        prediction_path = Path(
            save_count_predictions(
                model=model,
                dataset=dataset,
                num_samples=None,
                batch_size=BATCH_SIZE,
                out_dir=str(OUTPUT_DIR),
                suffix=PREDICTION_SUFFIX,
            )
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    evaluate_metagene_TIS_TTS_profile(
        dataset=dataset,
        pkl_path=str(prediction_path),
        window_size=METAGENE_WINDOW,
        out_dir=str(OUTPUT_DIR),
        suffix=PREDICTION_SUFFIX,
        unlog_data=True,
    )
    region_df = evaluate_region_specificity(
        truth_dataset=dataset,
        pkl_input=str(prediction_path),
        out_dir=str(OUTPUT_DIR),
        suffix=PREDICTION_SUFFIX,
    )
    periodicity_df = evaluate_periodicity_correlation(
        dataset=dataset,
        pkl_path=str(prediction_path),
        hk_genes_path=HOUSEKEEPING_GENES_PATH,
        gtf_path=GTF_PATH,
        out_dir=str(OUTPUT_DIR),
        suffix=PREDICTION_SUFFIX,
    )

    case_dir = OUTPUT_DIR / "cases"
    visualizer = PredictionVisualizer(
        str(prediction_path), dataset, out_dir=str(case_dir)
    )
    for tid, cell_type in CASE_TRANSCRIPTS:
        visualizer.plot_transcript(
            tid=tid,
            cell_type=cell_type,
            suffix=PREDICTION_SUFFIX,
        )

    full_corr_df = calculate_correlations_multitissue(
        dataset=dataset,
        pkl_input=str(prediction_path),
        output_dir=str(OUTPUT_DIR),
        suffix=PREDICTION_SUFFIX,
        for_cds=False,
    )
    cds_corr_df = calculate_correlations_multitissue(
        dataset=dataset,
        pkl_input=str(prediction_path),
        output_dir=str(OUTPUT_DIR),
        suffix=f"{PREDICTION_SUFFIX}.cds",
        for_cds=True,
    )
    full_cell_summary = plot_correlation_by_cell_type(
        full_corr_df,
        str(OUTPUT_DIR),
        suffix=PREDICTION_SUFFIX,
        metric="Spearman_R",
    )
    cds_cell_summary = plot_correlation_by_cell_type(
        cds_corr_df,
        str(OUTPUT_DIR),
        suffix=f"{PREDICTION_SUFFIX}.cds",
        metric="Spearman_R",
    )

    periodicity_summary = summarize_periodicity_results(periodicity_df)
    summary = {
        "dataset": str(DATASET_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "prediction_pkl": str(prediction_path),
        "region_rows": len(region_df),
        "periodicity": periodicity_summary,
        "full_positionwise_n": len(full_corr_df),
        "full_positionwise_mean_spearman": full_corr_df["Spearman_R"].mean(),
        "cds_positionwise_n": len(cds_corr_df),
        "cds_positionwise_mean_spearman": cds_corr_df["Spearman_R"].mean(),
        "full_cell_type_summary": full_cell_summary.to_dict(orient="records"),
        "cds_cell_type_summary": cds_cell_summary.to_dict(orient="records"),
    }
    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)

    print("\nEvaluation complete.")
    print(f"Prediction PKL: {prediction_path}")
    print(f"Summary: {summary_path}")
    print(f"Figures and tables: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

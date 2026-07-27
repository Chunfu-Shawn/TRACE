#!/usr/bin/env python3
"""Diagnose expression-condition generalization on identical test RNAs.

Each target RNA is predicted three times with:
1. its real cell-type expression vector;
2. an all-zero expression vector;
3. the nearest expression vector observed in the model's training datasets.

The script evaluates CDS position-wise correlation, CDS mean scale, and
CDS-anchored three-nucleotide periodicity for every condition.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.translation_dataset import TranslationDataset
from eval.evaluation_utils import (
    cds_slice,
    cds_with_stop_slice,
    to_1d_signal,
    transcript_id_from_uuid,
)
from eval.periodicity_corr import calculate_periodicity
from eval.psite_pos_wise_corr_depth import _correlation_pair
from eval.save_prediction_results import _extract_head_tensor
from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead


# -----------------------------------------------------------------------------
# Diagnostic configuration: edit this section before running the script.
# -----------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
TEST_DATASET_PATH = DATASET_DIR / "human_7c_6k_depth0.1_cov0.1_rpm1.test.h5"

# These files must match the expression environments seen by the checkpoint.
# For the tissue-only checkpoint, keep only the human tissue training file.
REFERENCE_DATASET_PATHS = [
    DATASET_DIR / "hs_22c_18c_26c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1.train.h5",
]

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

OUTPUT_DIR = PROJECT_ROOT.parent / "results/expression_generalization_diagnostic"
TARGET_CELL_TYPES = ("HeLa", "HEK293T")

# Optional explicit replacements, for example {"HeLa": "liver"}.
# Empty entries are resolved automatically by cosine distance.
REFERENCE_CELL_OVERRIDES: Dict[str, str] = {}

HEAD_HIDDEN_DIM = 384
BATCH_SIZE = 1
NUM_WORKERS = 4
MAX_TRANSCRIPTS_PER_CELL: Optional[int] = None
RANDOM_SEED = 42
HIGH_PERIODICITY_THRESHOLD = 0.60


CONDITION_LABELS = {
    "real_expression": "Real expression",
    "zero_expression": "Zero expression",
    "nearest_train_expression": "Nearest training expression",
}


def require_files(paths: Iterable[Path], label: str) -> None:
    """Raise one readable error listing all missing input files."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label}: " + ", ".join(missing))


def load_model(device: torch.device) -> Tuple[BaseModel, dict]:
    """Build BaseModel, attach its density head, and restore a checkpoint."""
    require_files([MODEL_CONFIG_PATH], "model config")
    require_files([CHECKPOINT_PATH], "checkpoint")

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
    return model, metadata


def load_expression_vectors(paths: Iterable[Path]) -> Dict[str, np.ndarray]:
    """Load unique cell-type expression vectors from one or more HDF5 files."""
    vectors: Dict[str, np.ndarray] = {}
    for path in paths:
        dataset = TranslationDataset.from_h5(str(path), lazy=True)
        for cell_type, vector in dataset.cell_expr_dict.items():
            value = np.asarray(vector, dtype=np.float32).reshape(-1)
            if cell_type in vectors and not np.allclose(vectors[cell_type], value):
                raise ValueError(
                    f"Cell type '{cell_type}' has inconsistent expression vectors "
                    f"across reference datasets."
                )
            vectors[cell_type] = value
    if not vectors:
        raise ValueError("No expression vectors were found in reference datasets.")
    return vectors


def expression_distance(first: np.ndarray, second: np.ndarray) -> Tuple[float, float]:
    """Return cosine distance and root-mean-square difference."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise ValueError(
            f"Expression shapes do not match: {first.shape} versus {second.shape}"
        )
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    cosine_distance = (
        float("inf") if denominator == 0 else 1.0 - float(np.dot(first, second) / denominator)
    )
    rmse = float(np.sqrt(np.mean(np.square(first - second))))
    return cosine_distance, rmse


def match_reference_cells(
    target_vectors: Dict[str, np.ndarray],
    reference_vectors: Dict[str, np.ndarray],
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Match every target cell to one expression environment seen in training."""
    matches: Dict[str, str] = {}
    records = []

    for target_cell in TARGET_CELL_TYPES:
        if target_cell not in target_vectors:
            raise KeyError(f"Target expression vector was not found: {target_cell}")

        candidates = []
        for reference_cell, reference_vector in reference_vectors.items():
            if reference_cell in TARGET_CELL_TYPES:
                continue
            cosine_distance, rmse = expression_distance(
                target_vectors[target_cell], reference_vector
            )
            candidates.append((cosine_distance, rmse, reference_cell))
        if not candidates:
            raise ValueError(f"No eligible reference cells were found for {target_cell}.")

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        automatic_cell = candidates[0][2]
        selected_cell = REFERENCE_CELL_OVERRIDES.get(target_cell, automatic_cell)
        if selected_cell not in reference_vectors:
            raise KeyError(
                f"Reference override '{selected_cell}' for {target_cell} was not found."
            )
        matches[target_cell] = selected_cell

        selected_cosine, selected_rmse = expression_distance(
            target_vectors[target_cell], reference_vectors[selected_cell]
        )
        records.append(
            {
                "Target_Cell_Type": target_cell,
                "Selected_Reference_Cell": selected_cell,
                "Automatic_Nearest_Cell": automatic_cell,
                "Cosine_Distance": selected_cosine,
                "Expression_RMSE": selected_rmse,
                "Used_Override": selected_cell != automatic_cell,
            }
        )

        print(
            f"Expression match: {target_cell} -> {selected_cell} "
            f"(cosine distance={selected_cosine:.6f}, RMSE={selected_rmse:.6f})"
        )

    return matches, pd.DataFrame.from_records(records)


def select_target_indices(dataset: TranslationDataset) -> List[int]:
    """Select the same deterministic RNA set used by all three conditions."""
    rng = np.random.default_rng(RANDOM_SEED)
    selected = []
    for cell_type in TARGET_CELL_TYPES:
        indices = np.asarray(
            [
                index
                for index, sample_cell in enumerate(dataset.cell_types)
                if str(sample_cell) == cell_type
            ],
            dtype=np.int64,
        )
        if MAX_TRANSCRIPTS_PER_CELL is not None and len(indices) > MAX_TRANSCRIPTS_PER_CELL:
            indices = np.sort(
                rng.choice(indices, size=MAX_TRANSCRIPTS_PER_CELL, replace=False)
            )
        print(f"Selected {len(indices):,} test RNAs for {cell_type}.")
        selected.extend(indices.tolist())
    if not selected:
        raise ValueError(f"No samples matched target cells: {TARGET_CELL_TYPES}")
    return selected


def collate_target_batch(batch):
    """Pad sequences and targets while preserving per-RNA metadata."""
    uuids, species, cell_types, expr_vectors, meta_infos, sequences, targets = zip(*batch)
    lengths = [int(sequence.shape[0]) for sequence in sequences]
    return (
        list(uuids),
        list(species),
        list(cell_types),
        torch.stack(expr_vectors),
        list(meta_infos),
        pad_sequence(sequences, batch_first=True, padding_value=-1),
        pad_sequence(targets, batch_first=True, padding_value=-1),
        lengths,
    )


def autocast_context(device: torch.device):
    """Use CUDA mixed precision when available and remain a no-op on CPU."""
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def condition_expression_batch(
    condition: str,
    real_expression: torch.Tensor,
    cell_types: List[str],
    reference_matches: Dict[str, str],
    reference_vectors: Dict[str, np.ndarray],
) -> torch.Tensor:
    """Construct one of the three expression inputs for the current batch."""
    if condition == "real_expression":
        return real_expression
    if condition == "zero_expression":
        return torch.zeros_like(real_expression)
    if condition == "nearest_train_expression":
        rows = [
            torch.from_numpy(reference_vectors[reference_matches[cell_type]])
            for cell_type in cell_types
        ]
        return torch.stack(rows).to(dtype=real_expression.dtype)
    raise KeyError(f"Unknown diagnostic condition: {condition}")


def safe_expm1(values: np.ndarray) -> np.ndarray:
    """Convert log1p signals to finite non-negative linear values."""
    linear = np.expm1(np.asarray(values, dtype=np.float32))
    linear = np.nan_to_num(linear, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(linear, 0.0, None)


def evaluate_one_prediction(
    uuid: str,
    cell_type: str,
    condition: str,
    reference_cell: Optional[str],
    meta_info: dict,
    target_log: np.ndarray,
    prediction_log: np.ndarray,
) -> Optional[dict]:
    """Calculate CDS profile and periodicity metrics for one RNA prediction."""
    length = min(len(target_log), len(prediction_log))
    if length < 3:
        return None
    target_log = np.asarray(target_log[:length], dtype=np.float32)
    prediction_log = np.asarray(prediction_log[:length], dtype=np.float32)

    profile_bounds = cds_with_stop_slice(meta_info, length)
    periodicity_bounds = cds_slice(meta_info, length)
    if profile_bounds is None or periodicity_bounds is None:
        return None

    profile_start, profile_end = profile_bounds
    periodicity_start, periodicity_end = periodicity_bounds
    if profile_end - profile_start < 3 or periodicity_end - periodicity_start < 3:
        return None

    target_cds_log = target_log[profile_start:profile_end]
    prediction_cds_log = prediction_log[profile_start:profile_end]
    _, _, cds_spearman, _ = _correlation_pair(prediction_cds_log, target_cds_log)

    target_linear = safe_expm1(target_log)
    prediction_linear = safe_expm1(prediction_log)
    observed_periodicity = calculate_periodicity(
        target_linear, periodicity_start, periodicity_end
    )
    predicted_periodicity = calculate_periodicity(
        prediction_linear, periodicity_start, periodicity_end
    )

    target_cds_linear = target_linear[profile_start:profile_end]
    prediction_cds_linear = prediction_linear[profile_start:profile_end]
    observed_cds_mean = float(np.mean(target_cds_linear))
    predicted_cds_mean = float(np.mean(prediction_cds_linear))

    return {
        "UUID": str(uuid),
        "Tid": transcript_id_from_uuid(uuid),
        "Cell_Type": cell_type,
        "Condition": condition,
        "Condition_Label": CONDITION_LABELS[condition],
        "Reference_Cell_Type": reference_cell,
        "Transcript_Length": length,
        "CDS_Length": periodicity_end - periodicity_start,
        "RPF_Depth": float(meta_info.get("rpf_depth", np.nan)),
        "RPF_Coverage": float(meta_info.get("rpf_coverage", np.nan)),
        "CDS_Profile_Spearman": cds_spearman,
        "Observed_Periodicity": observed_periodicity,
        "Predicted_Periodicity": predicted_periodicity,
        "Periodicity_Error": predicted_periodicity - observed_periodicity,
        "Periodicity_Absolute_Error": abs(
            predicted_periodicity - observed_periodicity
        ),
        "Observed_CDS_Mean": observed_cds_mean,
        "Predicted_CDS_Mean": predicted_cds_mean,
        "CDS_Mean_Absolute_Error": abs(predicted_cds_mean - observed_cds_mean),
    }


def run_diagnostic_inference(
    model: BaseModel,
    dataset: TranslationDataset,
    indices: List[int],
    reference_matches: Dict[str, str],
    reference_vectors: Dict[str, np.ndarray],
    device: torch.device,
) -> pd.DataFrame:
    """Run the three expression conditions on identical mini-batches."""
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_target_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    conditions = tuple(CONDITION_LABELS)
    records = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Expression generalization diagnostic"):
            (
                uuids,
                species,
                cell_types,
                real_expression,
                meta_infos,
                sequences,
                targets,
                lengths,
            ) = batch

            sequences = sequences.to(device, non_blocking=device.type == "cuda")
            positions = torch.arange(sequences.shape[1]).unsqueeze(0)
            src_mask = positions < torch.tensor(lengths).unsqueeze(1)
            src_mask = src_mask.to(device, non_blocking=device.type == "cuda")

            condition_predictions = {}
            for condition in conditions:
                expression_batch = condition_expression_batch(
                    condition,
                    real_expression,
                    cell_types,
                    reference_matches,
                    reference_vectors,
                ).to(device, non_blocking=device.type == "cuda")
                with autocast_context(device):
                    output = model(
                        seq_batch=sequences,
                        species=species,
                        expr_vector=expression_batch,
                        src_mask=src_mask,
                        head_names=["count"],
                    )
                condition_predictions[condition] = (
                    _extract_head_tensor(output, "count").detach().float().cpu()
                )

            targets = targets.float().cpu()
            for sample_index, uuid in enumerate(uuids):
                valid_length = lengths[sample_index]
                target_signal = to_1d_signal(targets[sample_index, :valid_length])
                cell_type = str(cell_types[sample_index])
                for condition in conditions:
                    prediction_signal = to_1d_signal(
                        condition_predictions[condition][sample_index, :valid_length]
                    )
                    reference_cell = (
                        reference_matches[cell_type]
                        if condition == "nearest_train_expression"
                        else None
                    )
                    record = evaluate_one_prediction(
                        uuid=str(uuid),
                        cell_type=cell_type,
                        condition=condition,
                        reference_cell=reference_cell,
                        meta_info=meta_infos[sample_index],
                        target_log=target_signal,
                        prediction_log=prediction_signal,
                    )
                    if record is not None:
                        records.append(record)

    return pd.DataFrame.from_records(records)


def safe_spearman(first: pd.Series, second: pd.Series) -> float:
    """Return a finite Spearman correlation or NaN for degenerate inputs."""
    values = pd.DataFrame({"first": first, "second": second}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(values) < 2:
        return float("nan")
    _, _, correlation, _ = _correlation_pair(
        values["first"].to_numpy(), values["second"].to_numpy()
    )
    return correlation


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate profile, periodicity, and scale metrics by cell and condition."""
    summaries = []
    for (cell_type, condition), group in results.groupby(
        ["Cell_Type", "Condition"], sort=False
    ):
        finite_profile = group["CDS_Profile_Spearman"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        finite_periodicity = group[
            ["Observed_Periodicity", "Predicted_Periodicity"]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        high_periodicity = finite_periodicity[
            finite_periodicity["Observed_Periodicity"] >= HIGH_PERIODICITY_THRESHOLD
        ]

        summaries.append(
            {
                "Cell_Type": cell_type,
                "Condition": condition,
                "Condition_Label": CONDITION_LABELS[condition],
                "N": int(len(group)),
                "CDS_Profile_N": int(len(finite_profile)),
                "Mean_CDS_Profile_Spearman": float(finite_profile.mean()),
                "Median_CDS_Profile_Spearman": float(finite_profile.median()),
                "Periodicity_N": int(len(finite_periodicity)),
                "Periodicity_Spearman": safe_spearman(
                    finite_periodicity["Observed_Periodicity"],
                    finite_periodicity["Predicted_Periodicity"],
                ),
                "Periodicity_MAE": float(
                    np.mean(
                        np.abs(
                            finite_periodicity["Predicted_Periodicity"]
                            - finite_periodicity["Observed_Periodicity"]
                        )
                    )
                ),
                "Periodicity_Bias": float(
                    np.mean(
                        finite_periodicity["Predicted_Periodicity"]
                        - finite_periodicity["Observed_Periodicity"]
                    )
                ),
                "High_Periodicity_N": int(len(high_periodicity)),
                "High_Periodicity_MAE": float(
                    np.mean(
                        np.abs(
                            high_periodicity["Predicted_Periodicity"]
                            - high_periodicity["Observed_Periodicity"]
                        )
                    )
                ),
                "High_Periodicity_Bias": float(
                    np.mean(
                        high_periodicity["Predicted_Periodicity"]
                        - high_periodicity["Observed_Periodicity"]
                    )
                ),
                "CDS_Mean_MAE": float(group["CDS_Mean_Absolute_Error"].mean()),
            }
        )
    return pd.DataFrame.from_records(summaries)


def plot_periodicity_comparison(results: pd.DataFrame, output_path: Path) -> None:
    """Plot observed versus predicted periodicity for all expression conditions."""
    cells = [cell for cell in TARGET_CELL_TYPES if cell in set(results["Cell_Type"])]
    fig, axes = plt.subplots(1, len(cells), figsize=(6 * len(cells), 5), squeeze=False)
    colors = {
        "real_expression": "#0072B2",
        "zero_expression": "#777777",
        "nearest_train_expression": "#D55E00",
    }

    for axis, cell_type in zip(axes[0], cells):
        cell_data = results[results["Cell_Type"] == cell_type]
        for condition in CONDITION_LABELS:
            condition_data = cell_data[cell_data["Condition"] == condition]
            axis.scatter(
                condition_data["Observed_Periodicity"],
                condition_data["Predicted_Periodicity"],
                s=8,
                alpha=0.18,
                linewidths=0,
                color=colors[condition],
                label=CONDITION_LABELS[condition],
            )
        axis.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1)
        axis.axvline(
            HIGH_PERIODICITY_THRESHOLD,
            color="black",
            linestyle=":",
            linewidth=1,
        )
        axis.set_xlim(0.3, 1.0)
        axis.set_ylim(0.3, 1.0)
        axis.set_title(cell_type)
        axis.set_xlabel("Observed CDS periodicity")
        axis.set_ylabel("Predicted CDS periodicity")
        axis.grid(alpha=0.2)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(results: pd.DataFrame, output_path: Path) -> None:
    """Plot paired distributions of profile accuracy and periodicity error."""
    plot_data = results.copy()
    condition_order = list(CONDITION_LABELS.values())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(
        data=plot_data,
        x="Condition_Label",
        y="CDS_Profile_Spearman",
        hue="Cell_Type",
        order=condition_order,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("CDS position-wise Spearman")
    axes[0].tick_params(axis="x", rotation=20)

    sns.boxplot(
        data=plot_data,
        x="Condition_Label",
        y="Periodicity_Error",
        hue="Cell_Type",
        order=condition_order,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Predicted - observed periodicity")
    axes[1].tick_params(axis="x", rotation=20)

    if axes[1].legend_ is not None:
        axes[1].legend_.remove()
    axes[0].legend(title="Cell type", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def json_safe(value):
    """Convert NumPy and non-finite values into JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def main() -> None:
    """Run the complete expression-generalization diagnostic."""
    require_files([TEST_DATASET_PATH], "test dataset")
    require_files(REFERENCE_DATASET_PATHS, "reference training datasets")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    test_dataset = TranslationDataset.from_h5(str(TEST_DATASET_PATH), lazy=True)

    reference_vectors = load_expression_vectors(REFERENCE_DATASET_PATHS)
    target_vectors = {
        cell_type: np.asarray(test_dataset.cell_expr_dict[cell_type], dtype=np.float32)
        for cell_type in TARGET_CELL_TYPES
        if cell_type in test_dataset.cell_expr_dict
    }
    reference_matches, match_table = match_reference_cells(
        target_vectors, reference_vectors
    )
    match_table.to_csv(OUTPUT_DIR / "expression_reference_matches.csv", index=False)

    indices = select_target_indices(test_dataset)
    model, checkpoint_metadata = load_model(device)
    print(
        f"Loaded checkpoint epoch={checkpoint_metadata.get('epoch', 'unknown')}: "
        f"{CHECKPOINT_PATH}"
    )

    results = run_diagnostic_inference(
        model=model,
        dataset=test_dataset,
        indices=indices,
        reference_matches=reference_matches,
        reference_vectors=reference_vectors,
        device=device,
    )
    if results.empty:
        raise RuntimeError("No CDS-containing target RNAs were available for evaluation.")

    details_path = OUTPUT_DIR / "expression_condition_metrics.csv"
    summary_path = OUTPUT_DIR / "expression_condition_summary.csv"
    results.to_csv(details_path, index=False)
    summary = summarize_results(results)
    summary.to_csv(summary_path, index=False)

    plot_periodicity_comparison(
        results, OUTPUT_DIR / "periodicity_by_expression_condition.pdf"
    )
    plot_metric_comparison(
        results, OUTPUT_DIR / "metrics_by_expression_condition.pdf"
    )

    report = {
        "test_dataset": str(TEST_DATASET_PATH),
        "reference_datasets": [str(path) for path in REFERENCE_DATASET_PATHS],
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "target_cell_types": list(TARGET_CELL_TYPES),
        "reference_matches": reference_matches,
        "evaluated_rnas": int(results["UUID"].nunique()),
        "high_periodicity_threshold": HIGH_PERIODICITY_THRESHOLD,
        "summary": summary.to_dict(orient="records"),
    }
    with (OUTPUT_DIR / "diagnostic_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_safe(report), handle, indent=2, ensure_ascii=False)

    print("\nExpression-condition diagnostic summary:")
    print(summary.to_string(index=False))
    print(f"\nDetailed metrics: {details_path}")
    print(f"Summary metrics: {summary_path}")
    print(f"Figures and JSON report: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

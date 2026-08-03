#!/usr/bin/env python3
"""Evaluate 5-cell architecture ablations on unseen cell environments.

The script resolves one validation-selected checkpoint per model, reuses or
creates test-set prediction PKLs, calculates RNA-level metrics, aggregates them
with equal weight per held-out cell type, and draws a four-panel comparison of
RNA profile Spearman, CDS profile Spearman, CDS-mean Spearman, and CDS-mean MAE.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.translation_dataset import TranslationDataset
from eval.evaluation_utils import (
    cds_slice,
    get_prediction,
    load_prediction_input,
    to_1d_signal,
    transcript_id_from_uuid,
)
from eval.save_prediction_results import save_count_predictions
from model.base_model import BaseModel
from model.base_model_hybrid import BaseModelHybrid
from model.prediction_heads import PsiteDensityHead
from model.translation_base_model_LN import BaseModelLN
from model.translation_base_model_conv import BaseModelConv


# -----------------------------------------------------------------------------
# Evaluation configuration
# -----------------------------------------------------------------------------
DATASET_DIR = Path("/public-supool/home/annie/translation_model/dataset")
CHECKPOINT_DIR = Path("/public-supool/home/annie/translation_model/checkpoint/train")
OUTPUT_DIR = Path(
    "/public-supool/home/annie/translation_model/results/ablation/architecture_zero_shot"
)

TRAIN_REFERENCE_DATASET = (
    DATASET_DIR / "human_5c_6k_depth0.1_cov0.1_rpm1.train.h5"
)
TEST_DATASET_PATH = (
    DATASET_DIR
    / "human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.test.h5"
)

# Use one validation criterion for every model. This avoids selecting models on
# either of the two test endpoints shown in the final figure.
CHECKPOINT_SUFFIX = ".best_total.pt"
HEAD_HIDDEN_DIM = 384
BATCH_SIZE = 30
NUM_TEST_SAMPLES: Optional[int] = None
MIN_RPF_DEPTH: Optional[float] = 0.5
MIN_RNA_PER_CELL = 50
BOOTSTRAP_ITERATIONS = 2000
RANDOM_SEED = 42
REUSE_EXISTING_PREDICTIONS = True
REUSE_EXISTING_METRICS = True
FAIL_FAST = False


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one 5-cell architecture or training ablation."""

    model_id: str
    label: str
    model_class: Type[torch.nn.Module]
    config_path: Path
    checkpoint_glob: str
    color: str
    force_zero_expression: bool = False
    enabled: bool = True
    checkpoint: Optional[Path] = None


MODEL_SPECS = (
    ModelSpec(
        model_id="trace_zero",
        label="TRACE (Zero)",
        model_class=BaseModel,
        config_path=SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml",
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*a2_b02_zero*"
        ),
        color="#777777",
        force_zero_expression=True,
    ),
    ModelSpec(
        model_id="trace_real",
        label="TRACE (Real)",
        model_class=BaseModel,
        config_path=SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml",
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*a2_b02_real*"
        ),
        color="#78A9CF",
        enabled=False,
    ),
    ModelSpec(
        model_id="trace_exp_aug",
        label="TRACE",
        model_class=BaseModel,
        config_path=SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml",
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*a2_b02_exp_aug*"
        ),
        color="#166A9A",
    ),
    ModelSpec(
        model_id="trace_exp_aug_no_ranking",
        label="TRACE (Mask + interpolation, no ranking)",
        model_class=BaseModel,
        config_path=SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml",
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*a2_b0_exp_aug*"
        ),
        color="#9A6FB0",
        enabled=False,
    ),
    ModelSpec(
        model_id="ln_transformer",
        label="LN Transformer",
        model_class=BaseModelLN,
        config_path=SRC_DIR / "config/base_model_LN_384d_16h_12l.yaml",
        checkpoint_glob="base_model_LN*hs_5c*",
        color="#C28548",
    ),
    ModelSpec(
        model_id="conv_model",
        label="Conv model",
        model_class=BaseModelConv,
        config_path=SRC_DIR / "config/base_model_conv_384d_12l_7k.yaml",
        checkpoint_glob="base_model_conv*hs_5c*",
        color="#5F9272",
    ),
    ModelSpec(
        model_id="hybrid_transformer",
        label="Hybrid Transformer",
        model_class=BaseModelHybrid,
        config_path=(
            SRC_DIR / "config/base_model_hybrid_384d_16h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob="base_model_hybrid*hs_5c*",
        color="#D67A3A",
        enabled=False,
    ),
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def file_fingerprint(path: Path) -> Dict[str, object]:
    """Return lightweight file provenance used for persistent caches."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def manifest_matches(path: Path, expected: Mapping[str, object]) -> bool:
    """Return whether a JSON manifest contains the expected provenance."""
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            observed = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    return observed == dict(expected)


def resolve_checkpoint(spec: ModelSpec) -> Tuple[Path, int]:
    """Resolve an explicit checkpoint or the newest matching checkpoint."""
    if spec.checkpoint is not None:
        checkpoint = spec.checkpoint.expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint.resolve(), 1

    pattern = spec.checkpoint_glob + CHECKPOINT_SUFFIX
    matches = sorted(
        CHECKPOINT_DIR.glob(pattern), key=lambda path: path.stat().st_mtime_ns
    )
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matched {pattern!r} inside {CHECKPOINT_DIR}"
        )
    return matches[-1].resolve(), len(matches)


def checkpoint_state_dict(checkpoint: object, path: Path) -> Tuple[dict, dict]:
    """Extract a state dictionary and metadata from supported checkpoints."""
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    if isinstance(checkpoint, dict) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint, {}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def load_model(
    spec: ModelSpec,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict]:
    """Build one configured model, attach its head, and restore weights."""
    if not spec.config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {spec.config_path}")
    model = spec.model_class.from_config(str(spec.config_path))
    model.add_head(
        "count",
        PsiteDensityHead.create_from_model(model, d_pred_h=HEAD_HIDDEN_DIM),
        overwrite=True,
        move_to_model_device=False,
    )
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, metadata = checkpoint_state_dict(checkpoint, checkpoint_path)
    state_dict = model._strip_head_module_prefix(state_dict)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module.") :]: value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, metadata


def prediction_manifest(spec: ModelSpec, checkpoint_path: Path) -> Dict[str, object]:
    """Describe every input that determines a prediction cache."""
    return {
        "model_id": spec.model_id,
        "test_dataset": file_fingerprint(TEST_DATASET_PATH),
        "checkpoint": file_fingerprint(checkpoint_path),
        "model_config": file_fingerprint(spec.config_path),
        "force_zero_expression": spec.force_zero_expression,
        "num_test_samples": NUM_TEST_SAMPLES,
        "storage_dtype": "float32",
    }


def cache_tag(manifest: Mapping[str, object]) -> str:
    """Create a stable cache tag from sorted provenance fields."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def prediction_cache_path(
    spec: ModelSpec,
    manifest: Mapping[str, object],
) -> Optional[Path]:
    """Locate a verified prediction cache for one model."""
    if not REUSE_EXISTING_PREDICTIONS:
        return None
    prediction_dir = OUTPUT_DIR / "predictions"
    tag = cache_tag(manifest)
    candidates = sorted(prediction_dir.glob(f"predictions_count.*.{spec.model_id}.{tag}.pkl"))
    for candidate in candidates:
        if manifest_matches(Path(str(candidate) + ".manifest.json"), manifest):
            return candidate
    return None


def predict_model(
    spec: ModelSpec,
    model: torch.nn.Module,
    dataset: TranslationDataset,
    checkpoint_path: Path,
) -> Tuple[Path, bool]:
    """Reuse or generate predictions for one architecture."""
    manifest = prediction_manifest(spec, checkpoint_path)
    cached = prediction_cache_path(spec, manifest)
    if cached is not None:
        print(f"Reusing predictions for {spec.label}: {cached}")
        return cached, True

    prediction_dir = OUTPUT_DIR / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    tag = cache_tag(manifest)
    prediction_path = Path(
        save_count_predictions(
            model=model,
            dataset=dataset,
            num_samples=NUM_TEST_SAMPLES,
            batch_size=BATCH_SIZE,
            out_dir=str(prediction_dir),
            suffix=f"{spec.model_id}.{tag}",
            force_zero_expression=spec.force_zero_expression,
            storage_dtype=np.float32,
        )
    )
    with Path(str(prediction_path) + ".manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    return prediction_path, False


def safe_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Spearman correlation or NaN for invalid or constant arrays."""
    first_array = np.asarray(first, dtype=np.float64).reshape(-1)
    second_array = np.asarray(second, dtype=np.float64).reshape(-1)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    first_array = first_array[valid]
    second_array = second_array[valid]
    if (
        len(first_array) < 2
        or np.ptp(first_array) == 0
        or np.ptp(second_array) == 0
    ):
        return float("nan")
    return float(spearmanr(first_array, second_array).statistic)


def evaluate_prediction_file(
    dataset: TranslationDataset,
    prediction_path: Path,
    spec: ModelSpec,
) -> pd.DataFrame:
    """Calculate RNA/CDS profile and CDS-mean values for every matched RNA."""
    predictions = load_prediction_input(str(prediction_path))
    records = []
    for index in tqdm(range(len(dataset)), desc=f"Evaluate {spec.label}"):
        uuid, _, cell_type, _, meta_info, _, target = dataset[index]
        tid = transcript_id_from_uuid(uuid)
        cell_type = str(cell_type)
        prediction = get_prediction(predictions, cell_type, tid)
        if prediction is None:
            continue

        target_signal = to_1d_signal(target)
        prediction_signal = to_1d_signal(prediction)
        length = min(len(target_signal), len(prediction_signal))
        bounds = cds_slice(meta_info, length)
        if bounds is None or length < 2:
            continue
        cds_start, cds_end = bounds
        if cds_end - cds_start < 3:
            continue

        target_signal = np.asarray(target_signal[:length], dtype=np.float32)
        prediction_signal = np.asarray(prediction_signal[:length], dtype=np.float32)
        target_cds = target_signal[cds_start:cds_end]
        prediction_cds = prediction_signal[cds_start:cds_end]
        observed_cds_mean = float(np.mean(target_cds))
        predicted_cds_mean = float(np.mean(prediction_cds))
        records.append(
            {
                "Model_ID": spec.model_id,
                "Model_Label": spec.label,
                "UUID": str(uuid),
                "Tid": tid,
                "Cell_Type": cell_type,
                "RPF_Depth": float(meta_info.get("rpf_depth", np.nan)),
                "RNA_Profile_Spearman": safe_spearman(
                    prediction_signal, target_signal
                ),
                "CDS_Profile_Spearman": safe_spearman(
                    prediction_cds, target_cds
                ),
                "Observed_CDS_Mean_Log1p": observed_cds_mean,
                "Predicted_CDS_Mean_Log1p": predicted_cds_mean,
                "CDS_Mean_Absolute_Error": abs(
                    predicted_cds_mean - observed_cds_mean
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def metric_manifest(prediction_path: Path, spec: ModelSpec) -> Dict[str, object]:
    """Describe inputs that determine one RNA-level metric cache."""
    return {
        "model_id": spec.model_id,
        "prediction": file_fingerprint(prediction_path),
        "test_dataset": file_fingerprint(TEST_DATASET_PATH),
        "metric_definition": "rna_cds_profile_and_cds_mean_log1p_v2",
    }


def load_or_evaluate_metrics(
    dataset: TranslationDataset,
    prediction_path: Path,
    spec: ModelSpec,
) -> Tuple[pd.DataFrame, Path, bool]:
    """Reuse or calculate RNA-level metrics for one prediction file."""
    metric_dir = OUTPUT_DIR / "rna_metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metric_dir / f"{prediction_path.stem}.rna_metrics.csv"
    manifest_path = Path(str(csv_path) + ".manifest.json")
    manifest = metric_manifest(prediction_path, spec)
    if (
        REUSE_EXISTING_METRICS
        and csv_path.is_file()
        and manifest_matches(manifest_path, manifest)
    ):
        print(f"Reusing RNA metrics for {spec.label}: {csv_path}")
        return pd.read_csv(csv_path), csv_path, True

    metrics = evaluate_prediction_file(dataset, prediction_path, spec)
    metrics.to_csv(csv_path, index=False)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return metrics, csv_path, False


def finite_mean(values: Sequence[float]) -> float:
    """Return the mean of finite values or NaN when none remain."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def summarize_cell_types(
    transcript_metrics: pd.DataFrame,
    spec: ModelSpec,
    expected_cell_types: Sequence[str],
) -> pd.DataFrame:
    """Calculate equal-status metrics for every held-out cell type."""
    rows = []
    for cell_type in expected_cell_types:
        raw = transcript_metrics[transcript_metrics["Cell_Type"] == cell_type]
        if MIN_RPF_DEPTH is None:
            filtered = raw
        else:
            depth = pd.to_numeric(raw["RPF_Depth"], errors="coerce")
            filtered = raw[np.isfinite(depth) & (depth >= MIN_RPF_DEPTH)]
        eligible = len(filtered) >= MIN_RNA_PER_CELL
        rna_profile_n = int(filtered["RNA_Profile_Spearman"].notna().sum())
        cds_profile_n = int(filtered["CDS_Profile_Spearman"].notna().sum())
        scale_columns = filtered[
            [
                "Observed_CDS_Mean_Log1p",
                "Predicted_CDS_Mean_Log1p",
                "CDS_Mean_Absolute_Error",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        scale_mask = np.isfinite(scale_columns).all(axis=1)
        scale_rows = scale_columns[scale_mask]
        rows.append(
            {
                "Model_ID": spec.model_id,
                "Model_Label": spec.label,
                "Cell_Type": cell_type,
                "RNA_N": int(len(raw)),
                "RNA_Passing_Depth_N": int(len(filtered)),
                "RNA_Excluded_By_Depth_N": int(len(raw) - len(filtered)),
                "Meets_Min_RNA_Per_Cell": bool(eligible),
                "RNA_Profile_N": rna_profile_n,
                "RNA_Profile_Excluded_N": int(len(filtered) - rna_profile_n),
                "CDS_Profile_N": cds_profile_n,
                "CDS_Profile_Excluded_N": int(len(filtered) - cds_profile_n),
                "CDS_Mean_N": int(len(scale_rows)),
                "CDS_Mean_Excluded_N": int(len(filtered) - len(scale_rows)),
                "Mean_RNA_Profile_Spearman": (
                    finite_mean(filtered["RNA_Profile_Spearman"])
                    if eligible
                    else float("nan")
                ),
                "Mean_CDS_Profile_Spearman": (
                    finite_mean(filtered["CDS_Profile_Spearman"])
                    if eligible
                    else float("nan")
                ),
                "CDS_Mean_Spearman": (
                    safe_spearman(
                        scale_rows["Observed_CDS_Mean_Log1p"],
                        scale_rows["Predicted_CDS_Mean_Log1p"],
                    )
                    if eligible
                    else float("nan")
                ),
                "CDS_Mean_MAE": (
                    finite_mean(scale_rows["CDS_Mean_Absolute_Error"])
                    if eligible
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def bootstrap_mean_ci(
    values: Sequence[float],
    rng: np.random.Generator,
) -> Tuple[float, float, float, int]:
    """Bootstrap a mean across held-out cell types."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(array.mean())
    if len(array) == 1 or BOOTSTRAP_ITERATIONS < 2:
        return mean, mean, mean, int(len(array))
    indices = rng.integers(
        0, len(array), size=(BOOTSTRAP_ITERATIONS, len(array))
    )
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(low), float(high), int(len(array))


def summarize_models(
    cell_metrics: pd.DataFrame,
    ordered_specs: Sequence[ModelSpec],
) -> pd.DataFrame:
    """Summarize all endpoints with equal weight per held-out cell type."""
    rows = []
    metrics = (
        "Mean_RNA_Profile_Spearman",
        "Mean_CDS_Profile_Spearman",
        "CDS_Mean_Spearman",
        "CDS_Mean_MAE",
    )
    for model_index, spec in enumerate(ordered_specs):
        group = cell_metrics[cell_metrics["Model_ID"] == spec.model_id]
        if group.empty:
            continue
        row = {
            "Model_ID": spec.model_id,
            "Model_Label": spec.label,
            "Cell_Type_N": int(group["Cell_Type"].nunique()),
            "Eligible_Cell_Type_N": int(
                group["Meets_Min_RNA_Per_Cell"].fillna(False).sum()
            ),
            "RNA_N": int(group["RNA_N"].sum()),
            "RNA_Passing_Depth_N": int(group["RNA_Passing_Depth_N"].sum()),
            "Min_RPF_Depth": MIN_RPF_DEPTH,
            "Min_RNA_Per_Cell": MIN_RNA_PER_CELL,
        }
        for metric_index, metric in enumerate(metrics):
            rng = np.random.Generator(
                np.random.PCG64(RANDOM_SEED + model_index * 100 + metric_index)
            )
            mean, low, high, n = bootstrap_mean_ci(group[metric], rng)
            row[metric] = mean
            row[f"{metric}_CI95_Low"] = low
            row[f"{metric}_CI95_High"] = high
            row[f"{metric}_Cell_N"] = n
        rows.append(row)
    return pd.DataFrame.from_records(rows)


PANEL_METRICS = (
    (
        "Mean_RNA_Profile_Spearman",
        "RNA profile shape",
        "Mean per-cell RNA profile Spearman",
    ),
    (
        "Mean_CDS_Profile_Spearman",
        "CDS profile shape",
        "Mean CDS profile Spearman",
    ),
    (
        "CDS_Mean_Spearman",
        "CDS signal scale",
        "CDS-mean Spearman",
    ),
    (
        "CDS_Mean_MAE",
        "CDS scale error",
        "CDS-mean MAE",
    ),
)


def plot_architecture_comparison(
    cell_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    ordered_specs: Sequence[ModelSpec],
) -> None:
    """Draw held-out cell points and equal-cell bootstrap summaries."""
    available_ids = set(model_metrics["Model_ID"])
    specs = [spec for spec in ordered_specs if spec.model_id in available_ids]
    y_positions = np.arange(len(specs))[::-1]
    figure, axes = plt.subplots(
        1,
        len(PANEL_METRICS),
        figsize=(12.2, max(2.8, 0.42 * len(specs) + 1.2)),
        sharey=True,
    )
    jitter_rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))

    for panel_index, (metric, title, x_label) in enumerate(PANEL_METRICS):
        axis = axes[panel_index]
        for y_position, spec in zip(y_positions, specs):
            cell_group = cell_metrics[cell_metrics["Model_ID"] == spec.model_id]
            values = pd.to_numeric(cell_group[metric], errors="coerce").to_numpy()
            values = values[np.isfinite(values)]
            jitter = jitter_rng.uniform(-0.10, 0.10, size=len(values))
            axis.scatter(
                values,
                np.full(len(values), y_position) + jitter,
                s=13,
                color=spec.color,
                alpha=0.28,
                edgecolors="none",
                zorder=1,
            )

            summary = model_metrics[model_metrics["Model_ID"] == spec.model_id].iloc[0]
            mean = float(summary[metric])
            low = float(summary[f"{metric}_CI95_Low"])
            high = float(summary[f"{metric}_CI95_High"])
            axis.errorbar(
                mean,
                y_position,
                xerr=np.asarray([[mean - low], [high - mean]]),
                fmt="D",
                markersize=5,
                markerfacecolor=spec.color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                color=spec.color,
                capsize=2.5,
                linewidth=1.2,
                zorder=3,
            )

        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_yticks(y_positions)
        axis.set_yticklabels([spec.label for spec in specs])
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.text(
            0.01,
            0.98,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    filter_text = f"RPF depth ≥ {MIN_RPF_DEPTH:g}" if MIN_RPF_DEPTH is not None else "all RNAs"
    figure.text(
        0.5,
        0.01,
        "Dots: held-out cell types; diamonds: equal-cell means; bars: "
        f"cell-bootstrap 95% CIs; {filter_text}; ≥{MIN_RNA_PER_CELL} RNAs per cell.",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )
    figure.subplots_adjust(
        left=0.20, right=0.99, bottom=0.18, top=0.88, wspace=0.24
    )

    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = figure_dir / "architecture_zero_shot_shape_and_scale"
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def verify_unseen_cell_types(
    train_dataset: TranslationDataset,
    test_dataset: TranslationDataset,
) -> List[str]:
    """Confirm that test cell types are absent from the 5-cell training set."""
    train_cells = set(map(str, train_dataset.cell_types))
    test_cells = sorted(set(map(str, test_dataset.cell_types)))
    overlap = sorted(train_cells.intersection(test_cells))
    if overlap:
        raise ValueError(
            "Test environments overlap the 5-cell training environments: "
            + ", ".join(overlap)
        )
    return test_cells


def main() -> None:
    """Run prediction, metric calculation, aggregation, and plotting."""
    for path in (TRAIN_REFERENCE_DATASET, TEST_DATASET_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = TranslationDataset.from_h5(
        str(TRAIN_REFERENCE_DATASET), lazy=True
    )
    test_dataset = TranslationDataset.from_h5(str(TEST_DATASET_PATH), lazy=True)
    test_cell_types = verify_unseen_cell_types(train_dataset, test_dataset)
    print(
        f"Verified {len(test_cell_types)} unseen test cell types: "
        + ", ".join(test_cell_types)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enabled_specs = [spec for spec in MODEL_SPECS if spec.enabled]
    transcript_tables = []
    cell_tables = []
    manifest_rows = []

    for spec in enabled_specs:
        row: Dict[str, object] = {
            "Model_ID": spec.model_id,
            "Model_Label": spec.label,
            "Model_Class": spec.model_class.__name__,
            "Config": str(spec.config_path),
            "Checkpoint": "",
            "Checkpoint_Match_N": 0,
            "Checkpoint_Epoch": np.nan,
            "Prediction_PKL": "",
            "Prediction_Reused": False,
            "RNA_Metrics_CSV": "",
            "Metrics_Reused": False,
            "Status": "pending",
            "Error": "",
        }
        try:
            checkpoint_path, match_count = resolve_checkpoint(spec)
            row["Checkpoint"] = str(checkpoint_path)
            row["Checkpoint_Match_N"] = match_count
            if match_count > 1:
                print(
                    f"[{spec.label}] {match_count} checkpoints matched; "
                    f"using newest: {checkpoint_path}"
                )

            model, checkpoint_metadata = load_model(spec, checkpoint_path, device)
            row["Checkpoint_Epoch"] = checkpoint_metadata.get("epoch", np.nan)
            prediction_path, prediction_reused = predict_model(
                spec, model, test_dataset, checkpoint_path
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            transcript_metrics, metric_path, metrics_reused = load_or_evaluate_metrics(
                test_dataset, prediction_path, spec
            )
            cell_metrics = summarize_cell_types(
                transcript_metrics, spec, test_cell_types
            )
            transcript_tables.append(transcript_metrics)
            cell_tables.append(cell_metrics)
            row.update(
                {
                    "Prediction_PKL": str(prediction_path),
                    "Prediction_Reused": prediction_reused,
                    "RNA_Metrics_CSV": str(metric_path),
                    "Metrics_Reused": metrics_reused,
                    "Status": "complete",
                }
            )
        except Exception as error:
            row["Status"] = "failed"
            row["Error"] = f"{type(error).__name__}: {error}"
            print(f"[{spec.label}] failed: {row['Error']}")
            if FAIL_FAST:
                raise
        manifest_rows.append(row)

    checkpoint_manifest = pd.DataFrame.from_records(manifest_rows)
    checkpoint_manifest.to_csv(OUTPUT_DIR / "checkpoint_manifest.csv", index=False)
    if not cell_tables:
        raise RuntimeError("No architecture produced evaluable test predictions")

    transcript_metrics = pd.concat(transcript_tables, ignore_index=True)
    cell_metrics = pd.concat(cell_tables, ignore_index=True)
    model_metrics = summarize_models(cell_metrics, enabled_specs)
    transcript_metrics.to_csv(OUTPUT_DIR / "rna_metrics.csv", index=False)
    cell_metrics.to_csv(OUTPUT_DIR / "cell_type_metrics.csv", index=False)
    model_metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    plot_architecture_comparison(cell_metrics, model_metrics, enabled_specs)

    print("\nZero-shot architecture summary:")
    print(
        model_metrics[
            [
                "Model_Label",
                "Eligible_Cell_Type_N",
                "Mean_RNA_Profile_Spearman",
                "Mean_CDS_Profile_Spearman",
                "CDS_Mean_Spearman",
                "CDS_Mean_MAE",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

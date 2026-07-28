#!/usr/bin/env python3
"""Evaluate TRACE environment diversity on held-out uncommon cell types.

The script performs four steps for every configured checkpoint:
1. predict the same uncommon-cell test dataset and save a float32 PKL file;
2. calculate transcript-level CDS profile, periodicity, and CDS-mean metrics;
3. aggregate metrics with equal weight per held-out cell type and save CSV tables;
4. draw environment-diversity and expression-distance figures for profile shape
   and signal scale.

Edit the configuration section and run this file directly on the server. The
default comparison contains 5/22/40 training environments crossed with zero,
real, and mask-plus-interpolation expression strategies.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None


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
from eval.periodicity_corr import calculate_periodicity
from eval.save_prediction_results import save_count_predictions
from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead


# -----------------------------------------------------------------------------
# Evaluation configuration: edit this section before running the script.
# -----------------------------------------------------------------------------
DATASET_DIR = Path("/public-supool/home/annie/translation_model/dataset")
CHECKPOINT_DIR = Path("/public-supool/home/annie/translation_model/TRACE/checkpoint/train")
OUTPUT_DIR = Path("/public-supool/home/annie/translation_model/results/environment_diversity")

TEST_DATASET_PATH = (
    DATASET_DIR
    / "human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.test.h5"
)

# These environment sets must match the datasets used to train the checkpoints.
# The 40-cell set is the union of the 22 tissues and 18 common cell lines.
TRAIN_ENVIRONMENT_DATASETS: Dict[int, List[Path]] = {
    5: [DATASET_DIR / "human_5c_6k_depth0.1_cov0.1_rpm1.train.h5"],
    22: [DATASET_DIR / "human_tissue_22c_6k_depth0.1_cov0.1_rpm1.train.h5"],
    40: [
        DATASET_DIR / "human_tissue_22c_6k_depth0.1_cov0.1_rpm1.train.h5",
        DATASET_DIR / "human_cell_line_18c_6k_depth0.1_cov0.1_rpm1.train.h5",
    ],
}

MODEL_CONFIG_PATH = SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
HEAD_HIDDEN_DIM = 384

# Use one checkpoint-selection rule for all nine runs. ``best_profile`` is the
# default because profile shape is the primary endpoint; change this to
# ``.best_total.pt`` if total-loss selection is preferred for the final study.
CHECKPOINT_SUFFIX = ".best_profile.pt"

# Exact paths take priority. Add entries here if a rule below matches more than
# one checkpoint, for example:
# (5, "zero"): CHECKPOINT_DIR / "exact_checkpoint.best_profile.pt"
EXACT_CHECKPOINTS: Dict[Tuple[int, str], Path] = {}

# Automatic matching is intentionally strict. Edit the tokens if the server-side
# checkpoint names use a different dataset tag.
CHECKPOINT_MATCH_RULES = {
    (5, "zero"): {"required": ("hs_5c", "zero"), "forbidden": ()},
    (5, "real"): {"required": ("hs_5c", "real"), "forbidden": ()},
    (5, "exp_aug"): {"required": ("hs_5c", "exp_aug"), "forbidden": ()},
    (22, "zero"): {"required": ("hs_22c", "zero"), "forbidden": ("18c",)},
    (22, "real"): {"required": ("hs_22c", "real"), "forbidden": ("18c",)},
    (22, "exp_aug"): {
        "required": ("hs_22c", "exp_aug"),
        "forbidden": ("18c",),
    },
    (40, "zero"): {"required": ("hs_22c_18c", "zero"), "forbidden": ()},
    (40, "real"): {"required": ("hs_22c_18c", "real"), "forbidden": ()},
    (40, "exp_aug"): {
        "required": ("hs_22c_18c", "exp_aug"),
        "forbidden": (),
    },
}

STRATEGY_LABELS = {
    "zero": "Zero expression",
    "real": "Real expression",
    "exp_aug": "Mask + interpolation",
}
STRATEGY_COLORS = {
    "zero": "#777777",
    "real": "#5B8DB8",
    "exp_aug": "#D67A3A",
}
STRATEGY_MARKERS = {"zero": "o", "real": "s", "exp_aug": "^"}
ENVIRONMENT_MARKERS = {5: "o", 22: "s", 40: "D"}

BATCH_SIZE = 1
REUSE_EXISTING_PREDICTIONS = True
EXPECTED_TEST_CELL_TYPES = 26
REQUIRE_DISJOINT_ENVIRONMENTS = True
HIGH_PERIODICITY_THRESHOLD = 0.60
BOOTSTRAP_ITERATIONS = 2000
RANDOM_SEED = 42

# Set to a positive integer for a quick pipeline test. Keep None for final results.
NUM_TEST_SAMPLES: Optional[int] = None


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint in the environment-count by strategy comparison."""

    environment_count: int
    strategy: str
    checkpoint: Path
    config_path: Path

    @property
    def model_id(self) -> str:
        return f"{self.environment_count}c_{self.strategy}"

    @property
    def strategy_label(self) -> str:
        return STRATEGY_LABELS[self.strategy]


def require_files(paths: Iterable[Path], label: str) -> None:
    """Raise one readable exception containing every missing path."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label}: " + ", ".join(missing))


def resolve_checkpoint(environment_count: int, strategy: str) -> Path:
    """Resolve one exact checkpoint without silently choosing among duplicates."""
    key = (environment_count, strategy)
    exact = EXACT_CHECKPOINTS.get(key)
    if exact is not None:
        if not exact.is_file():
            raise FileNotFoundError(f"Configured checkpoint does not exist: {exact}")
        return exact.resolve()

    if key not in CHECKPOINT_MATCH_RULES:
        raise KeyError(f"No checkpoint matching rule was configured for {key}")
    rule = CHECKPOINT_MATCH_RULES[key]
    required = tuple(token.lower() for token in rule.get("required", ()))
    forbidden = tuple(token.lower() for token in rule.get("forbidden", ()))
    matches = []
    if CHECKPOINT_DIR.is_dir():
        for path in CHECKPOINT_DIR.glob(f"*{CHECKPOINT_SUFFIX}"):
            name = path.name.lower()
            if all(token in name for token in required) and not any(
                token in name for token in forbidden
            ):
                matches.append(path.resolve())

    if len(matches) != 1:
        formatted = "\n  ".join(str(path) for path in sorted(matches)) or "none"
        raise RuntimeError(
            f"Expected exactly one checkpoint for {environment_count}c/{strategy}, "
            f"found {len(matches)}:\n  {formatted}\n"
            "Set an exact path in EXACT_CHECKPOINTS to make the comparison reproducible."
        )
    return matches[0]


def build_model_specs() -> List[ModelSpec]:
    """Create the nine ordered model specifications."""
    specs = []
    for environment_count in (5, 22, 40):
        for strategy in ("zero", "real", "exp_aug"):
            specs.append(
                ModelSpec(
                    environment_count=environment_count,
                    strategy=strategy,
                    checkpoint=resolve_checkpoint(environment_count, strategy),
                    config_path=MODEL_CONFIG_PATH,
                )
            )
    return specs


def checkpoint_state_dict(checkpoint: object, path: Path) -> Tuple[dict, dict]:
    """Extract a model state dictionary and metadata from supported formats."""
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    if isinstance(checkpoint, dict) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint, {}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def load_model(spec: ModelSpec, device: torch.device) -> Tuple[BaseModel, dict]:
    """Construct BaseModel, attach the density head, and restore one checkpoint."""
    require_files([spec.config_path, spec.checkpoint], "model inputs")
    model = BaseModel.from_config(str(spec.config_path))
    model.add_head(
        "count",
        PsiteDensityHead.create_from_model(model, d_pred_h=HEAD_HIDDEN_DIM),
        overwrite=True,
        move_to_model_device=False,
    )
    model.to(device)

    checkpoint = torch.load(spec.checkpoint, map_location=device)
    state_dict, metadata = checkpoint_state_dict(checkpoint, spec.checkpoint)
    state_dict = model._strip_head_module_prefix(state_dict)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(
        f"Loaded {spec.model_id}: epoch={metadata.get('epoch', 'unknown')}, "
        f"checkpoint={spec.checkpoint}"
    )
    return model, metadata


def file_fingerprint(path: Path) -> dict:
    """Return inexpensive provenance fields used to reject stale predictions."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def prediction_manifest(spec: ModelSpec) -> dict:
    """Describe every input that determines a saved prediction file."""
    return {
        "model_id": spec.model_id,
        "checkpoint": file_fingerprint(spec.checkpoint),
        "model_config": file_fingerprint(spec.config_path),
        "test_dataset": file_fingerprint(TEST_DATASET_PATH),
        "force_zero_expression": spec.strategy == "zero",
        "num_test_samples": NUM_TEST_SAMPLES,
        "storage_dtype": "float32",
    }


def manifest_matches(path: Path, expected: Mapping[str, object]) -> bool:
    """Return True only when an existing prediction has matching provenance."""
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            observed = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    return observed == expected


def predict_one_model(
    spec: ModelSpec,
    model: BaseModel,
    dataset: TranslationDataset,
) -> Path:
    """Create or safely reuse one float32 prediction PKL file."""
    prediction_dir = OUTPUT_DIR / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = (
        prediction_dir
        / f"predictions_count.{model.model_name}.{spec.model_id}.pkl"
    )
    manifest_path = Path(str(prediction_path) + ".manifest.json")
    expected_manifest = prediction_manifest(spec)

    if (
        REUSE_EXISTING_PREDICTIONS
        and prediction_path.is_file()
        and manifest_matches(manifest_path, expected_manifest)
    ):
        print(f"Reusing verified predictions: {prediction_path}")
        return prediction_path

    generated_path = Path(
        save_count_predictions(
            model=model,
            dataset=dataset,
            num_samples=NUM_TEST_SAMPLES,
            batch_size=BATCH_SIZE,
            out_dir=str(prediction_dir),
            suffix=spec.model_id,
            force_zero_expression=spec.strategy == "zero",
            storage_dtype=np.float32,
        )
    )
    if generated_path.resolve() != prediction_path.resolve():
        prediction_path = generated_path
        manifest_path = Path(str(prediction_path) + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(expected_manifest, handle, indent=2)
    return prediction_path


def read_cell_expression_vectors(paths: Sequence[Path]) -> Dict[str, np.ndarray]:
    """Read and validate unique cell-expression vectors from HDF5 datasets."""
    if h5py is None:
        raise ImportError("h5py is required for HDF5 environment-diversity evaluation")
    require_files(paths, "training environment datasets")
    vectors: Dict[str, np.ndarray] = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "cell_exprs" not in handle:
                raise KeyError(f"Dataset has no /cell_exprs group: {path}")
            for cell_type in handle["cell_exprs"].keys():
                value = np.asarray(
                    handle["cell_exprs"][cell_type][:], dtype=np.float32
                ).reshape(-1)
                if not np.isfinite(value).all():
                    raise ValueError(
                        f"Non-finite expression values for {cell_type} in {path}"
                    )
                if cell_type in vectors and not np.allclose(
                    vectors[cell_type], value, rtol=1e-5, atol=1e-6
                ):
                    raise ValueError(
                        f"Expression vector for {cell_type} differs across datasets"
                    )
                vectors[str(cell_type)] = value
    if not vectors:
        raise ValueError("No training expression vectors were found")
    return vectors


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine distance for two finite, non-zero expression vectors."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise ValueError(
            f"Expression dimensions differ: {first.shape} versus {second.shape}"
        )
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0 or not math.isfinite(denominator):
        return float("nan")
    return float(1.0 - np.dot(first, second) / denominator)


def calculate_expression_distances(
    test_vectors: Mapping[str, np.ndarray],
    training_vectors_by_count: Mapping[int, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    """Match each held-out cell to its nearest environment for 5/22/40 sets."""
    records = []
    for environment_count, training_vectors in training_vectors_by_count.items():
        overlap = sorted(set(test_vectors) & set(training_vectors))
        if overlap and REQUIRE_DISJOINT_ENVIRONMENTS:
            raise ValueError(
                f"{environment_count}c training environments overlap the zero-shot "
                f"test cells: {overlap}"
            )
        if len(training_vectors) != environment_count:
            raise ValueError(
                f"Expected {environment_count} training expression environments, "
                f"found {len(training_vectors)}"
            )

        for test_cell, test_vector in sorted(test_vectors.items()):
            candidates = []
            for train_cell, train_vector in training_vectors.items():
                distance = cosine_distance(test_vector, train_vector)
                if math.isfinite(distance):
                    candidates.append((distance, str(train_cell)))
            if not candidates:
                raise ValueError(
                    f"No finite expression-distance match for {test_cell} at "
                    f"{environment_count}c"
                )
            distance, nearest_cell = min(candidates, key=lambda item: (item[0], item[1]))
            records.append(
                {
                    "Environment_Count": environment_count,
                    "Cell_Type": str(test_cell),
                    "Nearest_Training_Cell": nearest_cell,
                    "Nearest_Cosine_Distance": distance,
                    "Expression_Coverage": 1.0 - distance,
                }
            )
    return pd.DataFrame.from_records(records)


def validate_nested_training_environments(
    training_vectors_by_count: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    """Verify that 5c is nested in 22c and 22c is nested in 40c."""
    ordered_counts = (5, 22, 40)
    missing = [count for count in ordered_counts if count not in training_vectors_by_count]
    if missing:
        raise KeyError(f"Missing training environment groups: {missing}")
    for smaller_count, larger_count in zip(ordered_counts[:-1], ordered_counts[1:]):
        smaller = training_vectors_by_count[smaller_count]
        larger = training_vectors_by_count[larger_count]
        absent = sorted(set(smaller) - set(larger))
        if absent:
            raise ValueError(
                f"The {smaller_count}c environments are not a subset of the "
                f"{larger_count}c environments: {absent}"
            )
        inconsistent = [
            cell_type
            for cell_type in smaller
            if not np.allclose(
                smaller[cell_type], larger[cell_type], rtol=1e-5, atol=1e-6
            )
        ]
        if inconsistent:
            raise ValueError(
                f"Nested datasets contain inconsistent expression vectors: {inconsistent}"
            )


def safe_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Spearman correlation or NaN for insufficient/constant inputs."""
    first_values = np.asarray(first, dtype=np.float64).reshape(-1)
    second_values = np.asarray(second, dtype=np.float64).reshape(-1)
    valid = np.isfinite(first_values) & np.isfinite(second_values)
    first_values = first_values[valid]
    second_values = second_values[valid]
    if (
        len(first_values) < 2
        or np.ptp(first_values) == 0
        or np.ptp(second_values) == 0
    ):
        return float("nan")
    return float(spearmanr(first_values, second_values).statistic)


def safe_expm1(values: np.ndarray) -> np.ndarray:
    """Convert log1p densities to finite, non-negative linear densities."""
    linear = np.expm1(np.asarray(values, dtype=np.float32))
    linear = np.nan_to_num(linear, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(linear, 0.0, None)


def evaluate_prediction_file(
    dataset: TranslationDataset,
    prediction_path: Path,
    spec: ModelSpec,
) -> pd.DataFrame:
    """Calculate requested transcript-level metrics for one checkpoint.

    Profile and CDS-mean metrics use the stored log1p-density scale, matching the
    Trainer validation metrics. Periodicity is calculated after returning both
    observed and predicted signals to linear density with expm1.
    """
    predictions = load_prediction_input(str(prediction_path))
    records = []
    iterator = tqdm(
        range(len(dataset)),
        desc=f"Evaluate {spec.model_id}",
    )
    for index in iterator:
        uuid, _, cell_type, _, meta_info, _, target = dataset[index]
        tid = transcript_id_from_uuid(uuid)
        cell_type = str(cell_type)
        prediction = get_prediction(predictions, cell_type, tid)
        if prediction is None:
            continue

        target_log = to_1d_signal(target)
        prediction_log = to_1d_signal(prediction)
        length = min(len(target_log), len(prediction_log))
        bounds = cds_slice(meta_info, length)
        if bounds is None:
            continue
        cds_start, cds_end = bounds
        if cds_end - cds_start < 3:
            continue

        target_log = np.asarray(target_log[:length], dtype=np.float32)
        prediction_log = np.asarray(prediction_log[:length], dtype=np.float32)
        target_cds_log = target_log[cds_start:cds_end]
        prediction_cds_log = prediction_log[cds_start:cds_end]

        profile_spearman = safe_spearman(prediction_cds_log, target_cds_log)
        observed_cds_mean = float(np.mean(target_cds_log))
        predicted_cds_mean = float(np.mean(prediction_cds_log))

        target_linear = safe_expm1(target_log)
        prediction_linear = safe_expm1(prediction_log)
        observed_periodicity = calculate_periodicity(
            target_linear, cds_start, cds_end
        )
        predicted_periodicity = calculate_periodicity(
            prediction_linear, cds_start, cds_end
        )
        periodicity_bias = (
            float(predicted_periodicity - observed_periodicity)
            if np.isfinite(observed_periodicity)
            and np.isfinite(predicted_periodicity)
            else float("nan")
        )

        records.append(
            {
                "Model_ID": spec.model_id,
                "Environment_Count": spec.environment_count,
                "Strategy": spec.strategy,
                "Strategy_Label": spec.strategy_label,
                "UUID": str(uuid),
                "Tid": tid,
                "Cell_Type": cell_type,
                "Transcript_Length": length,
                "CDS_Length": cds_end - cds_start,
                "RPF_Depth": float(meta_info.get("rpf_depth", np.nan)),
                "RPF_Coverage": float(meta_info.get("rpf_coverage", np.nan)),
                "CDS_Profile_Spearman": profile_spearman,
                "Observed_Periodicity": observed_periodicity,
                "Predicted_Periodicity": predicted_periodicity,
                "Periodicity_Bias": periodicity_bias,
                "Periodicity_Absolute_Error": abs(periodicity_bias)
                if math.isfinite(periodicity_bias)
                else float("nan"),
                "Observed_CDS_Mean_Log1p": observed_cds_mean,
                "Predicted_CDS_Mean_Log1p": predicted_cds_mean,
                "CDS_Mean_Absolute_Error_Log1p": abs(
                    predicted_cds_mean - observed_cds_mean
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def finite_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of finite values or NaN."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def summarize_by_cell_type(
    transcript_metrics: pd.DataFrame,
    spec: ModelSpec,
    expected_cells: Sequence[str],
) -> pd.DataFrame:
    """Aggregate RNA metrics within each cell while retaining missing cells."""
    rows = []
    for cell_type in expected_cells:
        group = transcript_metrics[transcript_metrics["Cell_Type"] == cell_type]
        periodicity = group[
            ["Observed_Periodicity", "Predicted_Periodicity", "Periodicity_Bias"]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        periodicity_excluded_count = int(len(group) - len(periodicity))
        high_periodicity = periodicity[
            periodicity["Observed_Periodicity"] >= HIGH_PERIODICITY_THRESHOLD
        ]
        profile_n = int(group["CDS_Profile_Spearman"].notna().sum())
        scale_n = int(
            group[
                ["Observed_CDS_Mean_Log1p", "Predicted_CDS_Mean_Log1p"]
            ]
            .dropna()
            .shape[0]
        )
        rows.append(
            {
                "Model_ID": spec.model_id,
                "Environment_Count": spec.environment_count,
                "Strategy": spec.strategy,
                "Strategy_Label": spec.strategy_label,
                "Cell_Type": cell_type,
                "RNA_N": int(len(group)),
                "CDS_Profile_N": profile_n,
                "CDS_Profile_Excluded_N": int(len(group) - profile_n),
                "Mean_CDS_Profile_Spearman": finite_mean(
                    group["CDS_Profile_Spearman"]
                ),
                "Median_CDS_Profile_Spearman": float(
                    group["CDS_Profile_Spearman"].median()
                ),
                "Periodicity_N": int(len(periodicity)),
                "Periodicity_Excluded_N": periodicity_excluded_count,
                "Periodicity_Spearman": safe_spearman(
                    periodicity["Observed_Periodicity"],
                    periodicity["Predicted_Periodicity"],
                ),
                "Periodicity_Bias": finite_mean(periodicity["Periodicity_Bias"]),
                "Periodicity_MAE": finite_mean(
                    np.abs(periodicity["Periodicity_Bias"])
                ),
                "High_Periodicity_N": int(len(high_periodicity)),
                "High_Periodicity_Bias": finite_mean(
                    high_periodicity["Periodicity_Bias"]
                ),
                "High_Periodicity_MAE": finite_mean(
                    np.abs(high_periodicity["Periodicity_Bias"])
                ),
                "CDS_Mean_Scale_N": scale_n,
                "CDS_Mean_Scale_Excluded_N": int(len(group) - scale_n),
                "CDS_Mean_Scale_Spearman": safe_spearman(
                    group["Observed_CDS_Mean_Log1p"],
                    group["Predicted_CDS_Mean_Log1p"],
                ),
                "CDS_Mean_MAE_Log1p": finite_mean(
                    group["CDS_Mean_Absolute_Error_Log1p"]
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


SUMMARY_METRICS = (
    "Mean_CDS_Profile_Spearman",
    "Periodicity_Bias",
    "Periodicity_MAE",
    "High_Periodicity_Bias",
    "High_Periodicity_MAE",
    "CDS_Mean_Scale_Spearman",
    "CDS_Mean_MAE_Log1p",
)


def bootstrap_mean_ci(
    values: Sequence[float],
    iterations: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, int]:
    """Estimate a mean and percentile CI by resampling held-out cell types."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(array))
    if len(array) == 1 or iterations < 2:
        return mean, mean, mean, int(len(array))
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(low), float(high), int(len(array))


def summarize_models(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create equal-cell-weighted model metrics and cell-bootstrap intervals."""
    rows = []
    for model_index, (model_id, group) in enumerate(
        cell_metrics.groupby("Model_ID", sort=False)
    ):
        first = group.iloc[0]
        row = {
            "Model_ID": model_id,
            "Environment_Count": int(first["Environment_Count"]),
            "Strategy": first["Strategy"],
            "Strategy_Label": first["Strategy_Label"],
            "Cell_Type_N": int(group["Cell_Type"].nunique()),
            "RNA_N": int(group["RNA_N"].sum()),
        }
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            rng = np.random.Generator(
                np.random.PCG64(RANDOM_SEED + model_index * 100 + metric_index)
            )
            mean, low, high, n = bootstrap_mean_ci(
                group[metric], BOOTSTRAP_ITERATIONS, rng
            )
            row[metric] = mean
            row[f"{metric}_CI95_Low"] = low
            row[f"{metric}_CI95_High"] = high
            row[f"{metric}_Cell_N"] = n
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(
        ["Environment_Count", "Strategy"]
    )


def cluster_bootstrap_regression(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    grid: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, int]:
    """Fit a line and cluster-bootstrap its band by held-out cell type."""
    valid = data[["Cell_Type", x_column, y_column]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(valid) < 2 or valid[x_column].nunique() < 2:
        nan_grid = np.full_like(grid, np.nan, dtype=np.float64)
        return nan_grid, nan_grid, nan_grid, np.nan, np.nan, np.nan, len(valid)

    slope, intercept = np.polyfit(valid[x_column], valid[y_column], 1)
    fitted = intercept + slope * grid
    distance_spearman = safe_spearman(valid[x_column], valid[y_column])

    cell_types = valid["Cell_Type"].drop_duplicates().to_numpy()
    bootstrap_lines = []
    for _ in range(iterations):
        sampled_indices = rng.integers(0, len(cell_types), size=len(cell_types))
        sampled_cells = cell_types[sampled_indices]
        sampled_groups = [
            valid[valid["Cell_Type"] == cell_type] for cell_type in sampled_cells
        ]
        sampled = pd.concat(sampled_groups, ignore_index=True)
        if sampled[x_column].nunique() < 2:
            continue
        sampled_slope, sampled_intercept = np.polyfit(
            sampled[x_column], sampled[y_column], 1
        )
        bootstrap_lines.append(sampled_intercept + sampled_slope * grid)

    if bootstrap_lines:
        bootstrap_array = np.asarray(bootstrap_lines)
        lower, upper = np.quantile(bootstrap_array, [0.025, 0.975], axis=0)
    else:
        lower = np.full_like(grid, np.nan, dtype=np.float64)
        upper = np.full_like(grid, np.nan, dtype=np.float64)
    return (
        fitted,
        lower,
        upper,
        float(slope),
        float(intercept),
        distance_spearman,
        int(len(valid)),
    )


def save_publication_figure(figure: plt.Figure, output_prefix: Path) -> None:
    """Export one figure with editable vector text and a high-resolution raster."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(
        output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight"
    )


PANEL_METRICS = (
    (
        "Mean_CDS_Profile_Spearman",
        "CDS profile shape",
        "Mean per-cell CDS profile Spearman",
    ),
    (
        "CDS_Mean_Scale_Spearman",
        "CDS signal scale",
        "Mean per-cell CDS-mean Spearman",
    ),
)


def plot_environment_diversity_curves(
    model_metrics: pd.DataFrame,
    output_prefix: Path,
) -> None:
    """Plot 5/22/40-cell curves for profile shape and signal scale."""
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True)
    for panel_index, (axis, (metric, title, ylabel)) in enumerate(
        zip(axes, PANEL_METRICS)
    ):
        for strategy in ("zero", "real", "exp_aug"):
            group = model_metrics[model_metrics["Strategy"] == strategy].sort_values(
                "Environment_Count"
            )
            x = group["Environment_Count"].to_numpy(dtype=float)
            y = group[metric].to_numpy(dtype=float)
            low = group[f"{metric}_CI95_Low"].to_numpy(dtype=float)
            high = group[f"{metric}_CI95_High"].to_numpy(dtype=float)
            axis.errorbar(
                x,
                y,
                yerr=np.vstack(
                    [np.maximum(0.0, y - low), np.maximum(0.0, high - y)]
                ),
                color=STRATEGY_COLORS[strategy],
                marker=STRATEGY_MARKERS[strategy],
                markersize=5,
                linewidth=1.7,
                capsize=2.5,
                label=STRATEGY_LABELS[strategy],
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xlabel("Training environments")
        axis.set_ylabel(ylabel)
        axis.set_xticks([5, 22, 40], ["5", "22", "40"])
        axis.grid(axis="y", color="#E7E7E7", linewidth=0.7, zorder=0)
        axis.text(
            0.01,
            0.98,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
            zorder=10,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.text(
        0.5,
        -0.01,
        "Points are equal-weight means across held-out cell types; error bars are cell-bootstrap 95% CIs.",
        ha="center",
        fontsize=6.5,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93), w_pad=2.0)
    save_publication_figure(figure, output_prefix)
    plt.close(figure)


def plot_zero_shot_distance_curves(
    cell_metrics: pd.DataFrame,
    output_prefix: Path,
) -> pd.DataFrame:
    """Plot expression distance against zero-shot shape and scale performance."""
    x_column = "Nearest_Cosine_Distance"
    finite_distance = cell_metrics[x_column].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_distance.empty or finite_distance.nunique() < 2:
        raise ValueError("At least two finite expression distances are required")
    x_grid = np.linspace(float(finite_distance.min()), float(finite_distance.max()), 200)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True)
    regression_rows = []
    for panel_index, (axis, (metric, title, ylabel)) in enumerate(
        zip(axes, PANEL_METRICS)
    ):
        for strategy_index, strategy in enumerate(("zero", "real", "exp_aug")):
            group = cell_metrics[cell_metrics["Strategy"] == strategy]
            for environment_count in (5, 22, 40):
                point_data = group[group["Environment_Count"] == environment_count]
                axis.scatter(
                    point_data[x_column],
                    point_data[metric],
                    s=18,
                    marker=ENVIRONMENT_MARKERS[environment_count],
                    facecolor=STRATEGY_COLORS[strategy],
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.42,
                    rasterized=True,
                    zorder=2,
                )
            result = cluster_bootstrap_regression(
                group,
                x_column,
                metric,
                x_grid,
                BOOTSTRAP_ITERATIONS,
                np.random.Generator(
                    np.random.PCG64(
                        RANDOM_SEED + panel_index * 100 + strategy_index
                    )
                ),
            )
            fitted, lower, upper, slope, intercept, rho, n = result
            axis.fill_between(
                x_grid,
                lower,
                upper,
                color=STRATEGY_COLORS[strategy],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                x_grid,
                fitted,
                color=STRATEGY_COLORS[strategy],
                linewidth=1.8,
                label=STRATEGY_LABELS[strategy],
                zorder=3,
            )
            regression_rows.append(
                {
                    "Metric": metric,
                    "Strategy": strategy,
                    "Strategy_Label": STRATEGY_LABELS[strategy],
                    "N_Cell_Model_Points": n,
                    "Excluded_Cell_Model_Points": int(len(group) - n),
                    "Linear_Slope": slope,
                    "Linear_Intercept": intercept,
                    "Distance_Performance_Spearman": rho,
                }
            )

        axis.set_title(title)
        axis.set_xlabel("Distance to nearest training environment")
        axis.set_ylabel(ylabel)
        axis.grid(color="#E7E7E7", linewidth=0.7, zorder=0)
        axis.text(
            0.01,
            0.98,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
            zorder=10,
        )

    method_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=STRATEGY_COLORS[strategy],
            linewidth=1.8,
            label=STRATEGY_LABELS[strategy],
        )
        for strategy in ("zero", "real", "exp_aug")
    ]
    environment_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color="#666666",
            marker=ENVIRONMENT_MARKERS[count],
            linestyle="none",
            markersize=4.5,
            label=f"{count} cells",
        )
        for count in (5, 22, 40)
    ]
    figure.legend(
        handles=method_handles + environment_handles,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.text(
        0.5,
        -0.01,
        "Points are held-out cell types; bands are cell-cluster bootstrap 95% CIs for the fitted lines.",
        ha="center",
        fontsize=6.5,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93), w_pad=2.0)
    save_publication_figure(figure, output_prefix)
    plt.close(figure)
    return pd.DataFrame.from_records(regression_rows)


def validate_test_dataset(dataset: TranslationDataset) -> List[str]:
    """Return actual test cells after checking the expected zero-shot scope."""
    cell_types = sorted(set(str(cell_type) for cell_type in dataset.cell_types))
    if EXPECTED_TEST_CELL_TYPES is not None and len(cell_types) != EXPECTED_TEST_CELL_TYPES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_CELL_TYPES} test cell types, found "
            f"{len(cell_types)}: {cell_types}"
        )
    missing_vectors = sorted(set(cell_types) - set(dataset.cell_expr_dict))
    if missing_vectors:
        raise ValueError(f"Test cells lack expression vectors: {missing_vectors}")
    return cell_types


def main() -> None:
    """Run prediction, metric aggregation, source-data export, and plotting."""
    require_files([TEST_DATASET_PATH, MODEL_CONFIG_PATH], "evaluation inputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "rna_metrics").mkdir(parents=True, exist_ok=True)

    model_specs = build_model_specs()
    test_dataset = TranslationDataset.from_h5(str(TEST_DATASET_PATH), lazy=True)
    test_cell_types = validate_test_dataset(test_dataset)
    print(
        f"Test dataset: {len(test_dataset):,} samples across "
        f"{len(test_cell_types)} held-out cell types"
    )

    training_vectors_by_count = {
        count: read_cell_expression_vectors(paths)
        for count, paths in TRAIN_ENVIRONMENT_DATASETS.items()
    }
    validate_nested_training_environments(training_vectors_by_count)
    test_vectors = {
        cell_type: np.asarray(
            test_dataset.cell_expr_dict[cell_type], dtype=np.float32
        ).reshape(-1)
        for cell_type in test_cell_types
    }
    distance_table = calculate_expression_distances(
        test_vectors, training_vectors_by_count
    )
    distance_table.to_csv(
        OUTPUT_DIR / "nearest_training_environment.csv", index=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    prediction_paths: Dict[str, Path] = {}
    checkpoint_metadata: Dict[str, dict] = {}

    print("\n=== Phase 1/2: predicting all checkpoints ===")
    for spec in model_specs:
        model, metadata = load_model(spec, device)
        prediction_paths[spec.model_id] = predict_one_model(
            spec, model, test_dataset
        )
        checkpoint_metadata[spec.model_id] = metadata
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Phase 2/2: evaluating all prediction files ===")
    all_cell_metrics = []
    checkpoint_rows = []
    for spec in model_specs:
        prediction_path = prediction_paths[spec.model_id]
        metadata = checkpoint_metadata[spec.model_id]
        transcript_metrics = evaluate_prediction_file(
            test_dataset, prediction_path, spec
        )
        if transcript_metrics.empty:
            raise RuntimeError(f"No valid CDS metrics were produced for {spec.model_id}")
        transcript_metrics.to_csv(
            OUTPUT_DIR / "rna_metrics" / f"{spec.model_id}.csv", index=False
        )
        cell_metrics = summarize_by_cell_type(
            transcript_metrics, spec, test_cell_types
        )
        all_cell_metrics.append(cell_metrics)
        checkpoint_rows.append(
            {
                "Model_ID": spec.model_id,
                "Environment_Count": spec.environment_count,
                "Strategy": spec.strategy,
                "Strategy_Label": spec.strategy_label,
                "Checkpoint": str(spec.checkpoint),
                "Checkpoint_Epoch": metadata.get("epoch"),
                "Prediction_PKL": str(prediction_path),
            }
        )

    cell_metrics = pd.concat(all_cell_metrics, ignore_index=True)
    cell_metrics = cell_metrics.merge(
        distance_table,
        on=["Environment_Count", "Cell_Type"],
        how="left",
        validate="many_to_one",
    )
    if cell_metrics["Nearest_Cosine_Distance"].isna().any():
        raise RuntimeError("Some model/cell rows lack expression-distance metadata")

    model_metrics = summarize_models(cell_metrics)
    checkpoint_table = pd.DataFrame.from_records(checkpoint_rows)
    checkpoint_table.to_csv(OUTPUT_DIR / "checkpoint_manifest.csv", index=False)
    cell_metrics.to_csv(OUTPUT_DIR / "cell_type_metrics.csv", index=False)
    model_metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    environment_source = model_metrics.copy()
    environment_source.to_csv(
        OUTPUT_DIR / "environment_diversity_curve_source.csv", index=False
    )
    zero_shot_source = cell_metrics[
        [
            "Model_ID",
            "Environment_Count",
            "Strategy",
            "Strategy_Label",
            "Cell_Type",
            "Nearest_Training_Cell",
            "Nearest_Cosine_Distance",
            "Expression_Coverage",
            "Mean_CDS_Profile_Spearman",
            "CDS_Mean_Scale_Spearman",
            "CDS_Mean_MAE_Log1p",
            "Periodicity_Bias",
            "High_Periodicity_Bias",
        ]
    ].copy()
    zero_shot_source.to_csv(
        OUTPUT_DIR / "zero_shot_distance_curve_source.csv", index=False
    )

    figure_dir = OUTPUT_DIR / "figures"
    plot_environment_diversity_curves(
        model_metrics,
        figure_dir / "environment_diversity_shape_and_scale",
    )
    regression_summary = plot_zero_shot_distance_curves(
        cell_metrics,
        figure_dir / "zero_shot_distance_shape_and_scale",
    )
    regression_summary.to_csv(
        OUTPUT_DIR / "zero_shot_distance_regression.csv", index=False
    )

    print("\nEnvironment-diversity evaluation complete.")
    print(f"Model summary: {OUTPUT_DIR / 'model_metrics.csv'}")
    print(f"Cell-type summary: {OUTPUT_DIR / 'cell_type_metrics.csv'}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()

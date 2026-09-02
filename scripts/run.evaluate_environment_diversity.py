#!/usr/bin/env python3
"""Evaluate TRACE environment diversity on held-out uncommon cell types.

The script performs four steps for every configured checkpoint:
1. reuse a verified prediction PKL, or optionally generate one when enabled;
2. calculate transcript-level CDS profile, periodicity, and CDS-mean metrics;
3. aggregate metrics with equal weight per held-out cell type and save CSV tables;
4. draw environment-diversity and expression-distance figures for profile shape
   and signal scale.

Edit the configuration section and run this file directly on the server. The
default comparison contains 5/22/40 training environments crossed with zero,
real, and mask-plus-interpolation expression strategies. Missing model-grid
positions remain explicit blank rows, and verified prediction/metric caches are
reused across runs.
"""

from __future__ import annotations

import hashlib
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
CHECKPOINT_DIR = Path("/public-supool/home/annie/translation_model/checkpoint/train")
OUTPUT_DIR = Path("/public-supool/home/annie/translation_model/results/ablation/environment_diversity")

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
CHECKPOINT_PREFIX = "base_model_384d_16h_12l_64env_16ad_bs"
CHECKPOINT_SUFFIX = ".best_total.pt"

# Exact paths take priority. Add entries here if a rule below matches more than
# one checkpoint, for example:
# (5, "zero"): CHECKPOINT_DIR / "exact_checkpoint.best_profile.pt"
EXACT_CHECKPOINTS: Dict[Tuple[int, str], Path] = {}

# Automatic matching is intentionally strict. Edit the tokens if the server-side
# checkpoint names use a different dataset tag.
CHECKPOINT_MATCH_RULES = {
    (5, "zero"): {"required": ("hs_5c", "a2_b02_zero"), "forbidden": ()},
    (5, "real"): {"required": ("hs_5c", "a2_b02_real"), "forbidden": ()},
    (5, "exp_aug"): {"required": ("hs_5c", "a2_b02_exp_aug"), "forbidden": ()},
    (22, "zero"): {"required": ("hs_22c", "a2_b02_zero"), "forbidden": ("18c",)},
    (22, "real"): {"required": ("hs_22c", "a2_b02_real"), "forbidden": ("18c",)},
    (22, "exp_aug"): {
        "required": ("hs_22c", "a2_b02_exp_aug"),
        "forbidden": ("18c",),
    },
    (40, "zero"): {"required": ("hs_22c_18c", "a2_b02_zero"), "forbidden": ()},
    (40, "real"): {"required": ("hs_22c_18c", "a2_b02_real"), "forbidden": ()},
    (40, "exp_aug"): {
        "required": ("hs_22c_18c", "a2_b02_exp_aug"),
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

BATCH_SIZE = 30
# Cache reuse requires matching dataset, model configuration, strategy, and
# checkpoint provenance. New checkpoint versions receive new prediction files.
REUSE_EXISTING_PREDICTIONS = True
REUSE_EXISTING_METRICS = True
# Reanalyze verified prediction PKLs without loading checkpoints or running
# inference. A missing or stale prediction remains an explicit blank grid row.
ANALYSIS_ONLY = True
# Keep False to finish all available model-grid positions and record failures.
FAIL_FAST = False
EXPECTED_TEST_CELL_TYPES = 26
# Cell-level correlations and means based on fewer RNAs are retained as source
# rows but excluded from model summaries, confidence intervals, and regressions.
MIN_RNA_PER_CELL = 50
# Apply the same transcript-quality gate to RNA-profile, CDS-profile, scale, and
# periodicity metrics. Set to None to disable additional depth filtering.
MIN_RPF_DEPTH: Optional[float] = 0.5
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
    checkpoint: Optional[Path]
    config_path: Path
    resolution_status: str = "resolved"
    resolution_message: str = ""

    @property
    def model_id(self) -> str:
        return f"{self.environment_count}c_{self.strategy}"

    @property
    def strategy_label(self) -> str:
        return STRATEGY_LABELS[self.strategy]

    @property
    def can_predict(self) -> bool:
        """Return whether all files required for fresh inference are present."""
        return (
            self.checkpoint is not None
            and self.checkpoint.is_file()
            and self.config_path.is_file()
        )


def require_files(paths: Iterable[Path], label: str) -> None:
    """Raise one readable exception containing every missing path."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label}: " + ", ".join(missing))


def resolve_checkpoint(
    environment_count: int,
    strategy: str,
) -> Tuple[Optional[Path], str, str]:
    """Resolve one checkpoint while retaining missing combinations as rows."""
    key = (environment_count, strategy)
    exact = EXACT_CHECKPOINTS.get(key)
    if exact is not None:
        if not exact.is_file():
            return None, "missing_checkpoint", f"Configured checkpoint is missing: {exact}"
        return exact.resolve(), "resolved", "Resolved from EXACT_CHECKPOINTS"

    if key not in CHECKPOINT_MATCH_RULES:
        return None, "missing_rule", f"No checkpoint matching rule is configured for {key}"
    rule = CHECKPOINT_MATCH_RULES[key]
    required = tuple(token.lower() for token in rule.get("required", ()))
    forbidden = tuple(token.lower() for token in rule.get("forbidden", ()))
    matches = []
    if CHECKPOINT_DIR.is_dir():
        for path in CHECKPOINT_DIR.glob(f"{CHECKPOINT_PREFIX}*{CHECKPOINT_SUFFIX}"):
            name = path.name.lower()
            if all(token in name for token in required) and not any(
                token in name for token in forbidden
            ):
                matches.append(path.resolve())

    if not matches:
        return (
            None,
            "missing_checkpoint",
            f"No checkpoint matched {environment_count}c/{strategy}",
        )
    if len(matches) > 1:
        formatted = "; ".join(str(path) for path in sorted(matches))
        return (
            None,
            "ambiguous_checkpoint",
            f"Multiple checkpoints matched; configure EXACT_CHECKPOINTS: {formatted}",
        )
    return matches[0], "resolved", "Resolved from checkpoint matching rule"


def build_model_specs() -> List[ModelSpec]:
    """Create the nine ordered model specifications."""
    specs = []
    for environment_count in (5, 22, 40):
        for strategy in ("zero", "real", "exp_aug"):
            checkpoint, status, message = resolve_checkpoint(
                environment_count, strategy
            )
            specs.append(
                ModelSpec(
                    environment_count=environment_count,
                    strategy=strategy,
                    checkpoint=checkpoint,
                    config_path=MODEL_CONFIG_PATH,
                    resolution_status=status,
                    resolution_message=message,
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
    if spec.checkpoint is None:
        raise FileNotFoundError(f"No checkpoint is available for {spec.model_id}")
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
    """Describe available inputs that determine a saved prediction file."""
    manifest = {
        "model_id": spec.model_id,
        "test_dataset": file_fingerprint(TEST_DATASET_PATH),
        "force_zero_expression": spec.strategy == "zero",
        "num_test_samples": NUM_TEST_SAMPLES,
        "storage_dtype": "float32",
    }
    if spec.checkpoint is not None and spec.checkpoint.is_file():
        manifest["checkpoint"] = file_fingerprint(spec.checkpoint)
    if spec.config_path.is_file():
        manifest["model_config"] = file_fingerprint(spec.config_path)
    return manifest


def manifest_matches(path: Path, expected: Mapping[str, object]) -> bool:
    """Return True when all expected provenance fields match a manifest."""
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            observed = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    return all(observed.get(key) == value for key, value in expected.items())


def prediction_directory() -> Path:
    """Return the persistent prediction cache directory."""
    return OUTPUT_DIR / "predictions"


def prediction_candidates(spec: ModelSpec) -> List[Path]:
    """Return saved prediction files for one model-grid position."""
    directory = prediction_directory()
    if not directory.is_dir():
        return []
    return sorted(
        directory.glob(f"predictions_count.*.{spec.model_id}*.pkl"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def cached_checkpoint_matches_spec(spec: ModelSpec, manifest: Mapping[str, object]) -> bool:
    """Validate the checkpoint identity stored in a cache-only prediction."""
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("path"):
        return False
    cached_path = Path(str(checkpoint["path"]))
    key = (spec.environment_count, spec.strategy)
    exact = EXACT_CHECKPOINTS.get(key)
    if exact is not None:
        return cached_path.expanduser().resolve() == exact.expanduser().resolve()
    rule = CHECKPOINT_MATCH_RULES.get(key)
    if rule is None or not cached_path.name.endswith(CHECKPOINT_SUFFIX):
        return False
    name = cached_path.name.lower()
    required = tuple(token.lower() for token in rule.get("required", ()))
    forbidden = tuple(token.lower() for token in rule.get("forbidden", ()))
    return all(token in name for token in required) and not any(
        token in name for token in forbidden
    )


def find_cached_prediction(spec: ModelSpec) -> Optional[Path]:
    """Find the newest prediction whose available provenance still matches."""
    if not REUSE_EXISTING_PREDICTIONS or not TEST_DATASET_PATH.is_file():
        return None
    expected_manifest = prediction_manifest(spec)
    for prediction_path in prediction_candidates(spec):
        manifest_path = Path(str(prediction_path) + ".manifest.json")
        if not manifest_matches(manifest_path, expected_manifest):
            continue
        observed_manifest = read_prediction_manifest(prediction_path)
        if spec.checkpoint is None and not cached_checkpoint_matches_spec(
            spec, observed_manifest
        ):
            continue
        return prediction_path
    return None


def prediction_cache_tag(spec: ModelSpec) -> str:
    """Create a stable short identifier for one checkpoint fingerprint."""
    if spec.checkpoint is None or not spec.checkpoint.is_file():
        raise FileNotFoundError(f"Cannot fingerprint missing checkpoint: {spec.model_id}")
    payload = json.dumps(
        file_fingerprint(spec.checkpoint), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def read_prediction_manifest(prediction_path: Path) -> dict:
    """Read saved prediction provenance or return an empty dictionary."""
    manifest_path = Path(str(prediction_path) + ".manifest.json")
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def json_scalar(value: object) -> object:
    """Convert common scalar containers to JSON-compatible values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.generic, torch.Tensor)):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            return str(value)
    return str(value)


def predict_one_model(
    spec: ModelSpec,
    model: BaseModel,
    dataset: TranslationDataset,
    checkpoint_metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    """Create or safely reuse one float32 prediction PKL file."""
    prediction_dir = prediction_directory()
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cached_path = find_cached_prediction(spec)
    if cached_path is not None:
        print(f"Reusing verified predictions: {cached_path}")
        return cached_path

    generated_path = Path(
        save_count_predictions(
            model=model,
            dataset=dataset,
            num_samples=NUM_TEST_SAMPLES,
            batch_size=BATCH_SIZE,
            out_dir=str(prediction_dir),
            suffix=f"{spec.model_id}.{prediction_cache_tag(spec)}",
            force_zero_expression=spec.strategy == "zero",
            storage_dtype=np.float32,
        )
    )
    prediction_path = generated_path
    manifest_path = Path(str(prediction_path) + ".manifest.json")
    expected_manifest = prediction_manifest(spec)
    if checkpoint_metadata is not None:
        expected_manifest["checkpoint_epoch"] = json_scalar(
            checkpoint_metadata.get("epoch")
        )
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
    """Verify nesting among the training environment groups that are available."""
    ordered_counts = sorted(training_vectors_by_count)
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


TRANSCRIPT_METRIC_COLUMNS = (
    "Model_ID",
    "Environment_Count",
    "Strategy",
    "Strategy_Label",
    "UUID",
    "Tid",
    "Cell_Type",
    "Transcript_Length",
    "CDS_Length",
    "RPF_Depth",
    "RPF_Coverage",
    "RNA_Profile_Spearman",
    "CDS_Profile_Spearman",
    "Observed_Periodicity",
    "Predicted_Periodicity",
    "Periodicity_Bias",
    "Periodicity_Absolute_Error",
    "Observed_CDS_Mean_Log1p",
    "Predicted_CDS_Mean_Log1p",
    "CDS_Mean_Absolute_Error_Log1p",
)


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

        rna_profile_spearman = safe_spearman(prediction_log, target_log)
        cds_profile_spearman = safe_spearman(prediction_cds_log, target_cds_log)
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
                "RNA_Profile_Spearman": rna_profile_spearman,
                "CDS_Profile_Spearman": cds_profile_spearman,
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
    return pd.DataFrame.from_records(records, columns=TRANSCRIPT_METRIC_COLUMNS)


def metric_cache_paths(prediction_path: Path) -> Tuple[Path, Path]:
    """Return persistent transcript-metric CSV and manifest paths."""
    metric_dir = OUTPUT_DIR / "rna_metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metric_dir / f"{prediction_path.stem}.rna_metrics.csv"
    return csv_path, Path(str(csv_path) + ".manifest.json")


def metric_manifest(prediction_path: Path, spec: ModelSpec) -> dict:
    """Describe inputs and settings that determine transcript-level metrics."""
    return {
        "model_id": spec.model_id,
        "prediction": file_fingerprint(prediction_path),
        "test_dataset": file_fingerprint(TEST_DATASET_PATH),
        "metric_scale": (
            "rna_and_cds_profile_and_cds_mean_log1p_periodicity_linear_v2"
        ),
    }


def load_or_evaluate_metrics(
    dataset: TranslationDataset,
    prediction_path: Path,
    spec: ModelSpec,
) -> Tuple[pd.DataFrame, Path, bool]:
    """Reuse verified transcript metrics or evaluate and persist them."""
    csv_path, manifest_path = metric_cache_paths(prediction_path)
    expected_manifest = metric_manifest(prediction_path, spec)
    if (
        REUSE_EXISTING_METRICS
        and csv_path.is_file()
        and manifest_matches(manifest_path, expected_manifest)
    ):
        print(f"Reusing verified RNA metrics: {csv_path}")
        return pd.read_csv(csv_path), csv_path, True

    transcript_metrics = evaluate_prediction_file(dataset, prediction_path, spec)
    transcript_metrics.to_csv(csv_path, index=False)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(expected_manifest, handle, indent=2)
    return transcript_metrics, csv_path, False


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
    """Aggregate RNA metrics while retaining cells below the RNA threshold."""
    rows = []
    for cell_type in expected_cells:
        raw_group = transcript_metrics[
            transcript_metrics["Cell_Type"] == cell_type
        ]
        if MIN_RPF_DEPTH is None:
            group = raw_group
        else:
            depth = pd.to_numeric(raw_group["RPF_Depth"], errors="coerce")
            group = raw_group[
                np.isfinite(depth) & (depth >= MIN_RPF_DEPTH)
            ]
        cell_is_eligible = len(group) >= MIN_RNA_PER_CELL
        periodicity = group[
            ["Observed_Periodicity", "Predicted_Periodicity", "Periodicity_Bias"]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        periodicity_excluded_count = int(len(group) - len(periodicity))
        high_periodicity = periodicity[
            periodicity["Observed_Periodicity"] >= HIGH_PERIODICITY_THRESHOLD
        ]
        rna_profile_n = int(group["RNA_Profile_Spearman"].notna().sum())
        cds_profile_n = int(group["CDS_Profile_Spearman"].notna().sum())
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
                "RNA_N": int(len(raw_group)),
                "RNA_Passing_Depth_N": int(len(group)),
                "RNA_Excluded_By_Depth_N": int(len(raw_group) - len(group)),
                "Meets_Min_RNA_Per_Cell": bool(cell_is_eligible),
                "RNA_Profile_N": rna_profile_n,
                "RNA_Profile_Excluded_N": int(len(group) - rna_profile_n),
                "Mean_RNA_Profile_Spearman": (
                    finite_mean(group["RNA_Profile_Spearman"])
                    if cell_is_eligible
                    else float("nan")
                ),
                "Median_RNA_Profile_Spearman": (
                    float(group["RNA_Profile_Spearman"].median())
                    if cell_is_eligible
                    else float("nan")
                ),
                "CDS_Profile_N": cds_profile_n,
                "CDS_Profile_Excluded_N": int(len(group) - cds_profile_n),
                "Mean_CDS_Profile_Spearman": (
                    finite_mean(group["CDS_Profile_Spearman"])
                    if cell_is_eligible
                    else float("nan")
                ),
                "Median_CDS_Profile_Spearman": (
                    float(group["CDS_Profile_Spearman"].median())
                    if cell_is_eligible
                    else float("nan")
                ),
                "Periodicity_N": int(len(periodicity)),
                "Periodicity_Excluded_N": periodicity_excluded_count,
                "Periodicity_Spearman": (
                    safe_spearman(
                        periodicity["Observed_Periodicity"],
                        periodicity["Predicted_Periodicity"],
                    )
                    if cell_is_eligible
                    else float("nan")
                ),
                "Periodicity_Bias": (
                    finite_mean(periodicity["Periodicity_Bias"])
                    if cell_is_eligible
                    else float("nan")
                ),
                "Periodicity_MAE": (
                    finite_mean(np.abs(periodicity["Periodicity_Bias"]))
                    if cell_is_eligible
                    else float("nan")
                ),
                "High_Periodicity_N": int(len(high_periodicity)),
                "High_Periodicity_Bias": (
                    finite_mean(high_periodicity["Periodicity_Bias"])
                    if cell_is_eligible
                    else float("nan")
                ),
                "High_Periodicity_MAE": (
                    finite_mean(np.abs(high_periodicity["Periodicity_Bias"]))
                    if cell_is_eligible
                    else float("nan")
                ),
                "CDS_Mean_Scale_N": scale_n,
                "CDS_Mean_Scale_Excluded_N": int(len(group) - scale_n),
                "CDS_Mean_Scale_Spearman": (
                    safe_spearman(
                        group["Observed_CDS_Mean_Log1p"],
                        group["Predicted_CDS_Mean_Log1p"],
                    )
                    if cell_is_eligible
                    else float("nan")
                ),
                "CDS_Mean_MAE_Log1p": (
                    finite_mean(group["CDS_Mean_Absolute_Error_Log1p"])
                    if cell_is_eligible
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def empty_cell_metrics(
    spec: ModelSpec,
    expected_cells: Sequence[str],
) -> pd.DataFrame:
    """Create explicit empty rows for an unavailable model-grid position."""
    empty_transcripts = pd.DataFrame(columns=TRANSCRIPT_METRIC_COLUMNS)
    return summarize_by_cell_type(empty_transcripts, spec, expected_cells)


SUMMARY_METRICS = (
    "Mean_RNA_Profile_Spearman",
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
            "Expected_Cell_Type_N": int(group["Cell_Type"].nunique()),
            "Cell_Type_N": int((group["RNA_N"] > 0).sum()),
            "Eligible_Cell_Type_N": int(
                group["Meets_Min_RNA_Per_Cell"].fillna(False).sum()
            ),
            "Min_RNA_Per_Cell": MIN_RNA_PER_CELL,
            "Min_RPF_Depth": (
                MIN_RPF_DEPTH if MIN_RPF_DEPTH is not None else np.nan
            ),
            "RNA_N": int(group["RNA_N"].sum()),
            "RNA_Passing_Depth_N": int(group["RNA_Passing_Depth_N"].sum()),
            "RNA_Excluded_By_Depth_N": int(
                group["RNA_Excluded_By_Depth_N"].sum()
            ),
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
    """Export editable vector files and a high-resolution PNG preview."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")


PANEL_METRICS = (
    (
        "Mean_RNA_Profile_Spearman",
        "RNA profile shape",
        "Mean per-cell RNA profile Spearman",
    ),
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

ZERO_SHOT_PANEL_METRICS = (
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

ZERO_SHOT_DELTA_PANEL_METRICS = (
    (
        "Delta_Mean_CDS_Profile_Spearman",
        "CDS profile shape",
        "Δ mean per-cell CDS profile Spearman",
    ),
    (
        "Delta_CDS_Mean_Scale_Spearman",
        "CDS signal scale",
        "Δ mean per-cell CDS-mean Spearman",
    ),
)


def quality_filter_description() -> str:
    """Return a concise description of the cell and RNA quality gates."""
    description = f"at least {MIN_RNA_PER_CELL} RNAs"
    if MIN_RPF_DEPTH is not None:
        description += f" passing RPF depth ≥ {MIN_RPF_DEPTH:g}"
    return description


def plot_environment_diversity_curves(
    model_metrics: pd.DataFrame,
    output_prefix: Path,
) -> None:
    """Plot 5/22/40-cell curves for RNA shape, CDS shape, and scale."""
    figure, axes = plt.subplots(
        1,
        len(PANEL_METRICS),
        figsize=(9, 3),
        sharex=True,
    )
    for panel_index, (axis, (metric, title, ylabel)) in enumerate(
        zip(axes, PANEL_METRICS)
    ):
        panel_has_data = False
        for strategy in ("zero", "real", "exp_aug"):
            group = model_metrics[model_metrics["Strategy"] == strategy].sort_values(
                "Environment_Count"
            )
            x = group["Environment_Count"].to_numpy(dtype=float)
            y = group[metric].to_numpy(dtype=float)
            low = group[f"{metric}_CI95_Low"].to_numpy(dtype=float)
            high = group[f"{metric}_CI95_High"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            panel_has_data = panel_has_data or bool(finite.any())
            finite_x = x[finite]
            finite_y = y[finite]
            finite_low = low[finite]
            finite_high = high[finite]
            lower_error = np.where(
                np.isfinite(finite_low),
                np.maximum(0.0, finite_y - finite_low),
                0.0,
            )
            upper_error = np.where(
                np.isfinite(finite_high),
                np.maximum(0.0, finite_high - finite_y),
                0.0,
            )
            axis.plot(
                x,
                y,
                color=STRATEGY_COLORS[strategy],
                marker=STRATEGY_MARKERS[strategy],
                markersize=5,
                linewidth=1.7,
                label=STRATEGY_LABELS[strategy],
                zorder=3,
            )
            axis.errorbar(
                finite_x,
                finite_y,
                yerr=np.vstack([lower_error, upper_error]),
                color=STRATEGY_COLORS[strategy],
                fmt="none",
                linewidth=1.0,
                capsize=2.5,
                zorder=2,
            )
        axis.set_title(title)
        axis.set_xlabel("Training environments")
        axis.set_ylabel(ylabel)
        axis.set_xticks([5, 22, 40], ["5", "22", "40"])
        axis.grid(axis="y", color="#E7E7E7", linewidth=0.7, zorder=0)
        if not panel_has_data:
            axis.text(
                0.5,
                0.5,
                "No evaluated model metrics",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )
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
        "Points are equal-weight means across held-out cell types with "
        f"{quality_filter_description()}; error bars are cell-bootstrap 95% CIs.",
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
    """Plot zero-shot CDS shape and scale separately for 5/22/40 environments."""
    x_column = "Nearest_Cosine_Distance"
    finite_distance = cell_metrics[x_column].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    x_grid = np.linspace(
        float(finite_distance.min()),
        float(finite_distance.max()),
        200,
    )

    figure, axes = plt.subplots(
        len(ZERO_SHOT_PANEL_METRICS),
        3,
        figsize=(9, 5.2),
        sharex=True,
        sharey="row",
    )
    regression_rows = []
    panel_index = 0
    for metric_index, (metric, row_title, ylabel) in enumerate(
        ZERO_SHOT_PANEL_METRICS
    ):
        for environment_index, environment_count in enumerate((5, 22, 40)):
            axis = axes[metric_index, environment_index]
            environment_data = cell_metrics[
                cell_metrics["Environment_Count"] == environment_count
            ]

            for strategy_index, strategy in enumerate(("zero", "real", "exp_aug")):
                group = environment_data[
                    environment_data["Strategy"] == strategy
                ]
                point_data = group[
                    np.isfinite(group[x_column]) & np.isfinite(group[metric])
                ]
                axis.scatter(
                    point_data[x_column],
                    point_data[metric],
                    s=18,
                    marker=STRATEGY_MARKERS[strategy],
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
                            RANDOM_SEED
                            + metric_index * 1000
                            + environment_index * 100
                            + strategy_index
                        )
                    ),
                )
                fitted, lower, upper, slope, intercept, rho, n = result
                if np.isfinite(fitted).any():
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
                        zorder=3,
                    )
                regression_rows.append(
                    {
                        "Metric": metric,
                        "Environment_Count": environment_count,
                        "Strategy": strategy,
                        "Strategy_Label": STRATEGY_LABELS[strategy],
                        "N_Cell_Model_Points": n,
                        "Excluded_Cell_Model_Points": int(len(group) - n),
                        "Linear_Slope": slope,
                        "Linear_Intercept": intercept,
                        "Distance_Performance_Spearman": rho,
                    }
                )

            if metric_index == 0:
                axis.set_title(f"{environment_count} environments")
            if environment_index == 0:
                axis.set_ylabel(f"{row_title}\n{ylabel}")
            axis.grid(color="#E7E7E7", linewidth=0.7, zorder=0)
            axis.text(
                0.01,
                0.98,
                chr(ord("a") + panel_index),
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                ha="left",
            )
            panel_index += 1

    method_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=STRATEGY_COLORS[strategy],
            marker=STRATEGY_MARKERS[strategy],
            linewidth=1.8,
            markersize=4.5,
            label=STRATEGY_LABELS[strategy],
        )
        for strategy in ("zero", "real", "exp_aug")
    ]
    figure.legend(
        handles=method_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.supxlabel(
        "Distance to nearest training environment",
        x=0.5,
        y=0.06,
        fontsize=8,
    )
    figure.text(
        0.5,
        -0.01,
        "Points are held-out cell types with "
        f"{quality_filter_description()}; bands are cell-cluster bootstrap 95% CIs.",
        ha="center",
        fontsize=6.5,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.93), h_pad=1.2, w_pad=1.2)
    save_publication_figure(figure, output_prefix)
    plt.close(figure)
    return pd.DataFrame.from_records(regression_rows)


def build_zero_shot_delta_source(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair ExpAug and Zero results within each environment count and cell type."""
    metrics = [item[0] for item in ZERO_SHOT_PANEL_METRICS]
    shared_columns = [
        "Environment_Count",
        "Cell_Type",
        "Nearest_Training_Cell",
        "Nearest_Cosine_Distance",
        "Expression_Coverage",
    ]
    zero = cell_metrics[cell_metrics["Strategy"] == "zero"][
        shared_columns + metrics
    ].copy()
    exp_aug = cell_metrics[cell_metrics["Strategy"] == "exp_aug"][
        ["Environment_Count", "Cell_Type"] + metrics
    ].copy()
    zero = zero.rename(columns={metric: f"Zero_{metric}" for metric in metrics})
    exp_aug = exp_aug.rename(
        columns={metric: f"ExpAug_{metric}" for metric in metrics}
    )
    paired = zero.merge(
        exp_aug,
        on=["Environment_Count", "Cell_Type"],
        how="inner",
        validate="one_to_one",
    )
    for metric in metrics:
        paired[f"Delta_{metric}"] = (
            paired[f"ExpAug_{metric}"] - paired[f"Zero_{metric}"]
        )
    return paired.sort_values(["Environment_Count", "Cell_Type"])


def plot_zero_shot_delta_curves(
    delta_source: pd.DataFrame,
    output_prefix: Path,
) -> pd.DataFrame:
    """Plot ExpAug-minus-Zero performance against expression distance."""
    x_column = "Nearest_Cosine_Distance"
    figure, axes = plt.subplots(
        len(ZERO_SHOT_DELTA_PANEL_METRICS),
        3,
        figsize=(6, 4),
        sharex=False,
        sharey="row",
    )
    regression_rows = []
    for metric_index, (metric, row_title, ylabel) in enumerate(
        ZERO_SHOT_DELTA_PANEL_METRICS
    ):
        for environment_index, environment_count in enumerate((5, 22, 40)):
            axis = axes[metric_index, environment_index]
            group = delta_source[
                delta_source["Environment_Count"] == environment_count
            ]
            point_data = group[
                np.isfinite(group[x_column]) & np.isfinite(group[metric])
            ]
            if len(point_data):
                x_grid = np.linspace(
                    float(point_data[x_column].min()),
                    float(point_data[x_column].max()),
                    200,
                )
            else:
                x_grid = np.linspace(0.0, 1.0, 200)
            axis.axhline(
                0.0,
                color="#999999",
                linewidth=0.9,
                linestyle="--",
                zorder=1,
            )
            axis.scatter(
                point_data[x_column],
                point_data[metric],
                s=20,
                marker=STRATEGY_MARKERS["exp_aug"],
                facecolor=STRATEGY_COLORS["exp_aug"],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.58,
                zorder=3,
            )
            result = cluster_bootstrap_regression(
                group,
                x_column,
                metric,
                x_grid,
                BOOTSTRAP_ITERATIONS,
                np.random.Generator(
                    np.random.PCG64(
                        RANDOM_SEED + metric_index * 1000 + environment_index * 100
                    )
                ),
            )
            fitted, lower, upper, slope, intercept, rho, n = result
            if n >= 2 and point_data[x_column].nunique() >= 2:
                correlation = spearmanr(
                    point_data[x_column], point_data[metric]
                )
                rho = float(correlation.statistic)
                p_value = float(correlation.pvalue)
            else:
                p_value = float("nan")
            if np.isfinite(fitted).any():
                axis.fill_between(
                    x_grid,
                    lower,
                    upper,
                    color=STRATEGY_COLORS["exp_aug"],
                    alpha=0.14,
                    linewidth=0,
                    zorder=2,
                )
                axis.plot(
                    x_grid,
                    fitted,
                    color=STRATEGY_COLORS["exp_aug"],
                    linewidth=1.8,
                    zorder=4,
                )
            regression_rows.append(
                {
                    "Metric": metric,
                    "Environment_Count": environment_count,
                    "N_Paired_Cell_Types": n,
                    "Excluded_Paired_Cell_Types": int(len(group) - n),
                    "Linear_Slope": slope,
                    "Linear_Intercept": intercept,
                    "Distance_Delta_Spearman": rho,
                    "Distance_Delta_Spearman_P": p_value,
                }
            )

            if np.isfinite(rho) and np.isfinite(p_value):
                p_text = f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.3f}"
                axis.text(
                    0.98,
                    0.97,
                    f"ρ = {rho:.2f}\nP = {p_text}",
                    transform=axis.transAxes,
                    fontsize=6.5,
                    va="top",
                    ha="right",
                )

            if metric_index == 0:
                axis.set_title(f"{environment_count} environments")
            if environment_index == 0:
                axis.set_ylabel(f"{row_title}\n{ylabel}")
            axis.grid(color="#E7E7E7", linewidth=0.7, zorder=0)

    figure.text(
        0.5,
        0.965,
        "Δ performance = Mask + interpolation − Zero expression",
        ha="center",
        va="top",
        fontsize=8,
    )
    figure.supxlabel(
        "Distance to nearest training environment",
        x=0.5,
        y=0.06,
        fontsize=8,
    )
    figure.text(
        0.5,
        -0.01,
        "Points are paired held-out cell types; bands are cell-bootstrap 95% CIs.",
        ha="center",
        fontsize=6.5,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.92), h_pad=1.2, w_pad=1.2)
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


DISTANCE_COLUMNS = (
    "Environment_Count",
    "Cell_Type",
    "Nearest_Training_Cell",
    "Nearest_Cosine_Distance",
    "Expression_Coverage",
)


def prepare_expression_distances(
    test_vectors: Mapping[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate every available distance group and report skipped groups."""
    training_vectors_by_count: Dict[int, Dict[str, np.ndarray]] = {}
    status_rows = []
    for environment_count, paths in TRAIN_ENVIRONMENT_DATASETS.items():
        try:
            vectors = read_cell_expression_vectors(paths)
            if len(vectors) != environment_count:
                raise ValueError(
                    f"Expected {environment_count} environments, found {len(vectors)}"
                )
        except Exception as error:
            if FAIL_FAST:
                raise
            print(
                f"[EnvironmentDiversity] Skipping {environment_count}c expression "
                f"distances: {error}"
            )
            status_rows.append(
                {
                    "Environment_Count": environment_count,
                    "Status": "unavailable",
                    "Environment_N": 0,
                    "Message": str(error),
                }
            )
            continue
        training_vectors_by_count[environment_count] = vectors
        status_rows.append(
            {
                "Environment_Count": environment_count,
                "Status": "available",
                "Environment_N": len(vectors),
                "Message": "",
            }
        )

    try:
        validate_nested_training_environments(training_vectors_by_count)
    except Exception as error:
        if FAIL_FAST:
            raise
        print(f"[EnvironmentDiversity] Training-set nesting warning: {error}")

    distance_parts = []
    for environment_count, vectors in training_vectors_by_count.items():
        try:
            distance_parts.append(
                calculate_expression_distances(
                    test_vectors,
                    {environment_count: vectors},
                )
            )
        except Exception as error:
            if FAIL_FAST:
                raise
            print(
                f"[EnvironmentDiversity] Skipping {environment_count}c distance "
                f"table: {error}"
            )
            for row in status_rows:
                if row["Environment_Count"] == environment_count:
                    row["Status"] = "invalid"
                    row["Message"] = str(error)

    distance_table = (
        pd.concat(distance_parts, ignore_index=True)
        if distance_parts
        else pd.DataFrame(columns=DISTANCE_COLUMNS)
    )
    return distance_table, pd.DataFrame.from_records(status_rows)


def initial_model_status(spec: ModelSpec) -> dict:
    """Create one persistent status row for a model-grid position."""
    return {
        "Model_ID": spec.model_id,
        "Environment_Count": spec.environment_count,
        "Strategy": spec.strategy,
        "Strategy_Label": spec.strategy_label,
        "Resolution_Status": spec.resolution_status,
        "Resolution_Message": spec.resolution_message,
        "Checkpoint": str(spec.checkpoint) if spec.checkpoint is not None else "",
        "Checkpoint_Epoch": "",
        "Prediction_Status": "pending",
        "Prediction_Reused": "",
        "Prediction_PKL": "",
        "Metrics_Status": "pending",
        "Metrics_Reused": "",
        "RNA_Metrics_CSV": "",
        "Error": "",
    }


def write_model_status(status_by_model: Mapping[str, dict]) -> pd.DataFrame:
    """Persist the latest model-grid status after each completed unit of work."""
    table = pd.DataFrame.from_records(list(status_by_model.values()))
    table.to_csv(OUTPUT_DIR / "checkpoint_manifest.csv", index=False)
    return table


def append_status_error(row: dict, error: object) -> None:
    """Append an error message without discarding an earlier diagnostic."""
    message = str(error)
    row["Error"] = f"{row['Error']} | {message}".strip(" |")


def main() -> None:
    """Run prediction, metric aggregation, source-data export, and plotting."""
    require_files([TEST_DATASET_PATH], "evaluation inputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "rna_metrics").mkdir(parents=True, exist_ok=True)
    prediction_directory().mkdir(parents=True, exist_ok=True)

    model_specs = build_model_specs()
    status_by_model = {
        spec.model_id: initial_model_status(spec) for spec in model_specs
    }
    write_model_status(status_by_model)

    test_dataset = TranslationDataset.from_h5(str(TEST_DATASET_PATH), lazy=True)
    test_cell_types = validate_test_dataset(test_dataset)
    print(
        f"Test dataset: {len(test_dataset):,} samples across "
        f"{len(test_cell_types)} held-out cell types"
    )

    test_vectors = {
        cell_type: np.asarray(
            test_dataset.cell_expr_dict[cell_type], dtype=np.float32
        ).reshape(-1)
        for cell_type in test_cell_types
    }
    distance_table, environment_status = prepare_expression_distances(
        test_vectors
    )
    distance_table.to_csv(
        OUTPUT_DIR / "nearest_training_environment.csv", index=False
    )
    environment_status.to_csv(
        OUTPUT_DIR / "training_environment_status.csv", index=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ANALYSIS_ONLY:
        print(
            "Analysis-only mode: reusing verified prediction PKLs and skipping "
            "all model inference."
        )
    else:
        print(f"Using device: {device}")
    prediction_paths: Dict[str, Path] = {}

    phase_label = (
        "locating cached predictions"
        if ANALYSIS_ONLY
        else "predicting all checkpoints"
    )
    print(f"\n=== Phase 1/2: {phase_label} ===")
    for spec in model_specs:
        row = status_by_model[spec.model_id]
        cached_path = find_cached_prediction(spec)
        if cached_path is not None:
            cached_manifest = read_prediction_manifest(cached_path)
            cached_checkpoint = cached_manifest.get("checkpoint", {})
            if not row["Checkpoint"] and isinstance(cached_checkpoint, dict):
                row["Checkpoint"] = cached_checkpoint.get("path", "")
            row["Checkpoint_Epoch"] = cached_manifest.get(
                "checkpoint_epoch", ""
            )
            row["Prediction_Status"] = "cached"
            row["Prediction_Reused"] = True
            row["Prediction_PKL"] = str(cached_path)
            prediction_paths[spec.model_id] = cached_path
            print(f"Reusing verified predictions: {cached_path}")
            write_model_status(status_by_model)
            continue

        if ANALYSIS_ONLY:
            row["Prediction_Status"] = "unavailable"
            row["Metrics_Status"] = "unavailable"
            append_status_error(
                row,
                "No verified cached prediction was found while ANALYSIS_ONLY=True",
            )
            write_model_status(status_by_model)
            print(
                f"[EnvironmentDiversity] Leaving {spec.model_id} blank: "
                "no verified cached prediction"
            )
            continue

        if not spec.can_predict:
            row["Prediction_Status"] = "unavailable"
            row["Metrics_Status"] = "unavailable"
            if not spec.config_path.is_file():
                append_status_error(row, f"Missing model config: {spec.config_path}")
            if spec.checkpoint is None:
                append_status_error(row, spec.resolution_message)
            write_model_status(status_by_model)
            print(
                f"[EnvironmentDiversity] Leaving {spec.model_id} blank: "
                f"{row['Error'] or spec.resolution_message}"
            )
            continue

        model = None
        try:
            model, metadata = load_model(spec, device)
            prediction_path = predict_one_model(
                spec,
                model,
                test_dataset,
                checkpoint_metadata=metadata,
            )
            prediction_paths[spec.model_id] = prediction_path
            row["Checkpoint_Epoch"] = metadata.get("epoch", "")
            row["Prediction_Status"] = "generated"
            row["Prediction_Reused"] = False
            row["Prediction_PKL"] = str(prediction_path)
        except Exception as error:
            row["Prediction_Status"] = "failed"
            row["Metrics_Status"] = "unavailable"
            append_status_error(row, error)
            print(f"[EnvironmentDiversity] Prediction failed for {spec.model_id}: {error}")
            if FAIL_FAST:
                raise
        finally:
            if model is not None:
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_model_status(status_by_model)

    print("\n=== Phase 2/2: evaluating all prediction files ===")
    all_cell_metrics = []
    for spec in model_specs:
        row = status_by_model[spec.model_id]
        prediction_path = prediction_paths.get(spec.model_id)
        if prediction_path is None:
            all_cell_metrics.append(empty_cell_metrics(spec, test_cell_types))
            row["Metrics_Status"] = "unavailable"
            write_model_status(status_by_model)
            continue

        try:
            transcript_metrics, metric_path, reused = load_or_evaluate_metrics(
                test_dataset,
                prediction_path,
                spec,
            )
            row["RNA_Metrics_CSV"] = str(metric_path)
            row["Metrics_Reused"] = reused
            if transcript_metrics.empty:
                row["Metrics_Status"] = "empty"
                all_cell_metrics.append(empty_cell_metrics(spec, test_cell_types))
            else:
                row["Metrics_Status"] = "cached" if reused else "generated"
                all_cell_metrics.append(
                    summarize_by_cell_type(
                        transcript_metrics,
                        spec,
                        test_cell_types,
                    )
                )
        except Exception as error:
            row["Metrics_Status"] = "failed"
            append_status_error(row, error)
            all_cell_metrics.append(empty_cell_metrics(spec, test_cell_types))
            print(f"[EnvironmentDiversity] Evaluation failed for {spec.model_id}: {error}")
            if FAIL_FAST:
                raise
        finally:
            write_model_status(status_by_model)

    cell_metrics = pd.concat(all_cell_metrics, ignore_index=True)
    excluded_cells = (
        cell_metrics[
            (cell_metrics["RNA_N"] > 0)
            & ~cell_metrics["Meets_Min_RNA_Per_Cell"]
        ][["Cell_Type", "RNA_N", "RNA_Passing_Depth_N"]]
        .drop_duplicates()
        .sort_values(["RNA_Passing_Depth_N", "Cell_Type"])
    )
    if not excluded_cells.empty:
        excluded_text = ", ".join(
            f"{row.Cell_Type} (passing={int(row.RNA_Passing_Depth_N)}/"
            f"{int(row.RNA_N)})"
            for row in excluded_cells.itertuples(index=False)
        )
        print(
            f"Excluded from cell-level summaries because n < "
            f"{MIN_RNA_PER_CELL}: {excluded_text}"
        )
    if distance_table.empty:
        cell_metrics["Nearest_Training_Cell"] = ""
        cell_metrics["Nearest_Cosine_Distance"] = np.nan
        cell_metrics["Expression_Coverage"] = np.nan
    else:
        cell_metrics = cell_metrics.merge(
            distance_table,
            on=["Environment_Count", "Cell_Type"],
            how="left",
            validate="many_to_one",
        )

    model_metrics = summarize_models(cell_metrics)
    checkpoint_table = write_model_status(status_by_model)
    status_columns = [
        "Model_ID",
        "Resolution_Status",
        "Prediction_Status",
        "Prediction_Reused",
        "Metrics_Status",
        "Metrics_Reused",
        "Error",
    ]
    model_metrics = model_metrics.merge(
        checkpoint_table[status_columns],
        on="Model_ID",
        how="left",
        validate="one_to_one",
    )
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
            "RNA_N",
            "RNA_Passing_Depth_N",
            "RNA_Excluded_By_Depth_N",
            "Meets_Min_RNA_Per_Cell",
            "Nearest_Training_Cell",
            "Nearest_Cosine_Distance",
            "Expression_Coverage",
            "Mean_RNA_Profile_Spearman",
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
    delta_source = build_zero_shot_delta_source(cell_metrics)
    delta_source.to_csv(
        OUTPUT_DIR / "zero_shot_distance_delta_curve_source.csv", index=False
    )
    delta_regression = plot_zero_shot_delta_curves(
        delta_source,
        figure_dir / "zero_shot_distance_delta_shape_and_scale",
    )
    delta_regression.to_csv(
        OUTPUT_DIR / "zero_shot_distance_delta_regression.csv", index=False
    )

    print("\nEnvironment-diversity evaluation complete.")
    print(f"Model summary: {OUTPUT_DIR / 'model_metrics.csv'}")
    print(f"Cell-type summary: {OUTPUT_DIR / 'cell_type_metrics.csv'}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()

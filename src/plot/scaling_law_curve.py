#!/usr/bin/env python3
"""Plot TRACE training and validation losses against epoch or estimated FLOPs.

Edit the configuration section and run this file directly on the server. The
script reads Trainer ``*.epoch_data.json`` files or plain-text training logs.
Training and validation losses are exported as separate figures.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT.parent / "log/train"
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
OUTPUT_PREFIX = (
    PROJECT_ROOT.parent / "results/ablation/loss_curves/model_loss_comparison"
)

COMPARISON_DATASET = "human_5c_6k_depth0.1_cov0.1_rpm1"
LOSS_DEFINITION = "micro + 2.0*macro + 0.2*ranking"
ALLOW_MIXED_DATASETS = False

X_AXIS = "flops"  # Choose "flops" or "epoch".
X_LOG = True
TRAIN_Y_LOG = True
VALID_Y_LOG = True
SHOW_FIGURE = False

FLOP_UNIT = 1e18
FLOP_UNIT_LABEL = "EFLOPs"
TRAINING_FLOP_MULTIPLIER = 3.0
HEAD_HIDDEN_DIM = 384

TRAIN_DATASETS = {
    "5c": ["human_5c_6k_depth0.1_cov0.1_rpm1.train.h5"],
    "22c": ["human_tissue_22c_6k_depth0.1_cov0.1_rpm1.train.h5"],
    "40c": [
        "human_tissue_22c_6k_depth0.1_cov0.1_rpm1.train.h5",
        "human_cell_line_18c_6k_depth0.1_cov0.1_rpm1.train.h5",
    ],
}

MODEL_CONFIGS = {
    "trace": "src/config/base_model_384d_16h_12l_64env_16ad_bs.yaml",
    "ln": "src/config/base_model_LN_384d_16h_12l.yaml",
    "conv": "src/config/base_model_conv_384d_12l_7k.yaml",
}

# Every run uses the same configuration fields. A relative log glob is resolved
# inside LOG_DIR; replace it with an exact ``path`` for final reproducibility.
MODEL_RUNS = [
    {
        "label": "TRACE-Zero (5 cell contexts)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*zero*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "train_dataset_files": TRAIN_DATASETS["5c"],
        "model_config_path": MODEL_CONFIGS["trace"],
        "loss_definition": LOSS_DEFINITION,
        "color": "#7A7A7A",
        "linestyle": "--",
        "enabled": True,
    },
    {
        "label": "TRACE-Real (5 cell contexts)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*real*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "train_dataset_files": TRAIN_DATASETS["5c"],
        "model_config_path": MODEL_CONFIGS["trace"],
        "loss_definition": LOSS_DEFINITION,
        "color": "#78A9CF",
        "linestyle": "-.",
        "enabled": True,
    },
    {
        "label": "TRACE-Mask+Interpolation (5 cell contexts)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*exp_aug*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "train_dataset_files": TRAIN_DATASETS["5c"],
        "model_config_path": MODEL_CONFIGS["trace"],
        "loss_definition": LOSS_DEFINITION,
        "color": "#166A9A",
        "linestyle": "-",
        "enabled": True,
    },
    {
        "label": "LayerNorm Transformer (5 cell contexts)",
        "glob": "base_model_LN*hs_5c*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "train_dataset_files": TRAIN_DATASETS["5c"],
        "model_config_path": MODEL_CONFIGS["ln"],
        "loss_definition": LOSS_DEFINITION,
        "color": "#C28548",
        "linestyle": "-",
        "enabled": True,
    },
    {
        "label": "Convolutional model (5 cell contexts)",
        "glob": "base_model_conv*hs_5c*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "train_dataset_files": TRAIN_DATASETS["5c"],
        "model_config_path": MODEL_CONFIGS["conv"],
        "loss_definition": LOSS_DEFINITION,
        "color": "#5F9272",
        "linestyle": "-",
        "enabled": False,
    },
    {
        "label": "TRACE-Mask+Interpolation (22 cell contexts)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_22c*exp_aug*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "train_dataset_files": TRAIN_DATASETS["22c"],
        "model_config_path": MODEL_CONFIGS["trace"],
        "loss_definition": LOSS_DEFINITION,
        "color": "#9A6FB0",
        "linestyle": "-.",
        "enabled": True,
    },
    {
        "label": "TRACE-Mask+Interpolation (22 cell contexts)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_40c*exp_aug*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "train_dataset_files": TRAIN_DATASETS["40c"],
        "model_config_path": MODEL_CONFIGS["trace"],
        "loss_definition": LOSS_DEFINITION,
        "color": "#6A3D78",
        "linestyle": "-",
        "enabled": True,
    },
]


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


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_LOSS_PATTERN = re.compile(
    rf"Epoch\s+(\d+)\s+(training|evaluating)\s+time:"
    rf"[^\n]*?mean\s+loss:\s*(?:tensor\(\[?)?\s*({FLOAT_PATTERN})",
    flags=re.IGNORECASE,
)
VALIDATION_METRICS_PATTERN = re.compile(
    rf"Epoch\s+(\d+)\s+validation\s+metrics:\s*"
    rf"profile\s+Spearman=({FLOAT_PATTERN}).*?"
    rf"CDS-mean\s+scale\s+Spearman=({FLOAT_PATTERN})",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ComputeEstimate:
    """Simple dataset and training-compute summary for one run."""

    training_dataset: str
    n_transcripts: int
    total_length: float
    total_length_squared: float
    flops_per_epoch: float


@dataclass
class RunHistory:
    """Clean epoch-level history for one model run."""

    label: str
    dataset: str
    loss_definition: str
    color: str
    linestyle: str
    epochs: np.ndarray
    train_loss: np.ndarray
    valid_loss: np.ndarray
    alpha: np.ndarray
    profile_spearman: np.ndarray
    scale_spearman: np.ndarray
    duplicate_epochs: int = 0
    compute: Optional[ComputeEstimate] = None

    def best_validation(self) -> tuple[int, float]:
        """Return the epoch and minimum finite validation loss."""
        finite = np.isfinite(self.valid_loss)
        if not finite.any():
            raise ValueError(f"Run {self.label!r} has no finite validation losses")
        indices = np.flatnonzero(finite)
        best_index = indices[np.argmin(self.valid_loss[finite])]
        return int(self.epochs[best_index]), float(self.valid_loss[best_index])

    @property
    def cumulative_flops(self) -> np.ndarray:
        """Return cumulative FLOPs at each completed epoch."""
        if self.compute is None:
            return np.full(self.epochs.shape, np.nan, dtype=float)
        return self.epochs.astype(float) * self.compute.flops_per_epoch


_DATASET_LENGTH_CACHE: Dict[str, np.ndarray] = {}


def _to_float(value: Any) -> float:
    """Convert common scalar representations to a finite float or NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return float("nan")
        return _to_float(value[0])
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return float("nan")
        return _to_float(value.reshape(-1)[0])
    if isinstance(value, str):
        match = re.search(FLOAT_PATTERN, value)
        if match is None:
            return float("nan")
        value = match.group(0)
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return converted if math.isfinite(converted) else float("nan")


def _first_value(record: Mapping[str, Any], names: Sequence[str]) -> float:
    """Return the first available metric alias."""
    for name in names:
        if name in record:
            return _to_float(record[name])
    return float("nan")


def _extract_json_entries(payload: Any) -> List[Dict[str, Any]]:
    """Normalize supported JSON history containers to a list of records."""
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = None
        for key in ("training_epoch_data", "epoch_data", "history", "data"):
            if isinstance(payload.get(key), list):
                entries = payload[key]
                break
        if entries is None and all(isinstance(value, dict) for value in payload.values()):
            entries = list(payload.values())
        if entries is None:
            raise ValueError("JSON does not contain an epoch-history list")
    else:
        raise TypeError("Epoch history JSON must contain a list or dictionary")
    if not all(isinstance(entry, dict) for entry in entries):
        raise TypeError("Every epoch entry must be a dictionary")
    return list(entries)


def _record_from_mapping(entry: Mapping[str, Any]) -> Dict[str, float]:
    """Normalize one current or legacy Trainer epoch record."""
    epoch = _first_value(entry, ("epoch", "epoch_num", "epoch_index"))
    if not math.isfinite(epoch):
        raise ValueError(f"Epoch entry has no valid epoch field: {entry}")
    return {
        "epoch": int(epoch),
        "train_loss": _first_value(
            entry, ("train_loss", "training_loss", "mean_train_loss")
        ),
        "valid_loss": _first_value(
            entry, ("valid_loss", "val_loss", "validation_loss", "mean_valid_loss")
        ),
        "alpha": _first_value(entry, ("alpha", "macro_loss_weight")),
        "profile_spearman": _first_value(
            entry, ("profile_spearman", "mean_profile_spearman")
        ),
        "scale_spearman": _first_value(
            entry, ("scale_spearman", "cds_mean_scale_spearman")
        ),
    }


def parse_epoch_json(path: Path) -> tuple[List[Dict[str, float]], int]:
    """Parse JSON history and retain the last occurrence of duplicate epochs."""
    with path.open(encoding="utf-8") as handle:
        entries = _extract_json_entries(json.load(handle))
    records: Dict[int, Dict[str, float]] = {}
    duplicates = 0
    for entry in entries:
        record = _record_from_mapping(entry)
        epoch = int(record["epoch"])
        duplicates += int(epoch in records)
        records[epoch] = record
    return [records[epoch] for epoch in sorted(records)], duplicates


def parse_text_log(path: Path) -> tuple[List[Dict[str, float]], int]:
    """Parse current Trainer console logs."""
    text = path.read_text(encoding="utf-8", errors="replace")
    records: Dict[int, Dict[str, float]] = {}
    seen_fields = set()
    duplicates = 0

    def get_record(epoch: int) -> Dict[str, float]:
        return records.setdefault(
            epoch,
            {
                "epoch": epoch,
                "train_loss": float("nan"),
                "valid_loss": float("nan"),
                "alpha": float("nan"),
                "profile_spearman": float("nan"),
                "scale_spearman": float("nan"),
            },
        )

    for match in EPOCH_LOSS_PATTERN.finditer(text):
        epoch = int(match.group(1))
        field = "train_loss" if match.group(2).lower() == "training" else "valid_loss"
        duplicates += int((epoch, field) in seen_fields)
        seen_fields.add((epoch, field))
        get_record(epoch)[field] = float(match.group(3))

    for match in VALIDATION_METRICS_PATTERN.finditer(text):
        epoch = int(match.group(1))
        get_record(epoch)["profile_spearman"] = float(match.group(2))
        get_record(epoch)["scale_spearman"] = float(match.group(3))

    if not records:
        raise ValueError(f"No epoch-level losses were recognized in {path}")
    return [records[epoch] for epoch in sorted(records)], duplicates


def _resolve_run_paths(config: Mapping[str, Any], log_dir: Path) -> List[Path]:
    """Resolve an exact path or the newest matching log file."""
    if "path" in config:
        raw_paths = [config["path"]]
    elif "paths" in config:
        raw_paths = list(config["paths"])
    elif "glob" in config:
        matches = sorted(
            log_dir.glob(str(config["glob"])), key=lambda path: path.stat().st_mtime
        )
        if not matches:
            raise FileNotFoundError(
                f"No log matched {config['glob']!r} inside {log_dir}"
            )
        return [matches[-1]]
    else:
        raise ValueError(f"Run {config.get('label', '<unnamed>')!r} needs path or glob")

    paths = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = log_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Training log not found: {path}")
        paths.append(path)
    return paths


def _resolve_project_path(raw_path: Any, base_dir: Path) -> Path:
    """Resolve a user-configured path relative to a stable project directory."""
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else base_dir / path


def _read_dataset_lengths(dataset_files: Sequence[str]) -> tuple[np.ndarray, List[Path]]:
    """Read per-transcript lengths from one or more HDF5 datasets."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for FLOP estimation") from exc

    arrays = []
    paths = []
    for file_name in dataset_files:
        path = _resolve_project_path(file_name, DATASET_DIR).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training dataset not found: {path}")
        paths.append(path)
        cache_key = str(path)
        if cache_key not in _DATASET_LENGTH_CACHE:
            with h5py.File(path, "r") as handle:
                if "samples" not in handle:
                    raise KeyError(f"HDF5 file has no /samples group: {path}")
                lengths = [
                    int(sample["count_emb"].shape[0])
                    for sample in handle["samples"].values()
                ]
            _DATASET_LENGTH_CACHE[cache_key] = np.asarray(lengths, dtype=np.float64)
        arrays.append(_DATASET_LENGTH_CACHE[cache_key])
    return np.concatenate(arrays), paths


def _load_model_config(config_path: str) -> Dict[str, Any]:
    """Load the small YAML mapping used to construct the model."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for FLOP estimation") from exc
    path = _resolve_project_path(config_path, PROJECT_ROOT).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Model config must contain a mapping: {path}")
    return payload


def estimate_flops_from_lengths(
    lengths: Sequence[float],
    model_config: Mapping[str, Any],
    head_hidden_dim: int = HEAD_HIDDEN_DIM,
) -> float:
    """Estimate one training epoch from sum(L) and sum(L squared).

    One multiply-add counts as two FLOPs. The estimate includes sequence and
    prediction-head projections plus Transformer attention/FFN or convolutional
    blocks. Training is approximated as three forward passes. Small elementwise
    operations and padding overhead are intentionally omitted.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError("Transcript lengths must be a non-empty positive vector")

    d_seq = int(model_config.get("d_seq", 4))
    d_model = int(model_config["d_model"])
    d_ff = int(model_config["d_ff"])
    n_layers = int(model_config["number_of_layers"])
    model_name = str(model_config.get("model_name", "")).lower()
    sum_length = float(lengths.sum())
    sum_length_squared = float(np.square(lengths).sum())

    sequence_flops = 2 * d_seq * d_model * sum_length
    head_flops = (
        2 * d_model * head_hidden_dim + 2 * head_hidden_dim
    ) * sum_length

    if "conv" in model_name:
        kernel_size = int(model_config.get("kernel_size", 7))
        backbone_flops = (
            2 * n_layers * d_model * d_ff * (kernel_size + 1) * sum_length
        )
    else:
        projection_ffn_flops = (
            n_layers * (8 * d_model**2 + 4 * d_model * d_ff) * sum_length
        )
        attention_flops = 4 * n_layers * d_model * sum_length_squared
        backbone_flops = projection_ffn_flops + attention_flops

    forward_flops = sequence_flops + backbone_flops + head_flops
    return float(forward_flops * TRAINING_FLOP_MULTIPLIER)


def _estimate_run_compute(config: Mapping[str, Any]) -> Optional[ComputeEstimate]:
    """Estimate FLOPs when dataset and model paths are configured."""
    dataset_files = config.get("train_dataset_files")
    model_config_path = config.get("model_config_path")
    if not dataset_files or not model_config_path:
        return None
    lengths, paths = _read_dataset_lengths(dataset_files)
    model_config = _load_model_config(str(model_config_path))
    return ComputeEstimate(
        training_dataset=" + ".join(path.name for path in paths),
        n_transcripts=int(lengths.size),
        total_length=float(lengths.sum()),
        total_length_squared=float(np.square(lengths).sum()),
        flops_per_epoch=estimate_flops_from_lengths(lengths, model_config),
    )


def load_run_history(
    config: Mapping[str, Any], log_dir: Path = LOG_DIR
) -> RunHistory:
    """Load, merge, and sort one configured training history."""
    merged: Dict[int, Dict[str, float]] = {}
    duplicate_epochs = 0

    if "loss_data" in config:
        entries = _extract_json_entries(config["loss_data"])
        groups = [([_record_from_mapping(entry) for entry in entries], 0)]
    else:
        groups = []
        for path in _resolve_run_paths(config, log_dir):
            groups.append(
                parse_epoch_json(path)
                if path.suffix.lower() == ".json"
                else parse_text_log(path)
            )

    for records, duplicates in groups:
        duplicate_epochs += duplicates
        for record in records:
            epoch = int(record["epoch"])
            duplicate_epochs += int(epoch in merged)
            merged[epoch] = record

    ordered = [merged[epoch] for epoch in sorted(merged)]
    if not ordered:
        raise ValueError(f"Run {config.get('label', '<unnamed>')!r} has no records")
    valid_loss = np.asarray([record["valid_loss"] for record in ordered], dtype=float)
    if not np.isfinite(valid_loss).any():
        raise ValueError(f"Run {config.get('label')!r} has no validation losses")

    return RunHistory(
        label=str(config.get("label", config.get("method", "Unknown"))),
        dataset=str(config.get("dataset", "")),
        loss_definition=str(config.get("loss_definition", "")),
        color=str(config.get("color", "#333333")),
        linestyle=str(config.get("linestyle", "-")),
        epochs=np.asarray([record["epoch"] for record in ordered], dtype=int),
        train_loss=np.asarray(
            [record["train_loss"] for record in ordered], dtype=float
        ),
        valid_loss=valid_loss,
        alpha=np.asarray([record["alpha"] for record in ordered], dtype=float),
        profile_spearman=np.asarray(
            [record["profile_spearman"] for record in ordered], dtype=float
        ),
        scale_spearman=np.asarray(
            [record["scale_spearman"] for record in ordered], dtype=float
        ),
        duplicate_epochs=duplicate_epochs,
        compute=_estimate_run_compute(config),
    )


def validate_comparison(
    histories: Sequence[RunHistory], allow_mixed_datasets: bool = False
) -> None:
    """Check that overlaid validation losses use comparable definitions."""
    if len(histories) < 2:
        raise ValueError("At least two model histories are required")
    datasets = {history.dataset for history in histories if history.dataset}
    if len(datasets) > 1 and not allow_mixed_datasets:
        raise ValueError(
            "Validation losses use different datasets: " + ", ".join(sorted(datasets))
        )
    definitions = {
        history.loss_definition for history in histories if history.loss_definition
    }
    if len(definitions) > 1:
        raise ValueError("Validation losses use different loss definitions")


def _x_values(history: RunHistory, x_axis: str) -> np.ndarray:
    """Return epoch or normalized cumulative-FLOP coordinates."""
    if x_axis == "epoch":
        return history.epochs.astype(float)
    if x_axis != "flops":
        raise ValueError("X_AXIS must be 'epoch' or 'flops'")
    if history.compute is None:
        raise ValueError(
            f"{history.label} needs train_dataset_files and model_config_path"
        )
    return history.cumulative_flops / FLOP_UNIT


def plot_loss_curve(
    histories: Sequence[RunHistory],
    metric: str,
    title: str,
    *,
    x_axis: str = X_AXIS,
    x_log: bool = X_LOG,
    y_log: bool = False,
):
    """Plot one loss type in its own single-panel figure."""
    figure, axis = plt.subplots(figsize=(3.5039, 2.8346))
    for history in histories:
        values = np.asarray(getattr(history, metric), dtype=float)
        x_values = _x_values(history, x_axis)
        finite = np.isfinite(values) & np.isfinite(x_values)
        if not finite.any():
            continue
        x_plot = x_values[finite]
        y_plot = values[finite]
        if x_log and np.any(x_plot <= 0):
            raise ValueError(f"{history.label} has non-positive x values for log scale")
        if y_log and np.any(y_plot <= 0):
            raise ValueError(f"{history.label} has non-positive losses for log scale")
        mark_every = max(1, int(math.ceil(len(x_plot) / 10)))
        axis.plot(
            x_plot,
            y_plot,
            label=history.label,
            color=history.color,
            linestyle=history.linestyle,
            linewidth=1.6,
            marker="o",
            markersize=2.8,
            markevery=mark_every,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )
        if metric == "valid_loss":
            best_index = int(np.argmin(y_plot))
            axis.scatter(
                x_plot[best_index],
                y_plot[best_index],
                s=18,
                color=history.color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )

    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(
        f"Cumulative estimated training compute ({FLOP_UNIT_LABEL})"
        if x_axis == "flops"
        else "Epoch"
    )
    axis.set_ylabel("Loss")
    axis.set_xscale("log" if x_log else "linear")
    axis.set_yscale("log" if y_log else "linear")
    if x_axis == "epoch" and not x_log:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="best", handlelength=2.2)
    return figure


def plot_model_loss_curves(
    histories: Sequence[RunHistory],
    *,
    x_axis: str = X_AXIS,
    x_log: bool = X_LOG,
    train_y_log: bool = TRAIN_Y_LOG,
    valid_y_log: bool = VALID_Y_LOG,
) -> Dict[str, Any]:
    """Create independent training- and validation-loss figures."""
    return {
        "train": plot_loss_curve(
            histories,
            "train_loss",
            "Training loss",
            x_axis=x_axis,
            x_log=x_log,
            y_log=train_y_log,
        ),
        "valid": plot_loss_curve(
            histories,
            "valid_loss",
            "Validation loss",
            x_axis=x_axis,
            x_log=x_log,
            y_log=valid_y_log,
        ),
    }


def write_source_data(histories: Sequence[RunHistory], output_prefix: Path) -> None:
    """Write the plotted values and a compact run summary."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    source_path = output_prefix.with_name(output_prefix.name + ".source_data.csv")
    summary_path = output_prefix.with_name(output_prefix.name + ".summary.csv")

    with source_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "model",
            "validation_dataset",
            "training_dataset",
            "epoch",
            "cumulative_training_flops",
            "cumulative_training_eflops",
            "train_loss",
            "valid_loss",
            "alpha",
            "profile_spearman",
            "scale_spearman",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for history in histories:
            for index, epoch in enumerate(history.epochs):
                cumulative = history.cumulative_flops[index]
                writer.writerow(
                    {
                        "model": history.label,
                        "validation_dataset": history.dataset,
                        "training_dataset": (
                            history.compute.training_dataset if history.compute else ""
                        ),
                        "epoch": int(epoch),
                        "cumulative_training_flops": cumulative,
                        "cumulative_training_eflops": cumulative / FLOP_UNIT,
                        "train_loss": history.train_loss[index],
                        "valid_loss": history.valid_loss[index],
                        "alpha": history.alpha[index],
                        "profile_spearman": history.profile_spearman[index],
                        "scale_spearman": history.scale_spearman[index],
                    }
                )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "model",
            "training_dataset",
            "n_transcripts",
            "total_length",
            "flops_per_epoch",
            "last_epoch",
            "best_validation_epoch",
            "best_validation_loss",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for history in histories:
            best_epoch, best_loss = history.best_validation()
            compute = history.compute
            writer.writerow(
                {
                    "model": history.label,
                    "training_dataset": compute.training_dataset if compute else "",
                    "n_transcripts": compute.n_transcripts if compute else "",
                    "total_length": compute.total_length if compute else "",
                    "flops_per_epoch": compute.flops_per_epoch if compute else "",
                    "last_epoch": int(history.epochs[-1]),
                    "best_validation_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                }
            )

    print(f"[LossCurve] Source data: {source_path}")
    print(f"[LossCurve] Summary: {summary_path}")


def save_figure(figure, output_prefix: Path) -> None:
    """Export one editable vector figure and raster previews."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(Path(f"{output_prefix}.svg"), bbox_inches="tight")
    figure.savefig(Path(f"{output_prefix}.pdf"), bbox_inches="tight")
    figure.savefig(Path(f"{output_prefix}.tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(Path(f"{output_prefix}.png"), dpi=300, bbox_inches="tight")
    print(f"[LossCurve] Figure prefix: {output_prefix}")


def plot_scaling_law_curves(
    models_config: Sequence[Dict[str, Any]],
    global_seq_len: int = 1024,
    save_path: Optional[str] = None,
):
    """Compatibility entry point retained for existing notebooks."""
    del global_seq_len
    histories = [load_run_history(config) for config in models_config]
    validate_comparison(histories, allow_mixed_datasets=ALLOW_MIXED_DATASETS)
    figures = plot_model_loss_curves(histories)
    if save_path is not None:
        prefix = Path(save_path).with_suffix("")
        for name, figure in figures.items():
            save_figure(figure, prefix.with_name(prefix.name + f".{name}"))
        write_source_data(histories, prefix)
    return figures, histories


def main() -> None:
    """Load configured runs and export independent loss figures."""
    enabled_runs = [config for config in MODEL_RUNS if config.get("enabled", True)]
    if len(enabled_runs) < 2:
        raise ValueError("Enable at least two entries in MODEL_RUNS")

    histories = [load_run_history(config) for config in enabled_runs]
    validate_comparison(histories, allow_mixed_datasets=ALLOW_MIXED_DATASETS)

    for history in histories:
        best_epoch, best_loss = history.best_validation()
        compute_text = ""
        if history.compute is not None:
            compute_text = (
                f", {history.compute.flops_per_epoch / FLOP_UNIT:.4f} "
                f"{FLOP_UNIT_LABEL}/epoch"
            )
        print(
            f"[LossCurve] {history.label}: best validation={best_loss:.6f} "
            f"at epoch {best_epoch}{compute_text}"
        )

    figures = plot_model_loss_curves(histories)
    for name, figure in figures.items():
        prefix = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + f".{name}")
        save_figure(figure, prefix)
    write_source_data(histories, OUTPUT_PREFIX)

    if SHOW_FIGURE:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    main()

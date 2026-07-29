#!/usr/bin/env python3
"""Plot configurable Trainer metrics against epoch for multiple TRACE runs.

Edit ``MODEL_RUNS`` and ``PLOT_METRICS`` before running this file on the
server. Each enabled metric is exported as an independent figure.
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
OUTPUT_PREFIX = (
    PROJECT_ROOT.parent / "results/ablation/loss_curves/model_metric_comparison"
)

# Validation metrics are directly comparable only when all runs use the same
# validation dataset and loss definition.
COMPARISON_DATASET = "human_5c_6k_depth0.1_cov0.1_rpm1"
LOSS_DEFINITION = "micro + 2.0*macro + 0.2*ranking"
ALLOW_MIXED_DATASETS = False

X_LOG = False
SHOW_FIGURE = False

SUPPORTED_METRICS = (
    "train_loss",
    "valid_loss",
    "profile_spearman",
    "scale_spearman",
    "cds_mean_mae",
    "calibration_slope",
    "calibration_intercept",
)

# Add, remove, or disable entries here to choose the y-axis of each figure.
# ``key`` must be one of the entries in ``SUPPORTED_METRICS``.
PLOT_METRICS = [
    {
        "key": "train_loss",
        "title": "Training loss",
        "y_label": "Loss",
        "filename": "train_loss",
        "y_log": True,
        "best": "min",
        "enabled": True,
    },
    {
        "key": "valid_loss",
        "title": "Validation loss",
        "y_label": "Loss",
        "filename": "valid_loss",
        "y_log": True,
        "best": "min",
        "enabled": True,
    },
    {
        "key": "profile_spearman",
        "title": "RNA profile Spearman",
        "y_label": "Spearman correlation",
        "filename": "profile_spearman",
        "y_log": False,
        "best": "max",
        "enabled": True,
    },
    {
        "key": "scale_spearman",
        "title": "CDS-mean scale Spearman",
        "y_label": "Spearman correlation",
        "filename": "cds_mean_scale_spearman",
        "y_log": False,
        "best": "max",
        "enabled": True,
    },
    {
        "key": "cds_mean_mae",
        "title": "CDS-mean MAE",
        "y_label": "Mean absolute error",
        "filename": "cds_mean_mae",
        "y_log": True,
        "best": "min",
        "enabled": True,
    },
]

# ``glob`` is resolved inside LOG_DIR and selects the newest matching history.
# Replace it with an exact ``path`` for final figure reproducibility.
MODEL_RUNS = [
    {
        "label": "TRACE Zero (5c)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*zero*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#7A7A7A",
        "linestyle": "--",
        "enabled": True,
    },
    {
        "label": "TRACE Real (5c)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*real*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#78A9CF",
        "linestyle": "-.",
        "enabled": True,
    },
    {
        "label": "TRACE Mask+Interp. (5c)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_5c*exp_aug*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#166A9A",
        "linestyle": "-",
        "enabled": True,
    },
    {
        "label": "LN Transformer (5c)",
        "glob": "base_model_LN*hs_5c*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#C28548",
        "linestyle": "-",
        "enabled": True,
    },
    {
        "label": "Conv model (5c)",
        "glob": "base_model_conv*hs_5c*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#5F9272",
        "linestyle": "-",
        "enabled": False,
    },
    {
        "label": "TRACE Mask+Interp. (22c)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_22c*exp_aug*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#9A6FB0",
        "linestyle": "-.",
        "enabled": True,
    },
    {
        "label": "TRACE Mask+Interp. (40c)",
        "glob": "base_model_384d_16h_12l_64env_16ad_bs*hs_40c*exp_aug*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
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
    rf"CDS-mean\s+scale\s+Spearman=({FLOAT_PATTERN})"
    rf"(?:.*?CDS-mean\s+MAE=({FLOAT_PATTERN}))?",
    flags=re.IGNORECASE,
)
CALIBRATION_PATTERN = re.compile(
    rf"Epoch\s+(\d+)\s+validation\s+metrics:.*?"
    rf"calibration\s+target=({FLOAT_PATTERN})\s*\+\s*"
    rf"({FLOAT_PATTERN})\s*\*\s*prediction",
    flags=re.IGNORECASE,
)


@dataclass
class RunHistory:
    """Clean epoch-level metrics for one model run."""

    label: str
    dataset: str
    loss_definition: str
    color: str
    linestyle: str
    epochs: np.ndarray
    train_loss: np.ndarray
    valid_loss: np.ndarray
    profile_spearman: np.ndarray
    scale_spearman: np.ndarray
    cds_mean_mae: np.ndarray
    calibration_slope: np.ndarray
    calibration_intercept: np.ndarray
    alpha: np.ndarray
    duplicate_epochs: int = 0

    def best_metric(self, metric: str, direction: str) -> tuple[int, float]:
        """Return the epoch and best finite value for a metric."""
        values = np.asarray(getattr(self, metric), dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError(f"Run {self.label!r} has no finite {metric} values")
        indices = np.flatnonzero(finite)
        selector = np.argmin if direction == "min" else np.argmax
        best_index = indices[selector(values[finite])]
        return int(self.epochs[best_index]), float(values[best_index])


def _to_float(value: Any) -> float:
    """Convert common scalar representations to a finite float or NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple)):
        return _to_float(value[0]) if len(value) == 1 else float("nan")
    if isinstance(value, np.ndarray):
        return _to_float(value.reshape(-1)[0]) if value.size == 1 else float("nan")
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


def _empty_record(epoch: int) -> Dict[str, float]:
    """Create one record with all supported metrics initialized to NaN."""
    return {
        "epoch": epoch,
        "train_loss": float("nan"),
        "valid_loss": float("nan"),
        "profile_spearman": float("nan"),
        "scale_spearman": float("nan"),
        "cds_mean_mae": float("nan"),
        "calibration_slope": float("nan"),
        "calibration_intercept": float("nan"),
        "alpha": float("nan"),
    }


def _extract_json_entries(payload: Any) -> List[Dict[str, Any]]:
    """Normalize supported JSON history containers to a list of records."""
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = next(
            (
                payload[key]
                for key in ("training_epoch_data", "epoch_data", "history", "data")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if entries is None and all(isinstance(value, dict) for value in payload.values()):
            entries = list(payload.values())
    else:
        entries = None
    if not isinstance(entries, list):
        raise ValueError("JSON does not contain an epoch-history list")
    if not all(isinstance(entry, dict) for entry in entries):
        raise TypeError("Every epoch entry must be a dictionary")
    return list(entries)


def _record_from_mapping(entry: Mapping[str, Any]) -> Dict[str, float]:
    """Normalize one current or legacy Trainer epoch record."""
    epoch = _first_value(entry, ("epoch", "epoch_num", "epoch_index"))
    if not math.isfinite(epoch):
        raise ValueError(f"Epoch entry has no valid epoch field: {entry}")
    record = _empty_record(int(epoch))
    record.update(
        {
            "train_loss": _first_value(
                entry, ("train_loss", "training_loss", "mean_train_loss")
            ),
            "valid_loss": _first_value(
                entry,
                ("valid_loss", "val_loss", "validation_loss", "mean_valid_loss"),
            ),
            "profile_spearman": _first_value(
                entry, ("profile_spearman", "mean_profile_spearman")
            ),
            "scale_spearman": _first_value(
                entry, ("scale_spearman", "cds_mean_scale_spearman")
            ),
            "cds_mean_mae": _first_value(
                entry, ("cds_mean_mae", "mean_cds_mae", "scale_mae")
            ),
            "calibration_slope": _first_value(
                entry, ("calibration_slope", "scale_calibration_slope")
            ),
            "calibration_intercept": _first_value(
                entry, ("calibration_intercept", "scale_calibration_intercept")
            ),
            "alpha": _first_value(entry, ("alpha", "macro_loss_weight")),
        }
    )
    return record


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
    """Parse current Trainer console logs, including validation MAE."""
    text = path.read_text(encoding="utf-8", errors="replace")
    records: Dict[int, Dict[str, float]] = {}
    seen_fields = set()
    duplicates = 0

    def get_record(epoch: int) -> Dict[str, float]:
        return records.setdefault(epoch, _empty_record(epoch))

    for match in EPOCH_LOSS_PATTERN.finditer(text):
        epoch = int(match.group(1))
        metric = "train_loss" if match.group(2).lower() == "training" else "valid_loss"
        duplicates += int((epoch, metric) in seen_fields)
        seen_fields.add((epoch, metric))
        get_record(epoch)[metric] = float(match.group(3))

    for match in VALIDATION_METRICS_PATTERN.finditer(text):
        epoch = int(match.group(1))
        record = get_record(epoch)
        record["profile_spearman"] = float(match.group(2))
        record["scale_spearman"] = float(match.group(3))
        if match.group(4) is not None:
            record["cds_mean_mae"] = float(match.group(4))

    for match in CALIBRATION_PATTERN.finditer(text):
        record = get_record(int(match.group(1)))
        record["calibration_intercept"] = float(match.group(2))
        record["calibration_slope"] = float(match.group(3))

    if not records:
        raise ValueError(f"No epoch-level metrics were recognized in {path}")
    return [records[epoch] for epoch in sorted(records)], duplicates


def _resolve_run_paths(config: Mapping[str, Any], log_dir: Path) -> List[Path]:
    """Resolve exact path(s) or the newest file matching one glob."""
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


def load_run_history(
    config: Mapping[str, Any], log_dir: Path = LOG_DIR
) -> RunHistory:
    """Load, merge, and sort one configured training history."""
    merged: Dict[int, Dict[str, float]] = {}
    duplicate_epochs = 0

    if "loss_data" in config:
        groups = [([_record_from_mapping(item) for item in config["loss_data"]], 0)]
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

    def values(name: str) -> np.ndarray:
        return np.asarray([record[name] for record in ordered], dtype=float)

    return RunHistory(
        label=str(config.get("label", config.get("method", "Unknown"))),
        dataset=str(config.get("dataset", "")),
        loss_definition=str(config.get("loss_definition", "")),
        color=str(config.get("color", "#333333")),
        linestyle=str(config.get("linestyle", "-")),
        epochs=np.asarray([record["epoch"] for record in ordered], dtype=int),
        train_loss=values("train_loss"),
        valid_loss=values("valid_loss"),
        profile_spearman=values("profile_spearman"),
        scale_spearman=values("scale_spearman"),
        cds_mean_mae=values("cds_mean_mae"),
        calibration_slope=values("calibration_slope"),
        calibration_intercept=values("calibration_intercept"),
        alpha=values("alpha"),
        duplicate_epochs=duplicate_epochs,
    )


def validate_comparison(
    histories: Sequence[RunHistory], allow_mixed_datasets: bool = False
) -> None:
    """Check that validation metrics use comparable datasets and loss definitions."""
    if len(histories) < 2:
        raise ValueError("At least two model histories are required")
    datasets = {history.dataset for history in histories if history.dataset}
    if len(datasets) > 1 and not allow_mixed_datasets:
        raise ValueError(
            "Validation metrics use different datasets: " + ", ".join(sorted(datasets))
        )
    definitions = {
        history.loss_definition for history in histories if history.loss_definition
    }
    if len(definitions) > 1:
        raise ValueError("Runs use different validation-loss definitions")


def _metric_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize one configurable y-axis entry."""
    key = str(config["key"])
    if key not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric key: {key}")
    direction = str(config.get("best", "min")).lower()
    if direction not in {"min", "max"}:
        raise ValueError("Metric best must be 'min' or 'max'")
    return {
        "key": key,
        "title": str(config.get("title", key)),
        "y_label": str(config.get("y_label", key)),
        "filename": str(config.get("filename", key)),
        "y_log": bool(config.get("y_log", False)),
        "best": direction,
    }


def plot_metric_curve(
    histories: Sequence[RunHistory], metric_config: Mapping[str, Any], *, x_log: bool = X_LOG
):
    """Plot one configured metric against epoch in an independent figure."""
    metric = _metric_config(metric_config)
    figure, axis = plt.subplots(figsize=(3.5039, 2.8346))
    for history in histories:
        y_values = np.asarray(getattr(history, metric["key"]), dtype=float)
        finite = np.isfinite(y_values)
        if not finite.any():
            continue
        x_values = history.epochs[finite].astype(float)
        y_values = y_values[finite]
        if x_log and np.any(x_values <= 0):
            raise ValueError(f"{history.label} has non-positive epochs for log scale")
        if metric["y_log"] and np.any(y_values <= 0):
            raise ValueError(
                f"{history.label} has non-positive {metric['key']} values for log scale"
            )
        mark_every = max(1, int(math.ceil(len(x_values) / 10)))
        axis.plot(
            x_values,
            y_values,
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
        best_index = int(
            np.argmin(y_values) if metric["best"] == "min" else np.argmax(y_values)
        )
        axis.scatter(
            x_values[best_index],
            y_values[best_index],
            s=18,
            color=history.color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )

    axis.set_title(metric["title"], loc="left", fontweight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric["y_label"])
    axis.set_xscale("log" if x_log else "linear")
    axis.set_yscale("log" if metric["y_log"] else "linear")
    if not x_log:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        columns = 2 if len(handles) > 1 else 1
        axis.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=columns,
            columnspacing=0.9,
            handlelength=1.8,
        )
        figure.subplots_adjust(bottom=0.30 if columns == 2 else 0.23)
    return figure


def plot_model_metric_curves(
    histories: Sequence[RunHistory],
    metric_configs: Sequence[Mapping[str, Any]] = PLOT_METRICS,
    *,
    x_log: bool = X_LOG,
) -> Dict[str, Any]:
    """Create one figure for every enabled y-axis metric."""
    figures = {}
    for raw_config in metric_configs:
        if not raw_config.get("enabled", True):
            continue
        metric = _metric_config(raw_config)
        figures[metric["filename"]] = plot_metric_curve(
            histories, metric, x_log=x_log
        )
    if not figures:
        raise ValueError("Enable at least one entry in PLOT_METRICS")
    return figures


def plot_model_loss_curves(
    histories: Sequence[RunHistory], *, x_log: bool = X_LOG
) -> Dict[str, Any]:
    """Compatibility alias for notebook code using the historical function name."""
    return plot_model_metric_curves(histories, x_log=x_log)


def write_source_data(histories: Sequence[RunHistory], output_prefix: Path) -> None:
    """Write all supported epoch metrics and a compact run summary."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    source_path = output_prefix.with_name(output_prefix.name + ".source_data.csv")
    summary_path = output_prefix.with_name(output_prefix.name + ".summary.csv")

    metric_keys = list(SUPPORTED_METRICS)
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "validation_dataset", "loss_definition", "epoch", *metric_keys, "alpha"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for history in histories:
            for index, epoch in enumerate(history.epochs):
                writer.writerow(
                    {
                        "model": history.label,
                        "validation_dataset": history.dataset,
                        "loss_definition": history.loss_definition,
                        "epoch": int(epoch),
                        **{key: getattr(history, key)[index] for key in metric_keys},
                        "alpha": history.alpha[index],
                    }
                )

    summary_fields = ["model", "validation_dataset", "epochs_recorded"]
    for raw_config in PLOT_METRICS:
        if not raw_config.get("enabled", True):
            continue
        metric = _metric_config(raw_config)
        summary_fields.extend(
            [f"best_{metric['filename']}_epoch", f"best_{metric['filename']}"]
        )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for history in histories:
            row: Dict[str, Any] = {
                "model": history.label,
                "validation_dataset": history.dataset,
                "epochs_recorded": len(history.epochs),
            }
            for raw_config in PLOT_METRICS:
                if not raw_config.get("enabled", True):
                    continue
                metric = _metric_config(raw_config)
                try:
                    best_epoch, best_value = history.best_metric(
                        metric["key"], metric["best"]
                    )
                except ValueError:
                    best_epoch, best_value = "", float("nan")
                row[f"best_{metric['filename']}_epoch"] = best_epoch
                row[f"best_{metric['filename']}"] = best_value
            writer.writerow(row)

    print(f"[MetricCurve] Source data: {source_path}")
    print(f"[MetricCurve] Summary: {summary_path}")


def save_figure(figure, output_prefix: Path) -> None:
    """Export one editable vector figure and raster previews."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(Path(f"{output_prefix}.svg"), bbox_inches="tight")
    figure.savefig(Path(f"{output_prefix}.pdf"), bbox_inches="tight")
    figure.savefig(Path(f"{output_prefix}.tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(Path(f"{output_prefix}.png"), dpi=300, bbox_inches="tight")
    print(f"[MetricCurve] Figure prefix: {output_prefix}")


def plot_scaling_law_curves(
    models_config: Sequence[Mapping[str, Any]],
    save_path: Optional[str] = None,
):
    """Load configured histories and return their epoch-metric figures."""
    histories = [load_run_history(config) for config in models_config]
    validate_comparison(histories, allow_mixed_datasets=ALLOW_MIXED_DATASETS)
    figures = plot_model_metric_curves(histories)
    if save_path is not None:
        prefix = Path(save_path).with_suffix("")
        for name, figure in figures.items():
            save_figure(figure, prefix.with_name(prefix.name + f".{name}"))
        write_source_data(histories, prefix)
    return figures, histories


def main() -> None:
    """Load configured runs and export every enabled epoch-metric figure."""
    enabled_runs = [config for config in MODEL_RUNS if config.get("enabled", True)]
    if len(enabled_runs) < 2:
        raise ValueError("Enable at least two entries in MODEL_RUNS")

    histories = [load_run_history(config) for config in enabled_runs]
    validate_comparison(histories, allow_mixed_datasets=ALLOW_MIXED_DATASETS)
    figures = plot_model_metric_curves(histories)
    for name, figure in figures.items():
        save_figure(figure, OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + f".{name}"))
    write_source_data(histories, OUTPUT_PREFIX)

    if SHOW_FIGURE:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    main()

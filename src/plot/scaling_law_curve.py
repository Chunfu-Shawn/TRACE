#!/usr/bin/env python3
"""Compare epoch-level training and validation losses across TRACE ablations.

The preferred inputs are Trainer ``*.epoch_data.json`` files. Plain-text logs
containing the current ``Epoch ... mean loss: tensor(...)`` messages are also
supported. Edit ``MODEL_RUNS`` and run this file directly on the server.
"""

from __future__ import annotations

import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT.parent / "log/train"
OUTPUT_PREFIX = (
    PROJECT_ROOT.parent / "results/ablation/loss_curves/model_loss_comparison"
)

# All runs overlaid in one validation-loss panel should use this same dataset.
COMPARISON_DATASET = "human_5c_6k_depth0.1_cov0.1_rpm1"
LOSS_DEFINITION = "micro + 2.0*macro + 0.2*ranking"
ALLOW_MIXED_DATASETS = False
SHOW_TRAINING_PANEL = True
Y_SCALE = "linear"  # Use "log" only when multiplicative differences are intended.
SHOW_FIGURE = False

# ``glob`` is resolved inside LOG_DIR and selects the newest match. Replace it
# with an exact ``path`` for final figure reproducibility.
MODEL_RUNS = [
    {
        "label": "TRACE-Zero",
        "glob": "*hs_5c*zero*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#7A7A7A",
        "linestyle": "--",
    },
    {
        "label": "TRACE-Real",
        "glob": "*hs_5c*real*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#78A9CF",
        "linestyle": "-.",
    },
    {
        "label": "TRACE-Mask+Interpolation",
        "glob": "*hs_5c*exp_aug*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#166A9A",
        "linestyle": "-",
    },
    {
        "label": "LayerNorm Transformer",
        "glob": "*base_model_LN*hs_5c*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#C28548",
        "linestyle": "-",
        "enabled": False,
    },
    {
        "label": "Convolutional model",
        "glob": "*base_model_conv*hs_5c*.epoch_data.json",
        "dataset": COMPARISON_DATASET,
        "loss_definition": LOSS_DEFINITION,
        "color": "#5F9272",
        "linestyle": "-",
        "enabled": False,
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

    def best_validation(self) -> tuple[int, float]:
        """Return the epoch and value of the minimum finite validation loss."""
        finite = np.isfinite(self.valid_loss)
        if not finite.any():
            raise ValueError(f"Run {self.label!r} has no finite validation losses")
        finite_indices = np.flatnonzero(finite)
        best_index = finite_indices[np.argmin(self.valid_loss[finite])]
        return int(self.epochs[best_index]), float(self.valid_loss[best_index])


def _to_float(value: Any) -> float:
    """Convert scalar, singleton-list, NumPy, or tensor-like text to float."""
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


def _first_value(record: Dict[str, Any], names: Sequence[str]) -> float:
    """Return the first present metric alias as a finite float or NaN."""
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

    invalid = [index for index, entry in enumerate(entries) if not isinstance(entry, dict)]
    if invalid:
        raise TypeError(f"Non-dictionary epoch entries at indices: {invalid[:10]}")
    return list(entries)


def _record_from_mapping(entry: Dict[str, Any]) -> Dict[str, float]:
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
            entry,
            ("valid_loss", "val_loss", "validation_loss", "mean_valid_loss"),
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
    """Parse a Trainer epoch-data JSON file and retain the last duplicate epoch."""
    with path.open(encoding="utf-8") as handle:
        entries = _extract_json_entries(json.load(handle))

    records: Dict[int, Dict[str, float]] = {}
    duplicates = 0
    for entry in entries:
        record = _record_from_mapping(entry)
        epoch = int(record["epoch"])
        if epoch in records:
            duplicates += 1
        records[epoch] = record
    return [records[epoch] for epoch in sorted(records)], duplicates


def parse_text_log(path: Path) -> tuple[List[Dict[str, float]], int]:
    """Parse current Trainer console logs and retain the last duplicate value."""
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
        phase = match.group(2).lower()
        field = "train_loss" if phase == "training" else "valid_loss"
        field_key = (epoch, field)
        if field_key in seen_fields:
            duplicates += 1
        seen_fields.add(field_key)
        get_record(epoch)[field] = float(match.group(3))

    for match in VALIDATION_METRICS_PATTERN.finditer(text):
        epoch = int(match.group(1))
        get_record(epoch)["profile_spearman"] = float(match.group(2))
        get_record(epoch)["scale_spearman"] = float(match.group(3))

    if not records:
        raise ValueError(
            f"No epoch-level losses were recognized in text log: {path}"
        )
    return [records[epoch] for epoch in sorted(records)], duplicates


def _resolve_run_paths(config: Dict[str, Any], log_dir: Path) -> List[Path]:
    """Resolve exact path(s) or the newest file matching one glob pattern."""
    if "paths" in config:
        raw_paths = list(config["paths"])
    elif "path" in config:
        raw_paths = [config["path"]]
    elif "loss_path" in config:
        raw_paths = [config["loss_path"]]
    elif "glob" in config:
        matches = sorted(
            log_dir.glob(str(config["glob"])),
            key=lambda path: path.stat().st_mtime,
        )
        if not matches:
            raise FileNotFoundError(
                f"No log matched {config['glob']!r} inside {log_dir}"
            )
        if len(matches) > 1:
            label = config.get("label", config.get("method", "Unknown"))
            print(
                f"[LossCurve] {label}: {len(matches)} files matched; "
                f"using newest {matches[-1].name}"
            )
        return [matches[-1]]
    else:
        raise ValueError(
            f"Run {config.get('label', '<unnamed>')!r} needs path, paths, or glob"
        )

    paths = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = log_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Training log not found: {path}")
        paths.append(path)
    return paths


def load_run_history(config: Dict[str, Any], log_dir: Path = LOG_DIR) -> RunHistory:
    """Load, merge, validate, and sort one configured model history."""
    merged: Dict[int, Dict[str, float]] = {}
    duplicate_epochs = 0

    if "loss_data" in config:
        entries = _extract_json_entries(config["loss_data"])
        record_groups = [([_record_from_mapping(entry) for entry in entries], 0)]
    else:
        paths = _resolve_run_paths(config, log_dir)
        record_groups = []
        for path in paths:
            if path.suffix.lower() == ".json":
                record_groups.append(parse_epoch_json(path))
            else:
                record_groups.append(parse_text_log(path))

    for records, duplicates in record_groups:
        duplicate_epochs += duplicates
        for record in records:
            epoch = int(record["epoch"])
            if epoch in merged:
                duplicate_epochs += 1
            merged[epoch] = record

    ordered = [merged[epoch] for epoch in sorted(merged)]
    if not ordered:
        raise ValueError(f"Run {config['label']!r} has no epoch records")

    epochs = np.asarray([record["epoch"] for record in ordered], dtype=int)
    train_loss = np.asarray([record["train_loss"] for record in ordered], dtype=float)
    valid_loss = np.asarray([record["valid_loss"] for record in ordered], dtype=float)
    if not np.isfinite(valid_loss).any():
        raise ValueError(f"Run {config['label']!r} has no finite validation losses")

    return RunHistory(
        label=str(config.get("label", config.get("method", "Unknown"))),
        dataset=str(config.get("dataset", "")),
        loss_definition=str(config.get("loss_definition", "")),
        color=str(config.get("color", "#333333")),
        linestyle=str(config.get("linestyle", "-")),
        epochs=epochs,
        train_loss=train_loss,
        valid_loss=valid_loss,
        alpha=np.asarray([record["alpha"] for record in ordered], dtype=float),
        profile_spearman=np.asarray(
            [record["profile_spearman"] for record in ordered], dtype=float
        ),
        scale_spearman=np.asarray(
            [record["scale_spearman"] for record in ordered], dtype=float
        ),
        duplicate_epochs=duplicate_epochs,
    )


def validate_comparison(
    histories: Sequence[RunHistory], allow_mixed_datasets: bool = False
) -> None:
    """Reject ambiguous overlays using differently labeled validation datasets."""
    if len(histories) < 2:
        raise ValueError("At least two model histories are required for comparison")
    datasets = {history.dataset for history in histories if history.dataset}
    if len(datasets) > 1 and not allow_mixed_datasets:
        raise ValueError(
            "Validation losses from different datasets cannot be directly overlaid: "
            + ", ".join(sorted(datasets))
        )
    loss_definitions = {
        history.loss_definition for history in histories if history.loss_definition
    }
    if len(loss_definitions) > 1:
        raise ValueError(
            "Runs use different validation-loss definitions and cannot be overlaid: "
            + "; ".join(sorted(loss_definitions))
        )


def _plot_one_metric(
    axis,
    histories: Sequence[RunHistory],
    metric_name: str,
    title: str,
    mark_best: bool,
) -> None:
    """Draw one unsmoothed epoch-level loss panel."""
    for history in histories:
        values = getattr(history, metric_name)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        epochs = history.epochs[finite]
        values = values[finite]
        mark_every = max(1, int(math.ceil(len(epochs) / 10)))
        axis.plot(
            epochs,
            values,
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
        if mark_best:
            best_position = int(np.argmin(values))
            axis.scatter(
                epochs[best_position],
                values[best_position],
                s=18,
                color=history.color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )

    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_yscale(Y_SCALE)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    axis.grid(axis="x", visible=False)


def plot_model_loss_curves(
    histories: Sequence[RunHistory],
    show_training_panel: bool = SHOW_TRAINING_PANEL,
):
    """Create a training/validation loss figure for multiple model runs."""
    has_training = any(np.isfinite(history.train_loss).any() for history in histories)
    show_training_panel = bool(show_training_panel and has_training)

    if show_training_panel:
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(7.2047, 3.0709),  # 183 x 78 mm at final size.
            sharey=True,
            gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.30},
        )
        _plot_one_metric(axes[0], histories, "train_loss", "a  Training", False)
        _plot_one_metric(axes[1], histories, "valid_loss", "b  Validation", True)
        legend_axis = axes[1]
    else:
        figure, axis = plt.subplots(figsize=(3.5039, 2.8346))  # 89 x 72 mm.
        axes = np.asarray([axis])
        _plot_one_metric(axis, histories, "valid_loss", "Validation", True)
        legend_axis = axis

    legend_axis.legend(
        loc="best",
        handlelength=2.2,
        borderaxespad=0.4,
        labelspacing=0.45,
    )
    figure.align_ylabels(axes)
    return figure


def write_source_data(histories: Sequence[RunHistory], output_prefix: Path) -> None:
    """Export plotted epoch data and best-loss summaries as CSV files."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    source_path = output_prefix.with_name(output_prefix.name + ".source_data.csv")
    summary_path = output_prefix.with_name(output_prefix.name + ".summary.csv")

    with source_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model",
            "dataset",
            "loss_definition",
            "epoch",
            "train_loss",
            "valid_loss",
            "alpha",
            "profile_spearman",
            "scale_spearman",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for history in histories:
            for index, epoch in enumerate(history.epochs):
                writer.writerow(
                    {
                        "model": history.label,
                        "dataset": history.dataset,
                        "loss_definition": history.loss_definition,
                        "epoch": int(epoch),
                        "train_loss": history.train_loss[index],
                        "valid_loss": history.valid_loss[index],
                        "alpha": history.alpha[index],
                        "profile_spearman": history.profile_spearman[index],
                        "scale_spearman": history.scale_spearman[index],
                    }
                )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model",
            "dataset",
            "loss_definition",
            "epochs_recorded",
            "last_epoch",
            "best_validation_epoch",
            "best_validation_loss",
            "final_validation_loss",
            "duplicate_epochs_replaced",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for history in histories:
            best_epoch, best_loss = history.best_validation()
            finite_validation = history.valid_loss[np.isfinite(history.valid_loss)]
            writer.writerow(
                {
                    "model": history.label,
                    "dataset": history.dataset,
                    "loss_definition": history.loss_definition,
                    "epochs_recorded": len(history.epochs),
                    "last_epoch": int(history.epochs[-1]),
                    "best_validation_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "final_validation_loss": float(finite_validation[-1]),
                    "duplicate_epochs_replaced": history.duplicate_epochs,
                }
            )

    print(f"[LossCurve] Source data: {source_path}")
    print(f"[LossCurve] Summary: {summary_path}")


def save_figure(figure, output_prefix: Path) -> None:
    """Export editable vector figures and high-resolution raster previews."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[LossCurve] Figure prefix: {output_prefix}")


def plot_scaling_law_curves(
    models_config: Sequence[Dict[str, Any]],
    global_seq_len: int = 1024,
    save_path: Optional[str] = None,
):
    """Compatibility wrapper for the historical notebook entry point.

    The current function plots loss against epoch rather than estimated FLOPs.
    ``global_seq_len`` is retained only to avoid breaking previous calls.
    """
    del global_seq_len
    warnings.warn(
        "plot_scaling_law_curves now compares epoch-level losses. "
        "Use plot_model_loss_curves for new code.",
        DeprecationWarning,
        stacklevel=2,
    )
    histories = [load_run_history(config) for config in models_config]
    validate_comparison(histories, allow_mixed_datasets=ALLOW_MIXED_DATASETS)
    figure = plot_model_loss_curves(histories)
    if save_path is not None:
        output_prefix = Path(save_path).with_suffix("")
        save_figure(figure, output_prefix)
        write_source_data(histories, output_prefix)
    return figure, histories


def main() -> None:
    """Load configured ablation runs and export the comparison figure."""
    enabled_runs = [config for config in MODEL_RUNS if config.get("enabled", True)]
    if len(enabled_runs) < 2:
        raise ValueError("Enable at least two entries in MODEL_RUNS")

    histories = [load_run_history(config) for config in enabled_runs]
    validate_comparison(histories, allow_mixed_datasets=ALLOW_MIXED_DATASETS)

    for history in histories:
        best_epoch, best_loss = history.best_validation()
        print(
            f"[LossCurve] {history.label}: epochs={len(history.epochs)}, "
            f"best validation={best_loss:.6f} at epoch {best_epoch}, "
            f"duplicates replaced={history.duplicate_epochs}"
        )

    figure = plot_model_loss_curves(histories)
    save_figure(figure, OUTPUT_PREFIX)
    write_source_data(histories, OUTPUT_PREFIX)
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()

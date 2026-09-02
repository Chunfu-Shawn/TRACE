#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import RNA
from scipy import stats


BASE_DIR = Path(
    "/Users/chunfu/Desktop/BGM_lab/translation_model/TE_optimization"
)
INPUT_FASTA = BASE_DIR / "Fluc_order_list_final_cDNA.fasta"
EXPRESSION_XLSX = BASE_DIR / "Fluc (6h-12h-24h-48h).xlsx"

# This file is both the ViennaRNA result table and the persistent prediction cache.
MFE_RESULT = BASE_DIR / "MFE.result.csv"
INTEGRATED_RESULT = BASE_DIR / "MFE_Fluc_integrated_results.csv"
CORRELATION_RESULT = BASE_DIR / "MFE_Fluc_correlation_statistics.csv"
MFE_BAR_PNG = BASE_DIR / "Fluc_mRNA_MFE_barplot.png"
MFE_BAR_PDF = BASE_DIR / "Fluc_mRNA_MFE_barplot.pdf"
CORRELATION_PNG = BASE_DIR / "MFE_Fluc_correlation.png"
CORRELATION_PDF = BASE_DIR / "MFE_Fluc_correlation.pdf"

TIME_POINTS = (6, 12, 24, 48)

GROUP_COLORS = {
    "BNT162b2": "#4C78A8",
    "TRACE": "#E45756",
    "ribodecode": "#59A14F",
    "gemorna": "#F2CF5B",
    "lineardesign": "#B279A2",
}

# The fluorescence workbook and FASTA use different spellings for this construct.
EXPRESSION_TO_FASTA_ID = {
    "gemorna_fullen": "gemorna_fulllen",
}

MFE_FIELDS = [
    "id",
    "group",
    "sequence_sha256",
    "length_nt",
    "mfe_kcal_mol",
    "mfe_per_100nt",
    "structure",
]


def read_fasta(fasta_path: Path) -> dict[str, str]:
    """Read FASTA records into an insertion-ordered dictionary."""
    sequences: dict[str, str] = {}
    current_id: str | None = None
    sequence_parts: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(sequence_parts)
                current_id = line[1:].split()[0]
                if current_id in sequences:
                    raise ValueError(f"Duplicate FASTA ID: {current_id}")
                sequence_parts = []
            else:
                if current_id is None:
                    raise ValueError("Sequence found before the first FASTA header.")
                sequence_parts.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(sequence_parts)

    if not sequences:
        raise ValueError(f"No FASTA records found in {fasta_path}")

    return sequences


def convert_cdna_to_rna(sequence: str) -> str:
    """Convert cDNA to RNA and validate the sequence alphabet."""
    rna_sequence = "".join(sequence.split()).upper().replace("T", "U")
    invalid_characters = set(rna_sequence) - set("ACGUN")
    if invalid_characters:
        invalid_text = ", ".join(sorted(invalid_characters))
        raise ValueError(f"Invalid RNA characters detected: {invalid_text}")
    return rna_sequence


def sequence_digest(rna_sequence: str) -> str:
    """Return a digest used to detect changed input sequences."""
    return hashlib.sha256(rna_sequence.encode("ascii")).hexdigest()


def assign_group(sequence_id: str) -> str:
    """Assign a construct to its sequence-design group."""
    normalized_id = sequence_id.lower()
    if normalized_id.startswith("bnt162b2"):
        return "BNT162b2"
    if normalized_id.startswith("trace"):
        return "TRACE"
    if normalized_id.startswith("ribodecode"):
        return "ribodecode"
    if normalized_id.startswith(("gemorna", "gemoran")):
        return "gemorna"
    if normalized_id.startswith("lineardesign"):
        return "lineardesign"
    raise ValueError(f"No group rule defined for sequence ID: {sequence_id}")


def load_mfe_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    """Load valid cached ViennaRNA predictions by sequence ID."""
    if not cache_path.exists():
        return {}

    cache: dict[str, dict[str, str]] = {}
    with cache_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "sequence_sha256", "mfe_kcal_mol", "structure"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            print(f"Ignoring incompatible cache format: {cache_path}")
            return {}

        for row in reader:
            try:
                mfe = float(row["mfe_kcal_mol"])
            except (TypeError, ValueError):
                continue
            if row.get("id") and row.get("sequence_sha256") and math.isfinite(mfe):
                cache[row["id"]] = row

    return cache


def write_mfe_cache(records: list[dict[str, object]], cache_path: Path) -> None:
    """Atomically update the persistent MFE result cache."""
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MFE_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(cache_path)


def calculate_mfe_with_cache(
    sequences: dict[str, str],
    cache_path: Path,
) -> pd.DataFrame:
    """Predict only missing or changed sequences and reuse valid cached results."""
    cache = load_mfe_cache(cache_path)
    records: list[dict[str, object]] = []
    predicted_count = 0
    reused_count = 0

    for sequence_id, cdna_sequence in sequences.items():
        rna_sequence = convert_cdna_to_rna(cdna_sequence)
        digest = sequence_digest(rna_sequence)
        cached = cache.get(sequence_id)

        if cached and cached.get("sequence_sha256") == digest:
            record = {
                "id": sequence_id,
                "group": assign_group(sequence_id),
                "sequence_sha256": digest,
                "length_nt": int(cached.get("length_nt") or len(rna_sequence)),
                "mfe_kcal_mol": float(cached["mfe_kcal_mol"]),
                "mfe_per_100nt": float(cached.get("mfe_per_100nt") or 0.0),
                "structure": cached["structure"],
            }
            if record["mfe_per_100nt"] == 0.0:
                record["mfe_per_100nt"] = (
                    float(record["mfe_kcal_mol"]) / len(rna_sequence) * 100
                )
            reused_count += 1
            print(f"Cache hit, skipping ViennaRNA prediction: {sequence_id}")
        else:
            fold_compound = RNA.fold_compound(rna_sequence)
            structure, mfe = fold_compound.mfe()
            record = {
                "id": sequence_id,
                "group": assign_group(sequence_id),
                "sequence_sha256": digest,
                "length_nt": len(rna_sequence),
                "mfe_kcal_mol": float(mfe),
                "mfe_per_100nt": float(mfe) / len(rna_sequence) * 100,
                "structure": structure,
            }
            predicted_count += 1
            print(f"ViennaRNA prediction completed: {sequence_id}")

        records.append(record)

        # Save after each sequence so an interrupted run can resume safely.
        write_mfe_cache(records, cache_path)

    print(f"MFE cache summary: reused={reused_count}, predicted={predicted_count}")
    return pd.DataFrame(records)


def read_expression_workbook(workbook_path: Path) -> pd.DataFrame:
    """Read four time-point blocks and summarize five technical replicates."""
    raw = pd.read_excel(
        workbook_path,
        sheet_name="Sheet1",
        header=None,
        usecols="A:F",
    )

    records: list[dict[str, object]] = []
    current_time: int | None = None

    for _, row in raw.iterrows():
        first_cell = row.iloc[0]
        if pd.isna(first_cell):
            continue

        label = str(first_cell).strip()
        time_text = label.lower().replace(" ", "")
        if time_text.endswith("h") and time_text[:-1].isdigit():
            current_time = int(time_text[:-1])
            continue

        if current_time not in TIME_POINTS or label.lower() == "blank":
            continue

        replicate_values = pd.to_numeric(row.iloc[1:6], errors="coerce").dropna()
        if replicate_values.empty:
            continue

        fasta_id = EXPRESSION_TO_FASTA_ID.get(label, label)
        records.append(
            {
                "expression_id": label,
                "id": fasta_id,
                "time_h": current_time,
                "replicate_n": int(replicate_values.size),
                "fluc_mean": float(replicate_values.mean()),
                "fluc_sd": float(replicate_values.std(ddof=1)),
                "fluc_median": float(replicate_values.median()),
                **{
                    f"replicate_{index + 1}": float(value)
                    for index, value in enumerate(replicate_values)
                },
            }
        )

    expression = pd.DataFrame(records)
    if expression.empty:
        raise ValueError(f"No fluorescence records found in {workbook_path}")

    duplicated = expression.duplicated(["id", "time_h"], keep=False)
    if duplicated.any():
        duplicate_rows = expression.loc[duplicated, ["id", "time_h"]]
        raise ValueError(f"Duplicate expression rows detected:\n{duplicate_rows}")

    missing_times = set(TIME_POINTS) - set(expression["time_h"].unique())
    if missing_times:
        raise ValueError(f"Missing time points: {sorted(missing_times)}")

    return expression


def integrate_results(
    mfe_results: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    """Join fluorescence summaries to MFE predictions and validate all IDs."""
    mfe_columns = [
        "id",
        "group",
        "length_nt",
        "mfe_kcal_mol",
        "mfe_per_100nt",
    ]
    integrated = expression.merge(
        mfe_results[mfe_columns],
        on="id",
        how="left",
        validate="many_to_one",
    )

    missing_mfe = sorted(
        integrated.loc[integrated["mfe_kcal_mol"].isna(), "id"].unique()
    )
    if missing_mfe:
        raise ValueError(f"Expression IDs without MFE predictions: {missing_mfe}")

    integrated = integrated.sort_values(["time_h", "id"]).reset_index(drop=True)
    return integrated


def calculate_correlations(integrated: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations at every time point."""
    records: list[dict[str, object]] = []

    for time_h in TIME_POINTS:
        subset = integrated.loc[integrated["time_h"] == time_h].dropna(
            subset=["mfe_kcal_mol", "fluc_mean"]
        )
        pearson = stats.pearsonr(subset["mfe_kcal_mol"], subset["fluc_mean"])
        spearman = stats.spearmanr(subset["mfe_kcal_mol"], subset["fluc_mean"])
        records.append(
            {
                "time_h": time_h,
                "n": len(subset),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )

    return pd.DataFrame(records)


def plot_mfe_bar(mfe_results: pd.DataFrame) -> None:
    """Plot group-colored MFE values for all FASTA constructs."""
    figure_width = max(12, len(mfe_results) * 0.82)
    fig, ax = plt.subplots(figsize=(figure_width, 7))

    colors = mfe_results["group"].map(GROUP_COLORS)
    bars = ax.bar(
        mfe_results["id"],
        mfe_results["mfe_kcal_mol"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    for bar, mfe in zip(bars, mfe_results["mfe_kcal_mol"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mfe,
            f"{mfe:.1f}",
            ha="center",
            va="top",
            rotation=90,
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("mRNA construct")
    ax.set_ylabel("Minimum free energy (kcal/mol)")
    ax.set_title("ViennaRNA-predicted MFE of full-length mRNA constructs")
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    present_groups = list(dict.fromkeys(mfe_results["group"]))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor="black",
            markersize=8,
            label=group,
        )
        for group in present_groups
    ]
    ax.legend(handles=handles, title="Sequence group", frameon=False)

    fig.tight_layout()
    fig.savefig(MFE_BAR_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(MFE_BAR_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(
    integrated: pd.DataFrame,
    correlation_results: pd.DataFrame,
) -> None:
    """Plot MFE against mean fluorescence for all four time points."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True)

    for ax, time_h in zip(axes.flat, TIME_POINTS):
        subset = integrated.loc[integrated["time_h"] == time_h].copy()
        statistics_row = correlation_results.loc[
            correlation_results["time_h"] == time_h
        ].iloc[0]

        for group, group_data in subset.groupby("group", sort=False):
            ax.errorbar(
                group_data["mfe_kcal_mol"],
                group_data["fluc_mean"],
                yerr=group_data["fluc_sd"],
                fmt="o",
                markersize=7,
                color=GROUP_COLORS[group],
                markeredgecolor="black",
                markeredgewidth=0.6,
                ecolor=GROUP_COLORS[group],
                elinewidth=1,
                capsize=3,
                alpha=0.9,
                label=group,
            )

        x = subset["mfe_kcal_mol"].to_numpy(dtype=float)
        y = subset["fluc_mean"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#333333",
            linewidth=1.4,
            linestyle="--",
        )

        for _, row in subset.iterrows():
            ax.annotate(
                row["expression_id"],
                (row["mfe_kcal_mol"], row["fluc_mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                alpha=0.85,
            )

        annotation = (
            f"n = {int(statistics_row['n'])}\n"
            f"Pearson r = {statistics_row['pearson_r']:.3f}, "
            f"p = {statistics_row['pearson_p']:.3g}\n"
            f"Spearman rho = {statistics_row['spearman_rho']:.3f}, "
            f"p = {statistics_row['spearman_p']:.3g}"
        )
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#BDBDBD",
                "alpha": 0.9,
            },
        )

        ax.set_title(f"{time_h} h")
        ax.set_ylabel("Fluc luminescence (mean ± SD)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(linestyle="--", linewidth=0.5, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Minimum free energy (kcal/mol)")

    present_groups = [
        group for group in GROUP_COLORS if group in set(integrated["group"])
    ]
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor="black",
            markersize=8,
            label=group,
        )
        for group in present_groups
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(legend_handles),
        frameon=False,
        title="Sequence group",
    )
    fig.suptitle(
        "Correlation between full-length mRNA MFE and Fluc luminescence",
        y=1.035,
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(CORRELATION_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(CORRELATION_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run cached MFE prediction, fluorescence integration, and correlation analysis."""
    for input_path in (INPUT_FASTA, EXPRESSION_XLSX):
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    sequences = read_fasta(INPUT_FASTA)
    mfe_results = calculate_mfe_with_cache(sequences, MFE_RESULT)
    expression = read_expression_workbook(EXPRESSION_XLSX)
    integrated = integrate_results(mfe_results, expression)
    correlation_results = calculate_correlations(integrated)

    integrated.to_csv(INTEGRATED_RESULT, index=False)
    correlation_results.to_csv(CORRELATION_RESULT, index=False)
    plot_mfe_bar(mfe_results)
    plot_correlations(integrated, correlation_results)

    print("\nCorrelation summary")
    print(correlation_results.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nMFE cache: {MFE_RESULT}")
    print(f"Integrated results: {INTEGRATED_RESULT}")
    print(f"Correlation statistics: {CORRELATION_RESULT}")
    print(f"Correlation plot: {CORRELATION_PNG}")


if __name__ == "__main__":
    main()

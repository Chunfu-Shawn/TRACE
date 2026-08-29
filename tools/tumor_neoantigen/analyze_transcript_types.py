#!/usr/bin/env python3
"""Classify and summarize tumor-associated transcript candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from cohort_annotation_utils import (
    clean_transcript_id,
    merge_annotations,
    natural_patient_key,
    normalize_patient_id,
    transcript_macro_category,
)


MACRO_ORDER = [
    "De novo Gene",
    "Novel Transcript",
    "Protein Coding",
    "Pseudogene",
    "lncRNA",
    "Other ncRNA",
    "Unknown ENST",
]
MICRO_ORDER = [
    "Intergenic (u)",
    "Variable TSS/TTS (k)",
    "Antisense (x)",
    "Intronic (i)",
    "Retained Intron (m/n)",
    "Novel Isoform (j)",
    "A5SS / A3SS",
    "MXE",
    "Skipped Exon (SE)",
    "Exonic Overlap (o)",
    "Other Novel",
]
MACRO_PALETTE = {
    "De novo Gene": "#C44E52",
    "Novel Transcript": "#E69F8C",
    "Protein Coding": "#4C78A8",
    "Pseudogene": "#F2A541",
    "lncRNA": "#59A14F",
    "Other ncRNA": "#8A6FB0",
    "Unknown ENST": "#9C755F",
}
MICRO_PALETTE = {
    "Intergenic (u)": "#E69F8C",
    "Variable TSS/TTS (k)": "#F3C969",
    "Antisense (x)": "#B39BC8",
    "Intronic (i)": "#8CBBD9",
    "Retained Intron (m/n)": "#9CCB86",
    "Novel Isoform (j)": "#F2A65A",
    "A5SS / A3SS": "#B8B8B8",
    "MXE": "#D8B07C",
    "Skipped Exon (SE)": "#B89A78",
    "Exonic Overlap (o)": "#E8DF7A",
    "Other Novel": "#D9A5C7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify tumor-associated transcripts after GTEx filtering."
    )
    parser.add_argument("--input", required=True, help="Final GTEx-filtered transcript CSV/TSV.")
    parser.add_argument("--reference_gtf", required=True)
    parser.add_argument("--novel_gtf", help="Optional intact novel/de novo transcript GTF.")
    parser.add_argument("--denovo_ids", help="Optional one-column de novo transcript ID file.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patient_order_file")
    parser.add_argument("--mode", choices=("count", "percent"), default="count")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png", "tiff"),
        default=("pdf", "svg", "png", "tiff"),
    )
    parser.add_argument("--width_per_patient", type=float, default=0.15)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Input table is missing required columns: {sorted(missing)}")


def read_identifier_file(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    identifiers = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            identifiers.add(clean_transcript_id(value.split("\t", 1)[0]))
    return identifiers


def resolve_patient_order(values: Iterable[object], order_file: Optional[str]) -> list[str]:
    observed = list(dict.fromkeys(normalize_patient_id(value) for value in values))
    if not order_file:
        return sorted(observed, key=natural_patient_key)
    requested = [
        normalize_patient_id(line)
        for line in Path(order_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observed_set = set(observed)
    ordered = [patient for patient in dict.fromkeys(requested) if patient in observed_set]
    requested_set = set(ordered)
    return ordered + sorted(
        [patient for patient in observed if patient not in requested_set],
        key=natural_patient_key,
    )


def micro_category(row: Mapping[str, object]) -> str:
    """Classify the structural subtype of one novel transcript."""
    if row.get("Broad_Category") != "Novel Transcript":
        return "Not Applicable"
    raw_code = row.get("Class_Code", "")
    code = "" if pd.isna(raw_code) else str(raw_code).strip().casefold()
    exact = {
        "u": "Intergenic (u)",
        "k": "Variable TSS/TTS (k)",
        "x": "Antisense (x)",
        "i": "Intronic (i)",
        "m": "Retained Intron (m/n)",
        "n": "Retained Intron (m/n)",
        "j": "Novel Isoform (j)",
        "o": "Exonic Overlap (o)",
    }
    if code in exact:
        return exact[code]
    if "a5ss" in code or "a3ss" in code:
        return "A5SS / A3SS"
    if "mxe" in code or "mutually exclusive" in code:
        return "MXE"
    if code == "se" or "skipped exon" in code:
        return "Skipped Exon (SE)"
    if "retained" in code or code == "ir":
        return "Retained Intron (m/n)"
    return "Other Novel"


def annotate_transcripts(
    frame: pd.DataFrame,
    reference_gtf: str,
    novel_gtf: Optional[str],
    denovo_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    require_columns(frame, {"Patient", "Transcript_ID"})
    result = frame.copy()
    input_rows = len(result)
    result["Patient"] = result["Patient"].map(normalize_patient_id)
    result["Clean_Transcript_ID"] = result["Transcript_ID"].map(clean_transcript_id)
    result = result.drop_duplicates(["Patient", "Clean_Transcript_ID"], keep="first").copy()

    annotations = merge_annotations(
        reference_gtf=reference_gtf,
        target_transcripts=set(result["Clean_Transcript_ID"]),
        novel_gtf=novel_gtf,
    ).rename(columns={"Transcript_ID": "Clean_Transcript_ID"})
    if annotations.empty:
        annotations = pd.DataFrame({"Clean_Transcript_ID": result["Clean_Transcript_ID"].unique()})
    overlapping_columns = [
        column
        for column in annotations.columns
        if column != "Clean_Transcript_ID" and column in result.columns
    ]
    result = result.merge(
        annotations,
        on="Clean_Transcript_ID",
        how="left",
        suffixes=("", "_GTF"),
    )
    for column in overlapping_columns:
        gtf_column = f"{column}_GTF"
        # Prefer the unified GTF annotation and retain input values as a fallback.
        result[column] = result[gtf_column].combine_first(result[column])
        result = result.drop(columns=gtf_column)

    result["Is_De_Novo"] = result.get("Is_De_Novo", False)
    result["Is_De_Novo"] = (
        result["Is_De_Novo"].fillna(False).astype(bool)
        | result["Clean_Transcript_ID"].isin(denovo_ids)
    )
    result["Broad_Category"] = result.apply(transcript_macro_category, axis=1)
    result["Sub_Category"] = result.apply(micro_category, axis=1)
    biotype = result.get("Biotype", pd.Series("", index=result.index)).fillna("").astype(str)
    structurally_resolved = result["Broad_Category"].isin({"De novo Gene", "Novel Transcript"})
    result["Annotation_Status"] = np.where(
        structurally_resolved | biotype.str.strip().ne(""),
        "resolved",
        "not_found_in_annotation",
    )
    metrics = {
        "Input_Rows": input_rows,
        "Unique_Patient_Transcript_Rows": len(result),
        "Duplicate_Rows_Removed": input_rows - len(result),
        "Patients": result["Patient"].nunique(),
        "Unresolved_Rows": int(result["Annotation_Status"].ne("resolved").sum()),
    }
    return result, metrics


def summarize(frame: pd.DataFrame, category: str, order: list[str], patients: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=pd.Index(patients, name="Patient"), dtype=int)
    counts = frame.groupby(["Patient", category], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(index=patients, fill_value=0)
    columns = [value for value in order if value in counts.columns]
    columns += [value for value in counts.columns if value not in columns]
    return counts.reindex(columns=columns, fill_value=0).astype(int)


def write_matrix(matrix: pd.DataFrame, prefix: Path) -> None:
    matrix.rename_axis("Patient").to_csv(prefix.with_name(prefix.name + ".wide.csv"))
    matrix.rename_axis("Patient").reset_index().melt(
        id_vars="Patient", var_name="Category", value_name="Count"
    ).to_csv(prefix.with_name(prefix.name + ".long.csv"), index=False)


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(figure, prefix: Path, formats: Iterable[str]) -> None:
    suffixes = {"pdf": ".pdf", "svg": ".svg", "png": ".png", "tiff": ".tiff"}
    for file_format in formats:
        dpi = 600 if file_format in {"png", "tiff"} else 300
        figure.savefig(prefix.with_suffix(suffixes[file_format]), dpi=dpi, bbox_inches="tight")


def plot_global_summary(
    frame: pd.DataFrame,
    category: str,
    order: list[str],
    palette: Mapping[str, str],
    output_prefix: Path,
    formats: Iterable[str],
) -> None:
    import matplotlib.pyplot as plt

    counts = frame[category].value_counts().reindex(order, fill_value=0)
    counts = counts[counts.gt(0)]
    percentages = counts / counts.sum() * 100.0
    figure_height = max(2.4, 0.32 * len(counts) + 0.8)
    figure, axis = plt.subplots(figsize=(3.5, figure_height))
    y_values = np.arange(len(counts))
    axis.barh(
        y_values,
        counts.to_numpy(),
        color=[palette.get(category_name, "#999999") for category_name in counts.index],
        height=0.72,
    )
    axis.set_yticks(y_values)
    axis.set_yticklabels(counts.index)
    axis.invert_yaxis()
    axis.set_xlabel("Number of transcripts")
    offset = max(float(counts.max()) * 0.015, 0.2)
    for index, (count, percentage) in enumerate(zip(counts, percentages)):
        axis.text(count + offset, index, f"{count:,} ({percentage:.1f}%)", va="center", fontsize=6.5)
    axis.set_xlim(0, float(counts.max()) * 1.28 if len(counts) else 1)
    axis.tick_params(width=0.7, length=2.5)
    figure.tight_layout()
    save_figure(figure, output_prefix, formats)
    plt.close(figure)


def plot_patient_stacked(
    matrix: pd.DataFrame,
    mode: str,
    palette: Mapping[str, str],
    ylabel: str,
    output_prefix: Path,
    formats: Iterable[str],
    width_per_patient: float,
) -> None:
    import matplotlib.pyplot as plt

    plot_matrix = matrix.astype(float)
    if mode == "percent":
        plot_matrix = plot_matrix.div(plot_matrix.sum(axis=1).replace(0, np.nan), axis=0)
        plot_matrix = plot_matrix.fillna(0.0) * 100.0
        ylabel = "Proportion of transcripts (%)"
    width = max(7.2, min(22.0, len(plot_matrix) * width_per_patient))
    figure, axis = plt.subplots(figsize=(width, 4.5))
    x_values = np.arange(len(plot_matrix))
    bottom = np.zeros(len(plot_matrix), dtype=float)
    for category in plot_matrix.columns:
        values = plot_matrix[category].to_numpy(dtype=float)
        axis.bar(
            x_values,
            values,
            bottom=bottom,
            width=0.84,
            color=palette.get(category, "#999999"),
            edgecolor="none",
            label=category,
        )
        bottom += values
    axis.set_xticks(x_values)
    axis.set_xticklabels(plot_matrix.index, rotation=90, ha="center", fontsize=5.5)
    axis.set_xlabel("Patient")
    axis.set_ylabel(ylabel)
    axis.set_xlim(-0.7, len(plot_matrix) - 0.3)
    axis.tick_params(width=0.7, length=2.5)
    axis.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=6.3)
    figure.tight_layout()
    save_figure(figure, output_prefix, formats)
    plt.close(figure)


def plot_expression_support(
    frame: pd.DataFrame,
    output_prefix: Path,
    formats: Iterable[str],
) -> None:
    import matplotlib.pyplot as plt

    required = {
        "Pass_TrackA_TPM",
        "Pass_TrackB_Junction",
        "Tumor_TPM",
        "Tumor_Junction_CPM",
    }
    if not required.issubset(frame.columns):
        return
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    specifications = [
        ("Pass_TrackA_TPM", "Tumor_TPM", "log2(TPM + 1)", "Transcript-expression track"),
        (
            "Pass_TrackB_Junction",
            "Tumor_Junction_CPM",
            "log2(junction CPM + 1)",
            "Junction track",
        ),
    ]
    for axis, (flag, value_column, ylabel, title) in zip(axes, specifications):
        mask = frame[flag].fillna(False).astype(bool)
        subset = frame.loc[mask].copy()
        subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
        subset = subset[subset[value_column].notna() & subset[value_column].ge(0)]
        categories = [category for category in MACRO_ORDER if category in set(subset["Broad_Category"])]
        if not categories:
            axis.text(0.5, 0.5, "No passing transcripts", ha="center", va="center")
            axis.set_axis_off()
            continue
        # A pseudocount of 1 is applied only after negative values are excluded.
        values = [
            np.log2(subset.loc[subset["Broad_Category"].eq(category), value_column].to_numpy() + 1.0)
            for category in categories
        ]
        positions = np.arange(1, len(categories) + 1)
        boxes = axis.boxplot(
            values,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#333333", "linewidth": 1.1},
            whiskerprops={"color": "#666666", "linewidth": 0.8},
            capprops={"color": "#666666", "linewidth": 0.8},
        )
        for patch, category in zip(boxes["boxes"], categories):
            patch.set_facecolor(MACRO_PALETTE.get(category, "#999999"))
            patch.set_alpha(0.55)
            patch.set_edgecolor("#555555")
            patch.set_linewidth(0.8)
        for position, group_values in zip(positions, values):
            jitter = np.linspace(-0.18, 0.18, num=len(group_values)) if len(group_values) > 1 else np.zeros(1)
            axis.scatter(
                position + jitter,
                group_values,
                s=5,
                color="#4D4D4D",
                alpha=0.35,
                linewidths=0,
                rasterized=True,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(categories, rotation=45, ha="right", fontsize=6)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=8)
        axis.tick_params(width=0.7, length=2.5)
    figure.tight_layout()
    save_figure(figure, output_prefix, formats)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    denovo_ids = read_identifier_file(args.denovo_ids)
    annotated, metrics = annotate_transcripts(
        read_table(input_path),
        reference_gtf=args.reference_gtf,
        novel_gtf=args.novel_gtf,
        denovo_ids=denovo_ids,
    )
    patients = resolve_patient_order(annotated["Patient"], args.patient_order_file)
    macro_matrix = summarize(annotated, "Broad_Category", MACRO_ORDER, patients)
    novel = annotated[annotated["Broad_Category"].eq("Novel Transcript")].copy()
    micro_matrix = summarize(novel, "Sub_Category", MICRO_ORDER, patients)

    annotated.to_csv(output_dir / "annotated_tumor_associated_transcripts.csv", index=False)
    annotated.loc[annotated["Annotation_Status"].ne("resolved")].to_csv(
        output_dir / "unresolved_transcript_annotations.csv", index=False
    )
    write_matrix(macro_matrix, output_dir / "transcript_type_macro_counts")
    write_matrix(micro_matrix, output_dir / "transcript_type_micro_counts")
    global_rows = []
    for level, frame, category in (
        ("macro", annotated, "Broad_Category"),
        ("micro", novel, "Sub_Category"),
    ):
        counts = frame[category].value_counts()
        total = int(counts.sum())
        for label, count in counts.items():
            global_rows.append(
                {
                    "Level": level,
                    "Category": label,
                    "Count": int(count),
                    "Percent": float(count / total * 100.0) if total else 0.0,
                }
            )
    pd.DataFrame(global_rows).to_csv(output_dir / "transcript_type_global_summary.csv", index=False)
    metrics.update(
        {
            "Input_File": str(input_path),
            "De_Novo_IDs_Loaded": len(denovo_ids),
            "Novel_Transcript_Rows": len(novel),
            "Stacked_Bar_Mode": args.mode,
        }
    )
    (output_dir / "transcript_type_analysis_qc.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    if not args.no_plots:
        configure_matplotlib()
        plot_global_summary(
            annotated,
            "Broad_Category",
            MACRO_ORDER,
            MACRO_PALETTE,
            output_dir / "transcript_type_macro_global",
            args.formats,
        )
        plot_patient_stacked(
            macro_matrix,
            args.mode,
            MACRO_PALETTE,
            "Number of tumor-associated transcripts",
            output_dir / f"transcript_type_macro_patient_stacked_{args.mode}",
            args.formats,
            args.width_per_patient,
        )
        if not novel.empty:
            plot_global_summary(
                novel,
                "Sub_Category",
                MICRO_ORDER,
                MICRO_PALETTE,
                output_dir / "transcript_type_micro_global",
                args.formats,
            )
            plot_patient_stacked(
                micro_matrix,
                args.mode,
                MICRO_PALETTE,
                "Number of novel transcripts",
                output_dir / f"transcript_type_micro_patient_stacked_{args.mode}",
                args.formats,
                args.width_per_patient,
            )
        plot_expression_support(
            annotated,
            output_dir / "transcript_type_expression_support",
            args.formats,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

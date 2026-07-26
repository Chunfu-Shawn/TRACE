#!/usr/bin/env python3
"""Summarize and plot cohort-level transcript and antigen compositions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from cohort_annotation_utils import (
    antigen_origin_category,
    clean_transcript_id,
    merge_annotations,
    natural_patient_key,
    normalize_patient_id,
    transcript_macro_category,
)


TRANSCRIPT_ORDER = [
    "De novo Gene",
    "Novel Transcript",
    "Protein Coding",
    "Pseudogene",
    "lncRNA",
    "Other ncRNA",
    "Unknown ENST",
]
ANTIGEN_ORDER = [
    "Canonical CDS",
    "Cryptic ORF",
    "Pseudogene",
    "lncRNA",
    "Other ncRNA",
    "Novel Transcript",
    "Unresolved protein-coding ORF",
    "Unresolved ENST",
    "Multiple Origins",
]
PALETTE = {
    "De novo Gene": "#8B1A1A",
    "Novel Transcript": "#D95F02",
    "Protein Coding": "#E6AB02",
    "Canonical CDS": "#1B9E77",
    "Cryptic ORF": "#7570B3",
    "Pseudogene": "#66A61E",
    "lncRNA": "#E7298A",
    "Other ncRNA": "#1F78B4",
    "Unknown ENST": "#A6761D",
    "Unresolved protein-coding ORF": "#BDBDBD",
    "Unresolved ENST": "#969696",
    "Multiple Origins": "#333333",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-patient stacked summaries without combining expression scales."
    )
    parser.add_argument("--transcript_table", required=True)
    parser.add_argument("--antigen_dir", required=True)
    parser.add_argument("--reference_gtf", required=True)
    parser.add_argument("--novel_gtf", help="Assembly GTF; entries are Novel Transcript.")
    parser.add_argument("--denovo_gtf", help="Independent de novo-gene GTF; entries are De novo Gene.")
    parser.add_argument("--denovo_fasta", help="De novo-gene transcript FASTA used to identify source IDs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patient_order_file")
    parser.add_argument("--mode", choices=("count", "percent"), default="count")
    parser.add_argument("--canonical_tolerance", type=int, default=6)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png"),
        default=["pdf", "svg", "png"],
    )
    parser.add_argument("--width_per_patient", type=float, default=0.13)
    parser.add_argument("--label_mode", choices=("auto", "all", "sparse", "none"), default="auto")
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def load_antigen_reports(directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted([*directory.glob("*.csv"), *directory.glob("*.tsv")]):
        frame = read_table(path)
        if frame.empty:
            continue
        if "Transcript_ID" not in frame.columns and "Identity" in frame.columns:
            frame["Transcript_ID"] = frame["Identity"]
        patient = frame["Patient"] if "Patient" in frame.columns else path.stem
        frame["Patient"] = pd.Series(patient, index=frame.index).map(normalize_patient_id)
        frame["Source_File"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError(f"No non-empty CSV/TSV antigen reports found in {directory}")
    result = pd.concat(frames, ignore_index=True)
    require_columns(result, {"Patient", "Peptide", "Transcript_ID"}, "Antigen reports")
    return result


def add_gtf_annotations(
    frame: pd.DataFrame,
    reference_gtf: str,
    novel_gtf: Optional[str],
    denovo_gtf: Optional[str],
    denovo_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    result = frame.copy()
    result["Clean_Transcript_ID"] = result["Transcript_ID"].map(clean_transcript_id)
    annotations = merge_annotations(
        reference_gtf=reference_gtf,
        target_transcripts=set(result["Clean_Transcript_ID"]),
        novel_gtf=novel_gtf,
        de_novo_gtf=denovo_gtf,
    ).rename(columns={"Transcript_ID": "Clean_Transcript_ID"})
    if annotations.empty:
        for column in (
            "Gene_ID", "Gene_Name", "Biotype", "Strand", "Is_De_Novo",
            "Is_Novel_Transcript", "Canonical_ORF_Start", "Canonical_ORF_Stop",
        ):
            result[column] = pd.NA
        if denovo_ids:
            result["Is_De_Novo"] = result["Clean_Transcript_ID"].isin(denovo_ids)
        return result

    overlapping = [column for column in annotations.columns if column in result.columns and column != "Clean_Transcript_ID"]
    result = result.merge(annotations, on="Clean_Transcript_ID", how="left", suffixes=("", "_GTF"))
    for column in overlapping:
        result[column] = result[f"{column}_GTF"].combine_first(result[column])
        result = result.drop(columns=f"{column}_GTF")
    if denovo_ids:
        result["Is_De_Novo"] = (
            result["Is_De_Novo"].fillna(False).astype(bool)
            | result["Clean_Transcript_ID"].isin(denovo_ids)
        )
    return result


def read_fasta_ids(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    identifiers = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                identifiers.add(clean_transcript_id(line[1:].split()[0]))
    return identifiers


def resolve_patient_order(values: Iterable[str], order_file: Optional[str]) -> list[str]:
    observed = list(dict.fromkeys(normalize_patient_id(value) for value in values))
    if not order_file:
        return sorted(observed, key=natural_patient_key)
    requested = [normalize_patient_id(line) for line in Path(order_file).read_text().splitlines() if line.strip()]
    requested_set = set(requested)
    return list(dict.fromkeys(requested)) + sorted(
        [value for value in observed if value not in requested_set], key=natural_patient_key
    )


def summarize(frame: pd.DataFrame, category_col: str, categories: list[str], patients: list[str]) -> pd.DataFrame:
    counts = frame.groupby(["Patient", category_col]).size().unstack(fill_value=0)
    counts = counts.reindex(index=patients, fill_value=0)
    ordered = [category for category in categories if category in counts.columns]
    ordered += [column for column in counts.columns if column not in ordered]
    return counts.reindex(columns=ordered, fill_value=0)


def unique_peptide_origins(antigens: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (patient, peptide), group in antigens.groupby(["Patient", "Peptide"], sort=False):
        origins = sorted(set(group["Antigen_Type"].dropna()))
        records.append(
            {
                "Patient": patient,
                "Peptide": peptide,
                "Antigen_Type": origins[0] if len(origins) == 1 else "Multiple Origins",
                "Source_Transcripts": ";".join(sorted(set(group["Clean_Transcript_ID"]))),
                "Source_Origin_Types": ";".join(origins),
                "Source_Count": group["Clean_Transcript_ID"].nunique(),
            }
        )
    return pd.DataFrame(records)


def write_summary(matrix: pd.DataFrame, prefix: Path) -> None:
    matrix.to_csv(prefix.with_name(prefix.name + ".wide.csv"))
    matrix.rename_axis("Patient").reset_index().melt(
        id_vars="Patient", var_name="Category", value_name="Count"
    ).to_csv(prefix.with_name(prefix.name + ".long.csv"), index=False)


def plot_stacked(matrix: pd.DataFrame, output_prefix: Path, ylabel: str, args: argparse.Namespace) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plot_matrix = matrix.astype(float)
    if args.mode == "percent":
        plot_matrix = plot_matrix.div(plot_matrix.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        plot_matrix = plot_matrix.fillna(0.0)
        ylabel = "Proportion (%)"

    width = max(7.2, min(18.0, len(plot_matrix) * args.width_per_patient))
    figure, axis = plt.subplots(figsize=(width, 4.6))
    bottom = np.zeros(len(plot_matrix))
    x_values = np.arange(len(plot_matrix))
    for category in plot_matrix.columns:
        values = plot_matrix[category].to_numpy()
        axis.bar(
            x_values,
            values,
            bottom=bottom,
            width=0.82,
            color=PALETTE.get(category, "#999999"),
            edgecolor="none",
            label=category,
        )
        bottom += values

    label_mode = args.label_mode
    if label_mode == "auto":
        label_mode = "sparse" if len(plot_matrix) > 40 else "all"
    if label_mode == "none":
        ticks = []
    elif label_mode == "sparse":
        step = max(1, int(np.ceil(len(plot_matrix) / 26)))
        ticks = x_values[::step]
    else:
        ticks = x_values
    axis.set_xticks(ticks)
    axis.set_xticklabels([plot_matrix.index[index] for index in ticks], rotation=90, ha="center")
    axis.set_xlabel("Patient ID")
    axis.set_ylabel(ylabel)
    axis.set_xlim(-0.7, len(plot_matrix) - 0.3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=0.7, length=2.5)
    axis.legend(frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=6.5)
    figure.tight_layout()
    if "pdf" in args.formats:
        figure.savefig(output_prefix.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    if "svg" in args.formats:
        figure.savefig(output_prefix.with_suffix(".svg"), dpi=600, bbox_inches="tight")
    if "png" in args.formats:
        figure.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    denovo_ids = read_fasta_ids(args.denovo_fasta)

    transcripts = read_table(Path(args.transcript_table))
    require_columns(transcripts, {"Patient", "Transcript_ID"}, "Transcript table")
    transcripts["Patient"] = transcripts["Patient"].map(normalize_patient_id)
    transcripts = add_gtf_annotations(
        transcripts, args.reference_gtf, args.novel_gtf, args.denovo_gtf, denovo_ids
    )
    transcripts["Transcript_Type"] = transcripts.apply(transcript_macro_category, axis=1)
    transcripts["Annotation_Status"] = np.where(
        transcripts["Biotype"].notna() & transcripts["Biotype"].astype(str).ne(""),
        "resolved",
        "not_found_in_annotation",
    )
    transcript_units = transcripts.drop_duplicates(["Patient", "Clean_Transcript_ID"])

    antigens = load_antigen_reports(Path(args.antigen_dir))
    antigens = add_gtf_annotations(
        antigens, args.reference_gtf, args.novel_gtf, args.denovo_gtf, denovo_ids
    )
    antigens["Antigen_Type"] = antigens.apply(
        antigen_origin_category, axis=1, tolerance=args.canonical_tolerance
    )
    antigens["Annotation_Status"] = np.where(
        antigens["Biotype"].notna() & antigens["Biotype"].astype(str).ne(""),
        "resolved",
        "not_found_in_annotation",
    )
    peptide_units = unique_peptide_origins(antigens)

    patients = resolve_patient_order(
        list(transcript_units["Patient"]) + list(peptide_units["Patient"]),
        args.patient_order_file,
    )
    transcript_matrix = summarize(transcript_units, "Transcript_Type", TRANSCRIPT_ORDER, patients)
    antigen_matrix = summarize(peptide_units, "Antigen_Type", ANTIGEN_ORDER, patients)

    transcripts.to_csv(output_dir / "tumor_specific_transcripts.annotated.csv", index=False)
    antigens.to_csv(output_dir / "tumor_associated_antigen_sources.annotated.csv", index=False)
    peptide_units.to_csv(output_dir / "tumor_associated_unique_peptides.annotated.csv", index=False)
    transcripts.loc[transcripts["Annotation_Status"] != "resolved"].to_csv(
        output_dir / "unresolved_transcript_annotations.csv", index=False
    )
    antigens.loc[antigens["Antigen_Type"].str.startswith("Unresolved")].to_csv(
        output_dir / "unresolved_antigen_annotations.csv", index=False
    )
    write_summary(transcript_matrix, output_dir / "transcript_type_counts")
    write_summary(antigen_matrix, output_dir / "antigen_type_unique_peptide_counts")

    if not args.no_plot:
        plot_stacked(
            transcript_matrix,
            output_dir / f"transcript_type_stacked_{args.mode}",
            "Number of Tumor-specific Transcripts",
            args,
        )
        plot_stacked(
            antigen_matrix,
            output_dir / f"antigen_type_stacked_{args.mode}",
            "Number of Unique Tumor-associated Peptides",
            args,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

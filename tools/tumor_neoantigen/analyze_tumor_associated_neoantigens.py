#!/usr/bin/env python3
"""Classify and summarize tumor-associated antigen candidates across a cohort."""

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
    "Canonical CDS",
    "De novo Gene",
    "Cryptic ORF",
    "Pseudogene",
    "lncRNA",
    "Other ncRNA",
    "Novel Transcript",
    "Unknown Source",
    "Multiple Origins",
]
MICRO_ORDER = [
    "Intergenic (u)",
    "Intronic (i)",
    "Novel Splice Junction",
    "Retained Intron Region",
    "Novel Exon/Alt Frame",
    "Antisense (x)",
    "Other Novel",
    "Multiple Novel Origins",
]
MACRO_PALETTE = {
    "Canonical CDS": "#4C78A8",
    "De novo Gene": "#C44E52",
    "Cryptic ORF": "#F2A541",
    "Pseudogene": "#8A6FB0",
    "lncRNA": "#59A14F",
    "Other ncRNA": "#B39BC8",
    "Novel Transcript": "#E69F8C",
    "Unknown Source": "#9C9C9C",
    "Multiple Origins": "#4D4D4D",
}
MICRO_PALETTE = {
    "Intergenic (u)": "#E69F8C",
    "Intronic (i)": "#8CBBD9",
    "Novel Splice Junction": "#D8B07C",
    "Retained Intron Region": "#9CCB86",
    "Novel Exon/Alt Frame": "#F2A65A",
    "Antisense (x)": "#B39BC8",
    "Other Novel": "#D9A5C7",
    "Multiple Novel Origins": "#6F6F6F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tumor-associated antigen reports after canonical-proteome filtering."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--reference_gtf", required=True)
    parser.add_argument("--novel_gtf")
    parser.add_argument("--canonical_protein_fasta", required=True)
    parser.add_argument("--transcript_meta")
    parser.add_argument("--denovo_ids")
    parser.add_argument("--denovo_protein_fasta")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patient_order_file")
    parser.add_argument("--target_patients_file")
    parser.add_argument("--mode", choices=("count", "percent"), default="count")
    parser.add_argument("--top_shared_peptides", type=int, default=30)
    parser.add_argument("--width_per_patient", type=float, default=0.15)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png", "tiff"),
        default=("pdf", "svg", "png", "tiff"),
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_patient_shards", action="store_true")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def load_reports(directory: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frames = []
    files_seen = 0
    empty_files = 0
    for path in sorted([*directory.glob("*.csv"), *directory.glob("*.tsv")]):
        files_seen += 1
        frame = read_table(path)
        if frame.empty:
            empty_files += 1
            continue
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        if "Transcript_ID" not in frame.columns and "Identity" in frame.columns:
            frame["Transcript_ID"] = frame["Identity"]
        if "Patient" not in frame.columns:
            frame["Patient"] = path.stem
        frame["Patient"] = frame["Patient"].map(normalize_patient_id)
        frame["Source_File"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError(f"No non-empty antigen CSV/TSV reports found in {directory}")
    result = pd.concat(frames, ignore_index=True, sort=False)
    require_columns(result, {"Patient", "Peptide", "Transcript_ID"}, "Antigen reports")
    result["Peptide"] = result["Peptide"].fillna("").astype(str).str.strip().str.upper()
    result = result[result["Peptide"].ne("")].copy()
    return result, {
        "Input_Files": files_seen,
        "Empty_Input_Files": empty_files,
        "Input_Rows_With_Peptide": len(result),
    }


def read_identifier_file(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    identifiers = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                identifiers.add(clean_transcript_id(value.split("\t", 1)[0]))
    return identifiers


def read_protein_fasta(path: Optional[str], ensembl_headers: bool = False) -> dict[str, str]:
    if not path:
        return {}
    proteins: dict[str, str] = {}
    current_id: Optional[str] = None
    sequence: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    proteins[current_id] = "".join(sequence).upper()
                tokens = line[1:].split()[0].split("|")
                if ensembl_headers:
                    identifier = next((token for token in tokens if token.startswith("ENST")), "")
                else:
                    identifier = tokens[0] if tokens else ""
                current_id = clean_transcript_id(identifier) if identifier else None
                sequence = []
            elif current_id:
                sequence.append(line)
    if current_id:
        proteins[current_id] = "".join(sequence).upper()
    return proteins


def load_class_codes(path: Optional[str]) -> dict[str, str]:
    if not path:
        return {}
    frame = read_table(Path(path))
    if "Transcript_ID" not in frame.columns or "Class_Code" not in frame.columns:
        return {}
    frame = frame.loc[frame["Transcript_ID"].notna(), ["Transcript_ID", "Class_Code"]].copy()
    frame["Clean_Transcript_ID"] = frame["Transcript_ID"].map(clean_transcript_id)
    frame = frame.drop_duplicates("Clean_Transcript_ID", keep="last")
    return dict(zip(frame["Clean_Transcript_ID"], frame["Class_Code"].fillna("unknown")))


def load_exon_boundaries(paths: Iterable[Optional[str]], targets: set[str]) -> dict[str, list[int]]:
    exon_records: dict[str, list[tuple[int, int, str]]] = {}
    for raw_path in paths:
        if not raw_path:
            continue
        with Path(raw_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9 or fields[2] != "exon":
                    continue
                transcript_match = None
                for attribute in fields[8].split(";"):
                    attribute = attribute.strip()
                    if attribute.startswith("transcript_id "):
                        transcript_match = attribute.split(None, 1)[1].strip().strip('"')
                        break
                transcript_id = clean_transcript_id(transcript_match or "")
                if transcript_id not in targets:
                    continue
                exon_records.setdefault(transcript_id, []).append(
                    (int(fields[3]), int(fields[4]), fields[6])
                )
    boundaries = {}
    for transcript_id, exons in exon_records.items():
        ordered = sorted(exons, key=lambda item: item[0], reverse=exons[0][2] == "-")
        lengths = [end - start + 1 for start, end, _ in ordered]
        boundaries[transcript_id] = list(np.cumsum(lengths[:-1], dtype=int))
    return boundaries


def annotate_sources(
    frame: pd.DataFrame,
    reference_gtf: str,
    novel_gtf: Optional[str],
    denovo_ids: set[str],
) -> pd.DataFrame:
    result = frame.copy()
    result["Clean_Transcript_ID"] = result["Transcript_ID"].map(clean_transcript_id)
    annotations = merge_annotations(
        reference_gtf=reference_gtf,
        novel_gtf=novel_gtf,
        target_transcripts=set(result["Clean_Transcript_ID"]),
    ).rename(columns={"Transcript_ID": "Clean_Transcript_ID"})
    if annotations.empty:
        annotations = pd.DataFrame({"Clean_Transcript_ID": result["Clean_Transcript_ID"].unique()})
    overlapping = [
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
    for column in overlapping:
        gtf_column = f"{column}_GTF"
        result[column] = result[gtf_column].combine_first(result[column])
        result = result.drop(columns=gtf_column)
    if "Is_De_Novo" not in result.columns:
        result["Is_De_Novo"] = False
    existing_denovo = result["Is_De_Novo"].map(
        lambda value: False if pd.isna(value) else bool(value)
    )
    result["Is_De_Novo"] = existing_denovo | result["Clean_Transcript_ID"].isin(denovo_ids)
    return result


def parse_interval(value: object) -> tuple[Optional[int], Optional[int]]:
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        return None, None
    try:
        start, stop = int(parts[0]), int(parts[1])
    except ValueError:
        return None, None
    return (start, stop) if 0 <= start < stop else (None, None)


def peptide_crosses_junction(
    transcript_id: str,
    peptide_position: object,
    exon_boundaries: Mapping[str, list[int]],
) -> bool:
    start, stop = parse_interval(peptide_position)
    if start is None:
        return False
    return any(start < boundary < stop for boundary in exon_boundaries.get(transcript_id, []))


def classify_source(
    row: Mapping[str, object],
    canonical_proteins: Mapping[str, str],
    denovo_proteins: Mapping[str, str],
    class_codes: Mapping[str, str],
    exon_boundaries: Mapping[str, list[int]],
) -> tuple[str, str, str]:
    transcript_id = clean_transcript_id(row.get("Transcript_ID", ""))
    peptide = str(row.get("Peptide", "")).strip().upper()
    is_de_novo = row.get("Is_De_Novo", False)
    is_de_novo = False if pd.isna(is_de_novo) else bool(is_de_novo)
    if is_de_novo:
        sequence = denovo_proteins.get(transcript_id, "")
        verified = bool(sequence and peptide in sequence)
        return "De novo Gene", "De novo Gene" if verified else "Other ORFs", (
            "verified_id_and_sequence" if verified else "id_only_not_sequence_verified"
        )

    canonical_sequence = canonical_proteins.get(transcript_id, "")
    if canonical_sequence and peptide in canonical_sequence:
        return "Canonical CDS", "Canonical CDS", "verified_source_sequence"

    transcript_category = transcript_macro_category(row)
    if transcript_category == "Protein Coding":
        return "Cryptic ORF", "Cryptic UTR/Frameshift", "classified_from_annotation"
    if transcript_category in {"Pseudogene", "lncRNA", "Other ncRNA"}:
        return transcript_category, transcript_category, "classified_from_annotation"
    if transcript_category in {"Unknown ENST", "Unknown Source"}:
        return "Unknown Source", "Unknown Source", "annotation_not_found"

    code = str(class_codes.get(transcript_id, row.get("Class_Code", "unknown"))).strip().casefold()
    if code == "u":
        micro = "Intergenic (u)"
    elif code == "i":
        micro = "Intronic (i)"
    elif peptide_crosses_junction(
        transcript_id,
        row.get("Peptide_Tx_Pos", ""),
        exon_boundaries,
    ):
        micro = "Novel Splice Junction"
    elif code in {"m", "n", "ir"} or "retained" in code:
        micro = "Retained Intron Region"
    elif code == "x":
        micro = "Antisense (x)"
    elif code and code != "unknown":
        micro = "Novel Exon/Alt Frame"
    else:
        micro = "Other Novel"
    return "Novel Transcript", micro, "classified_from_annotation"


def collapse_patient_peptides(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    numeric_specs = {
        "Best_Affinity_nM": ("Aff(nM)", "min"),
        "Best_Rank_EL": ("%Rank_EL", "min"),
        "Max_Mean_Intensity": ("mean_intensity", "max"),
        "Max_ORF_Score": ("ORF_Score", "max"),
        "Max_Tumor_TPM": ("Tumor_TPM", "max"),
        "Max_Junction_CPM": ("Junction_CPM", "max"),
    }
    for (patient, peptide), group in frame.groupby(["Patient", "Peptide"], sort=False):
        macro_origins = sorted({value for value in group["Macro_Origin"] if pd.notna(value)})
        novel_micro = sorted(
            {
                value
                for value in group.loc[
                    group["Macro_Origin"].eq("Novel Transcript"), "Micro_Origin"
                ]
                if pd.notna(value)
            }
        )
        record = {
            "Patient": patient,
            "Peptide": peptide,
            "Macro_Origin": macro_origins[0] if len(macro_origins) == 1 else "Multiple Origins",
            "Micro_Origin": (
                novel_micro[0]
                if len(novel_micro) == 1
                else "Multiple Novel Origins" if len(novel_micro) > 1 else "Not Applicable"
            ),
            "Source_Origins": ";".join(macro_origins),
            "Source_Transcripts": ";".join(sorted(set(group["Clean_Transcript_ID"]))),
            "Source_Transcript_Count": group["Clean_Transcript_ID"].nunique(),
            "HLA_Alleles": ";".join(
                sorted(
                    {
                        str(value)
                        for value in group.get("MHC", pd.Series(dtype=str))
                        if pd.notna(value)
                    }
                )
            ),
        }
        for output_column, (input_column, operation) in numeric_specs.items():
            values = pd.to_numeric(
                group.get(input_column, pd.Series(dtype=float)), errors="coerce"
            )
            values = values[values.notna()]
            record[output_column] = getattr(values, operation)() if not values.empty else np.nan
        records.append(record)
    return pd.DataFrame(records)


def add_sharing_statistics(peptides: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_patients = peptides["Patient"].nunique()
    records = []
    for peptide, group in peptides.groupby("Peptide", sort=False):
        records.append(
            {
                "Peptide": peptide,
                "Shared_Patient_Count": group["Patient"].nunique(),
                "Shared_Patient_Ratio": group["Patient"].nunique() / total_patients,
                "Patients": ";".join(sorted(set(group["Patient"]), key=natural_patient_key)),
                "Macro_Origins": ";".join(sorted(set(group["Macro_Origin"]))),
                "Source_Transcripts": ";".join(
                    sorted({item for values in group["Source_Transcripts"] for item in values.split(";")})
                ),
                "HLA_Alleles": ";".join(
                    sorted({item for values in group["HLA_Alleles"] for item in values.split(";") if item})
                ),
                "Best_Affinity_nM": pd.to_numeric(group["Best_Affinity_nM"], errors="coerce").min(),
                "Best_Rank_EL": pd.to_numeric(group["Best_Rank_EL"], errors="coerce").min(),
                "Max_Mean_Intensity": pd.to_numeric(
                    group["Max_Mean_Intensity"], errors="coerce"
                ).max(),
            }
        )
    sharing = pd.DataFrame(records).sort_values(
        ["Shared_Patient_Count", "Best_Rank_EL", "Best_Affinity_nM", "Peptide"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    return peptides.merge(
        sharing[["Peptide", "Shared_Patient_Count", "Shared_Patient_Ratio"]],
        on="Peptide",
        how="left",
    ), sharing


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
    ordered_set = set(ordered)
    return ordered + sorted(
        [patient for patient in observed if patient not in ordered_set],
        key=natural_patient_key,
    )


def summarize(frame: pd.DataFrame, category: str, order: list[str], patients: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=pd.Index(patients, name="Patient"), dtype=int)
    matrix = frame.groupby(["Patient", category], observed=True).size().unstack(fill_value=0)
    matrix = matrix.reindex(index=patients, fill_value=0)
    columns = [value for value in order if value in matrix.columns]
    columns += [value for value in matrix.columns if value not in columns]
    return matrix.reindex(columns=columns, fill_value=0).astype(int)


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


def plot_global_bar(
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
    figure, axis = plt.subplots(figsize=(3.5, max(2.4, len(counts) * 0.32 + 0.8)))
    y_values = np.arange(len(counts))
    axis.barh(
        y_values,
        counts,
        color=[palette.get(value, "#999999") for value in counts.index],
        height=0.72,
    )
    axis.set_yticks(y_values)
    axis.set_yticklabels(counts.index)
    axis.invert_yaxis()
    axis.set_xlabel("Number of unique patient–peptide pairs")
    offset = max(float(counts.max()) * 0.015, 0.2)
    for position, (count, percent) in enumerate(zip(counts, percentages)):
        axis.text(count + offset, position, f"{count:,} ({percent:.1f}%)", va="center", fontsize=6.5)
    axis.set_xlim(0, float(counts.max()) * 1.3 if len(counts) else 1)
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

    values = matrix.astype(float)
    if mode == "percent":
        values = values.div(values.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100.0
        ylabel = "Proportion of unique peptides (%)"
    width = max(7.2, min(22.0, len(values) * width_per_patient))
    figure, axis = plt.subplots(figsize=(width, 4.5))
    x_values = np.arange(len(values))
    bottom = np.zeros(len(values))
    for category in values.columns:
        heights = values[category].to_numpy(dtype=float)
        axis.bar(
            x_values,
            heights,
            bottom=bottom,
            width=0.84,
            color=palette.get(category, "#999999"),
            edgecolor="none",
            label=category,
        )
        bottom += heights
    axis.set_xticks(x_values)
    axis.set_xticklabels(values.index, rotation=90, ha="center", fontsize=5.5)
    axis.set_xlabel("Patient")
    axis.set_ylabel(ylabel)
    axis.set_xlim(-0.7, len(values) - 0.3)
    axis.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6.3)
    axis.tick_params(width=0.7, length=2.5)
    figure.tight_layout()
    save_figure(figure, output_prefix, formats)
    plt.close(figure)


def plot_metrics(frame: pd.DataFrame, output_prefix: Path, formats: Iterable[str]) -> None:
    import matplotlib.pyplot as plt

    specifications = [
        ("Max_Mean_Intensity", "log10(mean translation intensity + 1e-6)", False),
        ("Best_Affinity_nM", "log10(best HLA affinity, nM)", True),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for axis, (column, ylabel, invert) in zip(axes, specifications):
        subset = frame[["Macro_Origin", column]].copy()
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
        minimum = 0 if column == "Max_Mean_Intensity" else np.nextafter(0, 1)
        subset = subset[subset[column].notna() & subset[column].ge(minimum)]
        categories = [value for value in MACRO_ORDER if value in set(subset["Macro_Origin"])]
        if not categories:
            axis.text(0.5, 0.5, "No valid values", ha="center", va="center")
            axis.set_axis_off()
            continue
        # Strict positivity is required for affinity; intensity uses a 1e-6 pseudocount.
        transformed = []
        for category in categories:
            raw = subset.loc[subset["Macro_Origin"].eq(category), column].to_numpy(float)
            transformed.append(np.log10(raw + 1e-6) if column == "Max_Mean_Intensity" else np.log10(raw))
        positions = np.arange(1, len(categories) + 1)
        boxes = axis.boxplot(
            transformed,
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
        for position, points in zip(positions, transformed):
            jitter = np.linspace(-0.18, 0.18, len(points)) if len(points) > 1 else np.zeros(1)
            axis.scatter(
                position + jitter,
                points,
                s=5,
                color="#4D4D4D",
                alpha=0.32,
                linewidths=0,
                rasterized=True,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(categories, rotation=45, ha="right", fontsize=6)
        axis.set_ylabel(ylabel)
        if invert:
            axis.invert_yaxis()
        axis.tick_params(width=0.7, length=2.5)
    figure.tight_layout()
    save_figure(figure, output_prefix, formats)
    plt.close(figure)


def plot_sharing(
    peptide_units: pd.DataFrame,
    sharing: pd.DataFrame,
    patients: list[str],
    top_n: int,
    output_prefix: Path,
    formats: Iterable[str],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    distribution = sharing["Shared_Patient_Count"].value_counts().sort_index()
    axes[0].bar(distribution.index, distribution.values, color="#4C78A8", width=0.8)
    axes[0].set_xlabel("Patients sharing a peptide")
    axes[0].set_ylabel("Number of unique peptides")
    axes[0].tick_params(width=0.7, length=2.5)

    selected = sharing.head(max(1, top_n))["Peptide"].tolist()
    presence = (
        peptide_units[peptide_units["Peptide"].isin(selected)]
        .assign(Present=1)
        .pivot_table(index="Peptide", columns="Patient", values="Present", aggfunc="max", fill_value=0)
        .reindex(index=selected, columns=patients, fill_value=0)
    )
    axes[1].imshow(
        presence.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#F2F2F2", "#C44E52"]),
        vmin=0,
        vmax=1,
    )
    axes[1].set_xlabel("Patient")
    axes[1].set_ylabel("Shared peptide")
    patient_step = max(1, int(np.ceil(len(patients) / 24)))
    patient_ticks = np.arange(0, len(patients), patient_step)
    axes[1].set_xticks(patient_ticks)
    axes[1].set_xticklabels([patients[index] for index in patient_ticks], rotation=90, fontsize=5.2)
    peptide_step = max(1, int(np.ceil(len(selected) / 30)))
    peptide_ticks = np.arange(0, len(selected), peptide_step)
    axes[1].set_yticks(peptide_ticks)
    axes[1].set_yticklabels([selected[index] for index in peptide_ticks], fontsize=5.2)
    axes[1].spines[["top", "right"]].set_visible(True)
    figure.tight_layout()
    save_figure(figure, output_prefix, formats)
    plt.close(figure)


def export_target_patient_peptides(
    peptide_units: pd.DataFrame,
    target_file: Optional[str],
    output_path: Path,
) -> list[str]:
    if not target_file:
        return []
    targets = [
        normalize_patient_id(line)
        for line in Path(target_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    targets = list(dict.fromkeys(targets))
    observed = set(peptide_units["Patient"])
    missing = [patient for patient in targets if patient not in observed]
    if missing:
        raise ValueError(f"Target patients are absent from the antigen reports: {missing}")
    subset = peptide_units[peptide_units["Patient"].isin(targets)]
    counts = subset.groupby("Peptide")["Patient"].nunique()
    shared = set(counts[counts.eq(len(targets))].index)
    subset[subset["Peptide"].isin(shared)].sort_values(
        ["Peptide", "Patient"]
    ).to_csv(output_path, index=False)
    return targets


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports, metrics = load_reports(Path(args.input_dir))
    denovo_ids = read_identifier_file(args.denovo_ids)
    canonical_proteins = read_protein_fasta(args.canonical_protein_fasta, ensembl_headers=True)
    denovo_proteins = read_protein_fasta(args.denovo_protein_fasta)
    class_codes = load_class_codes(args.transcript_meta)
    annotated = annotate_sources(reports, args.reference_gtf, args.novel_gtf, denovo_ids)
    exon_boundaries = load_exon_boundaries(
        [args.reference_gtf, args.novel_gtf],
        set(annotated["Clean_Transcript_ID"]),
    )
    classifications = annotated.apply(
        lambda row: classify_source(
            row,
            canonical_proteins,
            denovo_proteins,
            class_codes,
            exon_boundaries,
        ),
        axis=1,
        result_type="expand",
    )
    classifications.columns = ["Macro_Origin", "Micro_Origin", "Origin_Evidence"]
    annotated = pd.concat([annotated, classifications], axis=1)
    source_keys = ["Patient", "Peptide", "Clean_Transcript_ID"]
    source_keys += [column for column in ("ORF_Pos", "MHC") if column in annotated.columns]
    source_units = annotated.drop_duplicates(source_keys, keep="first").copy()
    peptide_units = collapse_patient_peptides(source_units)
    peptide_units, sharing = add_sharing_statistics(peptide_units)
    patients = resolve_patient_order(peptide_units["Patient"], args.patient_order_file)
    macro_matrix = summarize(peptide_units, "Macro_Origin", MACRO_ORDER, patients)
    novel_units = peptide_units[peptide_units["Source_Origins"].eq("Novel Transcript")].copy()
    micro_matrix = summarize(novel_units, "Micro_Origin", MICRO_ORDER, patients)

    annotated.to_csv(output_dir / "tumor_associated_antigen_hits.annotated.csv", index=False)
    source_units.to_csv(output_dir / "tumor_associated_source_hla_hits.unique.csv", index=False)
    peptide_units.to_csv(output_dir / "tumor_associated_patient_peptides.unique.csv", index=False)
    sharing.to_csv(output_dir / "tumor_associated_shared_peptides.csv", index=False)
    source_units[source_units["Origin_Evidence"].isin({"annotation_not_found", "id_only_not_sequence_verified"})].to_csv(
        output_dir / "unresolved_or_unverified_antigen_sources.csv", index=False
    )
    write_matrix(macro_matrix, output_dir / "antigen_origin_macro_counts")
    write_matrix(micro_matrix, output_dir / "antigen_origin_micro_counts")
    global_rows = []
    for level, frame, category in (
        ("macro", peptide_units, "Macro_Origin"),
        ("micro", novel_units, "Micro_Origin"),
    ):
        counts = frame[category].value_counts()
        total = int(counts.sum())
        for label, count in counts.items():
            global_rows.append(
                {
                    "Level": level,
                    "Category": label,
                    "Count": int(count),
                    "Percent": float(count / total * 100) if total else 0.0,
                }
            )
    pd.DataFrame(global_rows).to_csv(output_dir / "antigen_origin_global_summary.csv", index=False)

    targets = export_target_patient_peptides(
        peptide_units,
        args.target_patients_file,
        output_dir / "shared_peptides_target_patients.csv",
    )
    if not args.no_patient_shards:
        shard_dir = output_dir / "per_patient_landscapes"
        shard_dir.mkdir(exist_ok=True)
        for patient, frame in peptide_units.groupby("Patient", sort=False):
            frame.to_csv(shard_dir / f"{normalize_patient_id(patient)}.csv", index=False)

    metrics.update(
        {
            "Patients": len(patients),
            "Annotated_Hit_Rows": len(annotated),
            "Unique_Source_HLA_Rows": len(source_units),
            "Unique_Patient_Peptide_Rows": len(peptide_units),
            "Unique_Cohort_Peptides": sharing["Peptide"].nunique(),
            "De_Novo_IDs_Loaded": len(denovo_ids),
            "Canonical_Proteins_Loaded": len(canonical_proteins),
            "De_Novo_Proteins_Loaded": len(denovo_proteins),
            "Unresolved_Or_Unverified_Source_Rows": int(
                source_units["Origin_Evidence"].isin(
                    {"annotation_not_found", "id_only_not_sequence_verified"}
                ).sum()
            ),
            "Target_Patients": targets,
            "Stacked_Bar_Mode": args.mode,
        }
    )
    (output_dir / "tumor_associated_neoantigen_analysis_qc.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    if not args.no_plots:
        configure_matplotlib()
        plot_global_bar(
            peptide_units,
            "Macro_Origin",
            MACRO_ORDER,
            MACRO_PALETTE,
            output_dir / "antigen_origin_macro_global",
            args.formats,
        )
        plot_patient_stacked(
            macro_matrix,
            args.mode,
            MACRO_PALETTE,
            "Number of unique tumor-associated peptides",
            output_dir / f"antigen_origin_macro_patient_stacked_{args.mode}",
            args.formats,
            args.width_per_patient,
        )
        if not novel_units.empty:
            plot_global_bar(
                novel_units,
                "Micro_Origin",
                MICRO_ORDER,
                MICRO_PALETTE,
                output_dir / "antigen_origin_micro_global",
                args.formats,
            )
            plot_patient_stacked(
                micro_matrix,
                args.mode,
                MICRO_PALETTE,
                "Number of unique novel-transcript peptides",
                output_dir / f"antigen_origin_micro_patient_stacked_{args.mode}",
                args.formats,
                args.width_per_patient,
            )
        plot_metrics(
            peptide_units,
            output_dir / "antigen_origin_translation_and_affinity",
            args.formats,
        )
        plot_sharing(
            peptide_units,
            sharing,
            patients,
            args.top_shared_peptides,
            output_dir / "shared_peptide_distribution_and_presence",
            args.formats,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

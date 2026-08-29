#!/usr/bin/env python3
"""Shared annotation helpers for cohort-level antigen analyses."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd


LNC_BIOTYPES = {
    "lncrna",
    "linc_rna",
    "antisense",
    "sense_intronic",
    "sense_overlapping",
    "processed_transcript",
    "3prime_overlapping_ncrna",
    "bidirectional_promoter_lncrna",
    "macro_lncrna",
    "non_coding",
}


def clean_transcript_id(value: object) -> str:
    """Remove an Ensembl version suffix while preserving novel transcript IDs."""
    transcript_id = str(value).strip().split("|")[0]
    if transcript_id.startswith("ENS"):
        return transcript_id.split(".")[0]
    if transcript_id.startswith("PB"):
        return transcript_id.split(":", 1)[0]
    return transcript_id


def normalize_patient_id(value: object) -> str:
    """Normalize spaces without changing the biological identifier."""
    return re.sub(r"\s+", "_", str(value).strip())


def natural_patient_key(value: object) -> Tuple:
    """Return a stable natural-sort key for mixed numeric patient identifiers."""
    parts = re.split(r"(\d+)", normalize_patient_id(value).casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def normalize_hla_a(value: object) -> str:
    """Normalize common HLA-A spellings to HLA-A*NN:NN resolution."""
    raw = str(value).strip().upper().replace("_", ":")
    if not raw or raw in {"NAN", "NONE", "NA"}:
        return ""
    raw = raw.replace("HLA-", "")
    match = re.search(r"A\*?(\d{2,3})[:]?([0-9]{2,3})", raw)
    if not match:
        return ""
    return f"HLA-A*{match.group(1)}:{match.group(2)}"


def parse_gtf_attributes(text: str) -> Dict[str, str]:
    """Parse GTF attributes without assuming a fixed attribute order."""
    attributes = {}
    for item in text.strip().strip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if " " in item:
            key, value = item.split(None, 1)
            attributes[key] = value.strip().strip('"')
        elif "=" in item:
            key, value = item.split("=", 1)
            attributes[key] = value.strip().strip('"')
    return attributes


def _map_genomic_position(
    position: int,
    exons: Sequence[Tuple[int, int]],
    strand: str,
) -> Optional[int]:
    """Map a 1-based genomic position to a 0-based spliced-transcript position."""
    ordered = sorted(exons, reverse=(strand == "-"))
    offset = 0
    for exon_start, exon_end in ordered:
        if exon_start <= position <= exon_end:
            if strand == "-":
                return offset + exon_end - position
            return offset + position - exon_start
        offset += exon_end - exon_start + 1
    return None


def load_gtf_annotations(
    gtf_path: str,
    target_transcripts: Optional[Set[str]] = None,
    annotation_kind: str = "reference",
) -> pd.DataFrame:
    """Extract transcript biotypes and canonical start/stop coordinates from a GTF."""
    target_transcripts = (
        {clean_transcript_id(value) for value in target_transcripts}
        if target_transcripts is not None
        else None
    )
    records = defaultdict(
        lambda: {
            "Gene_ID": "",
            "Gene_Name": "",
            "Biotype": "",
            "Strand": "",
            "Exons": [],
            "Start_Codon_Positions": [],
            "Stop_Codon_Positions": [],
            "Is_De_Novo": annotation_kind == "de_novo",
            "Is_Novel_Transcript": annotation_kind == "novel",
        }
    )

    with open(gtf_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature not in {"transcript", "exon", "start_codon", "stop_codon"}:
                continue
            attrs = parse_gtf_attributes(fields[8])
            transcript_id = clean_transcript_id(attrs.get("transcript_id", ""))
            if not transcript_id:
                continue
            if target_transcripts is not None and transcript_id not in target_transcripts:
                continue

            record = records[transcript_id]
            record["Gene_ID"] = attrs.get("gene_id", record["Gene_ID"])
            record["Gene_Name"] = attrs.get("gene_name", record["Gene_Name"])
            record["Biotype"] = (
                attrs.get("transcript_type")
                or attrs.get("transcript_biotype")
                or attrs.get("gene_type")
                or attrs.get("gene_biotype")
                or record["Biotype"]
            )
            record["Strand"] = fields[6]
            start, end = int(fields[3]), int(fields[4])
            if feature == "exon":
                record["Exons"].append((start, end))
            elif feature == "start_codon":
                record["Start_Codon_Positions"].extend(range(start, end + 1))
            elif feature == "stop_codon":
                record["Stop_Codon_Positions"].extend(range(start, end + 1))

    rows = []
    for transcript_id, record in records.items():
        exons = record.pop("Exons")
        start_positions = record.pop("Start_Codon_Positions")
        stop_positions = record.pop("Stop_Codon_Positions")
        strand = record["Strand"]
        mapped_start = [
            mapped
            for position in start_positions
            if (mapped := _map_genomic_position(position, exons, strand)) is not None
        ]
        mapped_stop = [
            mapped
            for position in stop_positions
            if (mapped := _map_genomic_position(position, exons, strand)) is not None
        ]
        rows.append(
            {
                "Transcript_ID": transcript_id,
                **record,
                "Canonical_ORF_Start": min(mapped_start) if mapped_start else pd.NA,
                "Canonical_ORF_Stop": min(mapped_stop) if mapped_stop else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def merge_annotations(
    reference_gtf: str,
    target_transcripts: Set[str],
    de_novo_gtf: Optional[str] = None,
    novel_gtf: Optional[str] = None,
) -> pd.DataFrame:
    """Merge reference and optional de novo annotations with de novo precedence."""
    frames = [load_gtf_annotations(reference_gtf, target_transcripts, "reference")]
    if novel_gtf:
        frames.append(load_gtf_annotations(novel_gtf, target_transcripts, "novel"))
    if de_novo_gtf:
        frames.append(load_gtf_annotations(de_novo_gtf, target_transcripts, "de_novo"))
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=["Transcript_ID"])
    merged = pd.concat(nonempty, ignore_index=True)
    priority = {"reference": 0, "novel": 1, "de_novo": 2}
    merged["_Priority"] = merged.apply(
        lambda row: priority[
            "de_novo" if row["Is_De_Novo"] else "novel" if row["Is_Novel_Transcript"] else "reference"
        ],
        axis=1,
    )
    merged = (
        merged.sort_values("_Priority")
        .drop_duplicates("Transcript_ID", keep="last")
        .drop(columns="_Priority")
    )
    return merged


def transcript_macro_category(row: Mapping[str, object]) -> str:
    """Assign the manuscript-level transcript origin category."""
    transcript_id = clean_transcript_id(row.get("Transcript_ID", ""))
    is_de_novo = row.get("Is_De_Novo", False)
    is_de_novo = False if pd.isna(is_de_novo) else bool(is_de_novo)
    if is_de_novo:
        return "De novo Gene"
    is_novel = row.get("Is_Novel_Transcript", False)
    is_novel = False if pd.isna(is_novel) else bool(is_novel)
    if is_novel or transcript_id.startswith(("MSTRG", "STRG", "PB.")):
        return "Novel Transcript"

    raw_biotype = row.get("Biotype", "")
    biotype = "" if pd.isna(raw_biotype) else str(raw_biotype).strip().casefold()
    if biotype == "protein_coding":
        return "Protein Coding"
    if "pseudogene" in biotype:
        return "Pseudogene"
    if biotype in LNC_BIOTYPES or "lncrna" in biotype or "linc" in biotype:
        return "lncRNA"
    if biotype:
        return "Other ncRNA"
    if transcript_id.startswith("ENST"):
        return "Unknown ENST"
    return "Novel Transcript"


def parse_orf_position(value: object) -> Tuple[Optional[int], Optional[int]]:
    """Parse the report's start:stop ORF coordinate field."""
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", str(value))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def antigen_origin_category(row: Mapping[str, object], tolerance: int = 6) -> str:
    """Classify a peptide source while keeping unresolved CDS status explicit."""
    macro = transcript_macro_category(row)
    if macro in {"De novo Gene", "Novel Transcript"}:
        return "Novel Transcript"
    if macro in {"Pseudogene", "lncRNA", "Other ncRNA"}:
        return macro
    if macro == "Unknown ENST":
        return "Unresolved ENST"

    orf_start, orf_stop = parse_orf_position(row.get("ORF_Pos", ""))
    canonical_start = pd.to_numeric(row.get("Canonical_ORF_Start"), errors="coerce")
    canonical_stop = pd.to_numeric(row.get("Canonical_ORF_Stop"), errors="coerce")
    if orf_start is None or not math.isfinite(canonical_start) or not math.isfinite(canonical_stop):
        return "Unresolved protein-coding ORF"
    if abs(orf_start - canonical_start) <= tolerance and abs(orf_stop - canonical_stop) <= tolerance:
        return "Canonical CDS"
    return "Cryptic ORF"


def allele_carrier_probability(allele_frequency: float) -> float:
    """Convert allele frequency to carrier probability under Hardy-Weinberg equilibrium."""
    frequency = min(max(float(allele_frequency), 0.0), 1.0)
    return 1.0 - (1.0 - frequency) ** 2


def population_coverage(covered_allele_frequency: float) -> float:
    """Approximate HLA-A genotype coverage from the summed covered allele frequency."""
    return allele_carrier_probability(min(float(covered_allele_frequency), 1.0))

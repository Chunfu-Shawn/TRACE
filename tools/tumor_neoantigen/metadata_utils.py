#!/usr/bin/env python3
"""Shared metadata label normalization and patient-run lookup utilities."""

import argparse
import csv
import re
import sys


def classify_tissue(value):
    """Map common tumor/normal labels to the pipeline's canonical classes."""
    label = str(value).strip().lower()
    normalized = re.sub(r"[_-]+", " ", label)
    normalized = re.sub(r"\s+", " ", normalized)

    normal_patterns = (
        r"\bnormal\b",
        r"\bnon\s*tumou?r\b",
        r"\badjacent\b",
        r"\bbenign\b",
        r"\bhealthy\b",
        r"\bcontrol\b",
        r"\bpara\s*tumou?r\b",
        r"\bparatumou?r\b",
    )
    if any(re.search(pattern, normalized) for pattern in normal_patterns):
        return "normal"

    tumor_patterns = (
        r"\btumou?r\b",
        r"\bmalignant\b",
        r"\bcancer\b",
        r"\bcarcinoma\b",
    )
    if any(re.search(pattern, normalized) for pattern in tumor_patterns):
        return "tumor"

    return "unknown"


def _find_column(fieldnames, aliases, contains=()):
    """Resolve a metadata column by case-insensitive aliases or substrings."""
    normalized = {str(name).strip().lower(): name for name in fieldnames or []}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for lowered, original in normalized.items():
        if any(token in lowered for token in contains):
            return original
    return None


def find_patient_runs(metadata_file, patient, tissue_class):
    """Return run IDs matching one patient and a canonical tissue class."""
    with open(metadata_file, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = reader.fieldnames or []
        run_col = _find_column(fieldnames, ("run", "run_id", "run accession"), ("run",))
        patient_col = _find_column(
            fieldnames,
            ("individual", "patient", "patient_id", "subject", "subject_id"),
            ("individual", "patient", "subject"),
        )
        tissue_col = _find_column(
            fieldnames,
            ("tissue", "tissue_type", "sample_type"),
            ("tissue", "sample type"),
        )
        missing = [
            name
            for name, column in (("run", run_col), ("patient", patient_col), ("tissue", tissue_col))
            if column is None
        ]
        if missing:
            raise ValueError(
                f"Metadata is missing required column(s): {', '.join(missing)}. "
                f"Available columns: {fieldnames}"
            )

        patient_key = str(patient).strip().casefold()
        runs = []
        for row in reader:
            row_patient = str(row.get(patient_col, "")).strip().casefold()
            if row_patient != patient_key:
                continue
            if classify_tissue(row.get(tissue_col, "")) != tissue_class:
                continue
            run_id = str(row.get(run_col, "")).strip()
            if run_id:
                runs.append(run_id)
        return list(dict.fromkeys(runs))


def main():
    parser = argparse.ArgumentParser(description="Resolve a patient run from metadata.")
    parser.add_argument("--metadata", required=True, help="CSV or TSV metadata file")
    parser.add_argument("--patient", required=True, help="Patient/individual identifier")
    parser.add_argument("--tissue", required=True, choices=("tumor", "normal"))
    args = parser.parse_args()

    try:
        runs = find_patient_runs(args.metadata, args.patient, args.tissue)
    except (OSError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    if len(runs) > 1:
        print(
            f"[Warning] Multiple {args.tissue} runs found for {args.patient}; using {runs[0]}.",
            file=sys.stderr,
        )
    if runs:
        print(runs[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

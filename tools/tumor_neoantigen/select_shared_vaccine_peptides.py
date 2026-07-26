#!/usr/bin/env python3
"""Select a shared HCC peptide panel using cohort and HLA-A coverage."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cohort_annotation_utils import (
    natural_patient_key,
    normalize_hla_a,
    normalize_patient_id,
    population_coverage,
)


def parse_args() -> argparse.Namespace:
    default_frequency = Path(__file__).with_name("hla_a_cwd22_common.csv")
    parser = argparse.ArgumentParser(
        description="Greedy selection of recurrent peptides with patient and HLA-A coverage."
    )
    parser.add_argument("--antigen_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hla_frequency_csv", default=str(default_frequency))
    parser.add_argument("--hla_typing_csv", help="Optional cohort HLA table with patient and HLA-A columns.")
    parser.add_argument("--patient_list_file", help="Optional one-patient-per-line cohort denominator.")
    parser.add_argument("--pan_hla_predictions", help="Optional CSV/TSV with Peptide, MHC and binding metrics.")
    parser.add_argument("--n_peptides", type=int, default=20)
    parser.add_argument("--min_patients", type=int, default=2)
    parser.add_argument("--max_rank_el", type=float, default=2.0)
    parser.add_argument("--max_aff_nm", type=float)
    parser.add_argument("--patient_weight", type=float, default=0.50)
    parser.add_argument("--population_weight", type=float, default=0.20)
    parser.add_argument("--hla_quota_weight", type=float, default=0.20)
    parser.add_argument("--recurrence_weight", type=float, default=0.08)
    parser.add_argument("--binding_weight", type=float, default=0.02)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt", ".xls"} else ","
    return pd.read_csv(path, sep=separator)


def find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    lowered = {str(column).strip().casefold(): column for column in frame.columns}
    for alias in aliases:
        if alias.casefold() in lowered:
            return lowered[alias.casefold()]
    return None


def standardize_binding_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    peptide_col = find_column(frame, ("Peptide", "pep"))
    hla_col = find_column(frame, ("MHC", "HLA", "Allele"))
    if not peptide_col or not hla_col:
        raise ValueError(f"{label} requires Peptide and MHC/HLA/Allele columns.")
    rename = {peptide_col: "Peptide", hla_col: "HLA"}
    rank_col = find_column(frame, ("%Rank_EL", "Rank_EL", "EL_Rank", "rank"))
    affinity_col = find_column(frame, ("Aff(nM)", "Affinity", "Affinity_nM"))
    if rank_col:
        rename[rank_col] = "Rank_EL"
    if affinity_col:
        rename[affinity_col] = "Affinity_nM"
    result = frame.rename(columns=rename).copy()
    result["Peptide"] = result["Peptide"].astype(str).str.strip().str.upper()
    result["HLA"] = result["HLA"].map(normalize_hla_a)
    result = result[result["Peptide"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", na=False)]
    result = result[result["HLA"].ne("")]
    if "Rank_EL" in result:
        result["Rank_EL"] = pd.to_numeric(result["Rank_EL"], errors="coerce")
    if "Affinity_nM" in result:
        result["Affinity_nM"] = pd.to_numeric(result["Affinity_nM"], errors="coerce")
    return result


def parse_netmhcpan_log(path: Path) -> pd.DataFrame:
    """Parse the standard whitespace netMHCpan EL+BA output."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 16 or not parts[0].isdigit():
                continue
            try:
                records.append(
                    {
                        "Peptide": parts[2],
                        "HLA": parts[1],
                        "Rank_EL": float(parts[12]),
                        "Affinity_nM": float(parts[15]),
                    }
                )
            except (ValueError, IndexError):
                continue
    if not records:
        raise ValueError(f"No netMHCpan prediction rows could be parsed from {path}")
    result = pd.DataFrame(records)
    result["HLA"] = result["HLA"].map(normalize_hla_a)
    return result[result["HLA"].ne("")]


def load_binding_predictions(path: Path) -> pd.DataFrame:
    """Load a long binding table, falling back to raw netMHCpan output."""
    try:
        return standardize_binding_table(read_table(path), "Pan-HLA predictions")
    except (ValueError, pd.errors.ParserError):
        return parse_netmhcpan_log(path)


def load_antigen_reports(directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted([*directory.glob("*.csv"), *directory.glob("*.tsv")]):
        frame = read_table(path)
        if frame.empty:
            continue
        patient_values = frame["Patient"] if "Patient" in frame.columns else path.stem
        frame["Patient"] = pd.Series(patient_values, index=frame.index).map(normalize_patient_id)
        frame["Source_File"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError(f"No non-empty antigen reports found in {directory}")
    combined = pd.concat(frames, ignore_index=True)
    binding = standardize_binding_table(combined, "Antigen reports")
    binding["Patient"] = combined.loc[binding.index, "Patient"]
    return binding


def filter_binders(frame: pd.DataFrame, max_rank: float, max_affinity: Optional[float]) -> pd.DataFrame:
    result = frame.copy()
    if "Rank_EL" in result.columns:
        result = result[result["Rank_EL"].isna() | result["Rank_EL"].le(max_rank)]
    if max_affinity is not None and "Affinity_nM" in result.columns:
        result = result[result["Affinity_nM"].isna() | result["Affinity_nM"].le(max_affinity)]
    return result


def load_hla_frequencies(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    allele_col = find_column(frame, ("Allele", "HLA", "MHC"))
    percent_col = find_column(frame, ("Allele_Frequency_Percent", "Frequency_Percent", "Percent"))
    fraction_col = find_column(frame, ("Allele_Frequency", "Frequency"))
    if not allele_col or (not percent_col and not fraction_col):
        raise ValueError("HLA frequency file needs Allele plus a percent or fractional frequency column.")
    result = pd.DataFrame({"HLA": frame[allele_col].map(normalize_hla_a)})
    if percent_col:
        result["Allele_Frequency"] = pd.to_numeric(frame[percent_col], errors="coerce") / 100.0
    else:
        result["Allele_Frequency"] = pd.to_numeric(frame[fraction_col], errors="coerce")
    result = result.dropna().query("HLA != '' and Allele_Frequency > 0")
    return result.groupby("HLA", as_index=False)["Allele_Frequency"].sum()


def load_patient_hlas(path: Path) -> dict[str, set[str]]:
    frame = read_table(path)
    patient_col = find_column(frame, ("Patient", "Patient_ID", "Individual", "Subject"))
    if not patient_col:
        raise ValueError("HLA typing table needs a Patient/Individual/Subject column.")
    hla_columns = []
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).casefold())
        if normalized.startswith("hlaa") or normalized in {"a1", "a2", "allelea1", "allelea2"}:
            hla_columns.append(column)
    if not hla_columns:
        raise ValueError("No HLA-A allele columns were detected in the HLA typing table.")
    patient_hlas: dict[str, set[str]] = defaultdict(set)
    for _, row in frame.iterrows():
        patient = normalize_patient_id(row[patient_col])
        for column in hla_columns:
            allele = normalize_hla_a(row[column])
            if allele:
                patient_hlas[patient].add(allele)
    return dict(patient_hlas)


def largest_remainder_quotas(frequencies: dict[str, float], panel_size: int) -> dict[str, int]:
    total = sum(frequencies.values())
    exact = {allele: panel_size * frequency / total for allele, frequency in frequencies.items()}
    quotas = {allele: int(math.floor(value)) for allele, value in exact.items()}
    remaining = panel_size - sum(quotas.values())
    order = sorted(exact, key=lambda allele: (exact[allele] - quotas[allele], frequencies[allele]), reverse=True)
    for allele in order[:remaining]:
        quotas[allele] += 1
    return quotas


def binding_quality(group: pd.DataFrame, max_rank: float) -> float:
    if "Rank_EL" not in group.columns or group["Rank_EL"].dropna().empty:
        return 0.5
    best_rank = float(group["Rank_EL"].min())
    return max(0.0, 1.0 - best_rank / max(max_rank, 1e-9))


def main() -> int:
    args = parse_args()
    if args.n_peptides < 1:
        raise ValueError("--n_peptides must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed = filter_binders(
        load_antigen_reports(Path(args.antigen_dir)), args.max_rank_el, args.max_aff_nm
    )
    cohort_patients = set(observed["Patient"])
    patient_hlas = load_patient_hlas(Path(args.hla_typing_csv)) if args.hla_typing_csv else {}
    cohort_patients.update(patient_hlas)
    if args.patient_list_file:
        cohort_patients.update(
            normalize_patient_id(line)
            for line in Path(args.patient_list_file).read_text().splitlines()
            if line.strip()
        )
    production = observed[["Patient", "Peptide"]].drop_duplicates()
    recurrence = production.groupby("Peptide")["Patient"].nunique()
    candidate_peptides = set(recurrence[recurrence.ge(args.min_patients)].index)
    if not candidate_peptides:
        raise ValueError("No peptide passed --min_patients after binding filters.")
    observed = observed[observed["Peptide"].isin(candidate_peptides)]
    production = production[production["Peptide"].isin(candidate_peptides)]

    binding_frames = [observed]
    pan_hla_used = bool(args.pan_hla_predictions)
    if pan_hla_used:
        pan = load_binding_predictions(Path(args.pan_hla_predictions))
        pan = filter_binders(pan, args.max_rank_el, args.max_aff_nm)
        binding_frames.append(pan[pan["Peptide"].isin(candidate_peptides)])
    bindings = pd.concat(binding_frames, ignore_index=True).drop_duplicates(["Peptide", "HLA"])

    hla_frequency_frame = load_hla_frequencies(Path(args.hla_frequency_csv))
    hla_frequencies = dict(zip(hla_frequency_frame["HLA"], hla_frequency_frame["Allele_Frequency"]))
    quotas = largest_remainder_quotas(hla_frequencies, args.n_peptides)
    peptide_hlas = bindings.groupby("Peptide")["HLA"].apply(set).to_dict()
    peptide_hlas = {
        peptide: {allele for allele in alleles if allele in hla_frequencies}
        for peptide, alleles in peptide_hlas.items()
    }

    patients = sorted(cohort_patients, key=natural_patient_key)
    production_by_peptide = production.groupby("Peptide")["Patient"].apply(set).to_dict()
    if patient_hlas:
        responsive_by_peptide = {
            peptide: {
                patient for patient in source_patients
                if patient_hlas.get(patient, set()).intersection(peptide_hlas.get(peptide, set()))
            }
            for peptide, source_patients in production_by_peptide.items()
        }
    else:
        responsive_by_peptide = (
            observed.groupby("Peptide")["Patient"].apply(set).to_dict()
        )

    quality = {
        peptide: binding_quality(group, args.max_rank_el)
        for peptide, group in bindings.groupby("Peptide")
    }
    selected: list[str] = []
    covered_patients: set[str] = set()
    covered_hlas: set[str] = set()
    hla_peptide_counts: dict[str, int] = defaultdict(int)
    selection_rows = []
    total_frequency = sum(hla_frequencies.values())

    for selection_rank in range(1, min(args.n_peptides, len(candidate_peptides)) + 1):
        best = None
        for peptide in sorted(candidate_peptides.difference(selected)):
            response_patients = responsive_by_peptide.get(peptide, set())
            bound_hlas = peptide_hlas.get(peptide, set())
            patient_gain = len(response_patients - covered_patients) / max(len(patients), 1)
            before_frequency = sum(hla_frequencies.get(allele, 0.0) for allele in covered_hlas)
            after_frequency = sum(hla_frequencies.get(allele, 0.0) for allele in covered_hlas | bound_hlas)
            population_gain = population_coverage(after_frequency) - population_coverage(before_frequency)
            quota_gain = sum(
                hla_frequencies.get(allele, 0.0)
                for allele in bound_hlas
                if hla_peptide_counts[allele] < quotas.get(allele, 0)
            ) / max(total_frequency, 1e-9)
            recurrence_score = len(production_by_peptide.get(peptide, set())) / max(len(patients), 1)
            score = (
                args.patient_weight * patient_gain
                + args.population_weight * population_gain
                + args.hla_quota_weight * quota_gain
                + args.recurrence_weight * recurrence_score
                + args.binding_weight * quality.get(peptide, 0.0)
            )
            tie_breaker = (
                score,
                patient_gain,
                quota_gain,
                recurrence_score,
                quality.get(peptide, 0.0),
                peptide,
            )
            if best is None or tie_breaker > best[0]:
                best = (tie_breaker, peptide, score, patient_gain, population_gain, quota_gain)
        if best is None:
            break

        _, peptide, score, patient_gain, population_gain, quota_gain = best
        selected.append(peptide)
        new_patient_count = len(responsive_by_peptide.get(peptide, set()) - covered_patients)
        covered_patients.update(responsive_by_peptide.get(peptide, set()))
        covered_hlas.update(peptide_hlas.get(peptide, set()))
        for allele in peptide_hlas.get(peptide, set()):
            hla_peptide_counts[allele] += 1
        covered_frequency = sum(hla_frequencies.get(allele, 0.0) for allele in covered_hlas)
        peptide_binding = bindings[bindings["Peptide"].eq(peptide)]
        selection_rows.append(
            {
                "Selection_Rank": selection_rank,
                "Peptide": peptide,
                "Cohort_Source_Patients": len(production_by_peptide.get(peptide, set())),
                "Cohort_Responsive_Patients": len(responsive_by_peptide.get(peptide, set())),
                "New_Patients_Added": new_patient_count,
                "Cumulative_Patients_Covered": len(covered_patients),
                "Cumulative_Cohort_Coverage": len(covered_patients) / max(len(patients), 1),
                "Bound_Common_HLA_A": ";".join(sorted(peptide_hlas.get(peptide, set()))),
                "Cumulative_HLA_A_Allele_Frequency": covered_frequency,
                "Cumulative_HLA_A_Carrier_Coverage": population_coverage(covered_frequency),
                "Best_Rank_EL": peptide_binding["Rank_EL"].min() if "Rank_EL" in peptide_binding else np.nan,
                "Greedy_Score": score,
                "Patient_Gain_Component": patient_gain,
                "Population_Gain_Component": population_gain,
                "HLA_Quota_Component": quota_gain,
                "Pan_HLA_Predictions_Used": pan_hla_used,
            }
        )

    selected_frame = pd.DataFrame(selection_rows)
    selected_frame.to_csv(output_dir / "selected_shared_vaccine_peptides.csv", index=False)
    selected_frame[
        [
            "Selection_Rank",
            "Cumulative_Patients_Covered",
            "Cumulative_Cohort_Coverage",
            "Cumulative_HLA_A_Allele_Frequency",
            "Cumulative_HLA_A_Carrier_Coverage",
        ]
    ].to_csv(output_dir / "panel_coverage_curve.csv", index=False)
    Path(output_dir / "selected_peptides.txt").write_text("\n".join(selected) + "\n")
    Path(output_dir / "pan_hla_candidate_peptides.txt").write_text(
        "\n".join(sorted(candidate_peptides)) + "\n"
    )
    Path(output_dir / "cwd22_hla_a_alleles.txt").write_text("\n".join(sorted(hla_frequencies)) + "\n")
    Path(output_dir / "cwd22_netmhcpan_alleles.txt").write_text(
        ",".join(sorted(allele.replace("*", "") for allele in hla_frequencies)) + "\n"
    )

    candidate_rows = []
    for peptide in sorted(candidate_peptides):
        peptide_binding = bindings[bindings["Peptide"].eq(peptide)]
        candidate_rows.append(
            {
                "Peptide": peptide,
                "Source_Patient_Count": len(production_by_peptide.get(peptide, set())),
                "Responsive_Patient_Count": len(responsive_by_peptide.get(peptide, set())),
                "Bound_Common_HLA_A": ";".join(sorted(peptide_hlas.get(peptide, set()))),
                "Best_Rank_EL": peptide_binding["Rank_EL"].min() if "Rank_EL" in peptide_binding else np.nan,
                "Selected": peptide in selected,
            }
        )
    pd.DataFrame(candidate_rows).to_csv(output_dir / "shared_peptide_candidates.csv", index=False)

    patient_rows = []
    for patient in patients:
        covering = [peptide for peptide in selected if patient in responsive_by_peptide.get(peptide, set())]
        patient_rows.append(
            {
                "Patient": patient,
                "Covered": bool(covering),
                "Covering_Peptide_Count": len(covering),
                "Covering_Peptides": ";".join(covering),
                "Patient_HLA_A": ";".join(sorted(patient_hlas.get(patient, set()))),
            }
        )
    patient_coverage_frame = pd.DataFrame(patient_rows)
    patient_coverage_frame.to_csv(output_dir / "patient_panel_coverage.csv", index=False)
    patient_coverage_frame.loc[~patient_coverage_frame["Covered"]].to_csv(
        output_dir / "uncovered_patients.csv", index=False
    )

    hla_rows = []
    for allele, frequency in sorted(hla_frequencies.items(), key=lambda item: item[1], reverse=True):
        hla_rows.append(
            {
                "HLA": allele,
                "Allele_Frequency": frequency,
                "Approximate_Carrier_Probability": population_coverage(frequency),
                "Target_Peptide_Quota": quotas.get(allele, 0),
                "Selected_Binding_Peptides": hla_peptide_counts.get(allele, 0),
                "Quota_Met": hla_peptide_counts.get(allele, 0) >= quotas.get(allele, 0),
            }
        )
    pd.DataFrame(hla_rows).to_csv(output_dir / "hla_a_frequency_quota_coverage.csv", index=False)

    metadata = pd.DataFrame(
        [
            {"Key": "Pan_HLA_Predictions_Used", "Value": pan_hla_used},
            {"Key": "Cohort_HLA_Typing_Used", "Value": bool(patient_hlas)},
            {"Key": "Cohort_Patient_Count", "Value": len(patients)},
            {"Key": "Candidate_Peptide_Count", "Value": len(candidate_peptides)},
            {"Key": "Selected_Peptide_Count", "Value": len(selected)},
            {"Key": "HLA_Carrier_Coverage_Is_Response_Rate", "Value": False},
        ]
    )
    metadata.to_csv(output_dir / "selection_run_metadata.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

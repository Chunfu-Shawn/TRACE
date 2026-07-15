#!/usr/bin/env python3
"""
Filter neoepitope candidates against each patient's own normal-tissue proteome.

For every patient's epitope report CSV, this script loads the corresponding
TRACE-predicted normal-tissue protein FASTA (produced by run_trace_prediction.py
run on normal-expressed transcripts), checks whether each candidate peptide
exists as a substring in any normal-expressed protein, and removes peptides
that match.  This catches:

  - Alternate isoforms of the same gene expressed in normal tissue
  - Homologous genes that produce identical peptide sequences
  - Novel transcripts expressed in normal tissue that share peptide sequences

Peptides that survive this filter are written to a new output directory and
proceed to the canonical proteome filter (filter_canonical_offtargets.py).
"""
import os
import re
import sys
import glob
import argparse


def clean_id(tid: str) -> str:
    tid_str = str(tid).strip()
    if tid_str.startswith("ENS"):
        return tid_str.split(".")[0]
    return tid_str


def read_fasta(file_path: str) -> list:
    """Read a protein FASTA and return a list of (header, sequence) tuples."""
    entries = []
    if not os.path.exists(file_path):
        return entries

    curr_header = ""
    curr_seq = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr_header:
                    entries.append((curr_header, "".join(curr_seq)))
                curr_header = line[1:]
                curr_seq = []
            else:
                curr_seq.append(line.upper())
        if curr_header:
            entries.append((curr_header, "".join(curr_seq)))
    return entries


def build_peptide_index(proteins: list, min_len: int = 8, max_len: int = 11) -> set:
    """
    Build a set of all k-mer peptides (k = min_len..max_len) from a list
    of protein sequences.
    """
    peptide_set = set()
    for _header, seq in proteins:
        seq_len = len(seq)
        for k in range(min_len, max_len + 1):
            if seq_len < k:
                continue
            for i in range(seq_len - k + 1):
                peptide_set.add(seq[i:i + k])
    return peptide_set


def extract_patient_id(filename: str) -> str:
    """Extract patient identifier from filename."""
    basename = os.path.basename(filename)
    match = re.search(r"(patient_?\d+)", basename, re.IGNORECASE)
    if match:
        return match.group(1)
    return basename.rsplit(".", 1)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Filter neoepitope candidates against patient-specific normal proteomes."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing per-patient epitope report CSVs.")
    parser.add_argument("--trace_base_dir", required=True,
                        help="Base directory containing per-patient TRACE output subdirs "
                             "(e.g., ${WORK_DIR}/translation/).")
    parser.add_argument("--trace_mode", default="short",
                        help="TRACE prediction mode, used to locate FASTA files (default: short).")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write filtered per-patient reports.")

    parser.add_argument("--min_pep_len", type=int, default=8,
                        help="Minimum peptide length for k-mer index (default: 8).")
    parser.add_argument("--max_pep_len", type=int, default=11,
                        help="Maximum peptide length for k-mer index (default: 11).")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Discover patient epitope reports
    search_csv = os.path.join(args.input_dir, "*.csv")
    patient_files = glob.glob(search_csv)

    if not patient_files:
        print(f"[Error] No CSV files found in {args.input_dir}")
        sys.exit(1)

    print(f"--- Found {len(patient_files)} patient epitope report(s) ---")

    total_before = 0
    total_after = 0

    for csv_path in sorted(patient_files):
        patient_id = extract_patient_id(csv_path)
        out_path = os.path.join(args.output_dir, os.path.basename(csv_path))

        # Locate the patient's normal proteome FASTA from TRACE output.
        fasta_path = os.path.join(
            args.trace_base_dir, patient_id,
            f"high_confidence_proteins.{patient_id}.{args.trace_mode}_mode.fasta"
        )

        if not os.path.exists(fasta_path):
            print(f"\n[Warning] No normal proteome found for {patient_id} "
                  f"(expected: {fasta_path}). Copying report unchanged.")
            with open(csv_path, "r") as fin, open(out_path, "w") as fout:
                fout.write(fin.read())
            continue

        # 2. Build normal k-mer index for this patient
        print(f"\n--- Processing {patient_id} ---")
        print(f" -> Loading normal proteome: {os.path.basename(fasta_path)}")
        normal_proteins = read_fasta(fasta_path)
        print(f" -> {len(normal_proteins)} protein entries loaded.")

        if not normal_proteins:
            print(" -> Empty proteome; copying report unchanged.")
            with open(csv_path, "r") as fin, open(out_path, "w") as fout:
                fout.write(fin.read())
            continue

        normal_peptides = build_peptide_index(
            normal_proteins, min_len=args.min_pep_len, max_len=args.max_pep_len
        )
        print(f" -> Built k-mer index with {len(normal_peptides)} unique peptides "
              f"(k={args.min_pep_len}-{args.max_pep_len}).")

        # 3. Filter the epitope report
        n_before = 0
        n_removed = 0

        with open(csv_path, "r") as fin, open(out_path, "w") as fout:
            header = fin.readline()
            fout.write(header)

            for line in fin:
                n_before += 1
                line = line.strip()
                if not line:
                    continue
                peptide = line.split(",")[0].strip()
                if peptide in normal_peptides:
                    n_removed += 1
                else:
                    fout.write(line + "\n")

        n_after = n_before - n_removed
        print(f" -> Before: {n_before}  |  Removed (present in normal proteome): {n_removed}")
        print(f" -> After:  {n_after}  ->  {out_path}")

        total_before += n_before
        total_after += n_after

    # 4. Summary
    removal_pct = (1 - total_after / max(total_before, 1)) * 100
    print(f"\n{'='*60}")
    print(f" Normal Proteome Filter Summary")
    print(f"  Total peptides before:  {total_before}")
    print(f"  Removed (normal match): {total_before - total_after}")
    print(f"  Retained:               {total_after}  ({100 - removal_pct:.1f}%)")
    print(f"  Output directory:       {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

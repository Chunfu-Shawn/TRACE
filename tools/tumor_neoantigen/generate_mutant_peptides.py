#!/usr/bin/env python3
"""
Generate 21aa mutant and reference peptide windows for netMHCpan prediction.

For each nonsynonymous mutation annotated by annotate_mutation_variants.py,
this script retrieves the reference protein sequence from TRACE-predicted
proteins (primary) or canonical proteome (fallback), substitutes the mutant
amino acid, and exports both the reference (wildtype) and mutant 21-aa window
centered on the mutation.  Only mutant windows are sent to netMHCpan; the
reference window is included in the mapping CSV for comparison.

Usage:
    python generate_mutant_peptides.py \
        --annotated_csv patient_annotated_variants.csv \
        --trace_fasta high_confidence_proteins.patient.short_mode.fasta \
        --canonical_fasta gencode.v49.pc_translations.fa \
        --output_fasta mutant_peptides.fasta \
        --output_csv mutant_peptide_map.csv
"""
import os, sys, argparse, pandas as pd

def safe_clean_id(tid):
    tid_str = str(tid).strip()
    if tid_str.startswith('ENS'): return tid_str.split('.')[0]
    return tid_str

def read_fasta(path):
    seqs = {}
    if not os.path.exists(path): return seqs
    curr_id = ""; curr_seq = []; curr_header = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if curr_id:
                    seqs[safe_clean_id(curr_id)] = (curr_header, "".join(curr_seq))
                curr_header = line[1:]
                raw_id = line[1:].split()[0].split('|')[0]
                curr_id = raw_id
                curr_seq = []
            else: curr_seq.append(line.upper())
        if curr_id: seqs[safe_clean_id(curr_id)] = (curr_header, "".join(curr_seq))
    return seqs

def extract_window(protein_seq, aa_pos_1based, window=21):
    """Extract a window of `window` aa centered on aa_pos_1based (1-based)."""
    center = aa_pos_1based - 1  # 0-based
    half = window // 2
    start = max(0, center - half)
    end = min(len(protein_seq), start + window)
    seq = protein_seq[start:end]
    # Pad with 'X' if window extends beyond protein boundaries
    if len(seq) < window:
        left_pad = max(0, half - center)
        right_pad = window - len(seq) - left_pad
        seq = 'X' * left_pad + seq + 'X' * right_pad
    return seq, start + 1  # 1-based start

def main():
    p = argparse.ArgumentParser(
        description="Generate 21aa mutant peptide windows for netMHCpan.")
    p.add_argument("--annotated_csv", required=True, help="CSV from annotate_mutation_variants.py")
    p.add_argument("--trace_fasta", required=True,
                   help="TRACE-predicted protein FASTA (primary protein source)")
    p.add_argument("--canonical_fasta", default=None,
                   help="Canonical proteome FASTA (fallback)")
    p.add_argument("--output_fasta", required=True, help="Output FASTA of 21aa mutant windows")
    p.add_argument("--output_csv", required=True, help="Output CSV with ref and mut peptides")
    p.add_argument("--window", type=int, default=21)
    args = p.parse_args()

    print("--- Loading annotated variants ---")
    df = pd.read_csv(args.annotated_csv)
    nonsyn = df[df['Mutation_Type'] == 'nonsynonymous'].copy()
    print(f"Nonsynonymous mutations: {len(nonsyn)}")
    if nonsyn.empty:
        open(args.output_fasta, 'w').close()
        pd.DataFrame().to_csv(args.output_csv, index=False)
        return

    print("\n--- Loading TRACE protein sequences ---")
    trace_prots = read_fasta(args.trace_fasta)
    print(f"TRACE proteins: {len(trace_prots)}")

    canonical_prots = {}
    if args.canonical_fasta:
        print("--- Loading canonical proteome (fallback) ---")
        canonical_prots = read_fasta(args.canonical_fasta)
        print(f"Canonical proteins: {len(canonical_prots)}")

    print("\n--- Generating 21aa ref/mut windows ---")
    fasta_out = []
    peptide_rows = []
    n_found = 0; n_miss = 0

    for _, row in nonsyn.iterrows():
        tx_id = str(row['Transcript_ID']).strip()
        tx_clean = safe_clean_id(tx_id)
        aa_pos = int(row['Codon_Pos'])
        ref_aa = str(row['Ref_AA']).strip()
        alt_aa = str(row['Alt_AA']).strip()
        gene = str(row.get('Gene_Name', 'Unknown'))
        aa_change = str(row.get('AA_Change', ''))

        # Look up reference protein
        prot_entry = (trace_prots.get(tx_clean) or canonical_prots.get(tx_clean)
                      or trace_prots.get(tx_id) or canonical_prots.get(tx_id))
        if not prot_entry:
            n_miss += 1
            continue

        prot_header, prot_seq = prot_entry
        if aa_pos > len(prot_seq):
            n_miss += 1
            continue

        # Reference peptide window (wildtype)
        ref_window, win_start = extract_window(prot_seq, aa_pos, args.window)

        # Mutant peptide window
        mut_seq = prot_seq[:aa_pos - 1] + alt_aa + prot_seq[aa_pos:]
        mut_window, _ = extract_window(mut_seq, aa_pos, args.window)

        # FASTA: write only mutant window (for netMHCpan)
        fasta_header = f">{tx_clean}|{gene}|{aa_change}"
        fasta_out.append(f"{fasta_header}\n{mut_window}\n")

        peptide_rows.append({
            'Transcript_ID': tx_id,
            'Gene': gene,
            'AA_Change': aa_change,
            'Codon_Pos': aa_pos,
            'Ref_AA': ref_aa,
            'Alt_AA': alt_aa,
            'Ref_Peptide': ref_window,
            'Mut_Peptide': mut_window,
            'Window_Start': win_start,
            'Window_End': win_start + args.window - 1,
            'Chr': row.get('Chrom', ''),
            'Pos': row.get('Pos', ''),
            'Ref': row.get('Ref', ''),
            'Alt': row.get('Alt', ''),
        })
        n_found += 1

    os.makedirs(os.path.dirname(args.output_fasta) or '.', exist_ok=True)
    with open(args.output_fasta, 'w') as f:
        f.writelines(fasta_out)
    print(f" -> {n_found} mutant windows -> {args.output_fasta}")

    pep_df = pd.DataFrame(peptide_rows)
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    pep_df.to_csv(args.output_csv, index=False)
    print(f" -> {len(pep_df)} mappings -> {args.output_csv}")
    print(f" -> Missed (no protein): {n_miss}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Integrate mutation neoantigen predictions into a final report.

Combines:
  - Annotated nonsynonymous mutations
  - TRACE translation predictions (TPM, mean intensity)
  - netMHCpan HLA binding results
  - Peptide-to-mutation mapping

Produces a per-patient CSV with columns mirroring the noncanonical pipeline format,
enabling direct comparison between noncanonical and mutation-derived neoantigens.
"""
import os, re, sys, argparse, pandas as pd

def safe_clean_id(tid):
    tid_str = str(tid).strip()
    if tid_str.startswith('ENS'): return tid_str.split('.')[0]
    return tid_str

def parse_netmhcpan_log(log_path, bind_levels=None, max_aff=None, max_rank_el=None):
    """Parse netMHCpan output log. Returns DataFrame."""
    data = []
    if not os.path.exists(log_path):
        print(f"[Warning] netMHCpan log not found: {log_path}")
        return pd.DataFrame()
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            if not parts or not parts[0].isdigit() or len(parts) < 16:
                continue
            raw_id = parts[10]
            match = re.split(r'_[sS]', raw_id)
            identity = match[0].replace('_', '.')
            try:
                pos_1b = int(parts[0])
                mhc = parts[1]; peptide = parts[2]; core = parts[3]
                score_el = float(parts[11]); rank_el = float(parts[12])
                score_ba = float(parts[13]); rank_ba = float(parts[14])
                aff_nm = float(parts[15])
            except (ValueError, IndexError):
                continue
            bind_level = parts[-1] if parts[-1] in ('SB','WB') else ''
            data.append([pos_1b, mhc, peptide, core, identity,
                         score_el, rank_el, score_ba, rank_ba, aff_nm, bind_level])
    cols = ['Pos','MHC','Peptide','Core','Identity',
            'Score_EL','%Rank_EL','Score_BA','%Rank_BA','Aff(nM)','BindLevel']
    df = pd.DataFrame(data, columns=cols)
    if df.empty: return df
    if bind_levels and 'ALL' not in [b.upper() for b in bind_levels]:
        df = df[df['BindLevel'].isin([b.upper() for b in bind_levels])]
    if max_aff is not None: df = df[df['Aff(nM)'] <= max_aff]
    if max_rank_el is not None: df = df[df['%Rank_EL'] <= max_rank_el]
    return df

def main():
    p = argparse.ArgumentParser(description="Mutation neoantigen integration report")
    p.add_argument("--mutation_csv", required=True,
                   help="CSV from annotate_mutation_variants.py")
    p.add_argument("--peptide_csv", required=True,
                   help="CSV from generate_mutant_peptides.py")
    p.add_argument("--trace_csv", required=True,
                   help="TRACE ORF CSV (high_confidence_orfs.{patient}.{mode}_mode.csv)")
    p.add_argument("--netmhcpan_log", required=True,
                   help="netMHCpan output log for mutant peptides")
    p.add_argument("--tpm_csv", required=True,
                   help="Transcript TPM matrix")
    p.add_argument("--patient_id", required=True, help="Patient identifier")
    p.add_argument("--tumor_run_id", required=True, help="Tumor Run ID for TPM lookup")
    p.add_argument("--output", required=True, help="Output report CSV")
    p.add_argument("--bind_levels", nargs='+', default=['SB','WB'])
    p.add_argument("--max_aff_nm", type=float, default=2000)
    p.add_argument("--max_rank_el", type=float, default=5.0)
    args = p.parse_args()

    # 1. Load mutation and peptide data
    mut_df = pd.read_csv(args.mutation_csv)
    pep_df = pd.read_csv(args.peptide_csv)
    if mut_df.empty or pep_df.empty:
        print("[Warning] Empty input, writing empty report.")
        pd.DataFrame().to_csv(args.output, index=False)
        return

    # 2. Load TRACE translation data
    trace_df = pd.read_csv(args.trace_csv)
    trace_df['Match_ID'] = trace_df['Tid'].apply(safe_clean_id)
    # Build per-transcript lookup
    trace_lookup = trace_df.groupby('Match_ID').agg(
        mean_intensity=('mean_intensity','mean'),
        tpm=('tpm','mean')
    ).to_dict('index')

    # 3. Load TPM matrix
    tpm_df = pd.read_csv(args.tpm_csv, index_col=0)
    tumor_tpm = pd.Series(dtype=float)
    if args.tumor_run_id in tpm_df.columns:
        tumor_tpm = tpm_df[args.tumor_run_id]

    # 4. Parse netMHCpan
    mhc_df = parse_netmhcpan_log(args.netmhcpan_log,
                                 bind_levels=args.bind_levels,
                                 max_aff=args.max_aff_nm,
                                 max_rank_el=args.max_rank_el)
    if mhc_df.empty:
        print("[Warning] No passing netMHCpan predictions.")
        # Write empty report with headers
        cols = ['Ref_Peptide','Mut_Peptide','MHC','Transcript_ID','Gene','AA_Change',
                'Tumor_TPM','mean_intensity','Total_Protein_Expression',
                'Aff(nM)','BindLevel','%Rank_EL','Score_EL',
                'Chr','Pos','Ref','Alt']
        pd.DataFrame(columns=cols).to_csv(args.output, index=False)
        return

    mhc_df['Identity'] = mhc_df['Identity'].apply(safe_clean_id)

    # 5. Merge and integrate
    rows = []
    for _, mhc_row in mhc_df.iterrows():
        peptide = mhc_row['Peptide']
        # Match peptide to mutation via peptide_csv
        pep_match = pep_df[pep_df['Mut_Peptide'] == peptide]
        if pep_match.empty: continue

        for _, pm in pep_match.iterrows():
            tx_id = pm['Transcript_ID']
            tx_clean = safe_clean_id(tx_id)

            # TRACE data
            trace_data = trace_lookup.get(tx_clean, {})
            mean_int = trace_data.get('mean_intensity', 0.0)
            trace_tpm = trace_data.get('tpm', 0.0)

            # TPM data
            tpm_val = 0.0
            if args.tumor_run_id in tpm_df.columns and tx_id in tpm_df.index:
                tpm_val = float(tpm_df.at[tx_id, args.tumor_run_id])
            elif tx_clean in tpm_df.index:
                tpm_val = float(tpm_df.at[tx_clean, args.tumor_run_id])

            # Use the better TPM value
            best_tpm = max(tpm_val, trace_tpm)
            total_expr = best_tpm * max(mean_int, 0.001)

            rows.append({
                'Ref_Peptide': pm['Ref_Peptide'],
                'Mut_Peptide': peptide,
                'MHC': mhc_row['MHC'],
                'Transcript_ID': tx_id,
                'Gene': pm['Gene'],
                'AA_Change': pm['AA_Change'],
                'Tumor_TPM': round(best_tpm, 3),
                'mean_intensity': round(mean_int, 4),
                'Total_Protein_Expression': round(total_expr, 4),
                'Aff(nM)': mhc_row['Aff(nM)'],
                'BindLevel': mhc_row['BindLevel'],
                '%Rank_EL': mhc_row['%Rank_EL'],
                'Score_EL': mhc_row['Score_EL'],
                'Chr': pm['Chrom'],
                'Pos': pm['Pos'],
                'Ref': pm['Ref'],
                'Alt': pm['Alt'],
                'Codon_Pos': pm['Codon_Pos'],
            })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        print("[Warning] No integrated results.")
        pd.DataFrame().to_csv(args.output, index=False)
        return

    # Sort by Total_Protein_Expression
    result_df.sort_values('Total_Protein_Expression', ascending=False, inplace=True)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"Report: {len(result_df)} neoantigen candidates saved to {args.output}")

if __name__ == "__main__":
    main()

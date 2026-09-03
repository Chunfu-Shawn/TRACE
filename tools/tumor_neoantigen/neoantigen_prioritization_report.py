#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import re


def calculate_orf_hla_index(frame, affinity_reference_nm=50000.0):
    """Combine ORF confidence and HLA affinity on a bounded 0-1 scale.

    ORF confidence is converted to a patient-level percentile across unique
    ORFs. HLA affinity uses the conventional log-scaled 50,000 nM reference,
    with lower affinity values receiving higher scores. Their harmonic mean
    favors candidates that are strong on both axes.
    """
    required = {'ORF_Score', 'Aff(nM)'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Cannot calculate ORF_HLA_Index; missing columns: {sorted(missing)}"
        )
    if not np.isfinite(affinity_reference_nm) or affinity_reference_nm <= 1.0:
        raise ValueError("affinity_reference_nm must be finite and greater than 1.")

    orf_scores = pd.to_numeric(frame['ORF_Score'], errors='coerce')
    orf_key_columns = [
        column for column in ('Identity', 'ORF_Pos') if column in frame.columns
    ]
    if orf_key_columns:
        orf_keys = frame[orf_key_columns].astype(str).agg('\x1f'.join, axis=1)
    else:
        orf_keys = pd.Series(frame.index.astype(str), index=frame.index)
    orf_units = pd.DataFrame({
        'ORF_Key': orf_keys,
        'ORF_Score': orf_scores,
    }).dropna(subset=['ORF_Score'])
    orf_unit_scores = orf_units.groupby('ORF_Key')['ORF_Score'].max()
    orf_percentiles = orf_unit_scores.rank(
        method='average',
        pct=True,
        ascending=True,
    )
    orf_component = orf_keys.map(orf_percentiles).astype(float)

    affinities = pd.to_numeric(frame['Aff(nM)'], errors='coerce')
    affinity_component = pd.Series(np.nan, index=frame.index, dtype=float)
    valid_affinity = np.isfinite(affinities) & affinities.gt(0)
    affinity_component.loc[valid_affinity] = np.clip(
        1.0
        - np.log10(affinities.loc[valid_affinity])
        / np.log10(float(affinity_reference_nm)),
        0.0,
        1.0,
    )

    combined = pd.Series(np.nan, index=frame.index, dtype=float)
    denominator = orf_component + affinity_component
    valid_components = (
        np.isfinite(orf_component)
        & np.isfinite(affinity_component)
        & denominator.gt(0)
    )
    combined.loc[valid_components] = (
        2.0
        * orf_component.loc[valid_components]
        * affinity_component.loc[valid_components]
        / denominator.loc[valid_components]
    )
    return combined

def safe_clean_id(tid):
    """
    Safely clean Transcript IDs to guarantee cross-file merging:
    """
    tid_str = str(tid).strip()
    
    # 尝试修复 NetMHCpan 替换字符后的格式，我们将其中的 _ 恢复为 . 
    tid_str = tid_str.replace('_', '.')
        
    # 已知 ENST 转录本：去除版本号
    if tid_str.startswith('ENS'):
        return tid_str.split('.')[0]
        
    return tid_str


def select_patient_candidate_rows(step2_df, tumor_run_id):
    """Select candidate rows belonging to one tumor run and normalize transcript IDs."""
    required = {'Transcript_ID', 'Tumor_Run', 'Tumor_Junction_CPM'}
    missing = required.difference(step2_df.columns)
    if missing:
        raise ValueError(f"Step 2 CSV is missing required columns: {sorted(missing)}")

    run_key = str(tumor_run_id).strip()
    run_mask = step2_df['Tumor_Run'].astype(str).str.strip() == run_key
    patient_step2 = step2_df.loc[run_mask].copy()
    if patient_step2.empty:
        raise ValueError(f"No Step 2 candidates found for tumor run '{run_key}'.")

    patient_step2['Clean_Tid'] = patient_step2['Transcript_ID'].apply(safe_clean_id)
    return patient_step2


def build_patient_jcpm_dict(step2_df, tumor_run_id):
    """Build a transcript-to-junction-CPM lookup for one tumor run only."""
    patient_step2 = select_patient_candidate_rows(step2_df, tumor_run_id)
    patient_step2['Tumor_Junction_CPM'] = pd.to_numeric(
        patient_step2['Tumor_Junction_CPM'], errors='coerce'
    ).fillna(0.0)
    return (
        patient_step2.groupby('Clean_Tid')['Tumor_Junction_CPM']
        .max()
        .to_dict()
    )


def build_patient_gtex_context(step2_df, tumor_run_id):
    """Build per-transcript GTEx assessment metadata for the final antigen report."""
    patient_step2 = select_patient_candidate_rows(step2_df, tumor_run_id)
    context_columns = [
        'GTEx_Transcript_TPM_Covered',
        'GTEx_Transcript_Filter_Source',
        'GTEx_Step2_Applied',
        'GTEx_Step2_Status',
        'GTEx_Background_Statistic',
        'Global_Max_GTEx_TPM',
        'Global_Max_GTEx_JCPM',
        'GTEx_Junction_Background_Assessed',
        'GTEx_Junction_Coverage',
        'GTEx_Junction_Filter_Source',
        'GTEx_Junction_IDs_Assessed',
        'GTEx_Junction_IDs_Missing',
    ]
    available_columns = [column for column in context_columns if column in patient_step2.columns]
    if not available_columns:
        return {}
    return (
        patient_step2.drop_duplicates('Clean_Tid')
        .set_index('Clean_Tid')[available_columns]
        .to_dict('index')
    )

def extract_filtered_binders_from_log(log_path, bind_levels, max_aff, max_rank_el, max_rank_ba):
    print(f"[Parser] Parsing NetMHCpan log: {log_path}")
    data = []
    
    if not os.path.exists(log_path):
        print(f"[Error] NetMHCpan log file not found at: {log_path}")
        sys.exit(1)
        
    with open(log_path, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()
            if not parts or not parts[0].isdigit() or len(parts) < 16:
                continue
                
            raw_identity = parts[10]
            
            # 模式: 任何以 _s 开头，或者单独一个 _ 结尾
            # 使用 re.split 提取 Tid 部分
            match = re.split(r'_[sS]', raw_identity)
            identity = match[0] # 取被截断前的部分

            pos_1b = int(parts[0])
            mhc, peptide, core = parts[1], parts[2], parts[3]
            
            try:
                score_el = float(parts[11])
                rank_el = float(parts[12])
                score_ba = float(parts[13])
                rank_ba = float(parts[14])
                aff_nm = float(parts[15])
            except ValueError:
                continue
                
            bind_level = ""
            if len(parts) >= 17 and parts[-1] in ["SB", "WB"]:
                bind_level = parts[-1]
            elif '<= SB' in line or '< SB' in line:
                bind_level = 'SB'
            elif '<= WB' in line or '< WB' in line:
                bind_level = 'WB'
                
            data.append([pos_1b, mhc, peptide, core, identity, score_el, rank_el, score_ba, rank_ba, aff_nm, bind_level])
            
    cols = ['Peptide_Protein_Pos', 'MHC', 'Peptide', 'Core', 'Identity', 'Score_EL', '%Rank_EL', 'Score_BA', '%Rank_BA', 'Aff(nM)', 'BindLevel']
    df = pd.DataFrame(data, columns=cols)
    
    if df.empty:
        print("[Warning] No predictions successfully parsed from the log.")
        return df
        
    print(f"[Filter] Total extracted predictions before filtering: {len(df)}")
    
    if bind_levels and 'ALL' not in [b.upper() for b in bind_levels]:
        target_levels = [b.upper() for b in bind_levels]
        df = df[df['BindLevel'].isin(target_levels)]
        print(f"  -> Retained {len(df)} candidates after applying BindLevel filter: {target_levels}")
        
    if max_aff is not None:
        df = df[df['Aff(nM)'] <= max_aff]
        print(f"  -> Retained {len(df)} candidates after Aff(nM) <= {max_aff}")
        
    if max_rank_el is not None:
        df = df[df['%Rank_EL'] <= max_rank_el]
        print(f"  -> Retained {len(df)} candidates after %Rank_EL <= {max_rank_el}")
        
    if max_rank_ba is not None:
        df = df[df['%Rank_BA'] <= max_rank_ba]
        print(f"  -> Retained {len(df)} candidates after %Rank_BA <= {max_rank_ba}")
        
    return df

def parse_trace_fasta(fasta_path):
    print(f"[FastaParser] Parsing sequence dictionary from: {fasta_path}")
    orf_dict = {}
    
    if not os.path.exists(fasta_path):
        print(f"[Error] FASTA file not found at: {fasta_path}")
        sys.exit(1)
        
    with open(fasta_path, 'r') as f:
        header = ""
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    parts = header.lstrip(">").split("|")
                    tid_base = safe_clean_id(parts[0])
                    start, stop = parts[1].split(":")
                    
                    orf_dict.setdefault(tid_base, []).append({
                        'start': int(start.split("-")[1]), 
                        'stop': int(stop.split("-")[1]), 
                        'sequence': "".join(seq)
                    })
                header = line
                seq = []
            else:
                seq.append(line)
                
        if header:
            parts = header.lstrip(">").split("|")
            tid_base = safe_clean_id(parts[0])
            start, stop = parts[1].split(":")
            orf_dict.setdefault(tid_base, []).append({
                'start': int(start.split("-")[1]), 
                'stop': int(stop.split("-")[1]), 
                'sequence': "".join(seq)
            })
            
    print(f"[FastaParser] Loaded translation spaces for {len(orf_dict)} unique transcripts.")
    return orf_dict

def main():
    parser = argparse.ArgumentParser(description="End-to-End Personalized Neoantigen Peptide Prioritization.")
    
    # Required Core Inputs
    parser.add_argument("-l", "--netmhcpan_log", required=True, help="Path to the patient's specific NetMHCpan log file.")
    parser.add_argument("-f", "--fasta_file", required=True, help="Path to the patient's specific protein FASTA.")
    parser.add_argument("-t", "--translation_csv", required=True, help="Path to the patient's specific high-confidence ORF data.")
    

    parser.add_argument("-s", "--step2_csv", required=True, help="Path to Step 2 valid targets CSV containing Tumor_Junction_CPM.")
    parser.add_argument("-p", "--patient_id", required=True, help="Patient identifier used for naming the output report.")
    parser.add_argument("--tumor_run_id", required=True, help="Tumor Run ID used to isolate patient-specific expression context.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save the prioritized neoantigen report.")
    
    # Optional Filtering Parameters
    parser.add_argument("--bind_levels", nargs='+', default=['SB'], help="Categorical bind levels to keep (e.g., SB WB). Use 'ALL' to disable.")
    parser.add_argument("--max_aff_nm", type=float, default=None, help="Hard filter: Maximum allowable HLA affinity in nM.")
    parser.add_argument("--max_rank_el", type=float, default=None, help="Hard filter: Maximum allowable Eluted Ligand %Rank.")
    parser.add_argument("--max_rank_ba", type=float, default=None, help="Hard filter: Maximum allowable Binding Affinity %Rank.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=== Phase 1: Parsing and Filtering Candidates for {args.patient_id} ===")
    
    df_mhc = extract_filtered_binders_from_log(
        args.netmhcpan_log, 
        bind_levels=args.bind_levels, 
        max_aff=args.max_aff_nm, 
        max_rank_el=args.max_rank_el, 
        max_rank_ba=args.max_rank_ba
    )
    
    if df_mhc.empty:
        print("[End] Pipeline stopped due to absence of valid candidate epitopes after filtering.")
        sys.exit(0)
        
    df_mhc['Identity'] = df_mhc['Identity'].apply(safe_clean_id)
    
    print("\n=== Phase 2: Constructing Expression Context Lookups (Dual-Track) ===")
    orf_dict = parse_trace_fasta(args.fasta_file)
    
    # 1. Extract patient-specific Junction CPM from Step 2.
    df_step2 = pd.read_csv(args.step2_csv)
    try:
        jcpm_dict = build_patient_jcpm_dict(df_step2, args.tumor_run_id)
        gtex_context = build_patient_gtex_context(df_step2, args.tumor_run_id)
    except ValueError as exc:
        print(f"[Error] {exc}")
        sys.exit(1)
    print(f"[Lookup] Loaded Junction CPM values for tumor run {args.tumor_run_id}.")

    # 2. 整合 TRACE 输出、ORF_Score 与 JCPM
    df_trans = pd.read_csv(args.translation_csv)

    required_translation_cols = {'Tid', 'start', 'stop', 'collapse_score'}
    missing_translation_cols = required_translation_cols.difference(df_trans.columns)
    if missing_translation_cols:
        print(
            "[Error] translation_csv is missing required columns: "
            f"{sorted(missing_translation_cols)}"
        )
        sys.exit(1)

    df_trans['Match_ID'] = df_trans['Tid'].apply(safe_clean_id)
    df_trans['collapse_score'] = pd.to_numeric(
        df_trans['collapse_score'], errors='coerce'
    ).fillna(0.0)
    
    trans_lookup = {}
    for _, row in df_trans.iterrows():
        tid_clean = row['Match_ID']
        key = (tid_clean, int(row['start']), int(row['stop']))
        
        tumor_tpm = pd.to_numeric(row.get('tpm', 0.0), errors='coerce')
        mean_int = pd.to_numeric(row.get('mean_intensity', 0.0), errors='coerce')
        orf_score = pd.to_numeric(row.get('collapse_score', 0.0), errors='coerce')

        tumor_tpm = 0.0 if pd.isna(tumor_tpm) else float(tumor_tpm)
        mean_int = 0.0 if pd.isna(mean_int) else float(mean_int)
        orf_score = 0.0 if pd.isna(orf_score) else float(orf_score)
        tumor_jcpm = jcpm_dict.get(tid_clean, 0.0)
        transcript_gtex_context = {
            'GTEx_Transcript_TPM_Covered': False,
            'GTEx_Transcript_Filter_Source': 'Unavailable',
            'GTEx_Step2_Applied': False,
            'GTEx_Step2_Status': 'Unavailable',
            'GTEx_Background_Statistic': 'Unavailable',
            'Global_Max_GTEx_TPM': 0.0,
            'Global_Max_GTEx_JCPM': 0.0,
            'GTEx_Junction_Background_Assessed': False,
            'GTEx_Junction_Coverage': 'Unavailable',
            'GTEx_Junction_Filter_Source': 'Unavailable',
            'GTEx_Junction_IDs_Assessed': '',
            'GTEx_Junction_IDs_Missing': '',
            **gtex_context.get(tid_clean, {}),
        }
        
        # Keep transcript and junction evidence on separate scales.
        prot_expr_t = tumor_tpm * mean_int
        prot_expr_c = tumor_jcpm * mean_int
        
        trans_lookup[key] = {
            'ORF_Score': orf_score,
            'Tumor_TPM': tumor_tpm,
            'Junction_CPM': tumor_jcpm,
            'mean_intensity': mean_int,
            'Protein_Expression_T': prot_expr_t,
            'Protein_Expression_C': prot_expr_c,
            **transcript_gtex_context,
        }
    print(f"[Lookup] Built integrated Dual-Track expression index covering {len(trans_lookup)} specific relative positions.")

    print("\n=== Phase 3: Resolving Peptides & Calculating Sequence Positions (0-based Safe) ===")
    mapped_peptides = []
    
    for _, row in df_mhc.iterrows():
        tid = row['Identity']
        peptide = row['Peptide']
        
        pep_prot_pos_1b = int(row['Peptide_Protein_Pos'])
        prot_start_0b = pep_prot_pos_1b - 1 
        
        # If one peptide maps to multiple compatible ORFs, select the ORF with
        # the highest TRACE collapse_score; use mean_intensity as a tie-breaker.
        best_orf_score = float('-inf')
        best_expr = float('-inf')
        best_metrics = {
            'ORF_Score': 0.0,
            'Tumor_TPM': 0.0, 'Junction_CPM': 0.0, 'mean_intensity': 0.0,
            'Protein_Expression_T': 0.0, 'Protein_Expression_C': 0.0,
            'GTEx_Transcript_TPM_Covered': False,
            'GTEx_Transcript_Filter_Source': 'Unavailable',
            'GTEx_Step2_Applied': False,
            'GTEx_Step2_Status': 'Unavailable',
            'GTEx_Background_Statistic': 'Unavailable',
            'Global_Max_GTEx_TPM': 0.0,
            'Global_Max_GTEx_JCPM': 0.0,
            'GTEx_Junction_Background_Assessed': False,
            'GTEx_Junction_Coverage': 'Unavailable',
            'GTEx_Junction_Filter_Source': 'Unavailable',
            'GTEx_Junction_IDs_Assessed': '',
            'GTEx_Junction_IDs_Missing': '',
        }
        
        mapped_orf_pos = "Unmapped"
        mapped_pep_tx_pos = "Unmapped"
        
        if tid in orf_dict:
            for orf in orf_dict[tid]:
                if prot_start_0b >= 0 and (prot_start_0b + len(peptide)) <= len(orf['sequence']):
                    extracted_pep = orf['sequence'][prot_start_0b : prot_start_0b + len(peptide)]
                    
                    if extracted_pep == peptide:
                        pep_tx_start = orf['start'] + (prot_start_0b * 3)
                        pep_tx_stop = pep_tx_start + (len(peptide) * 3)
                        
                        key = (tid, orf['start'], orf['stop'])
                        metrics = trans_lookup.get(key)
                        if metrics is None:
                            continue
                        orf_score = metrics.get('ORF_Score', 0.0)
                        expr = metrics.get('mean_intensity', 0.0)

                        is_better_orf = (
                            orf_score > best_orf_score
                            or (orf_score == best_orf_score and expr > best_expr)
                        )
                        if is_better_orf:
                            best_orf_score = orf_score
                            best_expr = expr
                            best_metrics = metrics
                            mapped_orf_pos = f"{orf['start']}:{orf['stop']}"
                            mapped_pep_tx_pos = f"{pep_tx_start}:{pep_tx_stop}"
                            
        row_dict = row.to_dict()
        row_dict['ORF_Pos'] = mapped_orf_pos
        row_dict['Peptide_Tx_Pos'] = mapped_pep_tx_pos
        row_dict.update(best_metrics)
        mapped_peptides.append(row_dict)
        
    df_mapped = pd.DataFrame(mapped_peptides)
    
    # Retain candidates supported by either independent expression track.
    df_mapped = df_mapped[
        (df_mapped['Protein_Expression_T'] > 0) |
        (df_mapped['Protein_Expression_C'] > 0)
    ].copy()
    print(f"[Mapping] Sequence positional alignment complete. Retained {len(df_mapped)} highly traceable candidates.")

    print("\n=== Phase 4: Final Prioritization and Export ===")

    df_mapped['ORF_HLA_Index'] = calculate_orf_hla_index(df_mapped)
    
    cols_order = [
        'Peptide', 'MHC', 'Identity', 
        'Peptide_Protein_Pos', 'Peptide_Tx_Pos', 'ORF_Pos', 'ORF_Score',
        'ORF_HLA_Index',
        'Protein_Expression_T', 'Tumor_TPM', 
        'Protein_Expression_C', 'Junction_CPM', 
        'mean_intensity', 
        'Score_EL', 'Aff(nM)', 'BindLevel', '%Rank_EL',
        'GTEx_Transcript_TPM_Covered', 'GTEx_Transcript_Filter_Source',
        'GTEx_Step2_Applied', 'GTEx_Step2_Status', 'GTEx_Background_Statistic',
        'Global_Max_GTEx_TPM', 'Global_Max_GTEx_JCPM',
        'GTEx_Junction_Background_Assessed', 'GTEx_Junction_Coverage',
        'GTEx_Junction_Filter_Source', 'GTEx_Junction_IDs_Assessed',
        'GTEx_Junction_IDs_Missing'
    ]
    remaining_cols = [c for c in df_mapped.columns if c not in cols_order]
    df_mapped = df_mapped[cols_order + remaining_cols]
    
    # Prioritize candidates that have both ORF and HLA-binding support.
    df_mapped.sort_values(
        by=['ORF_HLA_Index', 'ORF_Score', 'Score_EL', 'mean_intensity'],
        ascending=[False, False, False, False],
        inplace=True,
    )
    
    round_cols = [
        'Protein_Expression_T', 'Protein_Expression_C', 
        'Tumor_TPM', 'Junction_CPM', 'mean_intensity', 'ORF_Score',
        'ORF_HLA_Index', 'Score_EL'
    ]
    df_mapped[round_cols] = df_mapped[round_cols].round(4)
    
    output_filename = os.path.join(args.output_dir, f"{args.patient_id}.csv")
    df_mapped.to_csv(output_filename, index=False)
    
    print(f" -> [Success] Prioritized {len(df_mapped)} effective epitopes.")
    print(f" -> Report saved to: {output_filename}")
    print("\n==========================================================================")

if __name__ == "__main__":
    main()

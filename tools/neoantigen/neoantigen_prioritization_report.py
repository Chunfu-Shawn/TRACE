#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import re

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
    
    # 1. 提取 Step 2 中的 Junction CPM
    df_step2 = pd.read_csv(args.step2_csv)
    if 'Tumor_Junction_CPM' not in df_step2.columns:
        print("[Error] 'Tumor_Junction_CPM' not found in Step 2 CSV.")
        sys.exit(1)
        
    df_step2['Clean_Tid'] = df_step2['Transcript_ID'].apply(safe_clean_id)
    # 取该转录本中最高的 Junction CPM 以备用
    jcpm_dict = df_step2.groupby('Clean_Tid')['Tumor_Junction_CPM'].max().fillna(0.0).to_dict()

    # 2. 整合 TRACE 输出与 JCPM
    df_trans = pd.read_csv(args.translation_csv)
    df_trans['Match_ID'] = df_trans['Tid'].apply(safe_clean_id)
    
    trans_lookup = {}
    for _, row in df_trans.iterrows():
        tid_clean = row['Match_ID']
        key = (tid_clean, int(row['start']), int(row['stop']))
        
        tumor_tpm = row.get('tpm', 0.0)
        mean_int = row.get('mean_intensity', 0.0)
        tumor_jcpm = jcpm_dict.get(tid_clean, 0.0)
        
        # 计算双轨制蛋白质表达量
        prot_expr_t = tumor_tpm * mean_int
        prot_expr_c = tumor_jcpm * mean_int / 150 # read length
        total_prot_expr = prot_expr_t + prot_expr_c
        
        trans_lookup[key] = {
            'Tumor_TPM': tumor_tpm,
            'Junction_CPM': tumor_jcpm,
            'mean_intensity': mean_int,
            'Protein_Expression_T': prot_expr_t,
            'Protein_Expression_C': prot_expr_c,
            'Total_Protein_Expression': total_prot_expr
        }
    print(f"[Lookup] Built integrated Dual-Track expression index covering {len(trans_lookup)} specific relative positions.")

    print("\n=== Phase 3: Resolving Peptides & Calculating Sequence Positions (0-based Safe) ===")
    mapped_peptides = []
    
    for _, row in df_mhc.iterrows():
        tid = row['Identity']
        peptide = row['Peptide']
        
        pep_prot_pos_1b = int(row['Peptide_Protein_Pos'])
        prot_start_0b = pep_prot_pos_1b - 1 
        
        best_expr = -1.0
        best_metrics = {
            'Tumor_TPM': 0.0, 'Junction_CPM': 0.0, 'mean_intensity': 0.0,
            'Protein_Expression_T': 0.0, 'Protein_Expression_C': 0.0, 'Total_Protein_Expression': 0.0
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
                        metrics = trans_lookup.get(key, {})
                        expr = metrics.get('Total_Protein_Expression', 0.0)
                        
                        if expr > best_expr:
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
    
    # 只要由任意一条轨（Transcript 或 Junction）支撑蛋白质表达量，就将其保留
    df_mapped = df_mapped[df_mapped['Total_Protein_Expression'] > 0].copy()
    print(f"[Mapping] Sequence positional alignment complete. Retained {len(df_mapped)} highly traceable candidates.")

    print("\n=== Phase 4: Final Prioritization and Export ===")
    
    # 更新打分逻辑
    df_mapped['Combined_Score'] = df_mapped['Total_Protein_Expression'] * df_mapped['Score_EL']
    
    cols_order = [
        'Peptide', 'MHC', 'Identity', 
        'Peptide_Protein_Pos', 'Peptide_Tx_Pos', 'ORF_Pos', 
        'Total_Protein_Expression', 'Combined_Score',
        'Protein_Expression_T', 'Tumor_TPM', 
        'Protein_Expression_C', 'Junction_CPM', 
        'mean_intensity', 
        'Score_EL', 'Aff(nM)', 'BindLevel', '%Rank_EL'
    ]
    remaining_cols = [c for c in df_mapped.columns if c not in cols_order]
    df_mapped = df_mapped[cols_order + remaining_cols]
    
    # 严格按照 Total_Protein_Expression (Protein_Expression_T + Protein_Expression_C) 进行降序排序
    df_mapped.sort_values(by='Total_Protein_Expression', ascending=False, inplace=True)
    
    round_cols = [
        'Total_Protein_Expression', 'Combined_Score', 
        'Protein_Expression_T', 'Protein_Expression_C', 
        'Tumor_TPM', 'Junction_CPM', 'mean_intensity', 'Score_EL'
    ]
    df_mapped[round_cols] = df_mapped[round_cols].round(4)
    
    output_filename = os.path.join(args.output_dir, f"{args.patient_id}.csv")
    df_mapped.to_csv(output_filename, index=False)
    
    print(f" -> [Success] Prioritized {len(df_mapped)} effective epitopes.")
    print(f" -> Report saved to: {output_filename}")
    print("\n==========================================================================")

if __name__ == "__main__":
    main()
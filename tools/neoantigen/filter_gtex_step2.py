#!/usr/bin/env python3
import pandas as pd
import os
import argparse
import sys
import re

def clean_colname(col):
    """
    Clean column names: Revert full paths to GTEx SAMPID.
    """
    if '.bam' in col:
        basename = os.path.basename(col)
        samp_id = basename.replace('_uniq.sorted.bam', '').replace('.bam', '')
        return samp_id
    return col

def clean_id(tid):
    """
    Clean transcript/gene IDs: 
    - Remove version suffixes from Ensembl IDs.
    - Remove everything after the first ':' for PacBio (PB) IDs.
    Ensures perfect merging between Step 1 and GTEx featureCounts matrices.
    """
    tid_str = str(tid).strip()
    
    # 1. Ensembl ID: Remove version suffix
    if tid_str.startswith('ENS'):
        return tid_str.split('.')[0]
        
    # 2. PacBio ID: Remove trailing coordinate/info after colon
    elif tid_str.startswith('PB'):
        return tid_str.split(':')[0]
        
    return tid_str

def parse_featurecounts_log(log_path):
    """
    Parse featureCounts log file to extract the true Total Alignments for each BAM file.
    This prevents the "Shrunken Library Effect" when quantifying a subset of transcripts.
    Robust against trailing characters (like '||', '...', and spaces) in the log formatting.
    """
    print(f"[Log Parser] Extracting absolute library sizes from: {log_path}")
    alignments_dict = {}
    current_sample = None
    
    if not os.path.exists(log_path):
        print(f"[Error] featureCounts log file not found: {log_path}")
        sys.exit(1)
        
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 1. Capture the BAM filename
            if line.startswith("|| Process BAM file"):
                # line 示例: "|| Process BAM file GTEX-TSE9-3026-SM-3DB76_uniq.sorted.bam...                ||"
                # 剔除前缀、后缀 "||" 以及两端的空格
                raw_name = line.replace("|| Process BAM file", "").replace("||", "").strip()
                # 剔除尾部的 "..."
                if raw_name.endswith("..."):
                    raw_name = raw_name[:-3].strip()
                    
                # 交给统一的清洗函数处理
                current_sample = clean_colname(raw_name)
                
            # 2. Capture the Total Alignments for the current BAM
            elif line.startswith("||    Total alignments :"):
                if current_sample:
                    parts = line.split(":")
                    if len(parts) == 2:
                        # 使用正则剔除所有非数字字符
                        raw_number_str = parts[1]
                        clean_number_str = re.sub(r'\D', '', raw_number_str)
                        
                        if clean_number_str:
                            total_alignments = int(clean_number_str)
                            alignments_dict[current_sample] = total_alignments
                        else:
                            print(f"[Warning] Could not extract numeric total alignments for {current_sample}")
                            
                    current_sample = None # Reset for the next file
                    
    print(f"[Log Parser] Successfully extracted library sizes for {len(alignments_dict)} samples.")
    return alignments_dict

def main():
    parser = argparse.ArgumentParser(description="Calculate Absolute TPM for novel transcripts using true library size, and re-validate Dual-Track status using GTEx.")
    parser.add_argument("-i", "--step1_file", required=True, help="Input CSV from Step 1 (Dual-Track format)")
    parser.add_argument("-c", "--counts_file", required=True, help="Input TXT from featureCounts (novel transcripts)")
    parser.add_argument("-l", "--fc_log", required=True, help="Input featureCounts Log file to extract Total alignments")
    parser.add_argument("-a", "--anno_file", required=True, help="GTEx Annotations DS TXT file")
    parser.add_argument("-o", "--output", required=True, help="Final output CSV: safe_tumor_specific_transcripts_GTEx-step2.csv")
    
    # Dual-Track Thresholds
    parser.add_argument("--max_tpm", type=float, default=0.5, help="Track A: Max allowed median TPM in normal tissues (default: 0.5)")
    parser.add_argument("--veto_tpm", type=float, default=2.0, help="Veto Track B if GTEx TPM exceeds this value (default: 2.0)")
    parser.add_argument("--include_testis", action="store_true", help="Do not exclude Testis (disables CTA exemption if set)")
    
    args = parser.parse_args()

    print("### Phase 1: Load Step 1 Results & Separate ###")
    try:
        step1_df = pd.read_csv(args.step1_file)
    except Exception as e:
        print(f"Error loading Step 1 file: {e}")
        sys.exit(1)

    # Clean IDs strictly prior to processing
    step1_df['Transcript_ID'] = step1_df['Transcript_ID'].apply(clean_id)

    # Differentiate ENST and Novel transcripts
    is_known = step1_df['Transcript_ID'].astype(str).str.startswith('ENST')
    known_df = step1_df[is_known].copy()
    novel_df = step1_df[~is_known].copy()

    print(f" -> Known targets (ENST, bypassed Step 2 GTEx TPM filter): {len(known_df)}")
    print(f" -> Novel targets requiring Track A/B GTEx TPM verification: {len(novel_df)}")

    if novel_df.empty:
        print("No novel targets to process. Saving Step 1 results directly to Output.")
        step1_df.to_csv(args.output, index=False)
        sys.exit(0)

    print("\n### Phase 2: Process featureCounts Matrix to Absolute TPM ###")
    try:
        counts_df = pd.read_csv(args.counts_file, sep='\t', comment='#')
    except Exception as e:
        print(f"Error loading counts file: {e}")
        sys.exit(1)

    counts_df.rename(columns=clean_colname, inplace=True)
    counts_df['Geneid'] = counts_df['Geneid'].apply(clean_id)

    # Extract true library sizes from the log
    total_alignments_map = parse_featurecounts_log(args.fc_log)

    lengths = counts_df.set_index('Geneid')['Length']
    raw_counts = counts_df.drop(columns=['Chr', 'Start', 'End', 'Strand', 'Length']).set_index('Geneid')

    # Validate mapping coverage
    missing_samples = [col for col in raw_counts.columns if col not in total_alignments_map]
    if missing_samples:
        print(f"[Error] Log file is missing total alignments for {len(missing_samples)} samples.")
        print(f"Example missing samples: {missing_samples[:3]}")
        sys.exit(1)

    # =========================================================================
    # [CRITICAL UPGRADE: Absolute TPM Calculation]
    # scale_factors = True Total Alignments from BAM / 1,000,000
    # =========================================================================
    scale_factors = pd.Series({col: total_alignments_map[col] / 1e6 for col in raw_counts.columns})
    
    lengths_kb = lengths / 1000.0
    rpk = raw_counts.div(lengths_kb, axis=0)
    
    # Calculate TPM using the global true denominator, ensuring values are not artificially inflated
    tpm_df = rpk.div(scale_factors, axis=1).fillna(0)

    print(f" -> Absolute TPM matrix generated: {tpm_df.shape[0]} novel transcripts across {tpm_df.shape[1]} samples.")

    # Slice matrix to evaluate required novel transcripts only (Optimization)
    novel_ids = novel_df['Transcript_ID'].unique()
    tpm_df_novel = tpm_df[tpm_df.index.isin(novel_ids)].copy()
    print(f" -> Matrix optimized: Sliced down to strictly {tpm_df_novel.shape[0]} tracking novel targets.")

    print("\n### Phase 3: Calculate Tissue Medians & Re-evaluate Dual-Track Status ###")
    anno_df = pd.read_csv(args.anno_file, sep='\t', low_memory=False)
    samp2tissue = dict(zip(anno_df['SAMPID'], anno_df['SMTSD']))

    tissue_cols = {}
    for col in tpm_df_novel.columns:
        if col in samp2tissue:
            tissue = samp2tissue[col]
            tissue_cols.setdefault(tissue, []).append(col)

    # Compute median TPM per tissue ONLY on the localized subset
    medians_df = pd.DataFrame(index=tpm_df_novel.index)
    for tissue, cols in tissue_cols.items():
        medians_df[tissue] = tpm_df_novel[cols].median(axis=1)

    # CTA Exemption Logic
    tissue_list = list(medians_df.columns)
    if not args.include_testis:
        tissues_to_check = [t for t in tissue_list if 'Testis' not in t]
        print(f" -> CTA exemption enabled, evaluating {len(tissues_to_check)} core organs.")
    else:
        tissues_to_check = tissue_list

    # Retrieve max baseline expression for each novel transcript
    medians_df['Max_GTEx_Baseline_Actual'] = medians_df[tissues_to_check].max(axis=1).fillna(0)
    novel_baseline_map = medians_df['Max_GTEx_Baseline_Actual'].to_dict()

    # Update background expression values in novel_df
    novel_df['Global_Max_GTEx_TPM'] = novel_df['Transcript_ID'].map(novel_baseline_map).fillna(0.0)

    # Evaluate Track A: Must pass previously (local) AND GTEx TPM < max_tpm
    valid_A = novel_df['Pass_TrackA_TPM'] & (novel_df['Global_Max_GTEx_TPM'] < args.max_tpm)
    
    # Evaluate Track B: Must pass previously (local & GTEx JCPM) AND GTEx TPM < veto_tpm
    valid_B = novel_df['Pass_TrackB_Junction'] & (novel_df['Global_Max_GTEx_TPM'] < args.veto_tpm)
    
    novel_df['Pass_TrackA_TPM'] = valid_A
    novel_df['Pass_TrackB_Junction'] = valid_B
    
    # Survive if passes AT LEAST one track
    safe_novel_df = novel_df[novel_df['Pass_TrackA_TPM'] | novel_df['Pass_TrackB_Junction']].copy()

    elimination_rate = (1 - len(safe_novel_df)/len(novel_df))*100 if len(novel_df) > 0 else 0
    print(f" -> Novel targets remaining after re-validation: {len(safe_novel_df)} (Elimination rate: {elimination_rate:.2f}%)")

    print("\n### Phase 4: Final Merge & Export ###")
    # Re-merge the protected known ENST targets with the freshly filtered novel targets
    final_step2_targets = pd.concat([known_df, safe_novel_df], ignore_index=True)

    if 'Shared_Patient_Count' in final_step2_targets.columns:
        final_step2_targets = final_step2_targets.sort_values(
            by=['Shared_Patient_Count', 'Tumor_Junction_CPM', 'Tumor_TPM'], 
            ascending=[False, False, False] 
        )

    print(f"\n--- Ultimate Dual-Track Filter Report ---")
    print(f"Step 1 Input Target Count: {len(step1_df)}")
    print(f"Step 2 Ultimate Safe Target Count: {len(final_step2_targets)}")

    print("\nPreview of Ultimate Safe Candidates:")
    preview_cols = ['Transcript_ID']
    if 'Class_Code' in final_step2_targets.columns: preview_cols.append('Class_Code')
    preview_cols.extend(['Pass_TrackA_TPM', 'Global_Max_GTEx_TPM', 'Pass_TrackB_Junction', 'Global_Max_GTEx_JCPM'])
    print(final_step2_targets.head(10)[preview_cols].to_string(index=False))

    final_step2_targets.to_csv(args.output, index=False)
    print(f"\n✅ Ultimate verification list saved to: {args.output}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import sys

from quantification_utils import calculate_true_tpm, clean_featurecounts_sample_name

def clean_colname(col):
    """
    Clean column names: Revert full paths to GTEx SAMPID.
    """
    return clean_featurecounts_sample_name(col)

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


def permissive_background_pass(covered, values, threshold):
    """Pass missing re-quantification rows; threshold only rows with measurements."""
    return (~covered.astype(bool)) | (values < threshold)

def main():
    parser = argparse.ArgumentParser(description="Calculate true TPM from complete-GTF GTEx counts and re-validate Dual-Track status.")
    parser.add_argument("-i", "--step1_file", required=True, help="Input CSV from Step 1 (Dual-Track format)")
    parser.add_argument("-c", "--counts_file", required=True, help="Input TXT from featureCounts (novel transcripts)")
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
    known_df['GTEx_Step2_Applied'] = False
    known_df['GTEx_Step2_Status'] = 'Not_required_known_transcript'

    print(f" -> Known targets (ENST, bypassed Step 2 GTEx TPM filter): {len(known_df)}")
    print(f" -> Novel targets requiring Track A/B GTEx TPM verification: {len(novel_df)}")

    if novel_df.empty:
        print("No novel targets to process. Saving Step 1 results directly to Output.")
        known_df.to_csv(args.output, index=False)
        sys.exit(0)

    print("\n### Phase 2: Process Complete-GTF featureCounts Matrix to True TPM ###")
    try:
        counts_df = pd.read_csv(args.counts_file, sep='\t', comment='#')
    except Exception as e:
        print(f"Error loading counts file: {e}")
        sys.exit(1)

    counts_df.rename(columns=clean_colname, inplace=True)
    counts_df['Geneid'] = counts_df['Geneid'].apply(clean_id)

    metadata_cols = ['Chr', 'Start', 'End', 'Strand', 'Length']
    count_cols = [column for column in counts_df.columns if column not in metadata_cols + ['Geneid']]
    aggregation = {column: 'sum' for column in count_cols}
    aggregation.update({column: 'first' for column in metadata_cols if column in counts_df.columns})
    counts_df = counts_df.groupby('Geneid', as_index=False).agg(aggregation)

    lengths = counts_df.set_index('Geneid')['Length']
    raw_counts = counts_df.set_index('Geneid')[count_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    try:
        tpm_df = calculate_true_tpm(raw_counts, lengths)
    except ValueError as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    print(f" -> True TPM matrix generated: {tpm_df.shape[0]} transcripts across {tpm_df.shape[1]} samples.")

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
    medians_df['Max_GTEx_Baseline_Actual'] = medians_df[tissues_to_check].max(axis=1)
    novel_baseline_map = {
        transcript_id: value
        for transcript_id, value in medians_df['Max_GTEx_Baseline_Actual'].items()
        if pd.notna(value)
    }

    # Update background expression values in novel_df
    novel_df['GTEx_Transcript_TPM_Covered'] = novel_df['Transcript_ID'].isin(novel_baseline_map)
    novel_df['Global_Max_GTEx_TPM'] = novel_df['Transcript_ID'].map(novel_baseline_map)
    novel_df['GTEx_Transcript_Filter_Source'] = np.where(
        novel_df['GTEx_Transcript_TPM_Covered'],
        'Complete_GTF_GTEx_Step2_tissue_median_TPM',
        'Not_assessed_missing_from_step2_requantification_passed',
    )
    novel_df['GTEx_Step2_Applied'] = novel_df['GTEx_Transcript_TPM_Covered']
    novel_df['GTEx_Step2_Status'] = np.where(
        novel_df['GTEx_Transcript_TPM_Covered'],
        'Assessed_with_tissue_medians',
        'Missing_transcript_in_requantification_passed',
    )
    novel_df['GTEx_Background_Statistic'] = 'Maximum_across_tissue_medians'

    # Evaluate Track A: Must pass previously (local) AND GTEx TPM < max_tpm
    valid_A = novel_df['Pass_TrackA_TPM'] & \
              permissive_background_pass(
                  novel_df['GTEx_Transcript_TPM_Covered'],
                  novel_df['Global_Max_GTEx_TPM'],
                  args.max_tpm,
              )
    
    # Evaluate Track B: Must pass previously (local & GTEx JCPM) AND GTEx TPM < veto_tpm
    valid_B = novel_df['Pass_TrackB_Junction'] & \
              permissive_background_pass(
                  novel_df['GTEx_Transcript_TPM_Covered'],
                  novel_df['Global_Max_GTEx_TPM'],
                  args.veto_tpm,
              )
    
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

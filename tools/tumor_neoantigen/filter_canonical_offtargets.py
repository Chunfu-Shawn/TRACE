#!/usr/bin/env python3
import os
import re
import sys
import glob
import argparse
import pandas as pd

def clean_id(tid):
    """
    Safely clean Transcript IDs:
    Strip version numbers (decimals) ONLY for Ensembl IDs.
    """
    tid_str = str(tid).strip()
    if tid_str.startswith('ENS'):
        return tid_str.split('.')[0]
    return tid_str

def load_reference_proteome(fasta_path):
    """
    Reads the canonical protein FASTA file.
    Returns a dictionary mapping Clean_Transcript_ID -> Protein_Sequence.
    """
    print(f"[FASTA] Loading reference proteome from: {fasta_path}")
    ref_proteome = {}
    current_id = None
    seq_parts = []
    
    if not os.path.exists(fasta_path):
        print(f"[Error] FASTA file not found: {fasta_path}")
        sys.exit(1)
        
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    ref_proteome[current_id] = "".join(seq_parts)
                
                parts = line.lstrip('>').split('|')
                enst_part = next((p for p in parts if p.startswith('ENST')), None)
                
                if enst_part:
                    current_id = clean_id(enst_part)
                else:
                    current_id = None
                seq_parts = []
            else:
                if current_id:
                    seq_parts.append(line)
                    
        if current_id:
            ref_proteome[current_id] = "".join(seq_parts)
            
    print(f"[FASTA] Successfully loaded {len(ref_proteome)} canonical transcript sequences.")
    return ref_proteome

def main():
    parser = argparse.ArgumentParser(description="Batch process multi-patient neoantigen logs and strictly filter off-target self-peptides against normal proteome.")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing patient specific neoantigens CSV files.")
    parser.add_argument("-f", "--fasta", required=True, help="Canonical translations FASTA (e.g., gencode.v49.pc_translations.fa).")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory to save the filtered specific patient reports.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ==============================================================================
    # 1. Scan and Batch Load All Patient Files
    # ==============================================================================
    print("--- Phase 1: Scanning and Batch Loading Patient Datasets ---")
    if not os.path.exists(args.input_dir):
        print(f"[Error] Input directory does not exist: {args.input_dir}")
        sys.exit(1)
        
    search_csv = os.path.join(args.input_dir, "*.csv")
    search_tsv = os.path.join(args.input_dir, "*.tsv")
    patient_files = glob.glob(search_csv) + glob.glob(search_tsv)
    
    if not patient_files:
        print(f"[Error] No valid CSV/TSV neoantigen reports found under: {args.input_dir}")
        sys.exit(1)
        
    print(f" -> Found {len(patient_files)} total patient file pools to merge.")

    # Record map to remember the source file extension per patient for precise sharding later
    patient_file_info = {}
    df_list = []
    
    for f_path in patient_files:
        basename = os.path.basename(f_path)
        sep = '\t' if f_path.lower().endswith('.tsv') else ','
        tmp_df = pd.read_csv(f_path, sep=sep)
        
        if tmp_df.empty:
            continue
            
        # Standardize ID column tracking if necessary
        if 'Identity' in tmp_df.columns and 'Transcript_ID' not in tmp_df.columns:
            tmp_df['Transcript_ID'] = tmp_df['Identity']
            
        # Auto-extract precise Patient ID via standard prefix capture
        match = re.search(r'(patient_?\d+)', basename, re.IGNORECASE)
        patient_id = match.group(1) if match else "_".join(basename.split('_')[:2]).replace('.csv', '').replace('.tsv', '')
        
        tmp_df['Patient'] = patient_id
        patient_file_info[patient_id] = {
            'basename': basename,
            'sep': sep
        }
        df_list.append(tmp_df)
        
    global_df = pd.concat(df_list, ignore_index=True)
    print(f" -> Successfully mass-loaded {len(global_df)} aggregate neoantigen rows.")

    # ==============================================================================
    # 2. Load Canonical Reference and Cross-Reference Multi-peptides
    # ==============================================================================
    print("\n--- Phase 2: Building Normal Background Interception Indexes ---")
    ref_proteome = load_reference_proteome(args.fasta)
    
    global_df['Clean_ID'] = global_df['Transcript_ID'].apply(clean_id)
    unique_peptides = list(global_df['Peptide'].unique())
    print(f" -> Processing matrix-wide cross-referencing for {len(unique_peptides)} unique peptide variants...")
    
    # Pre-build structural map: Peptide -> Set of canonical Transcript IDs
    pep_to_canonical = {pep: set() for pep in unique_peptides}
    for tid, seq in ref_proteome.items():
        for pep in unique_peptides:
            if pep in seq:
                pep_to_canonical[pep].add(tid)

    # ==============================================================================
    # 3. Apply Multi-Patient Off-Target AND-Gate Filter
    # ==============================================================================
    print("\n--- Phase 3: Executing Global Self-Peptide Set-Difference Filtration ---")
    keep_mask = []
    
    for idx, row in global_df.iterrows():
        patient = row['Patient']
        pep = row['Peptide']
        
        # 1. Get the patient-specific cleared whitelist of tumor transcripts
        patient_tx_set = set(global_df[global_df['Patient'] == patient]['Clean_ID'])
        
        # 2. Get all normal canonical sources globally containing this specific peptide
        matched_canonical_tids = pep_to_canonical.get(pep, set())
        
        # 3. Set difference logic: find if there are canonical sources NOT verified for this patient
        off_targets = matched_canonical_tids - patient_tx_set
        
        if len(off_targets) == 0:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            
    global_df['Pass_Canonical_Safety'] = keep_mask
    filtered_global_df = global_df[global_df['Pass_Canonical_Safety']].copy()
    
    print(f"\n==================================================================")
    print(f" 🎯 Multi-Patient Cohort Filtration Summary")
    print(f"==================================================================")
    print(f" -> Initial Combined Rows Analyzed : {len(global_df)}")
    print(f" -> Excluded (Off-Target Autoimmune Risk) : {len(global_df) - len(filtered_global_df)}")
    print(f" -> Pristine Specific Epitopes Retained    : {len(filtered_global_df)}")
    print(f"==================================================================")

    # Clean processing tracking columns
    filtered_global_df = filtered_global_df.drop(columns=['Clean_ID', 'Pass_Canonical_Safety'])

    # ==============================================================================
    # 4. Patient Sharding and Multi-File Recovery Export
    # ==============================================================================
    print("\n--- Phase 4: Sharding and Shuffling Back to Individual Patient CSVs ---")
    grouped = filtered_global_df.groupby('Patient')
    
    for patient_id, patient_group in grouped:
        # Retrieve the exact filename metadata mapped during Phase 1
        file_meta = patient_file_info.get(patient_id)
        if file_meta:
            out_filename = file_meta['basename']
            out_sep = file_meta['sep']
        else:
            out_filename = f"{patient_id}_strictly_safe_neoepitopes.csv"
            out_sep = ','
            
        out_path = os.path.join(args.output_dir, out_filename)
        
        # Drop the runtime auxiliary 'Patient' column if it didn't exist in original shapes
        final_patient_export = patient_group.copy()
        
        final_patient_export.to_csv(out_path, sep=out_sep, index=False)
        print(f" -> Successfully saved ultra-safe specific file for [{patient_id}]: {out_filename} ({len(final_patient_export)} rows)")

    print(f"\n✅ All batch workflows completed successfully. Outputs structured under: {args.output_dir}/\n")

if __name__ == "__main__":
    main()
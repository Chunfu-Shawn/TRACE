#!/usr/bin/env python3
import os
import argparse
import sys
sys.path.append("/home/user/data3/rbase/translation_model/models/src")
import pandas as pd
import torch

# Custom model imports
from model.translation_base_model import TranslationBaseModel
from model.translation_predictor import TranslationProfilePredictor
from model.mask_heads import PsiteDensityHead
from model.generate_cell_env_expr_array import generate_cell_env_expr_dict
from model.orf_caller import TranslationSignalORFCaller

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser(description="Run TRACE translation prediction with dynamic expression array support.")
    
    parser.add_argument("-i", "--input_csv", required=True, help="Input CSV containing verified 'Transcript_ID's.")
    parser.add_argument("-o", "--out_dir", required=True, help="Base output directory.")
    parser.add_argument("-f", "--fasta_files", nargs='+', required=True, help="List of FASTA files.")
    
    parser.add_argument("--config_path", required=True, help="Path to TRACE YAML config.")
    parser.add_argument("--weights_path", required=True, help="Path to TRACE model weights.")
    
    expr_group = parser.add_mutually_exclusive_group(required=True)
    expr_group.add_argument("--expr_dict_path", type=str, help="Path to pre-built expression dict (.pt).")
    expr_group.add_argument("--patient_counts_file", type=str, help="Generate array on-the-fly from featureCounts TXT.")
    parser.add_argument("--counts_level", type=str, default="transcript", choices=["transcript", "gene"])
    
    parser.add_argument("--ref_order", type=str, help="Path to global_anchor_gene_order.txt")
    parser.add_argument("--mapping_json", type=str, help="Path to global_species_id_mapping.json")
    
    # === [MODIFIED 1: 更改了更具临床意义的变量名] ===
    parser.add_argument("--tumor_run_id", type=str, required=True, help="RUN ID to query matrices (e.g. 'SRR17593541').")
    parser.add_argument("--patient_id", type=str, required=True, help="Patient ID for output files (e.g. 'patient_10584').")
    # ==============================================================================

    parser.add_argument("--tpm_csv", type=str, default=None, help="Path to the patient-specific transcript TPM matrix.")
    parser.add_argument("--tpm_level", type=str, default="transcript", choices=["transcript", "gene"])
    parser.add_argument("--tx2gene_mapping", type=str, default=None, help="Path to Transcript-to-Gene mapping table.")

    parser.add_argument("--mode", type=str, default="balanced")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    print(f"\n--- Phase 1: Loading Target Transcripts & Resolving IDs ---")
    try:
        targets_df = pd.read_csv(args.input_csv)
        
        if 'Tumor_Run' not in targets_df.columns:
            print("[Error] Input CSV does not contain a 'Tumor_Run' column.")
            sys.exit(1)
            
        # === [MODIFIED 2: 使用 tumor_run_id 过滤] ===
        pt_df = targets_df[targets_df['Tumor_Run'] == args.tumor_run_id]
            
        if pt_df.empty:
            print(f"No specific transcripts found for Run ID '{args.tumor_run_id}'. Exiting smoothly.")
            sys.exit(0)

        tumor_specific_tx = pt_df['Transcript_ID'].dropna().unique().tolist()
        print(f"Loaded {len(tumor_specific_tx)} unique tumor-specific transcripts for Run {args.tumor_run_id}.")
        
        expr_key = args.tumor_run_id 
        
    except Exception as e:
        print(f"Failed to load input CSV: {e}")
        sys.exit(1)

    print(f"\n--- Phase 2: Preparing Expression Vector ---")
    if args.patient_counts_file:
        expr_dict = generate_cell_env_expr_dict(
            counts_file=args.patient_counts_file,
            ref_order_path=args.ref_order,
            quant_level=args.counts_level,
            tx2gene_file=args.tx2gene_mapping,
            mapping_json_path=args.mapping_json,
            min_tpm_threshold=0.0
        )
    else:
        expr_dict = torch.load(args.expr_dict_path, map_location='cpu')

    if expr_key not in expr_dict:
        print(f"Error: Resolved Sample key '{expr_key}' not found in the expression dictionary.")
        sys.exit(1)
        
    expr_vector = expr_dict[expr_key]
    print(f"Successfully extracted expression vector for '{expr_key}'.")

    print(f"\n--- Phase 3: Initializing TRACE Model ---")
    base_model = TranslationBaseModel.from_config(args.config_path).to(args.device)
    base_model.add_head(
        "count",
        PsiteDensityHead.create_from_model(base_model, d_pred_h=384),
        overwrite=True
    )
    base_model.load_pretrained_weights(args.weights_path, strict=False)
    print(f"Model loaded onto {args.device}.")

    print(f"\n--- Phase 4: Running Translation Profile Prediction ---")
    predictor = TranslationProfilePredictor(model=base_model, fasta_files=args.fasta_files)
    
    # === [MODIFIED 3: 内部接口传参保持不变，仍使用 cell_type，但传入 args.patient_id] ===
    pkl_path = predictor.run(
        species="human",
        cell_type=args.patient_id, 
        cell_expr_vector=expr_vector,
        target_tids=tumor_specific_tx,
        out_dir=args.out_dir,
        suffix=args.patient_id, 
        min_len=200,
        max_len=10000,
        batch_size=args.batch_size
    )

    print(f"\n--- Phase 5: Calling High-Confidence ORFs (With TPM Integration) ---")
    
    temp_tpm_path = None
    if args.tpm_csv and os.path.exists(args.tpm_csv):
        print(f"Adapting TPM Matrix column '{args.tumor_run_id}' to match Output patient_id '{args.patient_id}'...")
        temp_df = pd.read_csv(args.tpm_csv, index_col=0)
        # === [MODIFIED 4: TPM 矩阵重命名逻辑更新] ===
        if args.tumor_run_id in temp_df.columns:
            pt_tpm_df = temp_df[[args.tumor_run_id]].rename(columns={args.tumor_run_id: args.patient_id})
            temp_tpm_path = os.path.join(args.out_dir, f"temp_tpm_{args.patient_id}.csv")
            pt_tpm_df.to_csv(temp_tpm_path)
        else:
            print(f"[Warning] '{args.tumor_run_id}' not found in TPM matrix. TPM integration will be skipped.")

    # === [MODIFIED 5: ORF Caller 内部接口保持 cell_type 不变] ===
    orf_caller = TranslationSignalORFCaller(
        fasta_files=args.fasta_files,
        pkl_file=pkl_path,
        cell_type=args.patient_id,
        tpm_csv_path=temp_tpm_path,
        tpm_level=args.tpm_level,
        mapping_csv_path=args.tx2gene_mapping
    )

    df_orfs = orf_caller.run(
        out_dir=args.out_dir,
        start_codons=['ATG', 'CTG', 'GTG', 'TTG', 'ACG'],
        min_len=30,
        mode=args.mode,
        use_mane_filter=False,
        plot_density=False,
        hard_thresh_intensity=0,
        hard_thresh_periodicity=0.5,
        hard_thresh_uniformity=0.3,
        hard_thresh_step_up=0.6,
        hard_thresh_drop_off=0.6
    )
    
    if temp_tpm_path and os.path.exists(temp_tpm_path):
        os.remove(temp_tpm_path)
    
    print(f"\n✅ Pipeline complete! {len(df_orfs)} ORFs called and saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
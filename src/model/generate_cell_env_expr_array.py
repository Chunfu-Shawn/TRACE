#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
import json
import sys

def _clean_id(value):
    """Normalize an Ensembl identifier while preserving its namespace."""
    return str(value).strip().split(".", 1)[0]


def _clean_sample_name(value):
    """Remove known alignment suffixes without truncating dots in sample names."""
    name = os.path.basename(str(value)).strip()
    for suffix in (".bam", ".sam"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for suffix in ("_uniq.sorted", ".uniq.sorted", "_sorted", ".sorted"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name:
        raise ValueError(f"Could not derive a sample name from column {value!r}.")
    return name


def _sample_rename_map(columns):
    """Build a collision-safe mapping from featureCounts columns to sample names."""
    rename_dict = {column: _clean_sample_name(column) for column in columns}
    names_to_columns = {}
    for column, sample_name in rename_dict.items():
        names_to_columns.setdefault(sample_name, []).append(str(column))
    collisions = {
        sample_name: original_columns
        for sample_name, original_columns in names_to_columns.items()
        if len(original_columns) > 1
    }
    if collisions:
        raise ValueError(
            "Sample names are not unique after removing known alignment suffixes: "
            f"{collisions}"
        )
    return rename_dict


def load_id_mapping(mapping_json_path):
    """Load every supported native gene ID and map it to the human anchor ID."""
    with open(mapping_json_path, "r", encoding="utf-8") as handle:
        anchor_to_native = json.load(handle)

    if not isinstance(anchor_to_native, dict):
        raise ValueError("The mapping JSON must contain an anchor-to-species mapping object.")

    id_mapping = {}
    namespace_by_id = {}
    for anchor, species_dict in anchor_to_native.items():
        clean_anchor = _clean_id(anchor)
        id_mapping[clean_anchor] = clean_anchor
        namespace_by_id[clean_anchor] = "Human"
        if not isinstance(species_dict, dict):
            continue
        for species, native_id in species_dict.items():
            if native_id is None or str(native_id).strip() == "":
                continue
            clean_native = _clean_id(native_id)
            previous = id_mapping.get(clean_native)
            if previous is not None and previous != clean_anchor:
                raise ValueError(
                    f"Gene ID {clean_native!r} maps to multiple anchors: "
                    f"{previous!r} and {clean_anchor!r}."
                )
            id_mapping[clean_native] = clean_anchor
            namespace_by_id[clean_native] = str(species)
    return id_mapping, namespace_by_id

def generate_cell_env_expr_dict(
    counts_file, 
    ref_order_path, 
    mapping_json_path, 
    quant_level='transcript',  # quant_level: level of the input count matrix ('gene' or 'transcript')
    tx2gene_file=None,        # transcript-to-gene mapping file; used when quant_level='transcript'
    min_tpm_threshold=0.0, 
    output_pt_path=None
):
    """
    Generates personalized, Z-scored expression vectors directly in memory.
    Supports both direct Gene-level inputs and Transcript-level inputs with on-the-fly RPK aggregation.
    """
    print(f"\n[ExprBuilder] Generating expression array from: {counts_file}")
    print(f"[ExprBuilder] Input Quantification Level: {quant_level.upper()}")
    
    # 1. Load Reference Order
    try:
        with open(ref_order_path, 'r') as f:
            reference_anchor_ids = [_clean_id(line) for line in f if line.strip()]
        if len(reference_anchor_ids) != len(set(reference_anchor_ids)):
            raise ValueError("Reference anchor order contains duplicate gene IDs.")
    except Exception as e:
        print(f"[ExprBuilder] Error loading reference order: {e}")
        sys.exit(1)

    # 2. Load Global ID Mapping (any supported Ensembl gene ID -> human anchor)
    id_mapping, namespace_by_id = load_id_mapping(mapping_json_path)

    # 3. Read featureCounts Matrix
    try:
        df = pd.read_csv(counts_file, sep='\t', comment='#')
    except Exception as e:
        print(f"[ExprBuilder] Error reading counts file: {e}")
        sys.exit(1)
        
    if len(df.columns) <= 6:
        print("[ExprBuilder] Error: featureCounts file lacks sample columns.")
        sys.exit(1)
        
    bam_cols = df.columns[6:]
    rename_dict = _sample_rename_map(bam_cols)
    df = df.rename(columns=rename_dict)
    sample_cols = list(rename_dict.values())
    
    required_columns = {"Geneid", "Length"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required featureCounts columns: {sorted(missing_columns)}")

    # Strip version suffixes from Ensembl IDs.
    df['Clean_ID'] = df['Geneid'].map(_clean_id)
    
    # 4. Determine Gene-Level Grouping Target
    if quant_level == 'transcript':
        if not tx2gene_file:
            print("[ExprBuilder] Error: tx2gene_file is required when quant_level is 'transcript'.")
            sys.exit(1)
            
        print(f"[ExprBuilder] Loading Tx-to-Gene mapping from: {tx2gene_file}")
        try:
            tx2gene_df = pd.read_csv(tx2gene_file, sep='\t')
            col_tx = tx2gene_df.columns[0]
            col_gene = tx2gene_df.columns[1]
            
            # Strip version suffixes from the mapping table
            tx_keys = tx2gene_df[col_tx].map(_clean_id)
            gene_vals = tx2gene_df[col_gene].map(_clean_id)
            tx2gene_map = dict(zip(tx_keys, gene_vals))
            
            df['Target_Gene_ID'] = df['Clean_ID'].map(tx2gene_map)
            unmapped = df['Target_Gene_ID'].isna().sum()
            if unmapped > 0:
                print(f"[Warning] {unmapped} transcripts could not be mapped to genes and will be dropped.")
            df = df.dropna(subset=['Target_Gene_ID']).copy()
            
        except Exception as e:
            print(f"[ExprBuilder] Error processing Tx-to-Gene mapping: {e}")
            sys.exit(1)
            
    elif quant_level == 'gene':
        # Input is already at the gene level; clean ID is used directly as the target
        df['Target_Gene_ID'] = df['Clean_ID']
        if tx2gene_file:
            print("[ExprBuilder] Warning: tx2gene_file provided but ignored since quant_level is 'gene'.")
            
    else:
        print(f"[ExprBuilder] Error: Invalid quant_level '{quant_level}'. Choose 'transcript' or 'gene'.")
        sys.exit(1)

    mapped_gene_ids = df['Target_Gene_ID'].map(id_mapping)
    detected_namespaces = (
        df['Target_Gene_ID'].map(namespace_by_id).dropna().value_counts()
    )
    mapped_count = int(mapped_gene_ids.notna().sum())
    if mapped_count == 0:
        examples = df['Target_Gene_ID'].head(5).tolist()
        raise ValueError(
            "None of the input gene IDs were found in the cross-species mapping. "
            f"Example cleaned IDs: {examples}"
        )
    namespace_summary = ", ".join(f"{name}={count}" for name, count in detected_namespaces.items())
    print(
        f"[ExprBuilder] Auto-detected supported gene IDs: {namespace_summary}; "
        f"mapped {mapped_count}/{len(df)} rows."
    )
    df['Anchor_ID'] = mapped_gene_ids
    df = df.dropna(subset=['Anchor_ID']).copy()

    # 5. Calculate RPK & Aggregate to Human Anchor Level
    print(f"[ExprBuilder] Calculating true RPK and assembling Gene-level TPM...")
    length_kb = pd.to_numeric(df['Length'], errors='coerce') / 1000.0
    valid_length = np.isfinite(length_kb) & (length_kb > 0)
    if not bool(valid_length.all()):
        dropped = int((~valid_length).sum())
        print(f"[ExprBuilder] Warning: dropping {dropped} rows with invalid feature lengths.")
        df = df.loc[valid_length].copy()
        length_kb = length_kb.loc[valid_length]

    rpk_df = pd.DataFrame({'Anchor_ID': df['Anchor_ID']}, index=df.index)
    for col in sample_cols:
        counts = pd.to_numeric(df[col], errors='coerce').fillna(0.0).clip(lower=0.0)
        rpk_df[col] = counts / length_kb

    # groupby is safe for both transcript- and gene-level input.
    # For gene input most rows are 1-to-1 (no-op groupby), so this is naturally compatible.
    gene_rpk_df = rpk_df.groupby('Anchor_ID')[sample_cols].sum()

    # 6. Calculate TPM and Robust Z-score from Gene RPKs
    gene_tpm_df = pd.DataFrame(index=gene_rpk_df.index)
    zscore_cols = []
    
    for col in sample_cols:
        rpk = gene_rpk_df[col]
        scaling_factor = rpk.sum() / 1e6
        if not np.isfinite(scaling_factor) or scaling_factor <= 0:
            raise ValueError(f"Sample {col!r} has no positive counts after ID mapping.")
        tpm = rpk / scaling_factor
        
        log_tpm = np.log2(tpm + 1.0)
        
        active_mask = tpm > min_tpm_threshold
        if active_mask.sum() > 0:
            active_mean = log_tpm[active_mask].mean()
            active_std = log_tpm[active_mask].std(ddof=0)
        else:
            active_mean = log_tpm.mean()
            active_std = log_tpm.std(ddof=0)
        
        z_score = (log_tpm - active_mean) / (active_std + 1e-8)
        z_col_name = f"{col}_Zscore"
        gene_tpm_df[z_col_name] = z_score
        zscore_cols.append(z_col_name)

    # 7. IDs are already normalized to human anchors before aggregation.
    gene_tpm_df = gene_tpm_df.reset_index()
    aligned_df = gene_tpm_df
    
    # 8. Core array coverage radar
    covered_anchors = set(aligned_df['Anchor_ID'].unique())
    ref_anchors_set = set(reference_anchor_ids)
    
    found_anchors = covered_anchors.intersection(ref_anchors_set)
    missing_anchors = ref_anchors_set.difference(covered_anchors)
    
    total_ref = len(ref_anchors_set)
    found_ref = len(found_anchors)
    coverage_pct = (found_ref / total_ref * 100) if total_ref > 0 else 0
    
    print(f"\n=============================================")
    print(f" 🎯 Anchor Gene Expression Coverage Report")
    print(f"=============================================")
    print(f" -> Total Anchors Required by TRACE    : {total_ref}")
    print(f" -> Anchors Found in Input Count Matrix: {found_ref}")
    print(f" -> Global Vector Integrity            : {coverage_pct:.2f}%")
    
    if len(missing_anchors) > 0:
        print(f" -> Missing Anchors (Top 5 examples): {list(missing_anchors)[:5]}...")
    print(f"=============================================\n")
    
    if coverage_pct < 50.0:
        print("[Warning] INTEGRITY ALERT! Your vector integrity is below 50%.")
        print("          TRACE translation models perform poorly with sparse expression inputs.")
        print("          Please ensure your featureCounts file contains global genomic expression, not just filtered subsets.")

    # If multiple Ensembl IDs map to the same Anchor ID, take the mean
    grouped_zscore = aligned_df.groupby('Anchor_ID')[zscore_cols].mean()

    # 9. Align to Reference Coordinate System
    final_df = grouped_zscore.reindex(reference_anchor_ids).fillna(0.0)
    
    # 10. Pack into Dictionary
    expr_dict = {}
    for col, z_col in zip(sample_cols, zscore_cols):
        expr_dict[col] = torch.tensor(final_df[z_col].values, dtype=torch.float16)
        
    print(f"[ExprBuilder] ✅ Generated vectors for {len(expr_dict)} samples.")
    
    if output_pt_path:
        os.makedirs(os.path.dirname(output_pt_path) or '.', exist_ok=True)
        torch.save(expr_dict, output_pt_path)
        print(f"[ExprBuilder] Saved dict to: {output_pt_path}")
        
    return expr_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate patient-specific Z-scored expression vectors.")
    parser.add_argument("-c", "--counts_file", required=True, help="Path to featureCounts matrix")
    parser.add_argument("-r", "--ref_order", required=True, help="Path to reference anchor order list")
    parser.add_argument("-m", "--mapping_json", required=True, help="Path to species mapping JSON")
    
    # Expose quant_level as a CLI argument
    parser.add_argument("-q", "--quant_level", default="gene", choices=['transcript', 'gene'], help="Level of quantification in the counts_file (default: gene)")
    parser.add_argument("-t", "--tx2gene", default=None, help="Path to Transcript-to-Gene mapping TSV (Required if quant_level is 'transcript')")
    
    parser.add_argument("-o", "--output_pt", required=True, help="Path to save output .pt dictionary")
    parser.add_argument("--min_tpm", type=float, default=0.0, help="Minimum TPM to consider a gene active")
    args = parser.parse_args()
    
    generate_cell_env_expr_dict(
        counts_file=args.counts_file,
        ref_order_path=args.ref_order,
        mapping_json_path=args.mapping_json,
        quant_level=args.quant_level,
        tx2gene_file=args.tx2gene,
        output_pt_path=args.output_pt,
        min_tpm_threshold=args.min_tpm
    )

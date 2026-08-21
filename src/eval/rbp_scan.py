# ============================================================
# CISBP-RNA Parser and Data Standardizer
# ============================================================
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.prepare_rbp_database import pre_annotate_and_save_database
from eval.rbp_translation_effect import (
    RBPMotifMutagenesisEvaluator,
    build_motif_position_profiles,
    collect_rbp_motif_hits,
    collect_unique_transcript_samples,
    discover_de_novo_translation_motifs,
    extract_signed_translation_attribution_windows,
    run_rbp_translation_effect_analysis,
    scan_pwm_hits,
    summarize_rbp_motif_effects,
    validate_rbp_pwm_library,
)
from plot.rbp_scan import (
    plot_motif_position_preference_heatmap,
    plot_rbp_metagene_heatmap,
    plot_rbp_regulatory_bubble,
)

def parse_cisbp_pwms(pwm_dir):
    """
    Read individual PWM files from a CISBP directory.

    Skip the ``Pos A C G U`` header and return a dictionary compatible
    with the ATtRACT representation: ``{Motif_ID: np.array([L, 4])}``.
    """
    pwms = {}
    print(f"Scanning CISBP-RNA PWM directory: {pwm_dir} ...")
    
    file_list = [f for f in os.listdir(pwm_dir) if f.endswith('.txt') and f.startswith('M')]
    
    for filename in file_list:
        motif_id = filename.replace('.txt', '')
        filepath = os.path.join(pwm_dir, filename)
        try:
            # Skip the header and read only the A, C, G, and U columns.
            matrix = np.loadtxt(filepath, skiprows=1, usecols=(1, 2, 3, 4), dtype=np.float32)
            
            # Preserve a two-dimensional shape for single-position motifs.
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, 4)
                
            pwms[motif_id] = matrix
        except Exception as e:
            print(f"Warning: Failed to parse {filename} - {e}")
            
    print(f"Successfully loaded {len(pwms)} PWMs from CISBP-RNA.")
    return pwms


def load_cisbp_metadata(info_path):
    """
    Read CISBP RBP metadata and standardize it to the ATtRACT schema.
    """
    print(f"Parsing CISBP-RNA Metadata: {info_path}")
    df = pd.read_csv(info_path, sep='\t')
    
    # Remove RBPs that have no associated motif matrix.
    df = df[df['Motif_ID'] != '.']
    
    # Standardize field names to match the ATtRACT metadata table.
    std_df = pd.DataFrame({
        'Matrix_id': df['Motif_ID'],
        'Gene_name': df['RBP_Name'],
        'Gene_id': df['DBID'],         # Ensembl ID
        'Family': df['Family_Name'],
        # Retain the source type for downstream provenance tracking.
        'Database': 'CISBP (' + df['MSource_Type'].astype(str) + ')',
        # CISBP does not provide a consensus string, so retain the motif ID.
        'Motif': df['Motif_ID'] 
    })
    
    return std_df

# ============================================================
# Phase 3B: PWM Parsing and TOMTOM-like Alignment Engine
# ============================================================
def parse_attract_pwms(pwm_path):
    """
    Parses the ATtRACT pwm.txt file into a dictionary of numpy arrays.
    Each matrix row represents frequencies/probabilities for columns: [A, C, G, T].
    """
    pwms = {}
    current_id = None
    current_matrix = []
    
    print(f"Parsing ATtRACT PWM file: {pwm_path}")
    with open(pwm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Save previous matrix before switching to a new entry
                if current_id is not None and current_matrix:
                    pwms[current_id] = np.array(current_matrix, dtype=np.float32)
                
                # Header format: >matrix_id \t length
                parts = line[1:].split()
                current_id = parts[0]
                current_matrix = []
            else:
                # Convert the sequence rows of frequencies into numeric arrays
                freqs = [float(x) for x in line.split()]
                if len(freqs) == 4:
                    current_matrix.append(freqs)
                    
        # Don't forget to capture the last element in file
        if current_id is not None and current_matrix:
            pwms[current_id] = np.array(current_matrix, dtype=np.float32)
            
    print(f"Successfully loaded {len(pwms)} Position Weight Matrices from database.")
    return pwms




def compute_tomtom_similarity(matrix_q, matrix_t, min_overlap=4):
    """
    Calculates the maximum alignment similarity between a Query and Target matrix 
    using a sliding window Pearson Correlation Coefficient (similar to MEME TOMTOM).
    
    Args:
        matrix_q: Numpy array of shape [L_query, 4] (Cluster matrix)
        matrix_t: Numpy array of shape [L_target, 4] (Database PWM matrix)
        min_overlap: Minimum overlapping nucleotides required during alignment.
    Returns:
        max_pcc: Highest average column-to-column Pearson correlation observed.
        best_shift: Relative displacement coordinate maximizing the score.
    """
    l_q = len(matrix_q)
    l_t = len(matrix_t)
    max_pcc = -1.0
    best_shift = 0
    
    # Slide matrix_t relative to matrix_q
    # Shift represents the starting index of matrix_t on the timeline of matrix_q
    min_shift = -l_t + min_overlap
    max_shift = l_q - min_overlap
    
    for shift in range(min_shift, max_shift + 1):
        # Determine the boundaries of the overlapping segments
        overlap_q_start = max(0, shift)
        overlap_q_end = min(l_q, shift + l_t)
        
        overlap_t_start = max(0, -shift)
        overlap_t_end = min(l_t, l_q - shift)
        
        sub_q = matrix_q[overlap_q_start:overlap_q_end]
        sub_t = matrix_t[overlap_t_start:overlap_t_end]
        
        current_overlap_len = len(sub_q)
        if current_overlap_len < min_overlap:
            continue
            
        # Compute column-by-column Pearson Correlation over the 4-dimensional profiles
        pccs = []
        for col_q, col_t in zip(sub_q, sub_t):
            # Variance check to prevent division by zero in homogeneous columns
            if np.std(col_q) == 0 or np.std(col_t) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(col_q, col_t)[0, 1]
                if np.isnan(corr): corr = 0.0
            pccs.append(corr)
            
        # Take the mean correlation over the entire length of the active overlap segment
        mean_pcc = np.mean(pccs)
        if mean_pcc > max_pcc:
            max_pcc = mean_pcc
            best_shift = shift
            
    return float(max_pcc), best_shift


# ============================================================
# Unified TOMTOM Annotation Engine (With Functional Mapping)
# ============================================================
def annotate_motifs_with_unified_tomtom(all_motifs_df, combined_pwm_library, combined_metadata, 
                                        out_dir, min_pcc=0.75):
    """
    Executes TOMTOM matrix scanning against a unified memory database.
    [Feature]: Integrates a dual-key exact match lookup (ENSG ID -> Gene Symbol) 
    against a user-provided RBP functional annotation database to append biological context.
    """
    if all_motifs_df.empty or 'Motif_Name' not in all_motifs_df.columns:
        print("Empty motif dataframes, skipping TOMTOM annotation pipeline.")
        return None
        
    os.makedirs(out_dir, exist_ok=True)

    unique_clusters = all_motifs_df['Motif_Name'].unique()
    char_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    tomtom_records = []
    
    print("\nExecuting Matrix-level Alignment Scanner against Unified Database...")
    for cluster_name in tqdm(unique_clusters, desc="Scanning Clusters"):
        cluster_seqs = all_motifs_df[all_motifs_df['Motif_Name'] == cluster_name]['sequence'].tolist()
        if not cluster_seqs: continue
            
        # Reconstruct Query Matrix
        seq_len = len(cluster_seqs[0])
        query_matrix = np.zeros((seq_len, 4), dtype=np.float32)
        for seq in cluster_seqs:
            for idx, char in enumerate(seq):
                if char in char_idx:
                    query_matrix[idx, char_idx[char]] += 1
                    
        query_matrix = query_matrix / (len(cluster_seqs) + 1e-9)
        
        # Scan across all target PWMs
        for matrix_id, target_matrix in combined_pwm_library.items():
            pcc_score, alignment_shift = compute_tomtom_similarity(query_matrix, target_matrix)
            
            if pcc_score >= min_pcc:
                meta_matches = combined_metadata[combined_metadata['Matrix_id'] == matrix_id]
                if meta_matches.empty: continue
                    
                first_match = meta_matches.iloc[0]
                
                ensembl_id = str(first_match['Gene_id']).strip()
                gene_name = str(first_match['Gene_name']).strip()
                
                # Fetch biological function
                rbp_func = first_match.get('RBP_Function', 'Unannotated')
                rbp_go_bp = first_match.get('RBP_GO_BP', 'Unannotated') 
                
                tomtom_records.append({
                    'Discovered_Motif_Cluster': cluster_name,
                    'Predicted_RBP': gene_name,
                    'RBP_Ensembl_ID': ensembl_id,
                    'Database_Matrix_ID': matrix_id,
                    'TOMTOM_PCC_Score': round(pcc_score, 4),
                    'Alignment_Shift': alignment_shift,
                    'RBP_GO_BP': rbp_go_bp,
                    'RBP_Function': rbp_func,
                    'Database_Source': first_match['Database'],
                    'RBP_Family': first_match['Family'],
                    'Reference_Motif_String': first_match['Motif']
                })
                
    if tomtom_records:
        report_df = pd.DataFrame(tomtom_records)
        report_df = report_df.sort_values(['Discovered_Motif_Cluster', 'TOMTOM_PCC_Score'], ascending=[True, False])
        
        # Deduplicate to keep the highest scoring alignment per RBP per cluster
        report_df = report_df.drop_duplicates(subset=['Discovered_Motif_Cluster', 'Predicted_RBP'], keep='first')
        
        csv_out_path = os.path.join(out_dir, "Unified_TOMTOM_RBP_Annotations.csv")
        report_df.to_csv(csv_out_path, index=False)
        
        print(f"\n✅ Unified TOMTOM pipeline completed!")
        print(f"Master file saved to: {csv_out_path}")
        
        print("\n--- Top RBP Matrix Hits per Cluster ---")
        display_cols = ['Discovered_Motif_Cluster', 'Predicted_RBP', 'TOMTOM_PCC_Score', 'RBP_GO_BP', 'RBP_Function']
        print(report_df.groupby('Discovered_Motif_Cluster').head(2)[display_cols].to_string(index=False))
        return report_df
    else:
        print(f"\nNo target matrix matched the minimum threshold criteria (PCC >= {min_pcc}).")
        return pd.DataFrame()
    

def _score_sequence_with_pwm(seq, pwm, min_match_score=0.85):
    """
    Scan a sequence with a PWM and normalize the best score to [0, 1].
    """
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    W = len(pwm)
    L = len(seq)
    if L < W: return False
    
    # Sum row maxima to obtain the theoretical maximum PWM score.
    max_possible_score = np.sum(np.max(pwm, axis=1))
    if max_possible_score == 0: return False
    
    best_norm_score = 0.0
    
    # Scan all full-length windows.
    for i in range(L - W + 1):
        window = seq[i:i+W]
        score = 0.0
        valid = True
        for j, char in enumerate(window):
            if char in char_map:
                score += pwm[j, char_map[char]]
            else:
                valid = False
                break
        
        if valid:
            norm_score = score / max_possible_score
            if norm_score > best_norm_score:
                best_norm_score = norm_score
                
    return best_norm_score >= min_match_score


def rbp_centric_peak_scanner(high_te_dfs, low_te_dfs, unified_pwms, unified_meta,
                             out_dir, min_match_score=0.85):
    """
    Legacy descriptive scanner over High- and Low-TE attention peaks.

    Use ``run_rbp_translation_effect_analysis`` for directional matched
    perturbation evidence. This function is retained for descriptive analyses.

    For each RBP:
      - Scans High-TE and Low-TE attention peak sequences.
      - Computes mean attention score from High-TE matched peaks.
      - Computes Enrichment_Ratio: (High hit-rate) / (Low hit-rate),
        normalised by the total number of peaks in each group so that
        unequal group sizes do not bias the ratio.
      - Also tracks spatial distribution (5UTR/CDS/3UTR hits).

    Args:
        high_te_dfs: dict {region_name: DataFrame} from extract_attn_peaks_by_region (High-TE).
        low_te_dfs:  dict {region_name: DataFrame} from extract_attn_peaks_by_region (Low-TE).
        unified_pwms: {Motif_ID: np.array(L,4)} merged PWM library.
        unified_meta: DataFrame with columns [Matrix_id, Gene_name, ...].
        out_dir: output directory.
        min_match_score: minimum PWM match score (0-1).

    Returns:
        result_df: DataFrame with columns
            RBP_Name, Total_Hits, High_Hits, Low_Hits,
            5UTR_Hits, CDS_Hits, 3UTR_Hits,
            Mean_Attention, Enrichment_Ratio.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Merge High-TE peaks
    all_high = []
    for region, df in high_te_dfs.items():
        if not df.empty:
            all_high.append(df)
    master_high = pd.concat(all_high, ignore_index=True) if all_high else pd.DataFrame()

    # Merge Low-TE peaks
    all_low = []
    for region, df in low_te_dfs.items():
        if not df.empty:
            all_low.append(df)
    master_low = pd.concat(all_low, ignore_index=True) if all_low else pd.DataFrame()

    n_high_peaks = len(master_high)
    n_low_peaks = len(master_low)
    print(f"Scanning {len(unified_pwms)} RBP PWMs: "
          f"High-TE peaks={n_high_peaks}, Low-TE peaks={n_low_peaks}")

    if n_high_peaks == 0:
        print("No High-TE peaks provided — aborting.")
        return pd.DataFrame()

    has_attn = 'mean_attn' in master_high.columns

    # Helper: scan one master_peaks DataFrame, return {rbp_name: {hits_5utr, hits_cds, hits_3utr, total, attns}}
    def _scan_one(master):
        results = {}
        for rbp_name, matrix_ids in rbp_grouped.items():
            mats = [unified_pwms[mid] for mid in matrix_ids if mid in unified_pwms]
            if not mats:
                continue
            h5, hc, h3 = 0, 0, 0
            attns = []
            for _, row in master.iterrows():
                seq = row['sequence']
                region = row['Region']
                if any(_score_sequence_with_pwm(seq, p, min_match_score) for p in mats):
                    if region == '5UTR':
                        h5 += 1
                    elif region == 'CDS':
                        hc += 1
                    elif region == '3UTR':
                        h3 += 1
                    if has_attn:
                        attns.append(float(row['mean_attn']))
            total = h5 + hc + h3
            if total > 0:
                results[rbp_name] = {
                    'hits_5utr': h5, 'hits_cds': hc, 'hits_3utr': h3,
                    'total': total, 'attns': attns,
                }
        return results

    # Group by RBP once
    rbp_grouped = unified_meta.groupby('Gene_name')['Matrix_id'].apply(list).to_dict()

    # Scan both groups
    high_results = _scan_one(master_high)
    low_results = _scan_one(master_low) if n_low_peaks > 0 else {}

    # Merge and build final table
    all_rbps = set(high_results.keys()) | set(low_results.keys())
    rbp_results = []

    for rbp_name in all_rbps:
        h = high_results.get(rbp_name, {})
        l = low_results.get(rbp_name, {})
        h_total = h.get('total', 0)
        l_total = l.get('total', 0)
        total_hits = h_total + l_total

        if total_hits < 5:
            continue

        mean_attn = np.mean(h['attns']) if h.get('attns') else np.nan
        # Normalize by group size before computing ratio (peak counts differ
        # between High-TE and Low-TE groups — see logs).  +1e-5 pseudocount
        # avoids division by zero without meaningfully inflating the ratio.
        h_rate = h_total / n_high_peaks
        l_rate = l_total / n_low_peaks if n_low_peaks > 0 else 0.0
        ratio = (h_rate + 1e-5) / (l_rate + 1e-5)

        rbp_results.append({
            'RBP_Name': rbp_name,
            'Total_Hits': total_hits,
            'High_Hits': h_total,
            'Low_Hits': l_total,
            '5UTR_Hits': h.get('hits_5utr', 0) + l.get('hits_5utr', 0),
            'CDS_Hits': h.get('hits_cds', 0) + l.get('hits_cds', 0),
            '3UTR_Hits': h.get('hits_3utr', 0) + l.get('hits_3utr', 0),
            'Mean_Attention': mean_attn,
            'Enrichment_Ratio': ratio,
        })

    if not rbp_results:
        print("No RBPs passed the matching thresholds.")
        return pd.DataFrame()

    result_df = pd.DataFrame(rbp_results)
    result_df = result_df.sort_values('Total_Hits', ascending=False)

    save_path = os.path.join(out_dir, "RBP_Centric_Landscape.csv")
    result_df.to_csv(save_path, index=False)
    print(f"\n[RBP Scan] Saved {len(result_df)} RBPs to {save_path}")
    return result_df


def score_and_map_peaks(master_peaks, unified_pwms, unified_meta, min_match_score=0.85):
    """Map attention peaks to RBPs for downstream spatial plots."""
    rbp_grouped = unified_meta.groupby('Gene_name')['Matrix_id'].apply(list).to_dict()
    mapped_records = []
    
    print("Mapping Peaks to exact RBPs for spatial distribution...")
    for rbp_name, matrix_ids in rbp_grouped.items():
        rbp_matrices = [unified_pwms[mid] for mid in matrix_ids if mid in unified_pwms]
        if not rbp_matrices: continue
        
        for _, row in master_peaks.iterrows():
            seq = row['sequence']
            if any(_score_sequence_with_pwm(seq, pwm, min_match_score) for pwm in rbp_matrices):
                mapped_records.append({
                    'RBP_Name': rbp_name,
                    'x_pos': row['x_pos'],
                    'Region': row['Region'],
                    'sequence': seq,
                })
                
    return pd.DataFrame(mapped_records)

# ============================================================
# CISBP-RNA Parser and Data Standardizer
# ============================================================
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_cisbp_pwms(pwm_dir):
    """
    遍历 CISBP 目录，读取所有单独的 PWM 文件，
    跳过第一行 (Pos A C G U)，只提取数值矩阵。
    返回与 ATtRACT 格式完全一致的字典：{Motif_ID: np.array([L, 4])}
    """
    pwms = {}
    print(f"Scanning CISBP-RNA PWM directory: {pwm_dir} ...")
    
    file_list = [f for f in os.listdir(pwm_dir) if f.endswith('.txt') and f.startswith('M')]
    
    for filename in file_list:
        motif_id = filename.replace('.txt', '')
        filepath = os.path.join(pwm_dir, filename)
        try:
            # skiprows=1 跳过表头，usecols=(1,2,3,4) 只读取 A C G U 四列数值
            matrix = np.loadtxt(filepath, skiprows=1, usecols=(1, 2, 3, 4), dtype=np.float32)
            
            # 如果矩阵只有一行，确保它是二维的形状 (1, 4)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, 4)
                
            pwms[motif_id] = matrix
        except Exception as e:
            print(f"Warning: Failed to parse {filename} - {e}")
            
    print(f"Successfully loaded {len(pwms)} PWMs from CISBP-RNA.")
    return pwms


def load_cisbp_metadata(info_path):
    """
    读取 CISBP 的 RBP_Information_all_motifs.txt，
    筛选并重命名列，使其与 ATtRACT 的元数据格式无缝对接。
    """
    print(f"Parsing CISBP-RNA Metadata: {info_path}")
    df = pd.read_csv(info_path, sep='\t')
    
    # 过滤掉没有 Motif_ID 的行 (有些 RBP 在库里没有对应矩阵)
    df = df[df['Motif_ID'] != '.']
    
    # 统一命名映射字典，对齐之前 ATtRACT 的字段名
    std_df = pd.DataFrame({
        'Matrix_id': df['Motif_ID'],
        'Gene_name': df['RBP_Name'],
        'Gene_id': df['DBID'],         # Ensembl ID
        'Family': df['Family_Name'],
        # 拼接数据源标识，方便最后溯源
        'Database': 'CISBP (' + df['MSource_Type'].astype(str) + ')',
        # CISBP 没有直接提供 Consensus String，用固定占位符或 Motif_ID 替代
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
    import numpy as np
    
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


import os
import requests
import time
import pickle
import pandas as pd
from tqdm import tqdm

def pre_annotate_and_save_database(combined_pwms, combined_meta, out_dir):
    """
    通过 MyGene.info API 预先注释合并后的元数据，并将完整的超集数据库本地固化。
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n--- Phase 1: Pre-annotating Metadata with MyGene.info ---")
    
    # 提取所有去重的 Ensembl IDs (过滤掉空值)
    unique_ensgs = combined_meta['Gene_id'].dropna().unique()
    print(f"Found {len(unique_ensgs)} unique Ensembl IDs to annotate.")
    
    annotation_cache = {}
    
    # 遍历抓取 API
    for ensg in tqdm(unique_ensgs, desc="Fetching API"):
        # 确保是合法的 ENSG 格式
        ensg_clean = str(ensg).strip()
        if not ensg_clean.startswith("ENSG"):
            annotation_cache[ensg] = "Unannotated (Invalid ID)"
            continue
            
        url = f"https://mygene.info/v3/gene/{ensg_clean}?fields=summary,name"
        
        try:
            time.sleep(0.1)  # 遵守 API 速率限制
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # 优先取 summary，如果没有则取 name，都没有则给提示
                func_desc = data.get('summary', data.get('name', 'Summary unavailable in NCBI.'))
                annotation_cache[ensg] = func_desc
            elif response.status_code == 404:
                annotation_cache[ensg] = "Gene not found in MyGene database."
            else:
                annotation_cache[ensg] = f"HTTP {response.status_code}"
                
        except Exception as e:
            annotation_cache[ensg] = "API Fetch Error"

    # 将抓取到的注释映射回 combined_meta 表中，新建 RBP_Function 列
    combined_meta['RBP_Function'] = combined_meta['Gene_id'].map(annotation_cache)
    
    print("\n--- Phase 2: Saving Unified Database to Disk ---")
    
    # 1. 保存注释好的 Metadata 为 TSV
    meta_save_path = os.path.join(out_dir, "Unified_RBP_Metadata_Annotated.tsv")
    combined_meta.to_csv(meta_save_path, sep='\t', index=False)
    print(f"✅ Annotated Metadata saved: {meta_save_path}")
    
    # 2. 保存 PWM 字典为 Pickle (无损二进制格式，加载极快)
    pwm_save_path = os.path.join(out_dir, "Unified_RBP_PWMs.pkl")
    with open(pwm_save_path, 'wb') as f:
        pickle.dump(combined_pwms, f)
    print(f"✅ Unified PWM Dictionary saved: {pwm_save_path}")
    
    return combined_meta


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
    import numpy as np
    
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
    import os
    import pandas as pd
    import numpy as np
    
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
                
                tomtom_records.append({
                    'Discovered_Motif_Cluster': cluster_name,
                    'Predicted_RBP': gene_name,
                    'RBP_Ensembl_ID': ensembl_id,
                    'Database_Matrix_ID': matrix_id,
                    'TOMTOM_PCC_Score': round(pcc_score, 4),
                    'Alignment_Shift': alignment_shift,
                    'RBP_Function': rbp_func,         # <--- Newly appended functional context
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
        display_cols = ['Discovered_Motif_Cluster', 'Predicted_RBP', 'TOMTOM_PCC_Score', 'RBP_Function']
        print(report_df.groupby('Discovered_Motif_Cluster').head(2)[display_cols].to_string(index=False))
        return report_df
    else:
        print(f"\nNo target matrix matched the minimum threshold criteria (PCC >= {min_pcc}).")
        return pd.DataFrame()
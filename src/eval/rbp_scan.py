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


def pre_annotate_and_save_database(combined_pwms, combined_meta, out_dir):
    """
    Pre-annotate merged metadata using the MyGene.info API and solidify the database locally.
    [Updated]: Dynamically extracts GO Biological Process (BP) terms alongside standard summaries
               to facilitate downstream functional clustering of RNA-binding proteins.
    """
    import os
    import requests
    import time
    import pickle
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)
    print("\n--- Phase 1: Pre-annotating Metadata with MyGene.info ---")
    
    unique_ensgs = combined_meta['Gene_id'].dropna().unique()
    print(f"Found {len(unique_ensgs)} unique Ensembl IDs to annotate.")
    
    # 初始化缓存字典，分别存储Summary和GO_BP描述
    summary_cache = {}
    go_bp_cache = {}
    
    for ensg in tqdm(unique_ensgs, desc="Fetching API"):
        ensg_clean = str(ensg).strip()
        if not ensg_clean.startswith("ENSG"):
            summary_cache[ensg] = "Unannotated (Invalid ID)"
            go_bp_cache[ensg] = "None"
            continue
            
        # [修改]: 增加 go.BP 到请求字段中
        url = f"https://mygene.info/v3/gene/{ensg_clean}?fields=summary,name,go.BP"
        
        try:
            time.sleep(0.1)  # Rate limiting safety buffer
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # 1. 提取基础功能摘要
                func_desc = data.get('summary', data.get('name', 'Summary unavailable in NCBI.'))
                summary_cache[ensg] = func_desc
                
                # 2. [新增]: 提取 GO 生物学过程 (Biological Process)
                go_data = data.get('go', {})
                bp_entries = go_data.get('BP', [])
                
                # MyGene API 的返回习惯：当只有1条GO时为字典，多条时为列表
                bp_terms = []
                if isinstance(bp_entries, list):
                    for entry in bp_entries:
                        term = entry.get('term')
                        if term: bp_terms.append(term)
                elif isinstance(bp_entries, dict):
                    term = bp_entries.get('term')
                    if term: bp_terms.append(term)
                
                # 将该基因捕获到的所有 BP 词条用分号连接
                if bp_terms:
                    go_bp_cache[ensg] = "; ".join(sorted(list(set(bp_terms))))
                else:
                    go_bp_cache[ensg] = "No BP terms annotated"
                    
            elif response.status_code == 404:
                summary_cache[ensg] = "Gene not found in MyGene."
                go_bp_cache[ensg] = "None"
            else:
                summary_cache[ensg] = f"HTTP {response.status_code}"
                go_bp_cache[ensg] = "None"
                
        except Exception as e:
            summary_cache[ensg] = "API Fetch Error"
            go_bp_cache[ensg] = "None"

    # 3. 将两组新特征无缝映射回联合元数据表中
    combined_meta['RBP_Function'] = combined_meta['Gene_id'].map(summary_cache)
    combined_meta['RBP_GO_BP'] = combined_meta['Gene_id'].map(go_bp_cache)
    
    print("\n--- Phase 2: Saving Unified Database to Disk ---")
    
    # 保存包含功能和GO标签的完整的元数据表为 TSV
    meta_save_path = os.path.join(out_dir, "Unified_RBP_Metadata_Annotated.tsv")
    combined_meta.to_csv(meta_save_path, sep='\t', index=False)
    print(f"✅ Annotated Metadata saved: {meta_save_path}")
    
    # 无损保存 NumPy 概率矩阵字典为 Pickle
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
    

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def _score_sequence_with_pwm(seq, pwm, min_match_score=0.85):
    """
    使用 PWM 矩阵在序列上滑动扫描。
    将匹配得分归一化为 0~1 之间，大于 min_match_score 视为命中。
    """
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    W = len(pwm)
    L = len(seq)
    if L < W: return False
    
    # 计算该 PWM 的理论最大得分 (每行取最大概率相加)
    max_possible_score = np.sum(np.max(pwm, axis=1))
    if max_possible_score == 0: return False
    
    best_norm_score = 0.0
    
    # 滑动窗口扫描
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


def rbp_centric_peak_scanner(region_dfs, unified_pwms, unified_meta, transcript_te_dict, out_dir, min_match_score=0.85):
    """
    以 RBP 为主角：使用 unified_pwms 扫描所有的 Attention Peaks。
    统计每个 RBP 在不同区域的分布，并计算结合该 RBP 的转录本的平均 TE。
    
    transcript_te_dict: 字典，格式为 {tid: te_value}，用于评估调控方向。
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 整合所有区域的 Peak 数据
    all_peaks = []
    for region, df in region_dfs.items():
        if not df.empty:
            all_peaks.append(df)
    
    if not all_peaks:
        print("No peaks provided.")
        return None
        
    master_peaks = pd.concat(all_peaks, ignore_index=True)
    print(f"Scanning {len(unified_pwms)} RBP PWMs against {len(master_peaks)} Attention Peaks...")
    
    # 用于按 RBP 合并同源矩阵
    rbp_grouped = unified_meta.groupby('Gene_name')['Matrix_id'].apply(list).to_dict()
    
    rbp_results = []
    
    for rbp_name, matrix_ids in tqdm(rbp_grouped.items(), desc="RBP-Centric Scanning"):
        # 获取该 RBP 对应的所有变体矩阵
        rbp_matrices = [unified_pwms[mid] for mid in matrix_ids if mid in unified_pwms]
        if not rbp_matrices: continue
        
        hits_5utr = 0
        hits_cds = 0
        hits_3utr = 0
        hit_tids = set()
        
        # 扫描每一个 Peak
        for _, row in master_peaks.iterrows():
            seq = row['sequence']
            region = row['Region']
            tid = row['tid']
            
            # 只要有一个变体矩阵命中，就算该 RBP 结合
            matched = any(_score_sequence_with_pwm(seq, pwm, min_match_score) for pwm in rbp_matrices)
            
            if matched:
                if region == '5UTR': hits_5utr += 1
                elif region == 'CDS': hits_cds += 1
                elif region == '3UTR': hits_3utr += 1
                hit_tids.add(tid)
                
        total_hits = hits_5utr + hits_cds + hits_3utr
        
        # 如果命中次数超过阈值（如至少在 5 个峰里找到），则进行 TE 分析
        if total_hits >= 5:
            # 计算带有该 RBP 结合位点的转录本的平均 TE
            te_values = [transcript_te_dict.get(tid, np.nan) for tid in hit_tids]
            te_values = [te for te in te_values if not np.isnan(te)]
            avg_te = np.mean(te_values) if te_values else np.nan
            
            # 计算分布偏好 (百分比)
            rbp_results.append({
                'RBP_Name': rbp_name,
                'Total_Hits': total_hits,
                '5UTR_Hits': hits_5utr,
                'CDS_Hits': hits_cds,
                '3UTR_Hits': hits_3utr,
                '5UTR_Ratio': hits_5utr / total_hits,
                'CDS_Ratio': hits_cds / total_hits,
                '3UTR_Ratio': hits_3utr / total_hits,
                'Bound_Transcripts_Avg_TE': avg_te
            })
            
    if not rbp_results:
        print("No RBPs passed the matching thresholds.")
        return pd.DataFrame()
        
    # 汇总并保存
    result_df = pd.DataFrame(rbp_results)
    # 按命中总数降序排列
    result_df = result_df.sort_values('Total_Hits', ascending=False)
    
    save_path = os.path.join(out_dir, "RBP_Centric_Landscape.csv")
    result_df.to_csv(save_path, index=False)
    print(f"\n✅ RBP-Centric Landscape saved to: {save_path}")
    
    return result_df


# ============================================================
# Notebook Cell: RBP Landscape Bubble Plot & Metagene Heatmap
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import logomaker
from plotnine import (ggplot, aes, geom_tile, geom_vline, scale_fill_gradient, 
                      labs, theme_classic, theme, element_text, element_blank, element_line)

def _score_and_map_peaks(master_peaks, unified_pwms, unified_meta, min_match_score=0.85):
    """
    内部辅助函数：用 PWM 扫描所有的 Peaks，返回带有 RBP 注释的详细坐标表，供热图使用。
    """
    rbp_grouped = unified_meta.groupby('Gene_name')['Matrix_id'].apply(list).to_dict()
    mapped_records = []
    
    print("Mapping Peaks to exact RBPs for spatial distribution...")
    for rbp_name, matrix_ids in rbp_grouped.items():
        rbp_matrices = [unified_pwms[mid] for mid in matrix_ids if mid in unified_pwms]
        if not rbp_matrices: continue
        
        for _, row in master_peaks.iterrows():
            seq = row['sequence']
            # 使用上一个 Cell 中的 _score_sequence_with_pwm 函数
            if any(_score_sequence_with_pwm(seq, pwm, min_match_score) for pwm in rbp_matrices):
                mapped_records.append({
                    'RBP_Name': rbp_name,
                    'x_pos': row['x_pos'],
                    'Region': row['Region'],
                    'sequence': seq # 保留序列用于后续画 Logo
                })
                
    return pd.DataFrame(mapped_records)


def plot_rbp_metagene_heatmap(mapped_peaks_df, out_path, FIXED_CDS_LEN=900, bin_size=20, up_len=300, down_len=300):
    """
    绘制基于 RBP 真实结合位点的 Metagene 概率热图。
    """
    if mapped_peaks_df.empty: return
    
    df_plot = mapped_peaks_df[(mapped_peaks_df['x_pos'] >= -up_len) & (mapped_peaks_df['x_pos'] <= FIXED_CDS_LEN + down_len)].copy()
    df_plot['x_bin'] = (df_plot['x_pos'] // bin_size) * bin_size + (bin_size / 2)
    
    heatmap_data = df_plot.groupby(['RBP_Name', 'x_bin']).size().reset_index(name='count')
    
    # 构建完整网格填充
    unique_rbps = heatmap_data['RBP_Name'].unique()
    all_bins = np.arange((-up_len // bin_size) * bin_size + (bin_size / 2), 
                         ((FIXED_CDS_LEN + down_len) // bin_size) * bin_size + (bin_size / 2) + bin_size, 
                         bin_size)
    full_df = pd.DataFrame(index=pd.MultiIndex.from_product([unique_rbps, all_bins], names=['RBP_Name', 'x_bin'])).reset_index()
    heatmap_data = pd.merge(full_df, heatmap_data, on=['RBP_Name', 'x_bin'], how='left').fillna({'count': 0})
    
    # 行归一化计算空间偏好
    rbp_totals = heatmap_data.groupby('RBP_Name')['count'].transform('sum')
    heatmap_data['Probability'] = heatmap_data['count'] / (rbp_totals + 1e-9)
    
    # 按照 5' -> 3' 排序 RBP
    peak_bins = heatmap_data.loc[heatmap_data.groupby('RBP_Name')['Probability'].idxmax()]
    ordered_rbps = peak_bins.sort_values(['x_bin', 'RBP_Name'], ascending=[False, False])['RBP_Name'].tolist()
    heatmap_data['RBP_Name'] = pd.Categorical(heatmap_data['RBP_Name'], categories=ordered_rbps)
    
    p = (
        ggplot(heatmap_data, aes(x='x_bin', y='RBP_Name', fill='Probability'))
        + geom_tile(color='white', size=0.1) 
        + scale_fill_gradient(low='#E0F3F8', high='#08306B', limits=(0, heatmap_data['Probability'].max() or 1.0)) 
        + geom_vline(xintercept=[0, FIXED_CDS_LEN], linetype='dashed', color='red', size=0.8)
        + labs(x=f'Metagene Position', y='RNA Binding Proteins', fill='Spatial\nProbability', title='RBP Spatial Distribution')
        + theme_classic()
        + theme(figure_size=(10, max(4, 0.4 * len(unique_rbps))), axis_text_y=element_text(size=9, weight='bold'), axis_line_x=element_blank(), axis_line_y=element_blank())
    )
    p.save(out_path)
    print(f"RBP Heatmap saved to {out_path}")


def plot_rbp_bubble_with_logos(rbp_landscape_df, mapped_peaks_df, highlight_rbps, out_path):
    """
    复现顶刊气泡图：绘制 RBP 特征散点，并为指定的高亮 RBP 添加悬浮 Motif Logo。
    X 轴: 5' UTR 富集比例 (反映空间偏好)
    Y 轴: TE 影响 (Bound_Transcripts_Avg_TE)
    气泡大小: 结合次数 (Total_Hits)
    """
    # 1. 创建主画布
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算 X: 5UTR 偏好指数 (防止除以0，加上伪计数)
    x_data = rbp_landscape_df['5UTR_Ratio'] 
    y_data = rbp_landscape_df['Bound_Transcripts_Avg_TE']
    sizes  = rbp_landscape_df['Total_Hits'] * 3 # 调整气泡显示倍率
    
    # 绘制基础气泡
    scatter = ax.scatter(x_data, y_data, s=sizes, alpha=0.6, c='#D62728', edgecolors='white', linewidth=1.5)
    
    # 坐标轴美化
    ax.set_xlabel("5' UTR Enrichment Ratio (Spatial Preference)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Translation Efficiency (TE) Impact", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. 为指定的 RBP 绘制悬浮 Logo
    # 设置插图的相对大小和偏置角度，防止重叠
    offsets = [(-0.15, 0.15), (0.15, 0.15), (-0.15, -0.15), (0.15, -0.15), (0, 0.2)]
    
    for idx, rbp in enumerate(highlight_rbps):
        rbp_data = rbp_landscape_df[rbp_landscape_df['RBP_Name'] == rbp]
        if rbp_data.empty: continue
            
        x_c = rbp_data['5UTR_Ratio'].values[0]
        y_c = rbp_data['Bound_Transcripts_Avg_TE'].values[0]
        
        # 提取该 RBP 在数据集中真实命中的序列来画 Logo
        rbp_seqs = mapped_peaks_df[mapped_peaks_df['RBP_Name'] == rbp]['sequence'].tolist()
        if not rbp_seqs: continue
            
        counts_df = logomaker.alignment_to_matrix(sequences=rbp_seqs, to_type='information')
        
        # 确定浮动轴的位置 (x, y, width, height)
        offset_x, offset_y = offsets[idx % len(offsets)]
        logo_x = x_c + offset_x
        logo_y = y_c + offset_y
        
        # 创建内嵌坐标系 (width=0.2, height=0.1 of axes fraction)
        ax_inset = ax.inset_axes([logo_x, logo_y, 0.25, 0.12], transform=ax.transData)
        
        # 使用 logomaker 绘制
        logo = logomaker.Logo(counts_df, ax=ax_inset, font_name='Arial Rounded MT Bold')
        logo.style_spines(visible=False)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title(rbp, fontsize=10, fontweight='bold', pad=2)
        
        # 绘制连接箭头 (Dashed arrow)
        con = ConnectionPatch(xyA=(logo_x + 0.125, logo_y), xyB=(x_c, y_c),
                              coordsA="data", coordsB="data",
                              axesA=ax, axesB=ax,
                              arrowstyle="->", color="#A0A0A0", linestyle="dashed", linewidth=1.5)
        ax.add_artist(con)
        
        # 高亮对应的气泡
        ax.scatter([x_c], [y_c], s=rbp_data['Total_Hits'].values[0]*3, color='#2CA02C', edgecolors='black', linewidth=2, zorder=5)

    plt.title("RBP Regulatory Landscape: Motif Integration", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Bubble plot with logos saved to {out_path}")
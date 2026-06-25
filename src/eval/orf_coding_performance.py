import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from plotnine import *


# =====================================================================
# 辅助函数: 动态寻找评估分数列
# =====================================================================
def resolve_score_col(df: pd.DataFrame, target_col: Optional[str]) -> str:
    if target_col and target_col in df.columns:
        return target_col
    for col in ['expr_score', 'translation_score', 'score']:
        if col in df.columns:
            return col
    raise ValueError(f"No valid score column found! Available columns: {df.columns.tolist()}")

# =====================================================================
# Module 1: Data Loading and Preprocessing (Array Preds + Dict GTs)
# =====================================================================
def load_and_filter_data(
        pred_csv_paths: List[str],               
        gt_csv_paths: Dict[str, str],            
        target_transcript_ids: Optional[List[str]] = None,
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        target_score_col: Optional[str] = None):
    
    # 1. Load Ground Truths
    gt_dfs = []
    print("--- Loading Ground Truth Data ---")
    for cell_type, gt_path in gt_csv_paths.items():
        if not os.path.exists(gt_path):
            print(f"  [Warning] GT file not found: {gt_path}. Skipping '{cell_type}'.")
            continue
            
        try:
            gt_df = pd.read_csv(gt_path, sep='\t')
            if 'Tid' not in gt_df.columns:
                gt_df = pd.read_csv(gt_path, sep=',')
        except Exception as e:
            raise ValueError(f"Error reading GT for {cell_type}: {e}")
            
        gt_df['Tid_clean'] = gt_df['Tid'].astype(str).apply(lambda x: x.split('.')[0] if x.startswith('ENST') else x)
        gt_df['start_gt'] = gt_df['CDS_Start_0based']
        gt_df['stop_gt'] = gt_df['CDS_End_0based']
        gt_df['length'] = gt_df['stop_gt'] - gt_df['start_gt']
        gt_df['Cell_Type'] = cell_type
        gt_dfs.append(gt_df)
        print(f"  -> Loaded '{cell_type}' GT: {len(gt_df)} records.")

    if not gt_dfs: raise ValueError("No valid Ground Truth data loaded!")
    master_gt_df = pd.concat(gt_dfs, ignore_index=True)
    valid_cell_types = set(master_gt_df['Cell_Type'].unique())

    # 2. Load Predictions
    pred_dfs = []
    print("\n--- Loading Prediction Data ---")
    for pred_path in pred_csv_paths:
        if not os.path.exists(pred_path):
            print(f"  [Warning] Prediction file not found: {pred_path}. Skipping...")
            continue
            
        pred_df = pd.read_csv(pred_path)
        if 'Cell_Type' not in pred_df.columns:
            raise ValueError(f"Prediction file {pred_path} is missing the required 'Cell_Type' column!")
            
        pred_df['Tid_clean'] = pred_df['Tid'].astype(str).apply(lambda x: x.split('.')[0] if x.startswith('ENST') else x)
        if 'length' not in pred_df.columns: pred_df['length'] = pred_df['stop'] - pred_df['start']
        pred_dfs.append(pred_df)
        print(f"  -> Loaded Pred chunk: {len(pred_df)} records.")

    if not pred_dfs: raise ValueError("No valid Prediction data loaded!")
    master_pred_df = pd.concat(pred_dfs, ignore_index=True)

    # 3. Align Valid Cell Types
    initial_pred_len = len(master_pred_df)
    master_pred_df = master_pred_df[master_pred_df['Cell_Type'].isin(valid_cell_types)]
    dropped_preds = initial_pred_len - len(master_pred_df)
    if dropped_preds > 0:
        print(f"  -> Dropped {dropped_preds} predictions whose Cell_Type lacks Ground Truth data.")

    global_score_col = resolve_score_col(master_pred_df, target_score_col)
    print(f"  -> Decided primary score column: '{global_score_col}'")

    # 4. Filter by Transcript IDs
    if target_transcript_ids is not None:
        print(f"\nFiltering entire dataset to {len(target_transcript_ids)} target transcripts...")
        target_set = set(str(t).split('.')[0] if str(t).startswith('ENST') else str(t) for t in target_transcript_ids)
        master_gt_df = master_gt_df[master_gt_df['Tid_clean'].isin(target_set)].copy()
        master_pred_df = master_pred_df[master_pred_df['Tid_clean'].isin(target_set)].copy()
        
    # 5. Filter by ORF Length
    if min_orf_len is not None or max_orf_len is not None:
        lower_bound = min_orf_len if min_orf_len is not None else 0
        upper_bound = max_orf_len if max_orf_len is not None else float('inf')
        master_gt_df = master_gt_df[(master_gt_df['length'] >= lower_bound) & (master_gt_df['length'] <= upper_bound)].copy()
        master_pred_df = master_pred_df[(master_pred_df['length'] >= lower_bound) & (master_pred_df['length'] <= upper_bound)].copy()

    if len(master_gt_df) == 0: raise ValueError("No Ground Truth data left after filtering!")

    # 6. Indexing
    master_gt_df = master_gt_df.reset_index(drop=True)
    master_gt_df['gt_idx'] = master_gt_df.index
    master_pred_df = master_pred_df.sort_values(global_score_col, ascending=False).reset_index(drop=True)
    master_pred_df['pred_idx'] = master_pred_df.index
    
    return master_pred_df, master_gt_df, global_score_col


# =====================================================================
# Module 2: Cell-Aware NMS Matching
# =====================================================================
def match_and_build_eval_df(pred_df: pd.DataFrame, gt_df: pd.DataFrame, eval_metrics: List[str], overlap_threshold: float) -> pd.DataFrame:
    print(f"\nCell-Aware Memory-Safe Matching (Frame Consistent & Overlap > {overlap_threshold*100}%)...")
    
    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        key = (row.Cell_Type, row.Tid_clean)
        if key not in gt_dict: gt_dict[key] = []
        gt_dict[key].append((row.gt_idx, row.start_gt, row.stop_gt))
        
    pred_to_gt = {} 
    matched_gt_indices = set()
    
    for row in pred_df.itertuples(index=False):
        key = (row.Cell_Type, row.Tid_clean)
        if key not in gt_dict: continue
            
        p_start, p_stop, p_idx, p_len = row.start, row.stop, row.pred_idx, row.length
        
        for g_idx, g_start, g_stop in gt_dict[key]:
            if g_idx in matched_gt_indices: continue 
            if p_start % 3 != g_start % 3: continue
                
            overlap_s = max(p_start, g_start)
            overlap_e = min(p_stop, g_stop)
            overlap_l = max(0, overlap_e - overlap_s)
            
            if overlap_l > 0:
                g_len = g_stop - g_start
                if (overlap_l / (p_len + g_len - overlap_l)) >= overlap_threshold:
                    pred_to_gt[p_idx] = g_idx
                    matched_gt_indices.add(g_idx)
                    break 

    print("Assembling Unified Evaluation DataFrame...")
    eval_records = []
    gt_lengths = dict(zip(gt_df['gt_idx'], gt_df['length']))
    
    for row in pred_df.itertuples(index=False):
        is_tp = row.pred_idx in pred_to_gt
        eval_len = gt_lengths[pred_to_gt[row.pred_idx]] if is_tp else row.length
        
        record = {'Cell_Type': row.Cell_Type, 'y_true': 1 if is_tp else 0, 'length': eval_len}
        for m in eval_metrics: record[m] = float(getattr(row, m, 0.0) if hasattr(row, m) else 0.0)
        eval_records.append(record)
        
    for row in gt_df.itertuples(index=False):
        if row.gt_idx not in matched_gt_indices:
            record = {'Cell_Type': row.Cell_Type, 'y_true': 1, 'length': row.length}
            for m in eval_metrics: record[m] = -1.0 
            eval_records.append(record)
            
    eval_df = pd.DataFrame(eval_records)
    print("-" * 40)
    print(f"Total Evaluated MS Ground Truth : {len(gt_df)}")
    print(f"Successfully Matched (TP)       : {len(matched_gt_indices)}")
    print(f"Missed Ground Truths (FN)       : {len(gt_df) - len(matched_gt_indices)}")
    print(f"False Positives (FP)            : {len(pred_df) - len(matched_gt_indices)}")
    print("-" * 40)
    return eval_df

# =====================================================================
# [MODIFIED] Module 3: Global Evaluation Plotting (Comprehensive Integration)
# =====================================================================
def evaluate_and_plot_global(eval_df: pd.DataFrame, eval_metrics: List[str], display_names: dict, out_dir: str):
    print("\nCalculating comprehensive metrics (ROC-AUC, PR-AUC, Best F1) globally and per cell type...")
    
    comprehensive_records = []
    roc_dfs, pr_dfs = [], []
    
    def subsample_curve(x_array, y_array, max_points=2000):
        if len(x_array) <= max_points: return x_array, y_array
        indices = np.linspace(0, len(x_array) - 1, max_points).astype(int)
        return x_array[indices], y_array[indices]

    # ---------------------------------------------------------
    # 1. 计算 Overall (所有细胞系汇总) 的性能指标
    # ---------------------------------------------------------
    y_true_all = eval_df['y_true'].values
    baseline_all = np.sum(y_true_all) / len(y_true_all) if len(y_true_all) > 0 else 0

    for metric in eval_metrics:
        scores = eval_df[metric].values
        d_name = display_names.get(metric, metric)
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_true_all, scores)
        roc_auc = auc(fpr, tpr)
        fpr_plot, tpr_plot = subsample_curve(fpr, tpr)
        roc_dfs.append(pd.DataFrame({'FPR': fpr_plot, 'TPR': tpr_plot, 'Metric': d_name, 'AUC': roc_auc}))
        
        # PR-AUC & Best F1
        prec, rec, _ = precision_recall_curve(y_true_all, scores)
        pr_auc = average_precision_score(y_true_all, scores)
        
        # 防止除零警告
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
        best_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0
        
        rec_plot, prec_plot = subsample_curve(rec, prec)
        pr_dfs.append(pd.DataFrame({'Recall': rec_plot, 'Precision': prec_plot, 'Metric': d_name, 'AUC': pr_auc}))
        
        # 记录到总表
        comprehensive_records.append({
            'Cell_Type': 'Overall',
            'Feature': d_name,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Best_F1': best_f1
        })

    # ---------------------------------------------------------
    # 2. 计算按 Cell_Type 拆分的性能指标
    # ---------------------------------------------------------
    for cell_type, group_df in eval_df.groupby('Cell_Type'):
        y_c = group_df['y_true'].values
        # 必须同时存在正负样本才能计算 AUC
        if sum(y_c) == 0 or sum(y_c) == len(y_c):
            continue
            
        for metric in eval_metrics:
            scores_c = group_df[metric].values
            d_name = display_names.get(metric, metric)
            
            # ROC-AUC
            fpr_c, tpr_c, _ = roc_curve(y_c, scores_c)
            roc_auc_c = auc(fpr_c, tpr_c)
            
            # PR-AUC & Best F1
            prec_c, rec_c, _ = precision_recall_curve(y_c, scores_c)
            pr_auc_c = average_precision_score(y_c, scores_c)
            
            f1_scores_c = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-9)
            best_f1_c = np.max(f1_scores_c) if len(f1_scores_c) > 0 else 0.0
            
            # 记录到总表
            comprehensive_records.append({
                'Cell_Type': cell_type,
                'Feature': d_name,
                'ROC-AUC': roc_auc_c,
                'PR-AUC': pr_auc_c,
                'Best_F1': best_f1_c
            })

    # ---------------------------------------------------------
    # 3. 保存大满贯 CSV 并绘制图形
    # ---------------------------------------------------------
    comprehensive_df = pd.DataFrame(comprehensive_records)
    comprehensive_df.to_csv(os.path.join(out_dir, "comprehensive_metrics_summary.csv"), index=False)
    print("  -> Saved unified metrics table to 'comprehensive_metrics_summary.csv'")

    # --- 绘图：整体曲线 ---
    all_roc_df = pd.concat(roc_dfs, ignore_index=True)
    all_pr_df = pd.concat(pr_dfs, ignore_index=True)
    all_roc_df['Legend'] = all_roc_df.apply(lambda row: f"{row['Metric']} (AUC={row['AUC']:.3f})", axis=1)
    all_pr_df['Legend'] = all_pr_df.apply(lambda row: f"{row['Metric']} (AUC={row['AUC']:.3f})", axis=1)

    color_palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f1c40f", "#34495e", "#e67e22", "#1abc9c", "#7f8c8d"]
    
    p_roc = (
        ggplot(all_roc_df, aes(x='FPR', y='TPR', color='Legend'))
        + geom_line(size=1.2, alpha=0.8) + geom_abline(intercept=0, slope=1, linetype='dashed', color='gray')
        + scale_color_manual(values=color_palette) + theme_bw()
        + labs(title="Overall ROC Curves (All Cell Types)", x="False Positive Rate", y="True Positive Rate")
        + theme(figure_size=(7, 6), panel_border=element_rect(color="black", size=1), legend_position="bottom", legend_title=element_blank())
    )
    p_roc.save(os.path.join(out_dir, "Overall_ROC_Curves.pdf"), verbose=False)

    p_pr = (
        ggplot(all_pr_df, aes(x='Recall', y='Precision', color='Legend'))
        + geom_line(size=1.2, alpha=0.8) + geom_hline(yintercept=baseline_all, linetype='dashed', color='gray')
        + scale_color_manual(values=color_palette) + theme_bw()
        + labs(title="Overall PR Curves (All Cell Types)", x="Recall", y="Precision")
        + theme(figure_size=(7, 6), panel_border=element_rect(color="black", size=1), legend_position="bottom", legend_title=element_blank())
    )
    p_pr.save(os.path.join(out_dir, "Overall_PR_Curves.pdf"), verbose=False)

    # --- 绘图：三合一指标热图 (仅展示 Overall) ---
    overall_df = comprehensive_df[comprehensive_df['Cell_Type'] == 'Overall']
    heatmap_data = overall_df.set_index('Feature')[['ROC-AUC', 'PR-AUC', 'Best_F1']].sort_values(by='ROC-AUC', ascending=False)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=1, linecolor='white')
    plt.title("Overall Metrics (ROC-AUC, PR-AUC, Best F1)", pad=15, fontsize=14)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Overall_Metrics_Heatmap.pdf"), dpi=300)
    plt.close()


# =====================================================================
# Main Orchestrator
# =====================================================================
def evaluate_orf_level_predictions(
        pred_csv_paths: List[str],               
        gt_csv_paths: Dict[str, str],            
        target_transcript_ids: Optional[List[str]] = None,
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        out_dir: str = "./results/eval",
        overlap_threshold: float = 0.70,
        target_score_col: Optional[str] = None):
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Filter and Load
    pred_df, gt_df, score_col = load_and_filter_data(
        pred_csv_paths, gt_csv_paths, target_transcript_ids, min_orf_len, max_orf_len, target_score_col)
    
    all_possible_metrics = {
        'expr_score': 'Expression Score (TPM*Signal)',
        'translation_score': 'Pure Translation Score',
        'transcription_score': 'Pure Transcription Score',
        'seq_score': 'Pure ORF-structure Score',
        'score': 'Final Score', 
        'mean_intensity': 'Mean Intensity', 
        'tri_nucleotide_periodicity': 'Periodicity',
        'uniformity_of_signal': 'Uniformity', 
        'step_up_contrast': 'Step-up Contrast', 
        'drop_off': 'Drop-off'
    }
    eval_metrics = [m for m in all_possible_metrics.keys() if m in pred_df.columns]
    print(f"\nDynamically selected metrics for evaluation: {eval_metrics}")
    
    display_names = {k: all_possible_metrics[k] for k in eval_metrics}

    # 2. Match
    eval_df = match_and_build_eval_df(pred_df, gt_df, eval_metrics, overlap_threshold)
    eval_df.to_csv(os.path.join(out_dir, "unified_evaluation_table.csv"), index=False)
    
    # 3. Base Threshold Summary (基于主分数)
    print("\nCalculating Threshold Summary on Primary Score...")
    tp_count = ((eval_df['y_true'] == 1) & (eval_df[score_col] >= 0)).sum()
    fp_count = (eval_df['y_true'] == 0).sum()
    total_preds = tp_count + fp_count
    overall_prec = tp_count / total_preds if total_preds > 0 else 0.0

    prec, rec, threshs = precision_recall_curve(eval_df['y_true'].values, eval_df[score_col].values)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    opt_idx = np.argmax(f1)
    opt_thresh = threshs[opt_idx] if opt_idx < len(threshs) else threshs[-1]
    best_tp = ((eval_df['y_true'] == 1) & (eval_df[score_col] >= opt_thresh) & (eval_df[score_col] >= 0)).sum()
    best_fp = ((eval_df['y_true'] == 0) & (eval_df[score_col] >= opt_thresh) & (eval_df[score_col] >= 0)).sum()

    pd.DataFrame({
        'Total_Predictions': [total_preds],
        'True_Positives_TP': [tp_count],
        'False_Positives_FP': [fp_count],
        'Overall_Precision': [overall_prec],
        'Best_F1_Score': [f1[opt_idx]],
        'Best_Threshold': [opt_thresh],
        'TP_at_Best_Threshold': [best_tp],
        'FP_at_Best_Threshold': [best_fp]
    }).to_csv(os.path.join(out_dir, "primary_score_threshold_summary.csv"), index=False)
    
    # 4. Global Plots & Comprehensive CSV Output
    evaluate_and_plot_global(eval_df, eval_metrics, display_names, out_dir)
    
    print(f"\n✅ All Evaluation processes successfully finished! Output directory: {out_dir}")



# =====================================================================
# Module 1 (Top-K): Precision@K Calculation Engine
# =====================================================================
def calculate_top_k_precision(
        pred_csv_path: str, 
        gt_csv_path: str, 
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        overlap_threshold: float = 0.70,
        target_score_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate the Precision@K for predicted ORFs against the Ground Truth.
    Matches are based on Frame consistency and spatial overlap.
    """
    print(f"\nLoading and preparing data for Precision@K evaluation...")
    
    gt_df = pd.read_csv(gt_csv_path, sep='\t' if '\t' in open(gt_csv_path).readline() else ',')
    
    # =================================================================
    # [MODIFIED] 安全清理 Ground Truth ID：仅对 ENST/ENSG 截断版本号
    # =================================================================
    gt_df['Tid_clean'] = gt_df['Tid'].astype(str).apply(
        lambda x: x.split('.')[0] if (x.startswith('ENST') or x.startswith('ENSG')) and '.' in x else x
    )
    
    gt_df['start_gt'] = gt_df['CDS_Start_0based']
    gt_df['stop_gt'] = gt_df['CDS_End_0based']
    gt_df['length'] = gt_df['stop_gt'] - gt_df['start_gt'] 
    
    pred_df = pd.read_csv(pred_csv_path)
    
    # =================================================================
    # [MODIFIED] 安全清理 Predictions ID：仅对 ENST/ENSG 截断版本号
    # =================================================================
    pred_df['Tid_clean'] = pred_df['Tid'].astype(str).apply(
        lambda x: x.split('.')[0] if (x.startswith('ENST') or x.startswith('ENSG')) and '.' in x else x
    )
    
    if 'length' not in pred_df.columns:
        pred_df['length'] = pred_df['stop'] - pred_df['start']

    score_col = resolve_score_col(pred_df, target_score_col)
    print(f"  -> Ranking predictions using column: '{score_col}'")
        
    if min_orf_len is not None or max_orf_len is not None:
        if min_orf_len is not None and max_orf_len is not None and min_orf_len > max_orf_len:
            raise ValueError(f"Invalid length range.")
            
        lower_bound = min_orf_len if min_orf_len is not None else 0
        upper_bound = max_orf_len if max_orf_len is not None else float('inf')
        
        gt_df = gt_df[(gt_df['length'] >= lower_bound) & (gt_df['length'] <= upper_bound)].copy()
        pred_df = pred_df[(pred_df['length'] >= lower_bound) & (pred_df['length'] <= upper_bound)].copy()
        
    if len(gt_df) == 0 or len(pred_df) == 0:
        print("Warning: No Ground Truth or Predicted ORFs left after filtering. Returning empty dataframe.")
        return pd.DataFrame(columns=['K', 'TP_Count', 'Precision'])

    pred_df = pred_df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    pred_df['pred_idx'] = pred_df.index
    
    print(f"Executing ultra-fast coordinate matching (Overlap > {overlap_threshold*100}%)...")
    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        if row.Tid_clean not in gt_dict:
            gt_dict[row.Tid_clean] = []
        gt_dict[row.Tid_clean].append((row.start_gt, row.stop_gt))
        
    is_tp_list = []
    
    for row in pred_df.itertuples(index=False):
        tid = row.Tid_clean
        p_start, p_stop, p_len = row.start, row.stop, row.length
        
        matched = False
        if tid in gt_dict:
            for g_start, g_stop in gt_dict[tid]:
                if p_start % 3 != g_start % 3: 
                    continue
                    
                overlap_s = max(p_start, g_start)
                overlap_e = min(p_stop, g_stop)
                overlap_l = max(0, overlap_e - overlap_s)
                
                if overlap_l > 0:
                    g_len = g_stop - g_start
                    if (overlap_l / (p_len + g_len - overlap_l)) >= overlap_threshold:
                        matched = True
                        break
        
        is_tp_list.append(1 if matched else 0)

    print("Calculating Cumulative Precision@K...")
    is_tp_array = np.array(is_tp_list)
    tp_cumsum = np.cumsum(is_tp_array)
    k_array = np.arange(1, len(is_tp_array) + 1)
    precision_at_k = tp_cumsum / k_array
    
    pk_df = pd.DataFrame({
        'K': k_array,
        'TP_Count': tp_cumsum,
        'Precision': precision_at_k,
        'Score_Type': score_col 
    })
    
    print(f"Done! Evaluated Top {len(pk_df)} predictions.")
    return pk_df

# =====================================================================
# Module 2 (Top-K): Precision@K Plotting Function
# =====================================================================
def plot_top_k_precision(pk_df: pd.DataFrame, out_dir: str = "./results/eval", max_k: Optional[int] = None):
    if pk_df.empty:
        print("Dataframe is empty, skipping plot generation.")
        return

    print("\nGenerating Precision@K line chart...")
    os.makedirs(out_dir, exist_ok=True)
    
    plot_df = pk_df.copy()
    if max_k is not None:
        plot_df = plot_df[plot_df['K'] <= max_k]
        
    if len(plot_df) > 5000:
        indices = np.linspace(0, len(plot_df) - 1, 5000).astype(int)
        plot_df = plot_df.iloc[indices]
        
    baseline_precision = pk_df['Precision'].iloc[-1]
    
    score_label = plot_df['Score_Type'].iloc[0] if 'Score_Type' in plot_df.columns else 'Final Score'
    
    p = (
        ggplot(plot_df, aes(x='K', y='Precision'))
        + geom_line(color="#2980b9", size=1.5, alpha=0.9)
        + geom_hline(yintercept=baseline_precision, linetype="dashed", color="#e74c3c", size=1)
        + theme_classic()
        + labs(
            title="Precision@K: Top Predicted ORFs vs Ground Truth",
            x=f"Top K Predicted ORFs (Ranked by {score_label})", 
            y="Precision (Proportion of True Positives)"
        )
        + annotate("text", x=plot_df['K'].max() * 0.2, y=baseline_precision - 0.05, 
                   label=f"Overall Baseline: {baseline_precision:.3f}", color="#e74c3c", size=10)
        + scale_y_continuous(limits=(0, 1.05))
        + scale_x_log10()
        + theme(
            figure_size=(6, 5),
            axis_title=element_text(size=12),
            axis_text=element_text(size=10)
        )
    )
    
    filename = f"TopK_Precision_Curve_{'All' if max_k is None else max_k}.pdf"
    save_path = os.path.join(out_dir, filename)
    p.save(save_path, dpi=300, verbose=False)
    
    print(f"Chart successfully saved to: {save_path}")
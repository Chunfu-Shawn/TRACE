import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from plotnine import *

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'


def normalize_transcript_id(transcript_id) -> str:
    """Remove version suffixes from ENST IDs while preserving all other IDs."""
    transcript_id = str(transcript_id)
    if transcript_id.startswith('ENST'):
        return transcript_id.split('.', 1)[0]
    return transcript_id


# =====================================================================
# Helper function: Dynamically resolve evaluation score column
# =====================================================================
def resolve_score_col(df: pd.DataFrame, target_col: Optional[str]) -> str:
    """Prioritize target_col if specified, otherwise search by fallback priority."""
    if target_col and target_col in df.columns:
        return target_col
    
    fallback_candidates = [
        'expr_score', 
        'translation_score', 
        'transcription_score', 
        'seq_score', 
        'score'
    ]
    
    for col in fallback_candidates:
        if col in df.columns:
            return col
            
    raise ValueError(f"No valid score column found! Available columns: {df.columns.tolist()}")

# =====================================================================
# Module 1: Data Loading and Preprocessing (Array Preds + Dict GTs)
# =====================================================================
def load_and_filter_data(
        pred_csv_paths: List[str],               
        gt_csv_paths: Dict[str, str],            
        target_transcript_ids: Optional[Union[List[str], Dict[str, List[str]]]] = None,
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
            
        gt_df['Tid_clean'] = gt_df['Tid'].apply(normalize_transcript_id)
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
            
        pred_df['Tid_clean'] = pred_df['Tid'].apply(normalize_transcript_id)
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

    # =================================================================
    # [MODIFIED] 4. Filter by Transcript IDs (支持字典按细胞系过滤)
    # =================================================================
    if target_transcript_ids is not None:
        if isinstance(target_transcript_ids, dict):
            print("\nFiltering datasets using cell-specific active transcripts...")
            filtered_gt, filtered_pred = [], []
            
            for ct in valid_cell_types:
                if ct in target_transcript_ids:
                    # 提取该细胞系对应的活跃转录本
                    target_set = set(
                        normalize_transcript_id(t)
                        for t in target_transcript_ids[ct]
                    )
                    
                    filtered_gt.append(master_gt_df[(master_gt_df['Cell_Type'] == ct) & (master_gt_df['Tid_clean'].isin(target_set))])
                    filtered_pred.append(master_pred_df[(master_pred_df['Cell_Type'] == ct) & (master_pred_df['Tid_clean'].isin(target_set))])
                    print(f"  -> {ct}: Kept {len(target_set)} target transcripts.")
                else:
                    print(f"  [Warning] No target transcripts provided for '{ct}'. This cell type will be dropped.")
            
            # 重新组装大表
            master_gt_df = pd.concat(filtered_gt, ignore_index=True) if filtered_gt else pd.DataFrame(columns=master_gt_df.columns)
            master_pred_df = pd.concat(filtered_pred, ignore_index=True) if filtered_pred else pd.DataFrame(columns=master_pred_df.columns)
            
        else:
            # 兼容原有的单一列表全局过滤模式
            print(f"\nFiltering entire dataset globally to {len(target_transcript_ids)} target transcripts...")
            target_set = set(
                normalize_transcript_id(t) for t in target_transcript_ids
            )
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
# Module 3: Global Evaluation Plotting (Comprehensive Integration)
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
    # 1. Calculate overall performance metrics across all cell types
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
        
        # Prevent division by zero warning
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
        best_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0
        
        rec_plot, prec_plot = subsample_curve(rec, prec)
        pr_dfs.append(pd.DataFrame({'Recall': rec_plot, 'Precision': prec_plot, 'Metric': d_name, 'AUC': pr_auc}))
        
        # Append to comprehensive records
        comprehensive_records.append({
            'Cell_Type': 'Overall',
            'Feature': d_name,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Best_F1': best_f1
        })

    # ---------------------------------------------------------
    # 2. Calculate performance metrics split by Cell_Type
    # ---------------------------------------------------------
    for cell_type, group_df in eval_df.groupby('Cell_Type'):
        y_c = group_df['y_true'].values
        # Both positive and negative samples must exist to calculate AUC
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
            
            # Append to comprehensive records
            comprehensive_records.append({
                'Cell_Type': cell_type,
                'Feature': d_name,
                'ROC-AUC': roc_auc_c,
                'PR-AUC': pr_auc_c,
                'Best_F1': best_f1_c
            })

    # ---------------------------------------------------------
    # 3. Save comprehensive CSV and plot figures
    # ---------------------------------------------------------
    comprehensive_df = pd.DataFrame(comprehensive_records)
    comprehensive_df.to_csv(os.path.join(out_dir, "comprehensive_metrics_summary.csv"), index=False)
    print("  -> Saved unified metrics table to 'comprehensive_metrics_summary.csv'")

    # --- Plot: Overall curves ---
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

    # --- Plot: 3-in-1 metric heatmap (Overall only) ---
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
        target_transcript_ids: Optional[Union[List[str], Dict[str, List[str]]]] = None,
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
    
    # 3. Base Threshold Summary (Based on primary score)
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
        pred_csv_paths: Optional[Union[str, List[str]]] = None,
        gt_csv_paths: Optional[Union[str, List[str], Dict[str, str]]] = None,
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        overlap_threshold: float = 0.70,
        target_score_col: Optional[str] = None,
        cell_type: Optional[str] = None,
        pred_csv_path: Optional[str] = None,
        gt_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate cell-aware Precision@K and Recall@K for ranked predicted ORFs.

    ``pred_csv_paths`` accepts one path or a list of paths. ``gt_csv_paths``
    accepts one path, a list of paths with embedded Cell_Type values, or a
    ``{cell_type: path}`` dictionary matching evaluate_orf_level_predictions.
    The singular path arguments are retained for backward compatibility.

    Predictions are greedily matched in descending score order. Each ground
    truth ORF can be matched at most once, using the highest-IoU eligible ORF
    within the same cell type, transcript, and reading frame.
    """
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1.")

    if pred_csv_paths is not None and pred_csv_path is not None:
        raise ValueError("Use either pred_csv_paths or pred_csv_path, not both.")
    if gt_csv_paths is not None and gt_csv_path is not None:
        raise ValueError("Use either gt_csv_paths or gt_csv_path, not both.")
    if pred_csv_paths is None:
        pred_csv_paths = pred_csv_path
    if gt_csv_paths is None:
        gt_csv_paths = gt_csv_path
    if pred_csv_paths is None or gt_csv_paths is None:
        raise ValueError("Both prediction and ground-truth paths are required.")

    print(f"\nLoading and preparing data for Top-K evaluation...")

    pred_paths = [pred_csv_paths] if isinstance(pred_csv_paths, str) else list(pred_csv_paths)
    if isinstance(gt_csv_paths, str):
        gt_entries = [(None, gt_csv_paths)]
    elif isinstance(gt_csv_paths, dict):
        gt_entries = list(gt_csv_paths.items())
    else:
        gt_entries = [(None, path) for path in gt_csv_paths]

    if not pred_paths or not gt_entries:
        raise ValueError("Prediction and ground-truth path collections cannot be empty.")

    pred_dfs = []
    for path in pred_paths:
        frame = pd.read_csv(path)
        frame['Prediction_Source'] = str(path)
        pred_dfs.append(frame)
        print(f"  -> Loaded predictions: {path} ({len(frame)} records)")

    gt_dfs = []
    for assigned_cell_type, path in gt_entries:
        with open(path, encoding="utf-8") as handle:
            separator = '\t' if '\t' in handle.readline() else ','
        frame = pd.read_csv(path, sep=separator)
        if assigned_cell_type is not None:
            frame['Cell_Type'] = str(assigned_cell_type)
        frame['GT_Source'] = str(path)
        gt_dfs.append(frame)
        label = f" [{assigned_cell_type}]" if assigned_cell_type is not None else ""
        print(f"  -> Loaded ground truth{label}: {path} ({len(frame)} records)")

    common_score_columns = set(pred_dfs[0].columns)
    for frame in pred_dfs[1:]:
        common_score_columns &= set(frame.columns)
    if target_score_col is not None and target_score_col not in common_score_columns:
        raise ValueError(
            f"Score column '{target_score_col}' must exist in every prediction file."
        )

    pred_df = pd.concat(pred_dfs, ignore_index=True)
    gt_df = pd.concat(gt_dfs, ignore_index=True)

    required_gt = {'Tid', 'CDS_Start_0based', 'CDS_End_0based'}
    required_pred = {'Tid', 'start', 'stop'}
    missing_gt = required_gt - set(gt_df.columns)
    missing_pred = required_pred - set(pred_df.columns)
    if missing_gt:
        raise ValueError(f"Ground truth is missing columns: {sorted(missing_gt)}")
    if missing_pred:
        raise ValueError(f"Predictions are missing columns: {sorted(missing_pred)}")

    gt_df['Tid_clean'] = gt_df['Tid'].apply(normalize_transcript_id)
    pred_df['Tid_clean'] = pred_df['Tid'].apply(normalize_transcript_id)

    for column in ('CDS_Start_0based', 'CDS_End_0based'):
        gt_df[column] = pd.to_numeric(gt_df[column], errors='coerce')
    for column in ('start', 'stop'):
        pred_df[column] = pd.to_numeric(pred_df[column], errors='coerce')
    gt_df = gt_df.dropna(subset=['CDS_Start_0based', 'CDS_End_0based']).copy()
    pred_df = pred_df.dropna(subset=['start', 'stop']).copy()
    gt_df['start_gt'] = gt_df['CDS_Start_0based'].astype(int)
    gt_df['stop_gt'] = gt_df['CDS_End_0based'].astype(int)
    pred_df['start'] = pred_df['start'].astype(int)
    pred_df['stop'] = pred_df['stop'].astype(int)
    gt_df['length'] = gt_df['stop_gt'] - gt_df['start_gt']
    pred_df['length'] = pred_df['stop'] - pred_df['start']
    gt_df = gt_df[(gt_df['start_gt'] >= 0) & (gt_df['length'] > 0)].copy()
    pred_df = pred_df[(pred_df['start'] >= 0) & (pred_df['length'] > 0)].copy()

    if cell_type is not None:
        cell_type = str(cell_type)
        if 'Cell_Type' in gt_df.columns:
            gt_df = gt_df[gt_df['Cell_Type'].astype(str) == cell_type].copy()
        else:
            gt_df['Cell_Type'] = cell_type
        if 'Cell_Type' in pred_df.columns:
            pred_df = pred_df[pred_df['Cell_Type'].astype(str) == cell_type].copy()
        else:
            pred_df['Cell_Type'] = cell_type
    elif 'Cell_Type' in gt_df.columns and 'Cell_Type' in pred_df.columns:
        gt_df['Cell_Type'] = gt_df['Cell_Type'].astype(str)
        pred_df['Cell_Type'] = pred_df['Cell_Type'].astype(str)
    elif 'Cell_Type' not in gt_df.columns and 'Cell_Type' in pred_df.columns:
        pred_cell_types = pred_df['Cell_Type'].dropna().astype(str).unique()
        if len(pred_cell_types) != 1:
            raise ValueError(
                "Ground truth has no Cell_Type column but predictions contain "
                "multiple cell types. Supply cell_type explicitly."
            )
        gt_df['Cell_Type'] = pred_cell_types[0]
        pred_df['Cell_Type'] = pred_df['Cell_Type'].astype(str)
    elif 'Cell_Type' in gt_df.columns and 'Cell_Type' not in pred_df.columns:
        gt_cell_types = gt_df['Cell_Type'].dropna().astype(str).unique()
        if len(gt_cell_types) != 1:
            raise ValueError(
                "Predictions have no Cell_Type column but ground truth contains "
                "multiple cell types. Supply cell_type explicitly."
            )
        gt_df['Cell_Type'] = gt_df['Cell_Type'].astype(str)
        pred_df['Cell_Type'] = gt_cell_types[0]
    else:
        gt_df['Cell_Type'] = 'Unspecified'
        pred_df['Cell_Type'] = 'Unspecified'

    score_col = resolve_score_col(pred_df, target_score_col)
    if score_col not in common_score_columns:
        raise ValueError(
            f"Resolved score column '{score_col}' must exist in every prediction file."
        )
    pred_df[score_col] = pd.to_numeric(pred_df[score_col], errors='coerce')
    pred_df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col])
    print(f"  -> Ranking predictions using column: '{score_col}'")

    valid_cell_types = set(gt_df['Cell_Type'].astype(str).unique())
    prediction_count_before = len(pred_df)
    pred_df = pred_df[
        pred_df['Cell_Type'].astype(str).isin(valid_cell_types)
    ].copy()
    dropped_prediction_count = prediction_count_before - len(pred_df)
    if dropped_prediction_count:
        print(
            f"  -> Dropped {dropped_prediction_count} predictions without "
            "a matching ground-truth Cell_Type."
        )
        
    if min_orf_len is not None or max_orf_len is not None:
        if min_orf_len is not None and max_orf_len is not None and min_orf_len > max_orf_len:
            raise ValueError(f"Invalid length range.")
            
        lower_bound = min_orf_len if min_orf_len is not None else 0
        upper_bound = max_orf_len if max_orf_len is not None else float('inf')
        
        gt_df = gt_df[(gt_df['length'] >= lower_bound) & (gt_df['length'] <= upper_bound)].copy()
        pred_df = pred_df[(pred_df['length'] >= lower_bound) & (pred_df['length'] <= upper_bound)].copy()
        
    gt_df = gt_df.drop_duplicates(
        subset=['Cell_Type', 'Tid_clean', 'start_gt', 'stop_gt']
    ).reset_index(drop=True)
    gt_df['gt_idx'] = gt_df.index

    if len(gt_df) == 0 or len(pred_df) == 0:
        print("Warning: No Ground Truth or Predicted ORFs left after filtering. Returning empty dataframe.")
        return pd.DataFrame(
            columns=['K', 'TP_Count', 'Precision', 'Recall', 'Score_Type']
        )

    pred_df = pred_df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    pred_df['pred_idx'] = pred_df.index
    
    print(f"Executing ultra-fast coordinate matching (Overlap > {overlap_threshold*100}%)...")
    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        key = (row.Cell_Type, row.Tid_clean)
        if key not in gt_dict:
            gt_dict[key] = []
        gt_dict[key].append((row.gt_idx, row.start_gt, row.stop_gt))

    is_tp_list = []
    matched_gt_indices = set()
    matched_gt_list = []
    matched_iou_list = []

    for row in pred_df.itertuples(index=False):
        key = (row.Cell_Type, row.Tid_clean)
        p_start, p_stop, p_len = row.start, row.stop, row.length

        best_match = None
        if key in gt_dict:
            for gt_idx, g_start, g_stop in gt_dict[key]:
                if gt_idx in matched_gt_indices:
                    continue
                if p_start % 3 != g_start % 3:
                    continue

                overlap_s = max(p_start, g_start)
                overlap_e = min(p_stop, g_stop)
                overlap_l = max(0, overlap_e - overlap_s)

                if overlap_l > 0:
                    g_len = g_stop - g_start
                    iou = overlap_l / (p_len + g_len - overlap_l)
                    if iou >= overlap_threshold and (
                        best_match is None or iou > best_match[1]
                    ):
                        best_match = (gt_idx, iou)

        if best_match is not None:
            matched_gt_indices.add(best_match[0])
            is_tp_list.append(1)
            matched_gt_list.append(best_match[0])
            matched_iou_list.append(best_match[1])
        else:
            is_tp_list.append(0)
            matched_gt_list.append(np.nan)
            matched_iou_list.append(np.nan)

    print("Calculating global and cell-type-specific Precision@K and Recall@K...")
    is_tp_array = np.array(is_tp_list)
    tp_cumsum = np.cumsum(is_tp_array)
    k_array = np.arange(1, len(is_tp_array) + 1)
    precision_at_k = tp_cumsum / k_array
    recall_at_k = tp_cumsum / len(gt_df)

    cell_type_series = pred_df['Cell_Type'].astype(str).reset_index(drop=True)
    cell_type_k = cell_type_series.groupby(cell_type_series, sort=False).cumcount() + 1
    cell_type_tp_count = pd.Series(is_tp_array).groupby(
        cell_type_series, sort=False
    ).cumsum()
    cell_type_gt_counts = gt_df.groupby('Cell_Type').size().to_dict()
    cell_type_total_gt = cell_type_series.map(cell_type_gt_counts).astype(int)
    cell_type_precision = cell_type_tp_count / cell_type_k
    cell_type_recall = cell_type_tp_count / cell_type_total_gt

    pk_df = pd.DataFrame({
        'K': k_array,
        'TP_Count': tp_cumsum,
        'Precision': precision_at_k,
        'Recall': recall_at_k,
        'Precision_at_K': precision_at_k,
        'Recall_at_K': recall_at_k,
        'Total_GT_ORFs': len(gt_df),
        'Cell_Type': cell_type_series.to_numpy(),
        'Cell_Type_K': cell_type_k.to_numpy(),
        'Cell_Type_TP_Count': cell_type_tp_count.to_numpy(),
        'Cell_Type_Precision': cell_type_precision.to_numpy(),
        'Cell_Type_Recall': cell_type_recall.to_numpy(),
        'Cell_Type_Total_GT_ORFs': cell_type_total_gt.to_numpy(),
        'Tid': pred_df['Tid'].astype(str).to_numpy(),
        'Pred_Start': pred_df['start'].to_numpy(),
        'Pred_Stop': pred_df['stop'].to_numpy(),
        'Score': pred_df[score_col].to_numpy(),
        'Is_TP': is_tp_array,
        'Matched_GT_Index': matched_gt_list,
        'Match_IoU': matched_iou_list,
        'Prediction_Source': pred_df['Prediction_Source'].astype(str).to_numpy(),
        'Score_Type': score_col
    })
    gt_source_by_index = gt_df.set_index('gt_idx')['GT_Source'].to_dict()
    pk_df['Matched_GT_Source'] = pk_df['Matched_GT_Index'].map(gt_source_by_index)

    print(
        f"Done! Evaluated {len(pk_df)} predictions against "
        f"{len(gt_df)} unique cell-aware GT ORFs."
    )
    return pk_df

# =====================================================================
# Module 2 (Top-K): Precision@K Plotting Function
# =====================================================================
def plot_top_k_metric(
        pk_df: pd.DataFrame,
        metric: str,
        out_dir: str = "./results/eval",
        max_k: Optional[int] = None,
        rank_scope: str = "global") -> Optional[str]:
    """Plot global or cell-type-specific Precision@K/Recall@K as a PDF."""
    if pk_df.empty:
        print("Dataframe is empty, skipping plot generation.")
        return

    if metric not in {'Precision', 'Recall'}:
        raise ValueError("metric must be either 'Precision' or 'Recall'.")
    if rank_scope not in {'global', 'cell_type'}:
        raise ValueError("rank_scope must be either 'global' or 'cell_type'.")

    if rank_scope == 'global':
        k_column = 'K'
        metric_column = metric
    else:
        k_column = 'Cell_Type_K'
        metric_column = f'Cell_Type_{metric}'
    required_columns = {k_column, metric_column}
    if rank_scope == 'cell_type':
        required_columns.add('Cell_Type')
    missing_columns = required_columns - set(pk_df.columns)
    if missing_columns:
        raise ValueError(
            f"Input dataframe is missing columns: {sorted(missing_columns)}"
        )

    scope_label = "cell-type-specific" if rank_scope == 'cell_type' else "global"
    print(f"\nGenerating {scope_label} {metric}@K line chart...")
    os.makedirs(out_dir, exist_ok=True)
    
    plot_df = pk_df.copy()
    if max_k is not None:
        plot_df = plot_df[plot_df[k_column] <= max_k]
        
    if len(plot_df) > 5000:
        if rank_scope == 'global':
            indices = np.linspace(0, len(plot_df) - 1, 5000).astype(int)
            plot_df = plot_df.iloc[indices]
        else:
            sampled_frames = []
            number_of_cell_types = plot_df['Cell_Type'].nunique()
            points_per_cell_type = max(2, 5000 // number_of_cell_types)
            for _, frame in plot_df.groupby('Cell_Type', sort=False):
                if len(frame) > points_per_cell_type:
                    indices = np.linspace(
                        0, len(frame) - 1, points_per_cell_type
                    ).astype(int)
                    frame = frame.iloc[indices]
                sampled_frames.append(frame)
            plot_df = pd.concat(sampled_frames, ignore_index=True)
        
    score_label = plot_df['Score_Type'].iloc[0] if 'Score_Type' in plot_df.columns else 'Final Score'
    color = "#2980b9" if metric == 'Precision' else "#D55E00"

    mapping = (
        aes(x=k_column, y=metric_column)
        if rank_scope == 'global'
        else aes(x=k_column, y=metric_column, color='Cell_Type', group='Cell_Type')
    )
    line_layer = (
        geom_line(color=color, size=1.5, alpha=0.9)
        if rank_scope == 'global'
        else geom_line(size=1.1, alpha=0.85)
    )

    p = (
        ggplot(plot_df, mapping)
        + line_layer
        + theme_classic()
        + labs(
            title=(
                f"{metric}@K by cell type"
                if rank_scope == 'cell_type'
                else f"{metric}@K: Ranked predicted ORFs vs proteomics evidence"
            ),
            x=f"Top K Predicted ORFs (Ranked by {score_label})", 
            y=(
                "Fraction of top-K predictions supported by proteomics"
                if metric == 'Precision'
                else "Fraction of proteomics-supported ORFs recovered"
            )
        )
        + scale_y_continuous(limits=(0, 1.05))
        + scale_x_log10()
        + theme(
            figure_size=(6, 5),
            axis_title=element_text(size=12),
            axis_text=element_text(size=10),
            legend_title=element_blank(),
            legend_position=(
                "right" if rank_scope == 'cell_type' else "none"
            )
        )
    )
    
    limit_suffix = 'All' if max_k is None else max_k
    filename = (
        f"TopK_{metric}_Curve_By_Cell_Type_{limit_suffix}.pdf"
        if rank_scope == 'cell_type'
        else f"TopK_{metric}_Curve_{limit_suffix}.pdf"
    )
    save_path = os.path.join(out_dir, filename)
    p.save(save_path, dpi=300, verbose=False)

    print(f"Chart successfully saved to: {save_path}")
    return save_path


def plot_top_k_precision(
        pk_df: pd.DataFrame,
        out_dir: str = "./results/eval",
        max_k: Optional[int] = None,
        rank_scope: str = "global"):
    """Plot Precision@K while preserving the original public API."""
    return plot_top_k_metric(pk_df, 'Precision', out_dir, max_k, rank_scope)


def plot_top_k_recall(
        pk_df: pd.DataFrame,
        out_dir: str = "./results/eval",
        max_k: Optional[int] = None,
        rank_scope: str = "global"):
    """Plot Recall@K from calculate_top_k_precision output."""
    return plot_top_k_metric(pk_df, 'Recall', out_dir, max_k, rank_scope)

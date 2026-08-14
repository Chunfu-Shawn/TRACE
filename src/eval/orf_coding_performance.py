import os
from collections.abc import Iterable, Mapping
from itertools import combinations
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
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


def normalize_transcript_id(transcript_id) -> str:
    """Remove version suffixes from ENST IDs while preserving all other IDs."""
    transcript_id = str(transcript_id).strip()
    if transcript_id.startswith('ENST'):
        return transcript_id.split('.', 1)[0]
    return transcript_id


TranscriptTargetInput = Union[
    Iterable[str],
    Mapping[str, Iterable[str]],
]


def normalize_transcript_targets(
        target_transcript_ids: TranscriptTargetInput,
) -> Union[set, Dict[str, set]]:
    """Normalize global or cell-specific target transcript collections."""
    def build_target_set(values: Iterable[str], label: str) -> set:
        if isinstance(values, (str, bytes)):
            raise TypeError(
                f"{label} must be a collection of transcript IDs, not a string."
            )
        try:
            return {
                normalize_transcript_id(value)
                for value in values
                if pd.notna(value)
            }
        except TypeError as exc:
            raise TypeError(
                f"{label} must be an iterable of transcript IDs."
            ) from exc

    if isinstance(target_transcript_ids, Mapping):
        normalized_targets: Dict[str, set] = {}
        for raw_cell_type, transcript_ids in target_transcript_ids.items():
            cell_type = str(raw_cell_type)
            if cell_type in normalized_targets:
                raise ValueError(
                    f"Duplicate target-transcript key after string conversion: "
                    f"'{cell_type}'."
                )
            normalized_targets[cell_type] = build_target_set(
                transcript_ids,
                f"target_transcript_ids['{cell_type}']",
            )
        return normalized_targets

    return build_target_set(target_transcript_ids, "target_transcript_ids")


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
        target_transcript_ids: Optional[TranscriptTargetInput] = None,
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        target_score_col: Optional[str] = None,
        callable_start_codons: Optional[List[str]] = None):
    
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
        gt_df['start_gt'] = pd.to_numeric(
            gt_df['CDS_Start_0based'], errors='coerce'
        )
        gt_df['stop_gt'] = pd.to_numeric(
            gt_df['CDS_End_0based'], errors='coerce'
        )
        gt_df = gt_df.dropna(subset=['start_gt', 'stop_gt']).copy()
        gt_df['start_gt'] = gt_df['start_gt'].astype(int)
        gt_df['stop_gt'] = gt_df['stop_gt'].astype(int)
        gt_df['length'] = gt_df['stop_gt'] - gt_df['start_gt'] + 3
        gt_df['Cell_Type'] = cell_type
        gt_dfs.append(gt_df)
        print(f"  -> Loaded '{cell_type}' GT: {len(gt_df)} records.")

    if not gt_dfs: raise ValueError("No valid Ground Truth data loaded!")
    master_gt_df = pd.concat(gt_dfs, ignore_index=True)
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
        pred_df['start'] = pd.to_numeric(pred_df['start'], errors='coerce')
        pred_df['stop'] = pd.to_numeric(pred_df['stop'], errors='coerce')
        pred_df = pred_df.dropna(subset=['start', 'stop']).copy()
        pred_df['start'] = pred_df['start'].astype(int)
        pred_df['stop'] = pred_df['stop'].astype(int)
        pred_df['length'] = pred_df['stop'] - pred_df['start'] + 3
        pred_dfs.append(pred_df)
        print(f"  -> Loaded Pred chunk: {len(pred_df)} records.")

    if not pred_dfs: raise ValueError("No valid Prediction data loaded!")
    master_pred_df = pd.concat(pred_dfs, ignore_index=True)

    # 3. Align Valid Cell Types
    prediction_cell_types = set(master_pred_df['Cell_Type'].astype(str).unique())
    master_gt_df['Cell_Type'] = master_gt_df['Cell_Type'].astype(str)
    gt_count_before = len(master_gt_df)
    master_gt_df = master_gt_df[
        master_gt_df['Cell_Type'].isin(prediction_cell_types)
    ].copy()
    dropped_gt_count = gt_count_before - len(master_gt_df)
    if dropped_gt_count > 0:
        print(
            f"  -> Dropped {dropped_gt_count} GT ORFs from Cell Types "
            "without a prediction CSV."
        )
    valid_cell_types = master_gt_df['Cell_Type'].drop_duplicates().tolist()
    if not valid_cell_types:
        raise ValueError("No overlapping Cell Types between predictions and GT data.")

    initial_pred_len = len(master_pred_df)
    master_pred_df['Cell_Type'] = master_pred_df['Cell_Type'].astype(str)
    master_pred_df = master_pred_df[
        master_pred_df['Cell_Type'].isin(valid_cell_types)
    ]
    dropped_preds = initial_pred_len - len(master_pred_df)
    if dropped_preds > 0:
        print(f"  -> Dropped {dropped_preds} predictions whose Cell_Type lacks Ground Truth data.")

    global_score_col = resolve_score_col(master_pred_df, target_score_col)
    print(f"  -> Decided primary score column: '{global_score_col}'")

    # 4. Filter both GT and predictions by the callable transcript universe.
    if target_transcript_ids is not None:
        normalized_targets = normalize_transcript_targets(target_transcript_ids)
        if isinstance(normalized_targets, dict):
            print("\nFiltering datasets using cell-specific active transcripts...")
            filtered_gt, filtered_pred = [], []

            for ct in valid_cell_types:
                if ct not in normalized_targets:
                    print(
                        f"  [Warning] No target transcripts provided for "
                        f"'{ct}'. This cell type will be dropped."
                    )
                    continue

                target_set = normalized_targets[ct]
                gt_subset = master_gt_df[
                    (master_gt_df['Cell_Type'] == ct)
                    & (master_gt_df['Tid_clean'].isin(target_set))
                ].copy()
                pred_subset = master_pred_df[
                    (master_pred_df['Cell_Type'] == ct)
                    & (master_pred_df['Tid_clean'].isin(target_set))
                ].copy()
                filtered_gt.append(gt_subset)
                filtered_pred.append(pred_subset)
                print(
                    f"  -> {ct}: {len(target_set)} callable transcripts; "
                    f"kept {len(gt_subset)} GT ORFs and "
                    f"{len(pred_subset)} predicted ORFs."
                )

            unused_cell_types = sorted(
                set(normalized_targets).difference(valid_cell_types)
            )
            if unused_cell_types:
                print(
                    "  [Warning] Target transcript entries have no matching "
                    f"prediction/GT cell type: {unused_cell_types}"
                )

            master_gt_df = (
                pd.concat(filtered_gt, ignore_index=True)
                if filtered_gt else master_gt_df.iloc[0:0].copy()
            )
            master_pred_df = (
                pd.concat(filtered_pred, ignore_index=True)
                if filtered_pred else master_pred_df.iloc[0:0].copy()
            )
        else:
            print(
                f"\nFiltering all evaluated cell types globally to "
                f"{len(normalized_targets)} target transcripts..."
            )
            master_gt_df = master_gt_df[
                master_gt_df['Tid_clean'].isin(normalized_targets)
            ].copy()
            master_pred_df = master_pred_df[
                master_pred_df['Tid_clean'].isin(normalized_targets)
            ].copy()

    if callable_start_codons is not None:
        allowed_start_codons = {
            str(codon).upper() for codon in callable_start_codons
        }
        if 'Start_Codon' in master_gt_df.columns:
            master_gt_df = master_gt_df[
                master_gt_df['Start_Codon'].astype(str).str.upper().isin(
                    allowed_start_codons
                )
            ].copy()
        else:
            print(
                "  [Warning] GT data has no Start_Codon column; "
                "callable_start_codons was not applied to GT ORFs."
            )
        if 'start_codon' in master_pred_df.columns:
            master_pred_df = master_pred_df[
                master_pred_df['start_codon'].astype(str).str.upper().isin(
                    allowed_start_codons
                )
            ].copy()
        
    # 5. Filter by ORF Length
    if min_orf_len is not None or max_orf_len is not None:
        lower_bound = min_orf_len if min_orf_len is not None else 0
        upper_bound = max_orf_len if max_orf_len is not None else float('inf')
        master_gt_df = master_gt_df[(master_gt_df['length'] >= lower_bound) & (master_gt_df['length'] <= upper_bound)].copy()
        master_pred_df = master_pred_df[(master_pred_df['length'] >= lower_bound) & (master_pred_df['length'] <= upper_bound)].copy()

    if len(master_gt_df) == 0:
        raise ValueError("No Ground Truth data left after filtering!")
    if len(master_pred_df) == 0:
        raise ValueError("No predicted ORFs left after filtering!")

    # 6. Indexing
    master_gt_df = master_gt_df[
        (master_gt_df['start_gt'] >= 0) & (master_gt_df['length'] > 0)
    ].copy()
    master_pred_df = master_pred_df[
        (master_pred_df['start'] >= 0) & (master_pred_df['length'] > 0)
    ].copy()
    master_gt_df = master_gt_df.drop_duplicates(
        subset=['Cell_Type', 'Tid_clean', 'start_gt', 'stop_gt']
    ).reset_index(drop=True)
    master_gt_df['gt_idx'] = master_gt_df.index
    master_pred_df = master_pred_df.sort_values(
        global_score_col, ascending=False
    ).drop_duplicates(
        subset=['Cell_Type', 'Tid_clean', 'start', 'stop']
    ).reset_index(drop=True)
    master_pred_df['pred_idx'] = master_pred_df.index
    
    return master_pred_df, master_gt_df, global_score_col


# =====================================================================
# Module 2: Cell-Aware Many-to-One Matching
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
        
        best_match = None
        for g_idx, g_start, g_stop in gt_dict[key]:
            if p_start % 3 != g_start % 3: continue
                
            overlap_s = max(p_start, g_start)
            overlap_e = min(p_stop + 3, g_stop + 3)
            overlap_l = max(0, overlap_e - overlap_s)
            
            if overlap_l > 0:
                g_len = g_stop - g_start + 3
                iou = overlap_l / (p_len + g_len - overlap_l)
                if iou >= overlap_threshold and (
                    best_match is None or iou > best_match[1]
                ):
                    best_match = (g_idx, iou)

        if best_match is not None:
            pred_to_gt[p_idx] = best_match[0]
            matched_gt_indices.add(best_match[0])

    print("Assembling Unified Evaluation DataFrame...")
    eval_records = []
    gt_lengths = dict(zip(gt_df['gt_idx'], gt_df['length']))
    
    for row in pred_df.itertuples(index=False):
        is_tp = row.pred_idx in pred_to_gt
        eval_len = gt_lengths[pred_to_gt[row.pred_idx]] if is_tp else row.length
        
        record = {
            'Record_Type': 'Prediction',
            'Cell_Type': row.Cell_Type,
            'Tid': row.Tid_clean,
            'Pred_Index': row.pred_idx,
            'Matched_GT_Index': pred_to_gt.get(row.pred_idx, np.nan),
            'y_true': 1 if is_tp else 0,
            'length': eval_len
        }
        for m in eval_metrics: record[m] = float(getattr(row, m, 0.0) if hasattr(row, m) else 0.0)
        eval_records.append(record)
        
    for row in gt_df.itertuples(index=False):
        if row.gt_idx not in matched_gt_indices:
            record = {
                'Record_Type': 'Missed_GT',
                'Cell_Type': row.Cell_Type,
                'Tid': row.Tid_clean,
                'Pred_Index': np.nan,
                'Matched_GT_Index': row.gt_idx,
                'y_true': 1,
                'length': row.length
            }
            for m in eval_metrics: record[m] = -1.0 
            eval_records.append(record)
            
    eval_df = pd.DataFrame(eval_records)
    print("-" * 40)
    print(f"Total Evaluated MS Ground Truth : {len(gt_df)}")
    print(f"Matched Predictions (TP)        : {len(pred_to_gt)}")
    print(f"Unique Matched Ground Truths    : {len(matched_gt_indices)}")
    print(f"Missed Ground Truths (FN)       : {len(gt_df) - len(matched_gt_indices)}")
    print(f"False Positives (FP)            : {len(pred_df) - len(pred_to_gt)}")
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
    candidate_eval_df = (
        eval_df[eval_df['Record_Type'] == 'Prediction'].copy()
        if 'Record_Type' in eval_df.columns else eval_df.copy()
    )
    y_true_all = candidate_eval_df['y_true'].values
    baseline_all = np.sum(y_true_all) / len(y_true_all) if len(y_true_all) > 0 else 0

    for metric in eval_metrics:
        scores = candidate_eval_df[metric].values
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
    for cell_type, group_df in candidate_eval_df.groupby('Cell_Type'):
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

    color_palette = [
        mpl.colors.to_hex(color)
        for color in sns.color_palette(
            'colorblind', n_colors=max(len(eval_metrics), 1)
        )
    ]
    
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


def add_feature_combination_scores(
        pred_df: pd.DataFrame,
        base_score_columns: List[str],
        feature_columns: List[str],
        method: str = 'product',
        max_combination_size: Optional[int] = None):
    """Add systematically enumerated feature-combination scores."""
    valid_methods = {'product', 'geometric_mean', 'arithmetic_mean'}
    method = method.lower()
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}.")

    available_bases = [
        column for column in base_score_columns if column in pred_df.columns
    ]
    available_features = [
        column for column in feature_columns if column in pred_df.columns
    ]
    if not available_bases:
        raise ValueError("None of the requested base score columns are available.")
    if not available_features:
        raise ValueError("None of the requested combination features are available.")

    max_size = max_combination_size or len(available_features)
    max_size = min(max_size, len(available_features))
    if max_size < 1:
        raise ValueError("max_combination_size must be at least 1.")

    scored_df = pred_df.copy()
    feature_arrays = {}
    for feature in available_features:
        values = pd.to_numeric(scored_df[feature], errors='coerce').fillna(0.0)
        feature_arrays[feature] = values.clip(lower=0.0, upper=1.0).to_numpy()

    metadata_records = []
    score_index = 0
    for base_score in available_bases:
        base_values = pd.to_numeric(
            scored_df[base_score], errors='coerce'
        ).fillna(0.0).clip(lower=0.0).to_numpy()
        metadata_records.append({
            'Score_Column': base_score,
            'Base_Score': base_score,
            'Features': 'none',
            'Method': 'base',
            'Display_Name': base_score,
        })

        for combination_size in range(1, max_size + 1):
            for selected_features in combinations(
                    available_features, combination_size):
                stacked_values = np.vstack([
                    feature_arrays[feature] for feature in selected_features
                ])
                if method == 'product':
                    feature_factor = np.prod(stacked_values, axis=0)
                elif method == 'geometric_mean':
                    feature_factor = np.exp(
                        np.mean(np.log(np.clip(stacked_values, 1e-9, None)), axis=0)
                    )
                else:
                    feature_factor = np.mean(stacked_values, axis=0)

                score_index += 1
                score_column = f'feature_combo_score_{score_index:03d}'
                scored_df[score_column] = base_values * feature_factor
                metadata_records.append({
                    'Score_Column': score_column,
                    'Base_Score': base_score,
                    'Features': '+'.join(selected_features),
                    'Method': method,
                    'Display_Name': (
                        f"{base_score} × " + ' × '.join(selected_features)
                    ),
                })

    metadata_df = pd.DataFrame(metadata_records).drop_duplicates(
        subset=['Score_Column']
    )
    return scored_df, metadata_df


def summarize_feature_combination_performance(
        eval_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        score_metadata: pd.DataFrame,
        top_k_values: List[int]) -> pd.DataFrame:
    """Summarize candidate discrimination and Top-K GT recovery per score."""
    candidate_df = eval_df[eval_df['Record_Type'] == 'Prediction'].copy()
    top_k_values = sorted(set(int(k) for k in top_k_values if int(k) > 0))
    if not top_k_values:
        raise ValueError("top_k_values must contain at least one positive integer.")

    scopes = [('Overall', candidate_df)]
    scopes.extend(candidate_df.groupby('Cell_Type', sort=False))
    records = []

    for scope_name, scope_df in scopes:
        if scope_name == 'Overall':
            total_gt = len(gt_df)
        else:
            total_gt = int((gt_df['Cell_Type'] == scope_name).sum())

        for metadata in score_metadata.itertuples(index=False):
            score_column = metadata.Score_Column
            if score_column not in scope_df.columns:
                continue
            metric_df = scope_df.dropna(subset=[score_column]).copy()
            y_true = metric_df['y_true'].astype(int).to_numpy()
            scores = pd.to_numeric(
                metric_df[score_column], errors='coerce'
            ).to_numpy()
            finite_mask = np.isfinite(scores)
            metric_df = metric_df.loc[finite_mask].copy()
            y_true = y_true[finite_mask]
            scores = scores[finite_mask]

            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc_value = auc(fpr, tpr)
                precision_curve, recall_curve, _ = precision_recall_curve(
                    y_true, scores
                )
                pr_auc_value = average_precision_score(y_true, scores)
                f1_values = (
                    2 * precision_curve * recall_curve
                    / (precision_curve + recall_curve + 1e-9)
                )
                best_f1_value = float(np.max(f1_values))
            else:
                roc_auc_value = np.nan
                pr_auc_value = np.nan
                best_f1_value = np.nan

            record = {
                'Cell_Type': scope_name,
                'Score_Column': score_column,
                'Score_Label': metadata.Display_Name,
                'Base_Score': metadata.Base_Score,
                'Features': metadata.Features,
                'Method': metadata.Method,
                'Candidate_ROC_AUC': roc_auc_value,
                'Candidate_PR_AUC': pr_auc_value,
                'Candidate_Best_F1': best_f1_value,
                'Candidate_Count': len(metric_df),
                'Candidate_TP_Count': int(y_true.sum()),
                'Total_GT_ORFs': total_gt,
            }

            ranked_df = metric_df.sort_values(
                score_column, ascending=False
            ).reset_index(drop=True)
            for k in top_k_values:
                effective_k = min(k, len(ranked_df))
                top_df = ranked_df.iloc[:effective_k]
                precision_value = (
                    float(top_df['y_true'].sum() / effective_k)
                    if effective_k else np.nan
                )
                unique_gt_hits = int(
                    top_df['Matched_GT_Index'].dropna().nunique()
                )
                recall_value = (
                    unique_gt_hits / total_gt if total_gt else np.nan
                )
                record[f'Effective_K_at_{k}'] = effective_k
                record[f'Precision_at_{k}'] = precision_value
                record[f'Unique_GT_Recall_at_{k}'] = recall_value

            records.append(record)

    summary_df = pd.DataFrame(records)
    cell_df = summary_df[summary_df['Cell_Type'] != 'Overall'].copy()
    if not cell_df.empty:
        performance_columns = [
            column for column in summary_df.columns
            if column.startswith('Candidate_')
            or column.startswith('Precision_at_')
            or column.startswith('Unique_GT_Recall_at_')
        ]
        count_columns = [
            column for column in summary_df.columns
            if column.startswith('Effective_K_at_')
            or column in {'Candidate_Count', 'Candidate_TP_Count', 'Total_GT_ORFs'}
        ]
        macro_records = []
        for score_column, group_df in cell_df.groupby('Score_Column', sort=False):
            first_row = group_df.iloc[0]
            macro_record = {
                'Cell_Type': 'Macro_Average',
                'Score_Column': score_column,
                'Score_Label': first_row['Score_Label'],
                'Base_Score': first_row['Base_Score'],
                'Features': first_row['Features'],
                'Method': first_row['Method'],
            }
            for column in performance_columns:
                macro_record[column] = group_df[column].mean()
            for column in count_columns:
                macro_record[column] = group_df[column].sum()
            macro_records.append(macro_record)
        summary_df = pd.concat(
            [summary_df, pd.DataFrame(macro_records)], ignore_index=True
        )

    return summary_df


def plot_feature_combination_performance(
        summary_df: pd.DataFrame,
        out_dir: str,
        primary_metric: str,
        top_n: int = 20,
        ranking_scope: str = 'Overall') -> List[str]:
    """Plot overall metric profiles and per-cell performance for top scores."""
    if primary_metric not in summary_df.columns:
        raise ValueError(f"Unknown primary metric: {primary_metric}")
    os.makedirs(out_dir, exist_ok=True)

    if ranking_scope not in set(summary_df['Cell_Type']):
        raise ValueError(f"Unknown combination ranking scope: {ranking_scope}")
    overall_df = summary_df[
        summary_df['Cell_Type'] == ranking_scope
    ].copy()
    overall_df = overall_df.sort_values(
        primary_metric, ascending=False, na_position='last'
    ).head(top_n)
    if overall_df.empty:
        return []

    metric_columns = [
        column for column in [
            'Candidate_ROC_AUC', 'Candidate_PR_AUC', 'Candidate_Best_F1',
            *[c for c in summary_df.columns if c.startswith('Precision_at_')],
            *[c for c in summary_df.columns if c.startswith('Unique_GT_Recall_at_')],
        ] if column in summary_df.columns
    ]
    overall_matrix = overall_df.set_index('Score_Label')[metric_columns]
    figure_height = max(4.0, 0.34 * len(overall_matrix) + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, figure_height))
    sns.heatmap(
        overall_matrix,
        cmap='YlGnBu',
        vmin=0,
        vmax=1,
        annot=True,
        fmt='.3f',
        linewidths=0.4,
        linecolor='white',
        cbar_kws={'label': 'Performance'},
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(
        f'Feature combinations ranked by {primary_metric} ({ranking_scope})'
    )
    plt.tight_layout()
    overall_path = os.path.join(
        out_dir, 'feature_combination_performance.overall.pdf'
    )
    fig.savefig(overall_path, bbox_inches='tight')
    plt.close(fig)

    selected_scores = overall_df['Score_Column'].tolist()
    cell_df = summary_df[
        (~summary_df['Cell_Type'].isin(['Overall', 'Macro_Average']))
        & (summary_df['Score_Column'].isin(selected_scores))
    ].copy()
    output_paths = [overall_path]
    if not cell_df.empty:
        cell_matrix = cell_df.pivot(
            index='Score_Label', columns='Cell_Type', values=primary_metric
        ).reindex(overall_df['Score_Label'])
        fig_width = max(6.0, 0.65 * len(cell_matrix.columns) + 3.5)
        fig, ax = plt.subplots(figsize=(fig_width, figure_height))
        sns.heatmap(
            cell_matrix,
            cmap='YlGnBu',
            vmin=0,
            vmax=1,
            annot=True,
            fmt='.3f',
            linewidths=0.4,
            linecolor='white',
            cbar_kws={'label': primary_metric},
            ax=ax
        )
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('')
        ax.set_title(f'{primary_metric} across Cell Types')
        plt.tight_layout()
        cell_path = os.path.join(
            out_dir, 'feature_combination_performance.by_cell_type.pdf'
        )
        fig.savefig(cell_path, bbox_inches='tight')
        plt.close(fig)
        output_paths.append(cell_path)

    return output_paths


# =====================================================================
# Main Orchestrator
# =====================================================================
def evaluate_orf_level_predictions(
        pred_csv_paths: List[str],               
        gt_csv_paths: Dict[str, str],            
        target_transcript_ids: Optional[TranscriptTargetInput] = None,
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        out_dir: str = "./results/eval",
        overlap_threshold: float = 0.70,
        target_score_col: Optional[str] = None,
        callable_start_codons: Optional[List[str]] = None,
        evaluate_score_combinations: bool = False,
        combination_base_scores: Optional[List[str]] = None,
        combination_features: Optional[List[str]] = None,
        combination_method: str = 'product',
        combination_max_size: Optional[int] = None,
        combination_top_k_values: Optional[List[int]] = None,
        combination_primary_metric: Optional[str] = None,
        combination_top_n: int = 20,
        combination_ranking_scope: str = 'Overall'):
    """Evaluate predicted ORFs within a global or cell-specific transcript set.

    ``target_transcript_ids`` may be a single transcript collection applied to
    every evaluated cell type, or a ``{cell_type: transcript_collection}``
    mapping. Mapping keys must match prediction ``Cell_Type`` values and the
    keys used in ``gt_csv_paths``.
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Filter and Load
    pred_df, gt_df, score_col = load_and_filter_data(
        pred_csv_paths,
        gt_csv_paths,
        target_transcript_ids,
        min_orf_len,
        max_orf_len,
        target_score_col,
        callable_start_codons,
    )

    combination_metadata = pd.DataFrame()
    if evaluate_score_combinations:
        if 'base_translation_score' in pred_df.columns:
            default_base_scores = [
                'base_translation_score',
                'base_expr_score',
                'mean_intensity',
            ]
        else:
            default_base_scores = [
                score_col,
                'translation_score',
                'mean_intensity',
            ]
        base_scores = list(dict.fromkeys(
            combination_base_scores or default_base_scores
        ))
        feature_columns = combination_features or [
            'tri_nucleotide_periodicity',
            'uniformity_of_signal',
            'step_up_contrast',
            'drop_off',
        ]
        pred_df, combination_metadata = add_feature_combination_scores(
            pred_df=pred_df,
            base_score_columns=base_scores,
            feature_columns=feature_columns,
            method=combination_method,
            max_combination_size=combination_max_size,
        )
    
    all_possible_metrics = {
        'expr_score': 'Expression Score (TPM*Signal)',
        'base_translation_score': 'Base Translation Score',
        'base_expr_score': 'Base Expression-weighted Score',
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
    combination_score_columns = (
        combination_metadata['Score_Column'].tolist()
        if not combination_metadata.empty else []
    )
    match_metrics = list(dict.fromkeys(
        eval_metrics + combination_score_columns
    ))
    eval_df = match_and_build_eval_df(
        pred_df, gt_df, match_metrics, overlap_threshold
    )
    eval_df.to_csv(os.path.join(out_dir, "unified_evaluation_table.csv"), index=False)
    
    # 3. Base Threshold Summary (Based on primary score)
    print("\nCalculating Threshold Summary on Primary Score...")
    candidate_eval_df = eval_df[
        eval_df['Record_Type'] == 'Prediction'
    ].copy()
    tp_count = (candidate_eval_df['y_true'] == 1).sum()
    fp_count = (candidate_eval_df['y_true'] == 0).sum()
    total_preds = tp_count + fp_count
    overall_prec = tp_count / total_preds if total_preds > 0 else 0.0

    prec, rec, threshs = precision_recall_curve(
        candidate_eval_df['y_true'].values,
        candidate_eval_df[score_col].values
    )
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    opt_idx = np.argmax(f1)
    opt_thresh = threshs[opt_idx] if opt_idx < len(threshs) else threshs[-1]
    best_tp = (
        (candidate_eval_df['y_true'] == 1)
        & (candidate_eval_df[score_col] >= opt_thresh)
    ).sum()
    best_fp = (
        (candidate_eval_df['y_true'] == 0)
        & (candidate_eval_df[score_col] >= opt_thresh)
    ).sum()
    unique_gt_hits = int(
        candidate_eval_df['Matched_GT_Index'].dropna().nunique()
    )
    unique_gt_recall = unique_gt_hits / len(gt_df) if len(gt_df) else np.nan

    pd.DataFrame({
        'Total_Predictions': [total_preds],
        'True_Positives_TP': [tp_count],
        'False_Positives_FP': [fp_count],
        'Overall_Precision': [overall_prec],
        'Total_Unique_GT_ORFs': [len(gt_df)],
        'Recovered_Unique_GT_ORFs': [unique_gt_hits],
        'Unique_GT_Recall': [unique_gt_recall],
        'Best_F1_Score': [f1[opt_idx]],
        'Best_Threshold': [opt_thresh],
        'TP_at_Best_Threshold': [best_tp],
        'FP_at_Best_Threshold': [best_fp]
    }).to_csv(os.path.join(out_dir, "primary_score_threshold_summary.csv"), index=False)
    
    # 4. Global Plots & Comprehensive CSV Output
    evaluate_and_plot_global(eval_df, eval_metrics, display_names, out_dir)

    if evaluate_score_combinations:
        top_k_values = combination_top_k_values or [100, 500, 1000]
        combination_summary = summarize_feature_combination_performance(
            eval_df=eval_df,
            gt_df=gt_df,
            score_metadata=combination_metadata,
            top_k_values=top_k_values,
        )
        combination_metadata.to_csv(
            os.path.join(out_dir, 'feature_combination_definitions.csv'),
            index=False
        )
        combination_summary.to_csv(
            os.path.join(out_dir, 'feature_combination_metrics.csv'),
            index=False
        )
        primary_metric = combination_primary_metric or (
            f"Precision_at_{max(top_k_values)}"
        )
        plot_feature_combination_performance(
            summary_df=combination_summary,
            out_dir=out_dir,
            primary_metric=primary_metric,
            top_n=combination_top_n,
            ranking_scope=combination_ranking_scope,
        )
    
    print(f"\n✅ All Evaluation processes successfully finished! Output directory: {out_dir}")



# =====================================================================
# Module 1 (Top-K): Precision@K Calculation Engine
# =====================================================================
def add_requested_combined_score(
        pred_df: pd.DataFrame,
        combined_score: Union[str, Mapping[str, object], pd.Series],
        common_columns: Optional[set] = None):
    """Create one Top-K score from an evaluation combination definition."""
    if isinstance(combined_score, str):
        if combined_score not in pred_df.columns:
            raise ValueError(
                f"Combined score column '{combined_score}' was not found."
            )
        if common_columns is not None and combined_score not in common_columns:
            raise ValueError(
                f"Combined score column '{combined_score}' must exist in "
                "every prediction file."
            )
        return pred_df.copy(), combined_score, combined_score

    if isinstance(combined_score, pd.Series):
        definition = combined_score.to_dict()
    elif isinstance(combined_score, Mapping):
        definition = dict(combined_score)
    else:
        raise TypeError(
            "combined_score must be a column name or a combination-definition "
            "dictionary/Series."
        )

    base_score = definition.get('base_score', definition.get('Base_Score'))
    features = definition.get('features', definition.get('Features', []))
    method = definition.get('method', definition.get('Method', 'product'))
    if base_score is None or pd.isna(base_score):
        raise ValueError("combined_score must define Base_Score or base_score.")
    base_score = str(base_score)

    if isinstance(features, str):
        feature_columns = (
            [] if features.strip().lower() in {'', 'none', 'nan'}
            else [value.strip() for value in features.split('+') if value.strip()]
        )
    elif features is None:
        feature_columns = []
    else:
        feature_columns = [str(value) for value in features]

    method = str(method).lower()
    if method == 'base' and not feature_columns:
        method = 'product'
    valid_methods = {'product', 'geometric_mean', 'arithmetic_mean'}
    if method not in valid_methods:
        raise ValueError(f"Combined-score method must be one of {sorted(valid_methods)}.")

    required_columns = [base_score, *feature_columns]
    missing_columns = [
        column for column in required_columns if column not in pred_df.columns
    ]
    if common_columns is not None:
        missing_columns.extend(
            column for column in required_columns
            if column not in common_columns and column not in missing_columns
        )
    if missing_columns:
        raise ValueError(
            "Combined-score columns must exist in every prediction file. "
            f"Missing columns: {missing_columns}"
        )

    scored_df = pred_df.copy()
    base_values = pd.to_numeric(
        scored_df[base_score], errors='coerce'
    ).fillna(0.0).clip(lower=0.0).to_numpy()
    if feature_columns:
        feature_values = np.vstack([
            pd.to_numeric(scored_df[column], errors='coerce')
            .fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy()
            for column in feature_columns
        ])
        if method == 'product':
            feature_factor = np.prod(feature_values, axis=0)
        elif method == 'geometric_mean':
            feature_factor = np.exp(np.mean(
                np.log(np.clip(feature_values, 1e-9, None)), axis=0
            ))
        else:
            feature_factor = np.mean(feature_values, axis=0)
    else:
        feature_factor = np.ones(len(scored_df), dtype=np.float64)

    combined_column = '__top_k_combined_score'
    scored_df[combined_column] = base_values * feature_factor
    feature_label = '+'.join(feature_columns) if feature_columns else 'none'
    score_label = f"{base_score} | {method}({feature_label})"
    return scored_df, combined_column, score_label


def calculate_top_k_precision(
        pred_csv_paths: Optional[Union[str, List[str]]] = None,
        gt_csv_paths: Optional[Union[str, List[str], Dict[str, str]]] = None,
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        overlap_threshold: float = 0.70,
        target_score_col: Optional[str] = None,
        cell_type: Optional[str] = None,
        pred_csv_path: Optional[str] = None,
        gt_csv_path: Optional[str] = None,
        target_transcript_ids: Optional[TranscriptTargetInput] = None,
        callable_start_codons: Optional[List[str]] = None,
        score_col: Optional[str] = None,
        combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None) -> pd.DataFrame:
    """
    Calculate cell-aware Precision@K and Recall@K for ranked predicted ORFs.

    ``pred_csv_paths`` accepts one path or a list of paths. ``gt_csv_paths``
    accepts one path, a list of paths with embedded Cell_Type values, or a
    ``{cell_type: path}`` dictionary matching evaluate_orf_level_predictions.
    The singular path arguments are retained for backward compatibility.

    ``target_transcript_ids`` accepts either one transcript collection applied
    globally or a ``{cell_type: transcript_collection}`` mapping. Both GT and
    predictions are filtered to this callable transcript universe and to
    ``callable_start_codons`` before ranking and matching.

    ``score_col`` selects an existing prediction column and is an alias for
    ``target_score_col``. ``combined_score`` may be an existing column name or
    one row/dictionary from ``feature_combination_definitions.csv`` or
    ``feature_combination_metrics.csv`` using the ``Base_Score``, ``Features``,
    and ``Method`` fields.

    Predictions are greedily matched in descending score order. Each ground
    truth ORF may support multiple predictions. Each prediction is assigned to
    its highest-IoU eligible ORF within the same cell type, transcript, and
    reading frame. Precision counts all supported predictions, whereas Recall
    counts each recovered ground-truth ORF only once.
    """
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1.")

    if score_col is not None and target_score_col is not None:
        raise ValueError("Use either score_col or target_score_col, not both.")
    requested_score_col = score_col or target_score_col
    if combined_score is not None and requested_score_col is not None:
        raise ValueError(
            "Use either score_col/target_score_col or combined_score, not both."
        )

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
    if requested_score_col is not None and requested_score_col not in common_score_columns:
        raise ValueError(
            f"Score column '{requested_score_col}' must exist in every prediction file."
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
    gt_df['length'] = gt_df['stop_gt'] - gt_df['start_gt'] + 3
    pred_df['length'] = pred_df['stop'] - pred_df['start'] + 3
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

    if combined_score is not None:
        pred_df, ranking_score_col, score_type_label = add_requested_combined_score(
            pred_df=pred_df,
            combined_score=combined_score,
            common_columns=common_score_columns,
        )
    else:
        ranking_score_col = resolve_score_col(pred_df, requested_score_col)
        if ranking_score_col not in common_score_columns:
            raise ValueError(
                f"Resolved score column '{ranking_score_col}' must exist in "
                "every prediction file."
            )
        score_type_label = ranking_score_col

    pred_df[ranking_score_col] = pd.to_numeric(
        pred_df[ranking_score_col], errors='coerce'
    )
    pred_df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[ranking_score_col]
    )
    print(f"  -> Ranking predictions using: '{score_type_label}'")

    prediction_cell_types = set(pred_df['Cell_Type'].astype(str).unique())
    gt_df = gt_df[
        gt_df['Cell_Type'].astype(str).isin(prediction_cell_types)
    ].copy()
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

    if target_transcript_ids is not None:
        normalized_targets = normalize_transcript_targets(target_transcript_ids)
        gt_count_before = len(gt_df)
        pred_count_before = len(pred_df)

        if isinstance(normalized_targets, dict):
            print("  -> Applying cell-specific callable transcript sets.")
            evaluated_cell_types = set(gt_df['Cell_Type']).union(
                pred_df['Cell_Type']
            )
            missing_target_cells = sorted(
                evaluated_cell_types.difference(normalized_targets)
            )
            if missing_target_cells:
                print(
                    "  [Warning] No target transcripts were provided for "
                    f"these cell types; they will be dropped: "
                    f"{missing_target_cells}"
                )

            gt_mask = pd.Series(False, index=gt_df.index)
            pred_mask = pd.Series(False, index=pred_df.index)
            for target_cell_type, target_set in normalized_targets.items():
                gt_mask |= (
                    (gt_df['Cell_Type'] == target_cell_type)
                    & gt_df['Tid_clean'].isin(target_set)
                )
                pred_mask |= (
                    (pred_df['Cell_Type'] == target_cell_type)
                    & pred_df['Tid_clean'].isin(target_set)
                )
            gt_df = gt_df[gt_mask].copy()
            pred_df = pred_df[pred_mask].copy()
        else:
            print(
                f"  -> Applying one global callable set containing "
                f"{len(normalized_targets)} transcripts."
            )
            gt_df = gt_df[gt_df['Tid_clean'].isin(normalized_targets)].copy()
            pred_df = pred_df[
                pred_df['Tid_clean'].isin(normalized_targets)
            ].copy()

        print(
            f"     GT ORFs: {gt_count_before} -> {len(gt_df)}; "
            f"predicted ORFs: {pred_count_before} -> {len(pred_df)}"
        )

    if callable_start_codons is not None:
        allowed_start_codons = {
            str(codon).strip().upper() for codon in callable_start_codons
        }
        if not allowed_start_codons:
            raise ValueError("callable_start_codons cannot be empty.")

        gt_count_before = len(gt_df)
        pred_count_before = len(pred_df)
        if 'Start_Codon' in gt_df.columns:
            gt_df = gt_df[
                gt_df['Start_Codon'].astype(str).str.strip().str.upper().isin(
                    allowed_start_codons
                )
            ].copy()
        else:
            print(
                "  [Warning] GT data has no Start_Codon column; "
                "callable_start_codons was not applied to GT ORFs."
            )

        if 'start_codon' in pred_df.columns:
            pred_df = pred_df[
                pred_df['start_codon'].astype(str).str.strip().str.upper().isin(
                    allowed_start_codons
                )
            ].copy()
        else:
            print(
                "  [Warning] Prediction data has no start_codon column; "
                "callable_start_codons was not applied to predicted ORFs."
            )

        print(
            f"  -> Callable start codons {sorted(allowed_start_codons)}: "
            f"GT ORFs {gt_count_before} -> {len(gt_df)}; "
            f"predicted ORFs {pred_count_before} -> {len(pred_df)}"
        )
        
    if min_orf_len is not None or max_orf_len is not None:
        if min_orf_len is not None and max_orf_len is not None and min_orf_len > max_orf_len:
            raise ValueError(f"Invalid length range.")
            
        lower_bound = min_orf_len if min_orf_len is not None else 0
        upper_bound = max_orf_len if max_orf_len is not None else float('inf')
        
        gt_df = gt_df[(gt_df['length'] >= lower_bound) & (gt_df['length'] <= upper_bound)].copy()
        pred_df = pred_df[(pred_df['length'] >= lower_bound) & (pred_df['length'] <= upper_bound)].copy()

    callable_gt_cell_types = set(gt_df['Cell_Type'].astype(str).unique())
    pred_count_before = len(pred_df)
    pred_df = pred_df[
        pred_df['Cell_Type'].astype(str).isin(callable_gt_cell_types)
    ].copy()
    if len(pred_df) < pred_count_before:
        print(
            f"  -> Dropped {pred_count_before - len(pred_df)} predictions "
            "from cell types with no GT ORFs left in the callable universe."
        )
        
    gt_df = gt_df.drop_duplicates(
        subset=['Cell_Type', 'Tid_clean', 'start_gt', 'stop_gt']
    ).reset_index(drop=True)
    gt_df['gt_idx'] = gt_df.index

    if len(gt_df) == 0 or len(pred_df) == 0:
        print("Warning: No Ground Truth or Predicted ORFs left after filtering. Returning empty dataframe.")
        return pd.DataFrame(
            columns=['K', 'TP_Count', 'Precision', 'Recall', 'Score_Type']
        )

    pred_df = pred_df.sort_values(
        by=ranking_score_col, ascending=False
    ).reset_index(drop=True)
    pred_df['pred_idx'] = pred_df.index
    
    print(f"Executing ultra-fast coordinate matching (Overlap > {overlap_threshold*100}%)...")
    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        key = (row.Cell_Type, row.Tid_clean)
        if key not in gt_dict:
            gt_dict[key] = []
        gt_dict[key].append((row.gt_idx, row.start_gt, row.stop_gt))

    is_tp_list = []
    matched_gt_list = []
    matched_iou_list = []

    for row in pred_df.itertuples(index=False):
        key = (row.Cell_Type, row.Tid_clean)
        p_start, p_stop, p_len = row.start, row.stop, row.length

        best_match = None
        if key in gt_dict:
            for gt_idx, g_start, g_stop in gt_dict[key]:
                if p_start % 3 != g_start % 3:
                    continue

                overlap_s = max(p_start, g_start)
                overlap_e = min(p_stop + 3, g_stop + 3)
                overlap_l = max(0, overlap_e - overlap_s)

                if overlap_l > 0:
                    g_len = g_stop - g_start + 3
                    iou = overlap_l / (p_len + g_len - overlap_l)
                    if iou >= overlap_threshold and (
                        best_match is None or iou > best_match[1]
                    ):
                        best_match = (gt_idx, iou)

        if best_match is not None:
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
    unique_gt_hit_array = np.zeros(len(matched_gt_list), dtype=int)
    seen_gt_indices = set()
    for index, matched_gt_index in enumerate(matched_gt_list):
        if pd.notna(matched_gt_index) and matched_gt_index not in seen_gt_indices:
            unique_gt_hit_array[index] = 1
            seen_gt_indices.add(matched_gt_index)
    unique_gt_cumsum = np.cumsum(unique_gt_hit_array)
    k_array = np.arange(1, len(is_tp_array) + 1)
    precision_at_k = tp_cumsum / k_array
    recall_at_k = unique_gt_cumsum / len(gt_df)

    cell_type_series = pred_df['Cell_Type'].astype(str).reset_index(drop=True)
    cell_type_k = cell_type_series.groupby(cell_type_series, sort=False).cumcount() + 1
    cell_type_tp_count = pd.Series(is_tp_array).groupby(
        cell_type_series, sort=False
    ).cumsum()
    cell_type_unique_gt_count = pd.Series(unique_gt_hit_array).groupby(
        cell_type_series, sort=False
    ).cumsum()
    cell_type_gt_counts = gt_df.groupby('Cell_Type').size().to_dict()
    cell_type_total_gt = cell_type_series.map(cell_type_gt_counts).astype(int)
    cell_type_precision = cell_type_tp_count / cell_type_k
    cell_type_recall = cell_type_unique_gt_count / cell_type_total_gt

    pk_df = pd.DataFrame({
        'K': k_array,
        'TP_Count': tp_cumsum,
        'Unique_GT_Hit_Count': unique_gt_cumsum,
        'Precision': precision_at_k,
        'Recall': recall_at_k,
        'Precision_at_K': precision_at_k,
        'Recall_at_K': recall_at_k,
        'Total_GT_ORFs': len(gt_df),
        'Cell_Type': cell_type_series.to_numpy(),
        'Cell_Type_K': cell_type_k.to_numpy(),
        'Cell_Type_TP_Count': cell_type_tp_count.to_numpy(),
        'Cell_Type_Unique_GT_Hit_Count': cell_type_unique_gt_count.to_numpy(),
        'Cell_Type_Precision': cell_type_precision.to_numpy(),
        'Cell_Type_Recall': cell_type_recall.to_numpy(),
        'Cell_Type_Total_GT_ORFs': cell_type_total_gt.to_numpy(),
        'Tid': pred_df['Tid'].astype(str).to_numpy(),
        'Pred_Start': pred_df['start'].to_numpy(),
        'Pred_Stop': pred_df['stop'].to_numpy(),
        'Score': pred_df[ranking_score_col].to_numpy(),
        'Is_TP': is_tp_array,
        'Is_New_GT_Hit': unique_gt_hit_array,
        'Matched_GT_Index': matched_gt_list,
        'Match_IoU': matched_iou_list,
        'Prediction_Source': pred_df['Prediction_Source'].astype(str).to_numpy(),
        'Score_Type': score_type_label,
        'Score_Column': ranking_score_col,
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

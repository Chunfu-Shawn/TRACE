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

PredictionPathInput = Union[str, List[str]]
GroundTruthPathInput = Union[str, List[str], Dict[str, str]]


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
        'rank_score',
        'occupancy_expr_score',
        'occupancy_score',
        'expr_score', 
        'translation_score', 
        'transcription_score', 
        'seq_score',
        'kozak_score',
        'start_codon_score',
        'score'
    ]
    
    for col in fallback_candidates:
        if col in df.columns:
            return col
            
    raise ValueError(f"No valid score column found! Available columns: {df.columns.tolist()}")

# =====================================================================
# Module 1: Shared data loading and preprocessing
# =====================================================================
def _read_delimited_table(path: str) -> pd.DataFrame:
    """Read a comma- or tab-delimited table using its first line as a hint."""
    with open(path, encoding="utf-8") as handle:
        separator = '\t' if '\t' in handle.readline() else ','
    return pd.read_csv(path, sep=separator)


def _normalize_path_inputs(
        pred_csv_paths: PredictionPathInput,
        gt_csv_paths: GroundTruthPathInput):
    """Normalize flexible prediction and GT path inputs."""
    pred_paths = (
        [pred_csv_paths]
        if isinstance(pred_csv_paths, str)
        else list(pred_csv_paths)
    )
    if isinstance(gt_csv_paths, str):
        gt_entries = [(None, gt_csv_paths)]
    elif isinstance(gt_csv_paths, Mapping):
        gt_entries = [(str(cell_type), path)
                      for cell_type, path in gt_csv_paths.items()]
    else:
        gt_entries = [(None, path) for path in gt_csv_paths]
    if not pred_paths or not gt_entries:
        raise ValueError(
            "Prediction and ground-truth path collections cannot be empty."
        )
    return pred_paths, gt_entries


def _resolve_missing_cell_types(
        pred_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        cell_type: Optional[str] = None):
    """Resolve or validate cell-type labels before joint filtering."""
    pred_df = pred_df.copy()
    gt_df = gt_df.copy()
    if 'Cell_Type' not in pred_df.columns:
        pred_df['Cell_Type'] = np.nan
    if 'Cell_Type' not in gt_df.columns:
        gt_df['Cell_Type'] = np.nan

    if cell_type is not None:
        cell_type = str(cell_type)
        pred_has_label = pred_df['Cell_Type'].notna()
        gt_has_label = gt_df['Cell_Type'].notna()
        pred_df = pred_df[
            ~pred_has_label | (pred_df['Cell_Type'].astype(str) == cell_type)
        ].copy()
        gt_df = gt_df[
            ~gt_has_label | (gt_df['Cell_Type'].astype(str) == cell_type)
        ].copy()
        pred_df['Cell_Type'] = cell_type
        gt_df['Cell_Type'] = cell_type
        return pred_df, gt_df

    pred_labels = pred_df['Cell_Type'].dropna().astype(str).unique()
    gt_labels = gt_df['Cell_Type'].dropna().astype(str).unique()
    if len(pred_labels) == 0 and len(gt_labels) == 0:
        pred_df['Cell_Type'] = 'Unspecified'
        gt_df['Cell_Type'] = 'Unspecified'
        return pred_df, gt_df
    if pred_df['Cell_Type'].isna().any():
        candidates = gt_labels if len(gt_labels) else pred_labels
        if len(candidates) != 1:
            raise ValueError(
                "Predictions have missing Cell_Type values and they cannot be "
                "resolved uniquely. Supply cell_type explicitly."
            )
        pred_df['Cell_Type'] = pred_df['Cell_Type'].fillna(candidates[0])
    if gt_df['Cell_Type'].isna().any():
        candidates = pred_labels if len(pred_labels) else gt_labels
        if len(candidates) != 1:
            raise ValueError(
                "Ground truth has missing Cell_Type values and predictions "
                "contain multiple cell types. Supply cell_type explicitly."
            )
        gt_df['Cell_Type'] = gt_df['Cell_Type'].fillna(candidates[0])

    pred_df['Cell_Type'] = pred_df['Cell_Type'].astype(str)
    gt_df['Cell_Type'] = gt_df['Cell_Type'].astype(str)
    return pred_df, gt_df


def load_and_filter_data(
        pred_csv_paths: PredictionPathInput,
        gt_csv_paths: GroundTruthPathInput,
        target_transcript_ids: Optional[TranscriptTargetInput] = None,
        min_orf_len: Optional[int] = None,
        max_orf_len: Optional[int] = None,
        target_score_col: Optional[str] = None,
        callable_start_codons: Optional[List[str]] = None,
        cell_type: Optional[str] = None):
    """Load and identically preprocess inputs for every ORF evaluation path."""
    if min_orf_len is not None and max_orf_len is not None:
        if min_orf_len > max_orf_len:
            raise ValueError("Invalid length range.")

    pred_paths, gt_entries = _normalize_path_inputs(
        pred_csv_paths, gt_csv_paths
    )

    pred_dfs = []
    common_prediction_columns = None
    print("--- Loading Prediction Data ---")
    for pred_path in pred_paths:
        if not os.path.exists(pred_path):
            print(f"  [Warning] Prediction file not found: {pred_path}. Skipping.")
            continue
        pred_df = pd.read_csv(pred_path)
        required = {'Tid', 'start', 'stop'}
        missing = required.difference(pred_df.columns)
        if missing:
            raise ValueError(
                f"Prediction file {pred_path} is missing columns: "
                f"{sorted(missing)}"
            )
        pred_df['Prediction_Source'] = str(pred_path)
        pred_dfs.append(pred_df)
        columns = set(pred_df.columns)
        common_prediction_columns = (
            columns if common_prediction_columns is None
            else common_prediction_columns.intersection(columns)
        )
        print(f"  -> Loaded predictions: {pred_path} ({len(pred_df)} records)")
    if not pred_dfs:
        raise ValueError("No valid prediction data loaded.")
    master_pred_df = pd.concat(pred_dfs, ignore_index=True)

    gt_dfs = []
    print("\n--- Loading Ground Truth Data ---")
    for assigned_cell_type, gt_path in gt_entries:
        if not os.path.exists(gt_path):
            label = f" for '{assigned_cell_type}'" if assigned_cell_type else ''
            print(f"  [Warning] GT file not found{label}: {gt_path}. Skipping.")
            continue
        try:
            gt_df = _read_delimited_table(gt_path)
        except Exception as exc:
            raise ValueError(f"Error reading GT file {gt_path}: {exc}") from exc
        required = {'Tid', 'CDS_Start_0based', 'CDS_End_0based'}
        missing = required.difference(gt_df.columns)
        if missing:
            raise ValueError(
                f"Ground-truth file {gt_path} is missing columns: "
                f"{sorted(missing)}"
            )
        if assigned_cell_type is not None:
            gt_df['Cell_Type'] = assigned_cell_type
        gt_df['GT_Source'] = str(gt_path)
        gt_dfs.append(gt_df)
        label = f" [{assigned_cell_type}]" if assigned_cell_type else ''
        print(f"  -> Loaded ground truth{label}: {gt_path} ({len(gt_df)} records)")
    if not gt_dfs:
        raise ValueError("No valid ground-truth data loaded.")
    master_gt_df = pd.concat(gt_dfs, ignore_index=True)

    master_pred_df, master_gt_df = _resolve_missing_cell_types(
        master_pred_df, master_gt_df, cell_type=cell_type
    )

    master_pred_df['Tid_clean'] = master_pred_df['Tid'].apply(
        normalize_transcript_id
    )
    master_gt_df['Tid_clean'] = master_gt_df['Tid'].apply(
        normalize_transcript_id
    )
    for column in ('start', 'stop'):
        master_pred_df[column] = pd.to_numeric(
            master_pred_df[column], errors='coerce'
        )
    for column in ('CDS_Start_0based', 'CDS_End_0based'):
        master_gt_df[column] = pd.to_numeric(
            master_gt_df[column], errors='coerce'
        )
    master_pred_df = master_pred_df.dropna(
        subset=['start', 'stop']
    ).copy()
    master_gt_df = master_gt_df.dropna(
        subset=['CDS_Start_0based', 'CDS_End_0based']
    ).copy()
    master_pred_df[['start', 'stop']] = master_pred_df[
        ['start', 'stop']
    ].astype(int)
    master_gt_df['start_gt'] = master_gt_df['CDS_Start_0based'].astype(int)
    master_gt_df['stop_gt'] = master_gt_df['CDS_End_0based'].astype(int)
    master_pred_df['length'] = (
        master_pred_df['stop'] - master_pred_df['start'] + 3
    )
    master_gt_df['length'] = (
        master_gt_df['stop_gt'] - master_gt_df['start_gt'] + 3
    )
    master_pred_df = master_pred_df[
        (master_pred_df['start'] >= 0) & (master_pred_df['length'] > 0)
    ].copy()
    master_gt_df = master_gt_df[
        (master_gt_df['start_gt'] >= 0) & (master_gt_df['length'] > 0)
    ].copy()

    prediction_cell_types = set(master_pred_df['Cell_Type'].unique())
    gt_count_before = len(master_gt_df)
    master_gt_df = master_gt_df[
        master_gt_df['Cell_Type'].isin(prediction_cell_types)
    ].copy()
    if len(master_gt_df) < gt_count_before:
        print(
            f"  -> Dropped {gt_count_before - len(master_gt_df)} GT ORFs "
            "from cell types without predictions."
        )
    valid_cell_types = set(master_gt_df['Cell_Type'].unique())
    pred_count_before = len(master_pred_df)
    master_pred_df = master_pred_df[
        master_pred_df['Cell_Type'].isin(valid_cell_types)
    ].copy()
    if len(master_pred_df) < pred_count_before:
        print(
            f"  -> Dropped {pred_count_before - len(master_pred_df)} "
            "predictions from cell types without ground truth."
        )
    if not valid_cell_types:
        raise ValueError(
            "No overlapping Cell_Type values between predictions and GT data."
        )

    global_score_col = resolve_score_col(master_pred_df, target_score_col)
    if global_score_col not in common_prediction_columns:
        raise ValueError(
            f"Score column '{global_score_col}' must exist in every "
            "prediction file."
        )
    master_pred_df[global_score_col] = pd.to_numeric(
        master_pred_df[global_score_col], errors='coerce'
    )
    master_pred_df = master_pred_df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=[global_score_col])
    print(f"  -> Primary score column: '{global_score_col}'")

    if target_transcript_ids is not None:
        normalized_targets = normalize_transcript_targets(
            target_transcript_ids
        )
        gt_count_before = len(master_gt_df)
        pred_count_before = len(master_pred_df)
        if isinstance(normalized_targets, dict):
            gt_mask = pd.Series(False, index=master_gt_df.index)
            pred_mask = pd.Series(False, index=master_pred_df.index)
            evaluated_cell_types = set(master_gt_df['Cell_Type']).union(
                master_pred_df['Cell_Type']
            )
            missing_cells = sorted(
                evaluated_cell_types.difference(normalized_targets)
            )
            if missing_cells:
                print(
                    "  [Warning] Missing callable transcript sets for cell "
                    f"types that will be dropped: {missing_cells}"
                )
            for target_cell_type, target_set in normalized_targets.items():
                gt_mask |= (
                    (master_gt_df['Cell_Type'] == target_cell_type)
                    & master_gt_df['Tid_clean'].isin(target_set)
                )
                pred_mask |= (
                    (master_pred_df['Cell_Type'] == target_cell_type)
                    & master_pred_df['Tid_clean'].isin(target_set)
                )
            master_gt_df = master_gt_df[gt_mask].copy()
            master_pred_df = master_pred_df[pred_mask].copy()
        else:
            master_gt_df = master_gt_df[
                master_gt_df['Tid_clean'].isin(normalized_targets)
            ].copy()
            master_pred_df = master_pred_df[
                master_pred_df['Tid_clean'].isin(normalized_targets)
            ].copy()
        print(
            f"  -> Callable transcripts: GT {gt_count_before} -> "
            f"{len(master_gt_df)}; predictions {pred_count_before} -> "
            f"{len(master_pred_df)}"
        )

    if callable_start_codons is not None:
        allowed_start_codons = {
            str(codon).strip().upper() for codon in callable_start_codons
        }
        if not allowed_start_codons:
            raise ValueError("callable_start_codons cannot be empty.")
        gt_count_before = len(master_gt_df)
        pred_count_before = len(master_pred_df)
        if 'Start_Codon' in master_gt_df.columns:
            master_gt_df = master_gt_df[
                master_gt_df['Start_Codon'].astype(str).str.strip().str.upper()
                .isin(allowed_start_codons)
            ].copy()
        else:
            print(
                "  [Warning] GT has no Start_Codon column; the codon filter "
                "was not applied to GT ORFs."
            )
        if 'start_codon' in master_pred_df.columns:
            master_pred_df = master_pred_df[
                master_pred_df['start_codon'].astype(str).str.strip().str.upper()
                .isin(allowed_start_codons)
            ].copy()
        else:
            print(
                "  [Warning] Predictions have no start_codon column; the "
                "codon filter was not applied to predicted ORFs."
            )
        print(
            f"  -> Callable start codons {sorted(allowed_start_codons)}: "
            f"GT {gt_count_before} -> {len(master_gt_df)}; predictions "
            f"{pred_count_before} -> {len(master_pred_df)}"
        )

    lower_bound = min_orf_len if min_orf_len is not None else 0
    upper_bound = max_orf_len if max_orf_len is not None else float('inf')
    master_gt_df = master_gt_df[
        master_gt_df['length'].between(lower_bound, upper_bound)
    ].copy()
    master_pred_df = master_pred_df[
        master_pred_df['length'].between(lower_bound, upper_bound)
    ].copy()
    if master_gt_df.empty:
        raise ValueError("No ground-truth ORFs left after filtering.")
    if master_pred_df.empty:
        raise ValueError("No predicted ORFs left after filtering.")

    callable_gt_cells = set(master_gt_df['Cell_Type'].unique())
    master_pred_df = master_pred_df[
        master_pred_df['Cell_Type'].isin(callable_gt_cells)
    ].copy()
    master_gt_df = master_gt_df.drop_duplicates(
        subset=['Cell_Type', 'Tid_clean', 'start_gt', 'stop_gt']
    ).reset_index(drop=True)
    master_gt_df['gt_idx'] = master_gt_df.index
    master_pred_df = master_pred_df.sort_values(
        global_score_col, ascending=False, kind='mergesort'
    ).drop_duplicates(
        subset=['Cell_Type', 'Tid_clean', 'start', 'stop']
    ).reset_index(drop=True)
    master_pred_df['pred_idx'] = master_pred_df.index

    return master_pred_df, master_gt_df, global_score_col


# =====================================================================
# Module 2: Cell-Aware Many-to-One Matching
# =====================================================================
def match_and_build_eval_df(
        pred_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        eval_metrics: List[str],
        overlap_threshold: float) -> pd.DataFrame:
    """Build the single matched table used by all downstream metrics."""
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1.")
    print(
        "\nCell-aware matching "
        f"(frame-consistent, IoU >= {overlap_threshold:.2f})..."
    )

    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        key = (row.Cell_Type, row.Tid_clean)
        if key not in gt_dict:
            gt_dict[key] = []
        gt_dict[key].append((row.gt_idx, row.start_gt, row.stop_gt))

    pred_to_gt = {}
    pred_to_iou = {}
    matched_gt_indices = set()

    for row in pred_df.itertuples(index=False):
        key = (row.Cell_Type, row.Tid_clean)
        if key not in gt_dict:
            continue

        p_start = row.start
        p_stop = row.stop
        p_idx = row.pred_idx
        p_len = row.length
        best_match = None
        for g_idx, g_start, g_stop in gt_dict[key]:
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
                    best_match = (g_idx, iou)

        if best_match is not None:
            pred_to_gt[p_idx] = best_match[0]
            pred_to_iou[p_idx] = best_match[1]
            matched_gt_indices.add(best_match[0])

    print("Assembling Unified Evaluation DataFrame...")
    eval_records = []
    gt_lengths = dict(zip(gt_df['gt_idx'], gt_df['length']))
    gt_sources = (
        gt_df.set_index('gt_idx')['GT_Source'].to_dict()
        if 'GT_Source' in gt_df.columns else {}
    )
    gt_starts = gt_df.set_index('gt_idx')['start_gt'].to_dict()
    gt_stops = gt_df.set_index('gt_idx')['stop_gt'].to_dict()
    cell_gt_counts = gt_df.groupby('Cell_Type').size().to_dict()
    metric_values_by_prediction = (
        pred_df.set_index('pred_idx')[eval_metrics].to_dict(orient='index')
        if eval_metrics else {}
    )

    for row in pred_df.itertuples(index=False):
        is_tp = row.pred_idx in pred_to_gt
        matched_gt_index = pred_to_gt.get(row.pred_idx, np.nan)
        eval_len = (
            gt_lengths[matched_gt_index] if is_tp else row.length
        )

        record = {
            'Record_Type': 'Prediction',
            'Cell_Type': row.Cell_Type,
            'Tid': row.Tid_clean,
            'Tid_Original': str(getattr(row, 'Tid', row.Tid_clean)),
            'Pred_Index': row.pred_idx,
            'Pred_Start': row.start,
            'Pred_Stop': row.stop,
            'GT_Start': gt_starts.get(matched_gt_index, np.nan),
            'GT_Stop': gt_stops.get(matched_gt_index, np.nan),
            'Matched_GT_Index': matched_gt_index,
            'Match_IoU': pred_to_iou.get(row.pred_idx, np.nan),
            'y_true': 1 if is_tp else 0,
            'length': eval_len,
            'Prediction_Source': getattr(row, 'Prediction_Source', np.nan),
            'Matched_GT_Source': gt_sources.get(matched_gt_index, np.nan),
            'Total_GT_ORFs': len(gt_df),
            'Cell_Type_Total_GT_ORFs': cell_gt_counts[row.Cell_Type],
        }
        for metric in eval_metrics:
            value = metric_values_by_prediction[row.pred_idx].get(
                metric, np.nan
            )
            record[metric] = float(value) if pd.notna(value) else np.nan
        eval_records.append(record)

    for row in gt_df.itertuples(index=False):
        if row.gt_idx not in matched_gt_indices:
            record = {
                'Record_Type': 'Missed_GT',
                'Cell_Type': row.Cell_Type,
                'Tid': row.Tid_clean,
                'Tid_Original': str(getattr(row, 'Tid', row.Tid_clean)),
                'Pred_Index': np.nan,
                'Pred_Start': np.nan,
                'Pred_Stop': np.nan,
                'GT_Start': row.start_gt,
                'GT_Stop': row.stop_gt,
                'Matched_GT_Index': row.gt_idx,
                'Match_IoU': np.nan,
                'y_true': 1,
                'length': row.length,
                'Prediction_Source': np.nan,
                'Matched_GT_Source': getattr(row, 'GT_Source', np.nan),
                'Total_GT_ORFs': len(gt_df),
                'Cell_Type_Total_GT_ORFs': cell_gt_counts[row.Cell_Type],
            }
            for metric in eval_metrics:
                record[metric] = -1.0
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

    def finite_metric_arrays(frame, metric):
        """Return aligned labels and finite numeric scores for one metric."""
        scores = pd.to_numeric(frame[metric], errors='coerce').to_numpy(
            dtype=float
        )
        labels = pd.to_numeric(frame['y_true'], errors='coerce').to_numpy(
            dtype=float
        )
        finite_mask = np.isfinite(scores) & np.isfinite(labels)
        return labels[finite_mask].astype(int), scores[finite_mask]

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
        y_true, scores = finite_metric_arrays(candidate_eval_df, metric)
        d_name = display_names.get(metric, metric)

        if len(y_true) > 0 and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            fpr_plot, tpr_plot = subsample_curve(fpr, tpr)
            roc_dfs.append(pd.DataFrame({
                'FPR': fpr_plot, 'TPR': tpr_plot,
                'Metric': d_name, 'AUC': roc_auc,
            }))

            prec, rec, _ = precision_recall_curve(y_true, scores)
            pr_auc = average_precision_score(y_true, scores)
            f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
            best_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0

            rec_plot, prec_plot = subsample_curve(rec, prec)
            pr_dfs.append(pd.DataFrame({
                'Recall': rec_plot, 'Precision': prec_plot,
                'Metric': d_name, 'AUC': pr_auc,
            }))
        else:
            roc_auc = np.nan
            pr_auc = np.nan
            best_f1 = np.nan
            print(
                f"  [Warning] Skipping curves for '{metric}': fewer than "
                "two label classes remain after removing non-finite scores."
            )
        
        # Append to comprehensive records
        comprehensive_records.append({
            'Cell_Type': 'Overall',
            'Feature': d_name,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Best_F1': best_f1,
            'Candidate_Count': len(y_true),
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
            y_metric, scores_c = finite_metric_arrays(group_df, metric)
            d_name = display_names.get(metric, metric)
            if len(y_metric) == 0 or len(np.unique(y_metric)) < 2:
                continue

            fpr_c, tpr_c, _ = roc_curve(y_metric, scores_c)
            roc_auc_c = auc(fpr_c, tpr_c)

            prec_c, rec_c, _ = precision_recall_curve(y_metric, scores_c)
            pr_auc_c = average_precision_score(y_metric, scores_c)

            f1_scores_c = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-9)
            best_f1_c = np.max(f1_scores_c) if len(f1_scores_c) > 0 else 0.0
            
            # Append to comprehensive records
            comprehensive_records.append({
                'Cell_Type': cell_type,
                'Feature': d_name,
                'ROC-AUC': roc_auc_c,
                'PR-AUC': pr_auc_c,
                'Best_F1': best_f1_c,
                'Candidate_Count': len(y_metric),
            })

    # ---------------------------------------------------------
    # 3. Save comprehensive CSV and plot figures
    # ---------------------------------------------------------
    comprehensive_df = pd.DataFrame(comprehensive_records)
    comprehensive_df.to_csv(os.path.join(out_dir, "comprehensive_metrics_summary.csv"), index=False)
    print("  -> Saved unified metrics table to 'comprehensive_metrics_summary.csv'")

    # --- Plot: Overall curves ---
    if roc_dfs and pr_dfs:
        all_roc_df = pd.concat(roc_dfs, ignore_index=True)
        all_pr_df = pd.concat(pr_dfs, ignore_index=True)
        all_roc_df['Legend'] = all_roc_df.apply(
            lambda row: f"{row['Metric']} (AUC={row['AUC']:.3f})", axis=1
        )
        all_pr_df['Legend'] = all_pr_df.apply(
            lambda row: f"{row['Metric']} (AUC={row['AUC']:.3f})", axis=1
        )

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
    fig, ax = plt.subplots(figsize=(15, figure_height))
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


def plot_combined_vs_single_signature_performance(
        results: Optional[Union[Mapping[str, object], str, os.PathLike]] = None,
        combination_metrics_path: Optional[Union[str, os.PathLike]] = None,
        single_metrics_path: Optional[Union[str, os.PathLike]] = None,
        evaluation_path: Optional[Union[str, os.PathLike]] = None,
        out_dir: Optional[Union[str, os.PathLike]] = None,
        primary_metric: str = 'Candidate_PR_AUC',
        top_n_combinations: int = 10,
        selected_combinations: Optional[List[str]] = None,
        single_signatures: Optional[List[str]] = None,
        ranking_scope: str = 'Overall',
        metric_columns: Optional[List[str]] = None,
        output_prefix: str = 'combined_vs_single_signature_performance',
        cmap: str = 'YlGnBu') -> List[str]:
    """Plot selected combinations and individual signatures independently.

    ``results`` may be the dictionary returned by
    ``evaluate_orf_level_predictions`` or a result-directory path. Historical
    results can instead be supplied through ``combination_metrics_path`` and
    ``single_metrics_path``. The latter accepts either the comprehensive
    metrics schema or the normalized feature-combination metrics schema.
    """
    metric_aliases = {
        'ROC-AUC': 'Candidate_ROC_AUC',
        'PR-AUC': 'Candidate_PR_AUC',
        'Best_F1': 'Candidate_Best_F1',
    }
    single_signature_display_names = {
        'rank_score': 'Caller Ranking Score',
        'occupancy_expr_score': 'Expression-weighted Occupancy Score',
        'occupancy_score': 'Predicted Occupancy Score',
        'log_total_occupancy': 'Log Total Predicted Occupancy',
        'total_occupancy': 'Total Predicted Occupancy',
        'collapse_score': 'Boundary-aware Collapse Score',
        'expr_score': 'Expression Score (TPM*Signal)',
        'base_translation_score': 'Base Translation Score',
        'base_expr_score': 'Base Expression-weighted Score',
        'sequence_length_score': 'Sequence-length Score',
        'translation_score': 'Pure Translation Score',
        'transcription_score': 'Pure Transcription Score',
        'seq_score': 'Pure ORF-structure Score',
        'kozak_score': 'Kozak Context Score',
        'start_codon_score': 'Start-codon Prior Score',
        'score': 'Final Score',
        'mean_intensity': 'Mean Intensity',
        'tri_nucleotide_periodicity': 'Periodicity',
        'uniformity_of_signal': 'Uniformity',
        'step_up_contrast': 'Step-up Contrast',
        'drop_off': 'Drop-off',
    }
    display_name_to_column = {
        display_name: score_column
        for score_column, display_name in single_signature_display_names.items()
    }
    primary_metric = metric_aliases.get(primary_metric, primary_metric)
    if metric_columns is not None:
        metric_columns = [
            metric_aliases.get(column, column) for column in metric_columns
        ]

    combination_source = combination_metrics_path
    single_source = single_metrics_path
    evaluation_source = evaluation_path
    inferred_dir = None

    if isinstance(results, Mapping):
        if combination_source is None:
            combination_source = results.get('feature_combination_metrics')
        if single_source is None:
            single_source = results.get('comprehensive_metrics')
        if evaluation_source is None:
            evaluation_source = results.get('evaluation')
        returned_evaluation_path = results.get('evaluation_path')
        if evaluation_source is None:
            evaluation_source = returned_evaluation_path
        if returned_evaluation_path:
            inferred_dir = os.path.dirname(
                os.path.abspath(returned_evaluation_path)
            )
    elif results is not None:
        result_path = os.path.abspath(os.fspath(results))
        if os.path.isdir(result_path):
            inferred_dir = result_path
        elif os.path.isfile(result_path):
            inferred_dir = os.path.dirname(result_path)
            filename = os.path.basename(result_path)
            if filename == 'feature_combination_metrics.csv':
                combination_source = combination_source or result_path
            elif filename == 'comprehensive_metrics_summary.csv':
                single_source = single_source or result_path
            elif filename == 'unified_evaluation_table.csv':
                evaluation_source = evaluation_source or result_path
            else:
                raise ValueError(
                    "A results file must be feature_combination_metrics.csv, "
                    "comprehensive_metrics_summary.csv, or "
                    "unified_evaluation_table.csv."
                )
        else:
            raise FileNotFoundError(f"Results path not found: {result_path}")

    if inferred_dir is None:
        for source in (combination_source, single_source, evaluation_source):
            if source is not None and not isinstance(source, pd.DataFrame):
                inferred_dir = os.path.dirname(
                    os.path.abspath(os.fspath(source))
                )
                break

    if inferred_dir is not None:
        inferred_combination_path = os.path.join(
            inferred_dir, 'feature_combination_metrics.csv'
        )
        inferred_single_path = os.path.join(
            inferred_dir, 'comprehensive_metrics_summary.csv'
        )
        inferred_evaluation_path = os.path.join(
            inferred_dir, 'unified_evaluation_table.csv'
        )
        if combination_source is None and os.path.isfile(
                inferred_combination_path):
            combination_source = inferred_combination_path
        if single_source is None and os.path.isfile(inferred_single_path):
            single_source = inferred_single_path
        if evaluation_source is None and os.path.isfile(
                inferred_evaluation_path):
            evaluation_source = inferred_evaluation_path

    def load_table(source, source_name: str) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if source is None:
            return pd.DataFrame()
        path = os.path.abspath(os.fspath(source))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{source_name} not found: {path}")
        return pd.read_csv(path)

    combination_df = load_table(
        combination_source, 'Combination metrics table'
    )
    if combination_df.empty:
        raise ValueError(
            "No feature-combination metrics were found. Run the evaluation "
            "with evaluate_score_combinations=True first."
        )
    required_combination_columns = {
        'Cell_Type', 'Score_Column', 'Score_Label', 'Method', primary_metric,
    }
    missing = required_combination_columns.difference(combination_df.columns)
    if missing:
        raise ValueError(
            f"Combination metrics are missing columns: {sorted(missing)}"
        )

    def top_k_sort_key(column: str) -> int:
        try:
            return int(column.rsplit('_', 1)[-1])
        except ValueError:
            return 0

    precision_columns = sorted(
        [
            column for column in combination_df.columns
            if column.startswith('Precision_at_')
        ],
        key=top_k_sort_key,
    )
    recall_columns = sorted(
        [
            column for column in combination_df.columns
            if column.startswith('Unique_GT_Recall_at_')
        ],
        key=top_k_sort_key,
    )
    if metric_columns is None:
        metric_columns = [
            'Candidate_ROC_AUC',
            'Candidate_PR_AUC',
            'Candidate_Best_F1',
            *precision_columns,
            *recall_columns,
        ]

    raw_single_df = load_table(single_source, 'Single-signature metrics table')
    if not raw_single_df.empty and {
            'Feature', 'ROC-AUC', 'PR-AUC', 'Best_F1'
    }.issubset(raw_single_df.columns):
        single_df = raw_single_df.rename(columns={
            'Feature': 'Score_Label',
            'ROC-AUC': 'Candidate_ROC_AUC',
            'PR-AUC': 'Candidate_PR_AUC',
            'Best_F1': 'Candidate_Best_F1',
        }).copy()
        single_df['Score_Column'] = single_df['Score_Label'].map(
            display_name_to_column
        ).fillna(single_df['Score_Label'])
        single_df['Method'] = 'single'
    elif not raw_single_df.empty:
        required_single_columns = {
            'Cell_Type', 'Score_Column', 'Score_Label', primary_metric,
        }
        missing = required_single_columns.difference(raw_single_df.columns)
        if missing:
            raise ValueError(
                f"Single-signature metrics are missing columns: {sorted(missing)}"
            )
        single_df = raw_single_df.copy()
        single_df['Method'] = 'single'
    else:
        single_df = pd.DataFrame()

    base_df = combination_df[
        combination_df['Method'].astype(str).str.lower() == 'base'
    ].copy()
    base_df['Method'] = 'single'
    if single_df.empty:
        single_df = base_df
    elif not base_df.empty:
        common_columns = sorted(set(single_df.columns).union(base_df.columns))
        single_df = pd.concat([
            single_df.reindex(columns=common_columns),
            base_df.reindex(columns=common_columns),
        ], ignore_index=True)

    requested_top_k_columns = [
        column for column in metric_columns
        if column.startswith('Precision_at_')
        or column.startswith('Unique_GT_Recall_at_')
    ]
    top_k_values = sorted({
        top_k_sort_key(column)
        for column in requested_top_k_columns
        if top_k_sort_key(column) > 0
    })
    evaluation_df = load_table(
        evaluation_source, 'Unified evaluation table'
    )
    if top_k_values and evaluation_df.empty:
        raise ValueError(
            "Top-K columns are present in the combination metrics, but the "
            "unified evaluation table was not found. Supply evaluation_path "
            "or pass the results dictionary/result directory."
        )
    if top_k_values:
        required_evaluation_columns = {
            'Cell_Type', 'y_true', 'Matched_GT_Index'
        }
        missing = required_evaluation_columns.difference(evaluation_df.columns)
        if missing:
            raise ValueError(
                f"Unified evaluation table is missing columns: {sorted(missing)}"
            )
        candidate_df = (
            evaluation_df[
                evaluation_df['Record_Type'] == 'Prediction'
            ].copy()
            if 'Record_Type' in evaluation_df.columns
            else evaluation_df.copy()
        )
        candidate_df['y_true'] = pd.to_numeric(
            candidate_df['y_true'], errors='coerce'
        ).fillna(0.0)

        for row_index, row in single_df.iterrows():
            score_column = str(row['Score_Column'])
            if score_column not in candidate_df.columns:
                continue
            scope_name = str(row['Cell_Type'])
            if scope_name == 'Overall':
                scope_candidates = candidate_df.copy()
                scope_evaluation = evaluation_df
                total_column = 'Total_GT_ORFs'
            elif scope_name == 'Macro_Average':
                continue
            else:
                scope_candidates = candidate_df[
                    candidate_df['Cell_Type'].astype(str) == scope_name
                ].copy()
                scope_evaluation = evaluation_df[
                    evaluation_df['Cell_Type'].astype(str) == scope_name
                ]
                total_column = 'Cell_Type_Total_GT_ORFs'

            scope_candidates[score_column] = pd.to_numeric(
                scope_candidates[score_column], errors='coerce'
            )
            scope_candidates = scope_candidates.replace(
                [np.inf, -np.inf], np.nan
            ).dropna(subset=[score_column])
            ranked_df = scope_candidates.sort_values(
                score_column, ascending=False, kind='mergesort'
            ).reset_index(drop=True)

            total_gt_values = (
                pd.to_numeric(
                    scope_evaluation[total_column], errors='coerce'
                ).dropna()
                if total_column in scope_evaluation.columns
                else pd.Series(dtype=float)
            )
            if not total_gt_values.empty:
                total_gt = int(total_gt_values.iloc[0])
            else:
                total_gt = int(
                    scope_evaluation['Matched_GT_Index'].dropna().nunique()
                )

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
                single_df.loc[
                    row_index, f'Precision_at_{k}'
                ] = precision_value
                single_df.loc[
                    row_index, f'Unique_GT_Recall_at_{k}'
                ] = recall_value

    combined_df = combination_df[
        combination_df['Method'].astype(str).str.lower() != 'base'
    ].copy()
    combined_df['Signature_Group'] = 'Combined signatures'
    single_df['Signature_Group'] = 'Single signatures'
    combined_df['Plot_Key'] = (
        'combined::' + combined_df['Score_Column'].astype(str)
    )
    single_df['Plot_Key'] = 'single::' + single_df['Score_Column'].astype(str)

    def select_scores(
            frame: pd.DataFrame,
            requested: Optional[List[str]],
            label: str) -> pd.DataFrame:
        if requested is None:
            return frame
        if isinstance(requested, str):
            requested = [requested]
        requested = list(dict.fromkeys(str(value) for value in requested))
        available = set(frame['Score_Column'].astype(str)).union(
            frame['Score_Label'].astype(str)
        )
        missing_scores = [
            value for value in requested if value not in available
        ]
        if missing_scores:
            raise ValueError(
                f"Unknown {label}: {missing_scores}. Use Score_Column or "
                "Score_Label values from the corresponding metrics table."
            )
        return frame[
            frame['Score_Column'].astype(str).isin(requested)
            | frame['Score_Label'].astype(str).isin(requested)
        ].copy()

    combined_rank = combined_df[
        combined_df['Cell_Type'].astype(str) == str(ranking_scope)
    ].copy()
    single_rank = single_df[
        single_df['Cell_Type'].astype(str) == str(ranking_scope)
    ].copy()
    if combined_rank.empty:
        raise ValueError(f"No combinations found for scope '{ranking_scope}'.")

    if selected_combinations is None:
        if top_n_combinations < 1:
            raise ValueError("top_n_combinations must be at least 1.")
        combined_rank = combined_rank.sort_values(
            primary_metric, ascending=False, na_position='last'
        ).head(top_n_combinations)
    else:
        combined_rank = select_scores(
            combined_rank, selected_combinations, 'combined signatures'
        ).sort_values(primary_metric, ascending=False, na_position='last')

    single_rank = select_scores(
        single_rank, single_signatures, 'single signatures'
    ).sort_values(primary_metric, ascending=False, na_position='last')
    if single_rank.empty:
        raise ValueError(
            "No individual signatures were selected. Supply "
            "single_metrics_path or select base scores present in the "
            "combination metrics table."
        )

    selected_df = pd.concat(
        [combined_rank, single_rank], ignore_index=True
    ).drop_duplicates(subset=['Plot_Key'], keep='first')
    missing_metrics = [
        column for column in metric_columns if column not in selected_df.columns
    ]
    if missing_metrics:
        raise ValueError(f"Unknown heatmap metrics: {missing_metrics}")
    top_k_metric_columns = requested_top_k_columns
    if top_k_metric_columns:
        selected_single_df = selected_df[
            selected_df['Signature_Group'] == 'Single signatures'
        ]
        incomplete_single_mask = selected_single_df[
            top_k_metric_columns
        ].isna().any(axis=1)
        if incomplete_single_mask.any():
            incomplete_signatures = selected_single_df.loc[
                incomplete_single_mask, 'Score_Column'
            ].astype(str).tolist()
            raise ValueError(
                "Top-K metrics could not be calculated for single signatures: "
                f"{incomplete_signatures}. Confirm that these score columns "
                "exist in unified_evaluation_table.csv."
            )

    if out_dir is None:
        out_dir = inferred_dir or os.getcwd()
    out_dir = os.path.abspath(os.fspath(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    overall_matrix = selected_df.set_index('Score_Label')[metric_columns]
    figure_height = max(4.0, 0.36 * len(overall_matrix) + 1.8)
    figure_width = max(9.0, 0.7 * len(metric_columns) + 3.0)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(
        overall_matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        fmt='.3f',
        linewidths=0.4,
        linecolor='white',
        cbar_kws={'label': 'Performance'},
        ax=ax,
    )
    combination_count = len(combined_rank)
    ax.axhline(combination_count, color='black', linewidth=1.5)
    for tick_label in ax.get_yticklabels()[combination_count:]:
        tick_label.set_fontstyle('italic')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(
        f'Combined versus individual signatures ({ranking_scope})'
    )
    plt.tight_layout()
    overall_path = os.path.join(out_dir, f'{output_prefix}.overall.pdf')
    fig.savefig(overall_path, bbox_inches='tight')
    plt.close(fig)

    all_metrics_df = pd.concat(
        [combined_df, single_df], ignore_index=True, sort=False
    )
    selected_keys = selected_df['Plot_Key'].tolist()
    cell_df = all_metrics_df[
        (~all_metrics_df['Cell_Type'].isin(['Overall', 'Macro_Average']))
        & all_metrics_df['Plot_Key'].isin(selected_keys)
    ].copy()
    output_paths = [overall_path]
    if not cell_df.empty:
        cell_matrix = cell_df.pivot_table(
            index='Plot_Key',
            columns='Cell_Type',
            values=primary_metric,
            aggfunc='first',
        ).reindex(selected_keys)
        label_lookup = selected_df.set_index('Plot_Key')['Score_Label']
        cell_matrix.index = [
            label_lookup.loc[index] for index in cell_matrix.index
        ]
        figure_width = max(6.0, 0.65 * len(cell_matrix.columns) + 3.5)
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        sns.heatmap(
            cell_matrix,
            cmap=cmap,
            vmin=0,
            vmax=1,
            annot=True,
            fmt='.3f',
            linewidths=0.4,
            linecolor='white',
            cbar_kws={'label': primary_metric},
            ax=ax,
        )
        ax.axhline(combination_count, color='black', linewidth=1.5)
        for tick_label in ax.get_yticklabels()[combination_count:]:
            tick_label.set_fontstyle('italic')
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('')
        ax.set_title(f'{primary_metric} across Cell Types')
        plt.tight_layout()
        cell_path = os.path.join(out_dir, f'{output_prefix}.by_cell_type.pdf')
        fig.savefig(cell_path, bbox_inches='tight')
        plt.close(fig)
        output_paths.append(cell_path)

    return output_paths


# =====================================================================
# Main Orchestrator
# =====================================================================
def evaluate_orf_level_predictions(
        pred_csv_paths: PredictionPathInput,
        gt_csv_paths: GroundTruthPathInput,
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
        combination_ranking_scope: str = 'Overall',
        top_k_score_col: Optional[str] = None,
        top_k_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        save_top_k: bool = True,
        cell_type: Optional[str] = None):
    """Run comprehensive and Top-K evaluation from one matched candidate table.

    ``target_transcript_ids`` may be a single transcript collection applied to
    every evaluated cell type, or a ``{cell_type: transcript_collection}``
    mapping. Mapping keys must match prediction ``Cell_Type`` values and the
    keys used in ``gt_csv_paths``.

    ``top_k_score_col`` selects one existing score for the full Precision@K and
    Recall@K trajectory. ``top_k_combined_score`` accepts the same combination
    definition as ``calculate_top_k_precision``. When neither is supplied, the
    primary evaluation score is used. The function returns every core table so
    plotting can remain a separate notebook step.
    """
    if top_k_score_col is not None and top_k_combined_score is not None:
        raise ValueError(
            "Use either top_k_score_col or top_k_combined_score, not both."
        )
    os.makedirs(out_dir, exist_ok=True)

    pred_df, gt_df, score_col = load_and_filter_data(
        pred_csv_paths=pred_csv_paths,
        gt_csv_paths=gt_csv_paths,
        target_transcript_ids=target_transcript_ids,
        min_orf_len=min_orf_len,
        max_orf_len=max_orf_len,
        target_score_col=target_score_col,
        callable_start_codons=callable_start_codons,
        cell_type=cell_type,
    )

    combination_metadata = pd.DataFrame()
    if evaluate_score_combinations:
        if 'base_translation_score' in pred_df.columns:
            default_base_scores = [
                'rank_score',
                'occupancy_expr_score',
                'occupancy_score',
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

    if top_k_combined_score is not None:
        pred_df, selected_top_k_col, top_k_score_label = (
            add_requested_combined_score(
                pred_df=pred_df,
                combined_score=top_k_combined_score,
            )
        )
    else:
        selected_top_k_col = top_k_score_col or score_col
        if selected_top_k_col not in pred_df.columns:
            raise ValueError(
                f"Top-K score column '{selected_top_k_col}' was not found."
            )
        top_k_score_label = selected_top_k_col

    all_possible_metrics = {
        'rank_score': 'Caller Ranking Score',
        'occupancy_expr_score': 'Expression-weighted Occupancy Score',
        'occupancy_score': 'Predicted Occupancy Score',
        'log_total_occupancy': 'Log Total Predicted Occupancy',
        'total_occupancy': 'Total Predicted Occupancy',
        'collapse_score': 'Boundary-aware Collapse Score',
        'expr_score': 'Expression Score (TPM*Signal)',
        'base_translation_score': 'Base Translation Score',
        'base_expr_score': 'Base Expression-weighted Score',
        'sequence_length_score': 'Sequence-length Score',
        'translation_score': 'Pure Translation Score',
        'transcription_score': 'Pure Transcription Score',
        'seq_score': 'Pure ORF-structure Score',
        'kozak_score': 'Kozak Context Score',
        'start_codon_score': 'Start-codon Prior Score',
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

    combination_score_columns = (
        combination_metadata['Score_Column'].tolist()
        if not combination_metadata.empty else []
    )
    match_metrics = list(dict.fromkeys(
        eval_metrics + combination_score_columns + [selected_top_k_col]
    ))
    eval_df = match_and_build_eval_df(
        pred_df, gt_df, match_metrics, overlap_threshold
    )
    evaluation_path = os.path.join(out_dir, "unified_evaluation_table.csv")
    eval_df.to_csv(evaluation_path, index=False)

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
    
    evaluate_and_plot_global(eval_df, eval_metrics, display_names, out_dir)

    combination_summary = pd.DataFrame()
    top_k_values = combination_top_k_values or [100, 500, 1000]
    if evaluate_score_combinations:
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

    top_k_df = calculate_top_k_from_evaluation(
        eval_df=eval_df,
        score_col=selected_top_k_col,
        score_label=top_k_score_label,
    )
    top_k_summary = summarize_top_k_values(top_k_df, top_k_values)
    if save_top_k:
        top_k_df.to_csv(
            os.path.join(out_dir, 'Precision_at_K_data.csv'), index=False
        )
        top_k_summary.to_csv(
            os.path.join(out_dir, 'top_k_metrics_summary.csv'), index=False
        )

    print(
        "\nAll evaluation processes successfully finished. "
        f"Output directory: {out_dir}"
    )
    return {
        'evaluation': eval_df,
        'top_k': top_k_df,
        'top_k_summary': top_k_summary,
        'feature_combination_metrics': combination_summary,
        'feature_combination_definitions': combination_metadata,
        'primary_score_col': score_col,
        'top_k_score_col': selected_top_k_col,
        'evaluation_path': evaluation_path,
    }



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


def resolve_manifest_score_request(
        config: Mapping[str, object],
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        trace_model_name: str = 'TRACE'):
    """Resolve one model's score column or feature-combination request.

    A manifest-level ``combined_score`` has the highest priority. The global
    ``trace_combined_score`` then overrides ``score_col`` for the named TRACE
    model. Other models use their manifest-level ``score_col``.
    """
    model_name = str(config.get('model', ''))
    configured_combination = config.get('combined_score')
    configured_score_col = config.get('score_col')
    if configured_combination is not None and configured_score_col is not None:
        raise ValueError(
            f"Model '{model_name}' defines both score_col and combined_score. "
            "Specify only one."
        )

    if configured_combination is not None:
        return None, configured_combination
    if (
            trace_combined_score is not None
            and model_name.casefold() == str(trace_model_name).casefold()
    ):
        return None, trace_combined_score
    return configured_score_col, None


def prepare_evaluation_score(
        evaluation_df: pd.DataFrame,
        score_col: Optional[str] = None,
        combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None):
    """Filter prediction records and prepare one ranking score consistently."""
    if score_col is not None and combined_score is not None:
        raise ValueError("Specify either score_col or combined_score, not both.")

    candidate_df = evaluation_df.copy()
    if 'Record_Type' in candidate_df.columns:
        candidate_df = candidate_df[
            candidate_df['Record_Type'] == 'Prediction'
        ].copy()
    if candidate_df.empty:
        raise ValueError("The evaluation table contains no prediction records.")

    if combined_score is not None:
        return add_requested_combined_score(candidate_df, combined_score)

    if score_col is not None:
        if score_col not in candidate_df.columns:
            raise ValueError(
                f"Score column '{score_col}' was not found. Available columns: "
                f"{candidate_df.columns.tolist()}"
            )
        selected_score_col = score_col
    else:
        selected_score_col = resolve_score_col(candidate_df, None)
    return candidate_df, selected_score_col, selected_score_col


def calculate_top_k_from_evaluation(
        eval_df: pd.DataFrame,
        score_col: str,
        score_label: Optional[str] = None) -> pd.DataFrame:
    """Calculate Top-K trajectories from one already matched evaluation table."""
    required_columns = {
        'Record_Type', 'Cell_Type', 'Tid', 'Matched_GT_Index', 'y_true',
        score_col,
    }
    missing_columns = required_columns.difference(eval_df.columns)
    if missing_columns:
        raise ValueError(
            f"Evaluation table is missing columns: {sorted(missing_columns)}"
        )

    candidate_df = eval_df[
        eval_df['Record_Type'] == 'Prediction'
    ].copy()
    candidate_df[score_col] = pd.to_numeric(
        candidate_df[score_col], errors='coerce'
    )
    candidate_df = candidate_df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=[score_col])
    if candidate_df.empty:
        return pd.DataFrame(
            columns=['K', 'TP_Count', 'Precision', 'Recall', 'Score_Type']
        )
    candidate_df = candidate_df.sort_values(
        score_col, ascending=False, kind='mergesort'
    ).reset_index(drop=True)

    if 'Total_GT_ORFs' in eval_df.columns:
        total_gt = int(pd.to_numeric(
            eval_df['Total_GT_ORFs'], errors='coerce'
        ).dropna().max())
    else:
        total_gt = int(eval_df['Matched_GT_Index'].dropna().nunique())
    if total_gt <= 0:
        raise ValueError("The evaluation table contains no callable GT ORFs.")

    if 'Cell_Type_Total_GT_ORFs' in eval_df.columns:
        cell_gt_counts = (
            eval_df[['Cell_Type', 'Cell_Type_Total_GT_ORFs']]
            .dropna()
            .drop_duplicates('Cell_Type')
            .set_index('Cell_Type')['Cell_Type_Total_GT_ORFs']
            .astype(int)
            .to_dict()
        )
    else:
        cell_gt_counts = (
            eval_df.dropna(subset=['Matched_GT_Index'])
            .groupby('Cell_Type')['Matched_GT_Index']
            .nunique()
            .astype(int)
            .to_dict()
        )

    is_tp_array = candidate_df['y_true'].astype(int).to_numpy()
    matched_gt_indices = candidate_df['Matched_GT_Index'].to_numpy()
    tp_cumsum = np.cumsum(is_tp_array)
    new_gt_hits = np.zeros(len(candidate_df), dtype=int)
    seen_gt_indices = set()
    for index, matched_gt_index in enumerate(matched_gt_indices):
        if pd.notna(matched_gt_index) and matched_gt_index not in seen_gt_indices:
            new_gt_hits[index] = 1
            seen_gt_indices.add(matched_gt_index)
    unique_gt_cumsum = np.cumsum(new_gt_hits)
    k_array = np.arange(1, len(candidate_df) + 1)

    cell_type_series = candidate_df['Cell_Type'].astype(str)
    cell_type_k = cell_type_series.groupby(
        cell_type_series, sort=False
    ).cumcount() + 1
    cell_type_tp_count = pd.Series(is_tp_array).groupby(
        cell_type_series, sort=False
    ).cumsum()
    cell_type_unique_gt_count = pd.Series(new_gt_hits).groupby(
        cell_type_series, sort=False
    ).cumsum()
    cell_type_total_gt = cell_type_series.map(cell_gt_counts).astype(int)

    def values_or_default(column: str, default):
        if column in candidate_df.columns:
            return candidate_df[column].to_numpy()
        return np.full(len(candidate_df), default)

    score_label = score_label or score_col
    top_k_df = pd.DataFrame({
        'K': k_array,
        'TP_Count': tp_cumsum,
        'Unique_GT_Hit_Count': unique_gt_cumsum,
        'Precision': tp_cumsum / k_array,
        'Recall': unique_gt_cumsum / total_gt,
        'Precision_at_K': tp_cumsum / k_array,
        'Recall_at_K': unique_gt_cumsum / total_gt,
        'Total_GT_ORFs': total_gt,
        'Cell_Type': cell_type_series.to_numpy(),
        'Cell_Type_K': cell_type_k.to_numpy(),
        'Cell_Type_TP_Count': cell_type_tp_count.to_numpy(),
        'Cell_Type_Unique_GT_Hit_Count': (
            cell_type_unique_gt_count.to_numpy()
        ),
        'Cell_Type_Precision': (
            cell_type_tp_count / cell_type_k
        ).to_numpy(),
        'Cell_Type_Recall': (
            cell_type_unique_gt_count / cell_type_total_gt
        ).to_numpy(),
        'Cell_Type_Total_GT_ORFs': cell_type_total_gt.to_numpy(),
        'Tid': (
            candidate_df['Tid_Original'].astype(str).to_numpy()
            if 'Tid_Original' in candidate_df.columns
            else candidate_df['Tid'].astype(str).to_numpy()
        ),
        'Pred_Start': values_or_default('Pred_Start', np.nan),
        'Pred_Stop': values_or_default('Pred_Stop', np.nan),
        'Score': candidate_df[score_col].to_numpy(),
        'Is_TP': is_tp_array,
        'Is_New_GT_Hit': new_gt_hits,
        'Matched_GT_Index': matched_gt_indices,
        'Match_IoU': values_or_default('Match_IoU', np.nan),
        'Prediction_Source': values_or_default('Prediction_Source', np.nan),
        'Score_Type': score_label,
        'Score_Column': score_col,
        'Matched_GT_Source': values_or_default(
            'Matched_GT_Source', np.nan
        ),
    })
    return top_k_df


def summarize_top_k_values(
        top_k_df: pd.DataFrame,
        top_k_values: Iterable[int]) -> pd.DataFrame:
    """Extract exact global Precision@K and Recall@K values."""
    records = []
    for requested_k in sorted({int(k) for k in top_k_values if int(k) > 0}):
        effective_k = min(requested_k, len(top_k_df))
        if effective_k == 0:
            continue
        row = top_k_df.iloc[effective_k - 1]
        records.append({
            'Requested_K': requested_k,
            'Effective_K': effective_k,
            'TP_Count': int(row['TP_Count']),
            'Unique_GT_Hit_Count': int(row['Unique_GT_Hit_Count']),
            'Precision': float(row['Precision']),
            'Recall': float(row['Recall']),
            'Total_GT_ORFs': int(row['Total_GT_ORFs']),
            'Score_Type': row['Score_Type'],
            'Score_Column': row['Score_Column'],
        })
    return pd.DataFrame(records)


def calculate_top_k_precision(
        pred_csv_paths: Optional[PredictionPathInput] = None,
        gt_csv_paths: Optional[GroundTruthPathInput] = None,
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
        ] = None,
        evaluation_df: Optional[pd.DataFrame] = None,
        evaluation_csv_path: Optional[str] = None) -> pd.DataFrame:
    """Calculate Top-K metrics using the shared preprocessing and match table.

    For a one-run workflow, pass ``evaluation_df=results['evaluation']`` from
    ``evaluate_orf_level_predictions``. Raw prediction/GT paths remain
    supported and now use the same loader, filters, deduplication, and matcher
    as the comprehensive evaluation.
    """
    if score_col is not None and target_score_col is not None:
        raise ValueError("Use either score_col or target_score_col, not both.")
    requested_score_col = score_col or target_score_col
    if combined_score is not None and requested_score_col is not None:
        raise ValueError(
            "Use either score_col/target_score_col or combined_score, not both."
        )
    if evaluation_df is not None and evaluation_csv_path is not None:
        raise ValueError(
            "Use either evaluation_df or evaluation_csv_path, not both."
        )

    if evaluation_csv_path is not None:
        evaluation_df = pd.read_csv(evaluation_csv_path)

    if evaluation_df is not None:
        eval_df = evaluation_df.copy()
        if combined_score is not None:
            eval_df, ranking_score_col, score_type_label = (
                add_requested_combined_score(eval_df, combined_score)
            )
        else:
            ranking_score_col = resolve_score_col(
                eval_df, requested_score_col
            )
            score_type_label = ranking_score_col
        return calculate_top_k_from_evaluation(
            eval_df, ranking_score_col, score_type_label
        )

    if pred_csv_paths is not None and pred_csv_path is not None:
        raise ValueError("Use either pred_csv_paths or pred_csv_path, not both.")
    if gt_csv_paths is not None and gt_csv_path is not None:
        raise ValueError("Use either gt_csv_paths or gt_csv_path, not both.")
    pred_csv_paths = pred_csv_paths or pred_csv_path
    gt_csv_paths = gt_csv_paths or gt_csv_path
    if pred_csv_paths is None or gt_csv_paths is None:
        raise ValueError(
            "Supply evaluation_df/evaluation_csv_path or both prediction and "
            "ground-truth paths."
        )

    pred_df, gt_df, primary_score_col = load_and_filter_data(
        pred_csv_paths=pred_csv_paths,
        gt_csv_paths=gt_csv_paths,
        target_transcript_ids=target_transcript_ids,
        min_orf_len=min_orf_len,
        max_orf_len=max_orf_len,
        target_score_col=(requested_score_col if combined_score is None else None),
        callable_start_codons=callable_start_codons,
        cell_type=cell_type,
    )
    if combined_score is not None:
        pred_df, ranking_score_col, score_type_label = (
            add_requested_combined_score(pred_df, combined_score)
        )
    else:
        ranking_score_col = requested_score_col or primary_score_col
        score_type_label = ranking_score_col

    eval_df = match_and_build_eval_df(
        pred_df=pred_df,
        gt_df=gt_df,
        eval_metrics=[ranking_score_col],
        overlap_threshold=overlap_threshold,
    )
    return calculate_top_k_from_evaluation(
        eval_df, ranking_score_col, score_type_label
    )


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

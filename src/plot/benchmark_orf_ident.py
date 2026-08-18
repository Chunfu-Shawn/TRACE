import os
import pickle
import warnings
import numpy as np
import pandas as pd
from collections.abc import Mapping
from typing import Optional, Union
from plotnine import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, t as student_t
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from eval.orf_coding_performance import (
    prepare_evaluation_score,
    resolve_manifest_score_request,
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

# =================================================================
# [NEW] 定义全局配置：统一的颜色与顺序
# =================================================================
GLOBAL_MODEL_COLORS = {
    "TRACE": "#2C6B9A",
    "Convolution": "#637D96",
    "TranslationAI": "#555555",
    "RiboTIE": "#777777",
    "RibORF": "#BBBBBB",
    "RiboTISH": "#999999",
    "ORF-length": "#AF804F",
    "Transcription-level": "#EBC67F"
}

GLOBAL_MODEL_ORDER = [
    "TRACE", 
    "Convolution", 
    "TranslationAI", 
    "RiboTIE", 
    "RiboTISH", 
    "RibORF",
    "ORF-length", 
    "Transcription-level"
]


def compare_multi_model_roc_auc(
        manifest: list,
        out_dir: str = "./results/benchmark",
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        trace_model_name: str = 'TRACE',
        cell_type: Optional[str] = None,
        require_same_total_gt: bool = True,
        model_colors: Optional[Mapping[str, str]] = None,
        title: str = 'Candidate ORF ROC comparison',
        filename: str = 'Benchmark_Multi_Model_ROC_AUC.pdf',
        max_curve_points: int = 2000):
    """Compare candidate-level ROC-AUC across matched model tables.

    Each manifest item requires ``model`` and ``path`` and may specify either
    ``score_col`` or ``combined_score``. ``trace_combined_score`` provides a
    convenient global override for the named TRACE model. Combination inputs
    accept an existing score-column name or the same Base_Score/Features/Method
    definition used by ``evaluate_orf_level_predictions``.
    """
    if not manifest:
        raise ValueError("manifest cannot be empty.")
    if max_curve_points < 2:
        raise ValueError("max_curve_points must be at least 2.")
    if not filename.lower().endswith('.pdf'):
        filename = f"{filename}.pdf"

    curve_frames = []
    summary_records = []
    total_gt_by_model = {}
    for config in manifest:
        if 'model' not in config or 'path' not in config:
            raise ValueError("Every manifest item must define model and path.")
        model_name = str(config['model'])
        csv_path = str(config['path'])
        if not os.path.exists(csv_path):
            warnings.warn(
                f"ROC input was not found and will be skipped: {csv_path}"
            )
            continue

        source_df = pd.read_csv(csv_path)
        if cell_type is not None:
            if 'Cell_Type' not in source_df.columns:
                raise ValueError(
                    f"Model '{model_name}' has no Cell_Type column."
                )
            source_df = source_df[
                source_df['Cell_Type'].astype(str) == str(cell_type)
            ].copy()

        requested_score_col, requested_combination = (
            resolve_manifest_score_request(
                config,
                trace_combined_score=trace_combined_score,
                trace_model_name=trace_model_name,
            )
        )
        candidate_df, selected_score_col, score_label = (
            prepare_evaluation_score(
                source_df,
                score_col=requested_score_col,
                combined_score=requested_combination,
            )
        )
        if 'y_true' not in candidate_df.columns:
            raise ValueError(
                f"Model '{model_name}' evaluation table has no y_true column."
            )

        metric_df = candidate_df[['y_true', selected_score_col]].copy()
        candidate_count_before_filter = len(metric_df)
        metric_df['y_true'] = pd.to_numeric(
            metric_df['y_true'], errors='coerce'
        )
        metric_df[selected_score_col] = pd.to_numeric(
            metric_df[selected_score_col], errors='coerce'
        )
        metric_df = metric_df.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        nonfinite_count = candidate_count_before_filter - len(metric_df)
        metric_df['y_true'] = metric_df['y_true'].astype(int)
        unexpected_labels = set(metric_df['y_true'].unique()) - {0, 1}
        if unexpected_labels:
            raise ValueError(
                f"Model '{model_name}' has non-binary y_true values: "
                f"{sorted(unexpected_labels)}"
            )
        if metric_df['y_true'].nunique() != 2:
            warnings.warn(
                f"Model '{model_name}' lacks both positive and negative "
                "prediction records and will be skipped."
            )
            continue

        false_positive_rate, true_positive_rate, _ = roc_curve(
            metric_df['y_true'].to_numpy(),
            metric_df[selected_score_col].to_numpy(),
        )
        roc_auc_value = auc(false_positive_rate, true_positive_rate)
        if len(false_positive_rate) > max_curve_points:
            indices = np.unique(np.linspace(
                0,
                len(false_positive_rate) - 1,
                max_curve_points,
            ).astype(int))
            false_positive_rate = false_positive_rate[indices]
            true_positive_rate = true_positive_rate[indices]

        curve_frames.append(pd.DataFrame({
            'Model': model_name,
            'False_Positive_Rate': false_positive_rate,
            'True_Positive_Rate': true_positive_rate,
            'ROC_AUC': roc_auc_value,
            'Score_Type': score_label,
            'Score_Column': selected_score_col,
        }))

        total_gt = np.nan
        total_gt_column = (
            'Cell_Type_Total_GT_ORFs'
            if cell_type is not None
            and 'Cell_Type_Total_GT_ORFs' in source_df.columns
            else 'Total_GT_ORFs'
        )
        if total_gt_column in source_df.columns:
            total_values = pd.to_numeric(
                source_df[total_gt_column], errors='coerce'
            ).dropna()
            if not total_values.empty:
                total_gt = int(total_values.max())
                total_gt_by_model[model_name] = total_gt
        summary_records.append({
            'Model': model_name,
            'ROC_AUC': roc_auc_value,
            'Score_Type': score_label,
            'Score_Column': selected_score_col,
            'Candidate_Count': len(metric_df),
            'Excluded_Nonfinite_Count': nonfinite_count,
            'Positive_Count': int(metric_df['y_true'].sum()),
            'Negative_Count': int((metric_df['y_true'] == 0).sum()),
            'Total_GT_ORFs': total_gt,
            'Cell_Type': str(cell_type) if cell_type is not None else 'Overall',
        })

    if not curve_frames:
        raise ValueError("No valid model ROC curves could be calculated.")
    if require_same_total_gt and len(set(total_gt_by_model.values())) > 1:
        details = ', '.join(
            f"{model}={count}" for model, count in total_gt_by_model.items()
        )
        raise ValueError(
            "Models use different callable GT denominators, so their ROC-AUC "
            f"values are not directly comparable: {details}"
        )

    curve_df = pd.concat(curve_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_records).sort_values(
        'ROC_AUC', ascending=False
    ).reset_index(drop=True)
    model_order = [str(config['model']) for config in manifest]
    model_order = list(dict.fromkeys(
        model for model in model_order if model in set(curve_df['Model'])
    ))
    color_lookup = {
        model: (
            model_colors[model]
            if model_colors is not None and model in model_colors
            else GLOBAL_MODEL_COLORS.get(model, "#C0C0C0")
        )
        for model in model_order
    }

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for model in model_order:
        model_curve = curve_df[curve_df['Model'] == model]
        auc_value = float(model_curve['ROC_AUC'].iloc[0])
        ax.plot(
            model_curve['False_Positive_Rate'],
            model_curve['True_Positive_Rate'],
            color=color_lookup[model],
            linewidth=1.8,
            label=f"{model} (AUC={auc_value:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle='--', color='#888888', linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    ax.legend(frameon=False, loc='lower right')
    sns.despine(ax=ax)
    fig.tight_layout()
    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return summary_df, curve_df, save_path


def _resolve_tissue_model_path(
        tissue: str,
        config: Mapping[str, object]) -> str:
    """Resolve one tissue table from a model-specific manifest item."""
    path_by_tissue = config.get('path_by_tissue')
    tissue_result_dirs = config.get('tissue_result_dirs')
    if path_by_tissue is not None and tissue_result_dirs is not None:
        raise ValueError(
            f"Model '{config.get('model')}' defines both path_by_tissue and "
            "tissue_result_dirs. Specify only one."
        )
    if path_by_tissue is not None:
        if not isinstance(path_by_tissue, Mapping):
            raise TypeError("path_by_tissue must be a tissue-to-path mapping.")
        normalized_paths = {
            str(key): value for key, value in path_by_tissue.items()
        }
        if tissue not in normalized_paths:
            raise ValueError(
                f"Model '{config.get('model')}' has no path for tissue "
                f"'{tissue}'."
            )
        path_value = str(normalized_paths[tissue])
        result_dir = os.path.dirname(path_value)
    elif tissue_result_dirs is not None:
        if not isinstance(tissue_result_dirs, Mapping):
            raise TypeError(
                "tissue_result_dirs must be a tissue-to-directory mapping."
            )
        normalized_dirs = {
            str(key): value for key, value in tissue_result_dirs.items()
        }
        if tissue not in normalized_dirs:
            raise ValueError(
                f"Model '{config.get('model')}' has no result directory for "
                f"tissue '{tissue}'."
            )
        result_dir = str(normalized_dirs[tissue])
        path_value = str(
            config.get('path', 'unified_evaluation_table.csv')
        )
    else:
        raise ValueError(
            "Every model manifest item must define tissue_result_dirs or "
            "path_by_tissue."
        )

    path_value = path_value.format(
        tissue=tissue,
        result_dir=str(result_dir),
    )
    if not os.path.isabs(path_value):
        path_value = os.path.join(str(result_dir), path_value)
    return os.path.normpath(path_value)


def _get_manifest_tissues(config: Mapping[str, object]) -> list:
    """Return the tissue labels declared by one model manifest item."""
    path_by_tissue = config.get('path_by_tissue')
    tissue_result_dirs = config.get('tissue_result_dirs')
    if path_by_tissue is not None and tissue_result_dirs is not None:
        raise ValueError(
            f"Model '{config.get('model')}' defines both path_by_tissue and "
            "tissue_result_dirs. Specify only one."
        )
    tissue_mapping = (
        tissue_result_dirs
        if tissue_result_dirs is not None
        else path_by_tissue
    )
    if not isinstance(tissue_mapping, Mapping) or not tissue_mapping:
        raise ValueError(
            f"Model '{config.get('model')}' must define a non-empty "
            "tissue_result_dirs or path_by_tissue mapping."
        )
    return [str(tissue) for tissue in tissue_mapping]


def _extract_total_gt_count(
        source_df: pd.DataFrame,
        tissue: Optional[str] = None) -> float:
    """Extract the callable GT denominator recorded in an evaluation table."""
    preferred_columns = []
    if tissue is not None:
        preferred_columns.append('Cell_Type_Total_GT_ORFs')
    preferred_columns.append('Total_GT_ORFs')
    for column in preferred_columns:
        if column not in source_df.columns:
            continue
        values = pd.to_numeric(source_df[column], errors='coerce').dropna()
        if not values.empty:
            return float(values.max())
    return np.nan


def _normalize_tissue_metric_name(metric: str) -> tuple:
    """Normalize a requested across-tissue classification metric."""
    normalized = str(metric).strip().lower().replace('-', '_')
    normalized = '_'.join(normalized.split())
    aliases = {
        'pr_auc': ('PR_AUC', 'PR-AUC'),
        'prauc': ('PR_AUC', 'PR-AUC'),
        'average_precision': ('PR_AUC', 'PR-AUC'),
        'roc_auc': ('ROC_AUC', 'ROC-AUC'),
        'rocauc': ('ROC_AUC', 'ROC-AUC'),
        'best_f1': ('Best_F1', 'Best F1'),
        'bestf1': ('Best_F1', 'Best F1'),
        'f1': ('Best_F1', 'Best F1'),
    }
    if normalized not in aliases:
        raise ValueError(
            "metric must be one of PR_AUC, ROC_AUC, or Best_F1."
        )
    return aliases[normalized]


def _collect_multi_tissue_pr_auc(
        manifest: list,
        trace_score_col: Optional[str] = None,
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        trace_model_name: str = 'TRACE',
        require_same_total_gt: bool = True,
        require_complete_grid: bool = True) -> pd.DataFrame:
    """Calculate candidate metrics for each tissue and model."""
    if not manifest:
        raise ValueError("manifest cannot be empty.")
    if trace_score_col is not None and trace_combined_score is not None:
        raise ValueError(
            "Use either trace_score_col or trace_combined_score, not both."
        )

    model_names = []
    tissues_by_model = {}
    tissue_order = []
    for config in manifest:
        if not isinstance(config, Mapping) or 'model' not in config:
            raise ValueError("Every manifest item must define model.")
        model_name = str(config['model'])
        model_names.append(model_name)
        model_tissues = _get_manifest_tissues(config)
        tissues_by_model[model_name] = set(model_tissues)
        tissue_order.extend(
            tissue for tissue in model_tissues if tissue not in tissue_order
        )
    duplicate_models = sorted({
        model for model in model_names if model_names.count(model) > 1
    })
    if duplicate_models:
        raise ValueError(
            f"Manifest model names must be unique: {duplicate_models}"
        )

    records = []
    incomplete_entries = []
    gt_counts_by_tissue = {}
    for tissue in tissue_order:
        gt_counts_by_tissue[tissue] = {}
        for config in manifest:
            model_name = str(config['model'])
            if tissue not in tissues_by_model[model_name]:
                incomplete_entries.append(
                    f"{tissue} / {model_name}: tissue is not declared"
                )
                continue
            csv_path = _resolve_tissue_model_path(
                tissue=tissue,
                config=config,
            )
            if not os.path.exists(csv_path):
                incomplete_entries.append(
                    f"{tissue} / {model_name}: file not found"
                )
                continue

            source_df = pd.read_csv(csv_path)
            cell_type_by_tissue = config.get('cell_type_by_tissue')
            if cell_type_by_tissue is not None:
                if not isinstance(cell_type_by_tissue, Mapping):
                    raise TypeError(
                        "cell_type_by_tissue must be a tissue-to-cell-type "
                        "mapping."
                    )
                normalized_cell_types = {
                    str(key): value
                    for key, value in cell_type_by_tissue.items()
                }
                table_tissue = normalized_cell_types.get(
                    tissue, config.get('cell_type')
                )
            else:
                table_tissue = config.get('cell_type')
            if table_tissue is not None:
                if 'Cell_Type' not in source_df.columns:
                    raise ValueError(
                        f"Model '{model_name}' in tissue '{tissue}' has no "
                        "Cell_Type column for the requested filter."
                    )
                source_df = source_df[
                    source_df['Cell_Type'].astype(str) == str(table_tissue)
                ].copy()

            is_trace = (
                model_name.casefold() == str(trace_model_name).casefold()
            )
            if is_trace and trace_score_col is not None:
                requested_score_col = trace_score_col
                requested_combination = None
            else:
                requested_score_col, requested_combination = (
                    resolve_manifest_score_request(
                        config,
                        trace_combined_score=trace_combined_score,
                        trace_model_name=trace_model_name,
                    )
                )

            candidate_df, selected_score_col, score_label = (
                prepare_evaluation_score(
                    source_df,
                    score_col=requested_score_col,
                    combined_score=requested_combination,
                )
            )
            if 'y_true' not in candidate_df.columns:
                raise ValueError(
                    f"Model '{model_name}' in tissue '{tissue}' has no "
                    "y_true column."
                )

            metric_df = candidate_df[['y_true', selected_score_col]].copy()
            candidate_count_before_filter = len(metric_df)
            metric_df['y_true'] = pd.to_numeric(
                metric_df['y_true'], errors='coerce'
            )
            metric_df[selected_score_col] = pd.to_numeric(
                metric_df[selected_score_col], errors='coerce'
            )
            metric_df = metric_df.replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            excluded_nonfinite_count = (
                candidate_count_before_filter - len(metric_df)
            )
            metric_df['y_true'] = metric_df['y_true'].astype(int)
            unexpected_labels = set(metric_df['y_true'].unique()) - {0, 1}
            if unexpected_labels:
                raise ValueError(
                    f"Model '{model_name}' in tissue '{tissue}' has "
                    f"non-binary y_true values: {sorted(unexpected_labels)}"
                )
            if metric_df['y_true'].nunique() != 2:
                incomplete_entries.append(
                    f"{tissue} / {model_name}: y_true lacks both classes"
                )
                continue

            y_true = metric_df['y_true'].to_numpy()
            scores = metric_df[selected_score_col].to_numpy()
            pr_auc_value = average_precision_score(y_true, scores)
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_true, scores
            )
            roc_auc_value = auc(false_positive_rate, true_positive_rate)
            precision_values, recall_values, _ = precision_recall_curve(
                y_true, scores
            )
            f1_values = np.divide(
                2 * precision_values * recall_values,
                precision_values + recall_values,
                out=np.zeros_like(precision_values, dtype=float),
                where=(precision_values + recall_values) > 0,
            )
            best_f1_value = (
                float(np.max(f1_values)) if len(f1_values) else 0.0
            )
            total_gt = _extract_total_gt_count(
                source_df,
                tissue=str(config.get('cell_type', tissue)),
            )
            if np.isfinite(total_gt):
                gt_counts_by_tissue[tissue][model_name] = int(total_gt)

            positive_count = int(y_true.sum())
            candidate_count = len(metric_df)
            records.append({
                'Tissue': tissue,
                'Model': model_name,
                'PR_AUC': float(pr_auc_value),
                'ROC_AUC': float(roc_auc_value),
                'Best_F1': best_f1_value,
                'Score_Type': score_label,
                'Score_Column': selected_score_col,
                'Candidate_Count': candidate_count,
                'Excluded_Nonfinite_Count': excluded_nonfinite_count,
                'Positive_Count': positive_count,
                'Negative_Count': int(candidate_count - positive_count),
                'Positive_Prevalence': positive_count / candidate_count,
                'Total_GT_ORFs': total_gt,
            })

    if incomplete_entries and require_complete_grid:
        details = '; '.join(incomplete_entries)
        raise ValueError(
            "The tissue-by-model benchmark grid is incomplete: " + details
        )
    for entry in incomplete_entries:
        warnings.warn(f"Skipping incomplete metric entry: {entry}")
    if not records:
        raise ValueError("No valid tissue-by-model metric values were found.")

    if require_same_total_gt:
        mismatches = []
        for tissue, model_counts in gt_counts_by_tissue.items():
            if len(set(model_counts.values())) > 1:
                details = ', '.join(
                    f"{model}={count}"
                    for model, count in model_counts.items()
                )
                mismatches.append(f"{tissue}: {details}")
        if mismatches:
            raise ValueError(
                "Models use different callable GT denominators within the "
                "same tissue, so their metric values are not directly "
                "comparable: " + '; '.join(mismatches)
            )
    return pd.DataFrame(records)


def plot_multi_model_pr_auc_across_tissues(
        manifest: list,
        out_dir: str = "./results/benchmark",
        trace_score_col: Optional[str] = None,
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        trace_model_name: str = 'TRACE',
        tissue_order: Optional[list] = None,
        require_same_total_gt: bool = True,
        require_complete_grid: bool = True,
        model_colors: Optional[Mapping[str, str]] = None,
        y_limits: tuple = (0, 1),
        title: Optional[str] = None,
        filename: Optional[str] = None,
        w: float = 6.0,
        h: float = 5.0,
        metric: str = 'PR_AUC'):
    """Plot a mean metric bar with one tissue point for every model.

    Every manifest item defines its own ``tissue_result_dirs`` mapping. Its
    relative ``path`` is resolved inside each tissue directory. TRACE can be
    reranked with one existing ``trace_score_col`` or a dynamic
    ``trace_combined_score`` definition. Other models use ``score_col`` or
    ``combined_score`` from their own manifest item. ``cell_type_by_tissue``
    optionally maps display tissue labels to table ``Cell_Type`` values.
    ``metric`` accepts PR_AUC, ROC_AUC, or Best_F1.
    """
    if len(y_limits) != 2 or not 0 <= y_limits[0] < y_limits[1] <= 1:
        raise ValueError("y_limits must satisfy 0 <= lower < upper <= 1.")
    if w <= 0 or h <= 0:
        raise ValueError("w and h must be greater than 0.")
    metric_column, metric_label = _normalize_tissue_metric_name(metric)

    plot_df = _collect_multi_tissue_pr_auc(
        manifest=manifest,
        trace_score_col=trace_score_col,
        trace_combined_score=trace_combined_score,
        trace_model_name=trace_model_name,
        require_same_total_gt=require_same_total_gt,
        require_complete_grid=require_complete_grid,
    )
    plot_df['Metric'] = metric_column
    plot_df['Metric_Value'] = plot_df[metric_column]
    actual_models = plot_df['Model'].drop_duplicates().tolist()
    model_order = [
        model for model in GLOBAL_MODEL_ORDER if model in actual_models
    ]
    model_order.extend(
        model for model in actual_models if model not in model_order
    )
    requested_tissue_order = (
        [str(tissue) for tissue in tissue_order]
        if tissue_order is not None
        else plot_df['Tissue'].drop_duplicates().tolist()
    )
    observed_tissues = set(plot_df['Tissue'])
    invalid_tissues = [
        tissue for tissue in requested_tissue_order
        if tissue not in observed_tissues
    ]
    if invalid_tissues:
        raise ValueError(
            f"tissue_order contains unavailable tissues: {invalid_tissues}"
        )
    ordered_tissues = [
        tissue for tissue in requested_tissue_order
        if tissue in observed_tissues
    ]
    ordered_tissues.extend(
        tissue for tissue in plot_df['Tissue'].drop_duplicates()
        if tissue not in ordered_tissues
    )

    plot_df['Model'] = pd.Categorical(
        plot_df['Model'], categories=model_order, ordered=True
    )
    plot_df['Tissue'] = pd.Categorical(
        plot_df['Tissue'], categories=ordered_tissues, ordered=True
    )
    summary_df = (
        plot_df.groupby('Model', observed=True)['Metric_Value']
        .agg(['mean', 'sem', 'count'])
        .reset_index()
        .rename(columns={'count': 'Tissue_Count'})
    )
    summary_df['sem'] = summary_df['sem'].fillna(0.0)
    summary_df['Metric'] = metric_column
    summary_df['ymin'] = (
        summary_df['mean'] - summary_df['sem']
    ).clip(lower=y_limits[0])
    summary_df['ymax'] = (
        summary_df['mean'] + summary_df['sem']
    ).clip(upper=y_limits[1])

    color_lookup = {
        model: (
            model_colors[model]
            if model_colors is not None and model in model_colors
            else GLOBAL_MODEL_COLORS.get(model, "#C0C0C0")
        )
        for model in model_order
    }
    marker_values = [
        'o', '^', 's', 'D', 'v', 'p', '*', 'h', 'X', 'P', '<', '>',
        '8', 'd', 'H',
    ]
    if len(ordered_tissues) > len(marker_values):
        raise ValueError(
            f"At most {len(marker_values)} tissues can be assigned unique "
            "point shapes."
        )
    tissue_shapes = {
        tissue: marker_values[index]
        for index, tissue in enumerate(ordered_tissues)
    }

    plot = (
        ggplot()
        + geom_col(
            data=summary_df,
            mapping=aes(x='Model', y='mean', fill='Model'),
            width=0.8,
            size=0.35,
        )
        + geom_errorbar(
            data=summary_df,
            mapping=aes(x='Model', ymin='ymin', ymax='ymax'),
            width=0.24,
            size=0.8,
            color='black',
        )
        + geom_jitter(
            data=plot_df,
            mapping=aes(x='Model', y='Metric_Value', shape='Tissue'),
            fill='#202020',
            color='#202020',
            size=3.5,
            width=0.18,
            alpha=0.85,
            stroke=0,
        )
        + scale_fill_manual(values=color_lookup)
        + scale_shape_manual(values=tissue_shapes)
        + scale_y_continuous(limits=y_limits)
        + guides(fill=None, shape=guide_legend(title="Tissue"))
        + theme_bw()
        + labs(x="", y=metric_label, title=title)
        + theme(
            figure_size=(w, h),
            axis_text_x=element_text(angle=45, hjust=1, color="black"),
            axis_title_x=element_blank(),
            panel_grid_major_x=element_blank(),
            legend_position='right',
            legend_title=element_text(fontweight='bold'),
        )
    )

    if filename is None:
        filename = f"Benchmark_Multi_Model_{metric_column}_by_Tissue.pdf"
    output_stem, output_extension = os.path.splitext(filename)
    if output_extension.lower() != '.pdf':
        filename = f"{output_stem or filename}.pdf"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    plot.save(save_path, width=w, height=h, verbose=False)
    return summary_df, plot_df, save_path


def _extract_incomplete_curve_endpoints(
        curve_df: pd.DataFrame,
        max_k: Optional[int]) -> pd.DataFrame:
    """Return the last displayed point for curves that end before max_k."""
    if max_k is None or curve_df.empty:
        return curve_df.iloc[0:0].copy()
    required_columns = {'Model', 'K'}
    missing_columns = required_columns.difference(curve_df.columns)
    if missing_columns:
        raise ValueError(
            f"Curve table is missing columns: {sorted(missing_columns)}"
        )
    endpoints = (
        curve_df.sort_values('K', kind='mergesort')
        .groupby('Model', group_keys=False, observed=False)
        .tail(1)
        .copy()
    )
    return endpoints[endpoints['K'] < max_k].copy()


def _expand_top_k_manifest(
        manifest: list,
        require_complete_grid: bool = True) -> list:
    """Expand legacy or multi-tissue model entries into table-level inputs."""
    if not manifest:
        raise ValueError("manifest cannot be empty.")
    expanded_inputs = []
    model_tissues = {}
    for config in manifest:
        if not isinstance(config, Mapping) or 'model' not in config:
            raise ValueError("Every manifest item must define model.")
        model_name = str(config['model'])
        if model_name in model_tissues:
            raise ValueError(
                f"Manifest model names must be unique: '{model_name}'."
            )
        if (
                config.get('tissue_result_dirs') is not None
                or config.get('path_by_tissue') is not None
        ):
            tissues = _get_manifest_tissues(config)
            table_inputs = [
                (tissue, _resolve_tissue_model_path(tissue, config))
                for tissue in tissues
            ]
        else:
            if config.get('path') is None:
                raise ValueError(
                    f"Model '{model_name}' must define path, "
                    "tissue_result_dirs, or path_by_tissue."
                )
            tissue = str(
                config.get('tissue', config.get('cell_type', 'Overall'))
            )
            tissues = [tissue]
            table_inputs = [(tissue, str(config['path']))]
        model_tissues[model_name] = set(tissues)
        expanded_inputs.extend({
            'config': config,
            'model': model_name,
            'tissue': tissue,
            'path': path,
        } for tissue, path in table_inputs)

    if require_complete_grid:
        tissue_sets = list(model_tissues.values())
        if tissue_sets and any(
                tissue_set != tissue_sets[0]
                for tissue_set in tissue_sets[1:]
        ):
            details = '; '.join(
                f"{model}={sorted(tissues)}"
                for model, tissues in model_tissues.items()
            )
            raise ValueError(
                "Models do not define the same tissue set: " + details
            )
    return expanded_inputs


def _filter_top_k_table_for_tissue(
        source_df: pd.DataFrame,
        config: Mapping[str, object],
        tissue: str) -> pd.DataFrame:
    """Apply an optional model-specific Cell_Type filter to one table."""
    cell_type_by_tissue = config.get('cell_type_by_tissue')
    if cell_type_by_tissue is not None:
        if not isinstance(cell_type_by_tissue, Mapping):
            raise TypeError(
                "cell_type_by_tissue must be a tissue-to-cell-type mapping."
            )
        normalized_cell_types = {
            str(key): value for key, value in cell_type_by_tissue.items()
        }
        table_cell_type = normalized_cell_types.get(
            tissue, config.get('cell_type')
        )
    else:
        table_cell_type = config.get('cell_type')
    if table_cell_type is None:
        return source_df
    if 'Cell_Type' not in source_df.columns:
        raise ValueError(
            f"Model '{config.get('model')}' tissue '{tissue}' has no "
            "Cell_Type column for the requested filter."
        )
    return source_df[
        source_df['Cell_Type'].astype(str) == str(table_cell_type)
    ].copy()


def _resolve_top_k_score_request(
        config: Mapping[str, object],
        trace_score_col: Optional[str],
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ],
        trace_model_name: str):
    """Resolve one model score while supporting a TRACE score override."""
    if trace_score_col is not None and trace_combined_score is not None:
        raise ValueError(
            "Use either trace_score_col or trace_combined_score, not both."
        )
    model_name = str(config.get('model', ''))
    if (
            trace_score_col is not None
            and model_name.casefold() == str(trace_model_name).casefold()
    ):
        return trace_score_col, None
    return resolve_manifest_score_request(
        config,
        trace_combined_score=trace_combined_score,
        trace_model_name=trace_model_name,
    )


def _summarize_top_k_tissue_curves(
        tissue_curve_df: pd.DataFrame,
        value_col: str,
        confidence_level: float = 0.95,
        common_k_only: bool = True) -> pd.DataFrame:
    """Summarize tissue curves using a mean and t-based confidence interval."""
    required_columns = {'Model', 'Tissue', 'K', value_col}
    missing_columns = required_columns.difference(tissue_curve_df.columns)
    if missing_columns:
        raise ValueError(
            f"Tissue curve table is missing columns: {sorted(missing_columns)}"
        )
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    grouped = (
        tissue_curve_df.groupby(['Model', 'K'], observed=False)[value_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    total_tissues = (
        tissue_curve_df.groupby('Model', observed=False)['Tissue']
        .nunique()
        .to_dict()
    )
    grouped['Total_Tissues'] = grouped['Model'].map(total_tissues).astype(int)
    grouped = grouped.rename(columns={'count': 'Tissue_Count'})
    if common_k_only:
        grouped = grouped[
            grouped['Tissue_Count'] == grouped['Total_Tissues']
        ].copy()
    if grouped.empty:
        raise ValueError(
            "No common K values remain across the requested tissue curves."
        )

    grouped['SEM'] = (
        grouped['std'].fillna(0.0)
        / np.sqrt(grouped['Tissue_Count'].clip(lower=1))
    )
    has_ci = grouped['Tissue_Count'] > 1
    critical_values = np.ones(len(grouped), dtype=float)
    critical_values[has_ci.to_numpy()] = student_t.ppf(
        (1 + confidence_level) / 2,
        grouped.loc[has_ci, 'Tissue_Count'].to_numpy() - 1,
    )
    margin = critical_values * grouped['SEM'].to_numpy()
    grouped[value_col] = grouped['mean']
    grouped['CI_Lower'] = np.clip(grouped['mean'] - margin, 0, 1)
    grouped['CI_Upper'] = np.clip(grouped['mean'] + margin, 0, 1)
    grouped['Has_CI'] = has_ci
    grouped['Confidence_Level'] = confidence_level

    score_labels = (
        tissue_curve_df.groupby('Model', observed=False)['Score_Type']
        .agg(lambda values: ' | '.join(dict.fromkeys(map(str, values))))
        .to_dict()
        if 'Score_Type' in tissue_curve_df.columns
        else {}
    )
    grouped['Score_Type'] = grouped['Model'].map(score_labels)
    return grouped.drop(columns=['mean', 'std'])


def _downsample_top_k_curves(
        curve_df: pd.DataFrame,
        max_points_per_model: int = 3000) -> pd.DataFrame:
    """Downsample displayed curve points without changing metric calculation."""
    sampled_frames = []
    for _, model_df in curve_df.groupby(
            'Model', sort=False, observed=False
    ):
        if len(model_df) > max_points_per_model:
            indices = np.unique(np.linspace(
                0, len(model_df) - 1, max_points_per_model
            ).astype(int))
            model_df = model_df.iloc[indices]
        sampled_frames.append(model_df)
    return pd.concat(sampled_frames, ignore_index=True)


def plot_multi_model_top_k_precision(
        manifest: list, 
        out_dir: str = "./results/benchmark", 
        min_k: Optional[int] = None, 
        max_k: Optional[int] = None, 
        suffix: str = "",
        y_limits: tuple = (0, 1),
        trace_score_col: Optional[str] = None,
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        trace_model_name: str = 'TRACE',
        confidence_level: float = 0.95,
        common_k_only: bool = True,
        require_complete_grid: bool = True,
        smoothing_window: int = 20,
        mark_incomplete_endpoints: bool = True,
        endpoint_size: float = 3.2,
):
    """Plot tissue-mean Precision@K with optional confidence intervals."""
    if min_k is not None and min_k < 1:
        raise ValueError("min_k must be at least 1 for a logarithmic K axis.")
    if max_k is not None and max_k < 1:
        raise ValueError("max_k must be at least 1 for a logarithmic K axis.")
    if min_k is not None and max_k is not None and min_k > max_k:
        raise ValueError("min_k cannot be greater than max_k.")
    if len(y_limits) != 2 or not 0 <= y_limits[0] < y_limits[1] <= 1:
        raise ValueError("y_limits must satisfy 0 <= lower < upper <= 1.")
    if endpoint_size <= 0:
        raise ValueError("endpoint_size must be greater than 0.")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least 1.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    os.makedirs(out_dir, exist_ok=True)
    all_tissue_curves = []
    missing_inputs = []
    expanded_inputs = _expand_top_k_manifest(
        manifest,
        require_complete_grid=require_complete_grid,
    )

    for table_input in expanded_inputs:
        config = table_input['config']
        model_name = table_input['model']
        tissue = table_input['tissue']
        csv_path = table_input['path']
        score_col, combined_score = _resolve_top_k_score_request(
            config=config,
            trace_score_col=trace_score_col,
            trace_combined_score=trace_combined_score,
            trace_model_name=trace_model_name,
        )
        if not os.path.exists(csv_path):
            missing_inputs.append(f"{tissue} / {model_name}: {csv_path}")
            continue

        source_df = pd.read_csv(csv_path)
        source_df = _filter_top_k_table_for_tissue(
            source_df, config, tissue
        )
        if source_df.empty:
            missing_inputs.append(
                f"{tissue} / {model_name}: no rows after Cell_Type filtering"
            )
            continue

        if 'Precision' in source_df.columns and 'K' in source_df.columns:
            if combined_score is not None or (
                    trace_score_col is not None
                    and model_name.casefold()
                    == str(trace_model_name).casefold()
            ):
                raise ValueError(
                    f"Model '{model_name}' tissue '{tissue}' uses a "
                    "precomputed Precision@K "
                    "table, which cannot be reranked by a new score or "
                    "combination. "
                    "Use unified_evaluation_table.csv or regenerate the "
                    "Top-K table with top_k_combined_score."
                )
            precision_df = source_df[['K', 'Precision']].copy()
            score_label = (
                str(source_df['Score_Type'].dropna().iloc[0])
                if 'Score_Type' in source_df.columns
                and not source_df['Score_Type'].dropna().empty
                else 'precomputed'
            )
        elif 'y_true' in source_df.columns:
            had_record_type = 'Record_Type' in source_df.columns
            ranked_df, score_col, score_label = prepare_evaluation_score(
                source_df,
                score_col=score_col,
                combined_score=combined_score,
            )
            ranked_df[score_col] = pd.to_numeric(
                ranked_df[score_col], errors='coerce'
            )
            ranked_df['y_true'] = pd.to_numeric(
                ranked_df['y_true'], errors='coerce'
            )
            ranked_df = ranked_df.replace(
                [np.inf, -np.inf], np.nan
            ).dropna(subset=[score_col, 'y_true'])
            if not had_record_type:
                ranked_df = ranked_df[ranked_df[score_col] >= 0].copy()
            unexpected_labels = set(ranked_df['y_true'].unique()) - {0, 1}
            if unexpected_labels:
                raise ValueError(
                    f"Model '{model_name}' tissue '{tissue}' has non-binary "
                    f"y_true values: {sorted(unexpected_labels)}"
                )
            ranked_df = ranked_df.sort_values(
                by=score_col,
                ascending=False,
                kind='mergesort',
            ).reset_index(drop=True)
            if ranked_df.empty:
                missing_inputs.append(
                    f"{tissue} / {model_name}: no valid ranked predictions"
                )
                continue
            k_array = np.arange(1, len(ranked_df) + 1)
            tp_cumsum = ranked_df['y_true'].cumsum()
            precision_df = pd.DataFrame({
                'K': k_array,
                'Precision': tp_cumsum / k_array,
            })
        else:
            missing_inputs.append(
                f"{tissue} / {model_name}: unsupported Top-K table"
            )
            continue

        precision_df['K'] = pd.to_numeric(
            precision_df['K'], errors='coerce'
        )
        precision_df['Precision'] = pd.to_numeric(
            precision_df['Precision'], errors='coerce'
        )
        precision_df = precision_df.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        precision_df = precision_df[
            precision_df['K'] > 0
        ].sort_values('K').drop_duplicates('K', keep='last')
        if precision_df.empty:
            missing_inputs.append(
                f"{tissue} / {model_name}: empty Precision@K curve"
            )
            continue
        if not precision_df['Precision'].between(0, 1).all():
            raise ValueError(
                f"Model '{model_name}' tissue '{tissue}' has Precision "
                "values outside [0, 1]."
            )
        precision_df['Precision_Smooth'] = (
            precision_df['Precision']
            .rolling(window=smoothing_window, min_periods=1)
            .mean()
        )
        precision_df['Model'] = model_name
        precision_df['Tissue'] = tissue
        precision_df['Score_Type'] = score_label
        all_tissue_curves.append(precision_df)

    if missing_inputs and require_complete_grid:
        raise ValueError(
            "The tissue-by-model Precision@K grid is incomplete: "
            + '; '.join(missing_inputs)
        )
    for entry in missing_inputs:
        warnings.warn(f"Skipping incomplete Precision@K input: {entry}")
    if not all_tissue_curves:
        raise ValueError("No valid Top-K precision data processed.")

    tissue_curve_df = pd.concat(all_tissue_curves, ignore_index=True)
    if min_k is not None:
        tissue_curve_df = tissue_curve_df[tissue_curve_df['K'] >= min_k]
    if max_k is not None:
        tissue_curve_df = tissue_curve_df[tissue_curve_df['K'] <= max_k]
    if tissue_curve_df.empty:
        raise ValueError(
            "No Precision@K observations remain in the requested K range."
        )
    plot_df = _summarize_top_k_tissue_curves(
        tissue_curve_df=tissue_curve_df,
        value_col='Precision_Smooth',
        confidence_level=confidence_level,
        common_k_only=common_k_only,
    )
    endpoint_df = (
        _extract_incomplete_curve_endpoints(plot_df, max_k)
        if mark_incomplete_endpoints
        else plot_df.iloc[0:0].copy()
    )

    plot_df = _downsample_top_k_curves(plot_df)

    # =================================================================
    # [MODIFIED] 动态过滤类别顺序
    # =================================================================
    actual_models = plot_df['Model'].unique().tolist()
    valid_order = [m for m in GLOBAL_MODEL_ORDER if m in actual_models]
    for m in actual_models:
        if m not in valid_order: valid_order.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=valid_order, ordered=True)
    endpoint_df['Model'] = pd.Categorical(
        endpoint_df['Model'], categories=valid_order, ordered=True
    )

    color_mapping = {m: GLOBAL_MODEL_COLORS.get(m, "#C0C0C0") for m in valid_order}
    linetype_mapping = {m: "solid" if "TRACE" in m else "dashed" for m in valid_order}
    ci_df = plot_df[plot_df['Has_CI']].copy()

    if min_k is not None and max_k is not None:
        title_suffix = f"(K: {min_k} to {max_k})"
        file_suffix = f"{suffix}_{min_k}_to_{max_k}"
    elif min_k is not None:
        title_suffix = f"(K >= {min_k})"
        file_suffix = f"{suffix}_{min_k}_to_All"
    elif max_k is not None:
        title_suffix = f"(Top {max_k})"
        file_suffix = f"{suffix}_1_to_{max_k}"
    else:
        title_suffix = "(All Predictions)"
        file_suffix = f"{suffix}_All"

    p = (
        ggplot(plot_df, aes(x='K', y='Precision_Smooth', color='Model'))
    )
    if not ci_df.empty:
        p += geom_ribbon(
            data=ci_df,
            mapping=aes(
                x='K', ymin='CI_Lower', ymax='CI_Upper',
                fill='Model', group='Model',
            ),
            alpha=0.18,
            color=None,
            inherit_aes=False,
        )
    p += (
        geom_line(aes(linetype='Model'), size=1.5, alpha=0.85)
        + scale_color_manual(values=color_mapping)
        + scale_fill_manual(values=color_mapping, guide=None)
        + scale_linetype_manual(values=linetype_mapping, guide=None)
        + scale_y_continuous(limits=y_limits)
        + scale_x_log10() 
        + theme_classic()
        + labs(
            title=f"Precision@K Benchmark {title_suffix}",
            x="Top K Predicted ORFs (Log Scale, Ranked by Conf. Score)",
            y="Precision (Proportion of True Positives)"
        )
        + theme(
            figure_size=(7, 5),
            axis_title=element_text(size=12, face="bold"),
            axis_text=element_text(size=10),
            legend_position="right",
            legend_text=element_text(size=10),
            legend_title=element_blank()
        )
    )
    if not endpoint_df.empty:
        p += geom_point(
            data=endpoint_df,
            mapping=aes(x='K', y='Precision_Smooth', color='Model'),
            shape='o',
            size=endpoint_size,
            fill='white',
            stroke=1.2,
            alpha=1.0,
            show_legend=False,
        )
    save_path = os.path.join(
        out_dir,
        f"Benchmark_TopK_Precision_Curve_{file_suffix}.pdf",
    )
    p.save(save_path, dpi=300, verbose=False)
    return plot_df, save_path


def _extract_top_k_recall_curve(
        df: pd.DataFrame,
        score_col: str,
        total_gt_override: Optional[int] = None):
    """Extract or calculate global unique-GT Recall@K from one model table."""
    if 'K' in df.columns and ('Recall' in df.columns or 'Recall_at_K' in df.columns):
        recall_col = 'Recall' if 'Recall' in df.columns else 'Recall_at_K'
        curve_df = df[['K', recall_col]].rename(
            columns={recall_col: 'Recall'}
        ).copy()
        curve_df['K'] = pd.to_numeric(curve_df['K'], errors='coerce')
        curve_df['Recall'] = pd.to_numeric(
            curve_df['Recall'], errors='coerce'
        )
        curve_df = curve_df.replace([np.inf, -np.inf], np.nan).dropna()
        curve_df = curve_df[curve_df['K'] > 0].sort_values('K')
        curve_df = curve_df.drop_duplicates('K', keep='last')
        if not curve_df['Recall'].between(0, 1).all():
            raise ValueError("Precomputed Recall values must be between 0 and 1.")
        known_total_gt = total_gt_override
        if known_total_gt is None and 'Total_GT_ORFs' in df.columns:
            total_values = pd.to_numeric(
                df['Total_GT_ORFs'], errors='coerce'
            ).dropna()
            if not total_values.empty:
                known_total_gt = int(total_values.max())
        return curve_df, known_total_gt

    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' was not found.")
    if 'Matched_GT_Index' not in df.columns:
        raise ValueError(
            "Recall@K requires Matched_GT_Index, or a precomputed table with "
            "K and Recall columns."
        )

    total_gt = total_gt_override
    if total_gt is None and 'Total_GT_ORFs' in df.columns:
        total_values = pd.to_numeric(
            df['Total_GT_ORFs'], errors='coerce'
        ).dropna()
        if not total_values.empty:
            total_gt = int(total_values.max())
    if total_gt is None:
        total_gt = int(df['Matched_GT_Index'].dropna().nunique())
    if total_gt <= 0:
        raise ValueError("No callable GT ORFs were found for Recall@K.")
    observed_gt_count = int(df['Matched_GT_Index'].dropna().nunique())
    if observed_gt_count > total_gt:
        raise ValueError(
            f"Observed {observed_gt_count} GT identifiers, exceeding the "
            f"declared total_gt={total_gt}."
        )

    candidate_df = df.copy()
    if 'Record_Type' in candidate_df.columns:
        candidate_df = candidate_df[
            candidate_df['Record_Type'] == 'Prediction'
        ].copy()
    candidate_df[score_col] = pd.to_numeric(
        candidate_df[score_col], errors='coerce'
    )
    candidate_df = candidate_df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=[score_col])
    if 'Record_Type' not in df.columns:
        candidate_df = candidate_df[candidate_df[score_col] >= 0].copy()
    candidate_df = candidate_df.sort_values(
        score_col, ascending=False, kind='mergesort'
    ).reset_index(drop=True)
    if candidate_df.empty:
        return pd.DataFrame(columns=['K', 'Recall']), total_gt

    matched_gt = candidate_df['Matched_GT_Index']
    new_gt_hit = matched_gt.notna() & ~matched_gt.duplicated(keep='first')
    k_array = np.arange(1, len(candidate_df) + 1)
    curve_df = pd.DataFrame({
        'K': k_array,
        'Recall': new_gt_hit.astype(int).cumsum().to_numpy() / total_gt,
    })
    return curve_df, total_gt


def _resolve_total_gt_override(
        config: Mapping[str, object],
        tissue: str) -> Optional[int]:
    """Resolve an optional tissue-specific callable GT denominator."""
    total_gt_by_tissue = config.get('total_gt_by_tissue')
    if total_gt_by_tissue is not None:
        if not isinstance(total_gt_by_tissue, Mapping):
            raise TypeError(
                "total_gt_by_tissue must be a tissue-to-count mapping."
            )
        normalized_gt_counts = {
            str(key): value for key, value in total_gt_by_tissue.items()
        }
        total_gt = normalized_gt_counts.get(
            tissue, config.get('total_gt')
        )
    else:
        total_gt = config.get('total_gt')
    return None if total_gt is None else int(total_gt)


def plot_multi_model_top_k_recall(
        manifest: list,
        out_dir: str = "./results/benchmark",
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
        suffix: str = "",
        require_same_total_gt: bool = True,
        y_limits: tuple = (0, 1),
        trace_score_col: Optional[str] = None,
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        trace_model_name: str = 'TRACE',
        confidence_level: float = 0.95,
        common_k_only: bool = True,
        require_complete_grid: bool = True,
        mark_incomplete_endpoints: bool = True,
        endpoint_size: float = 3.2):
    """Plot tissue-mean Recall@K with optional confidence intervals."""
    if min_k is not None and min_k < 1:
        raise ValueError("min_k must be at least 1 for a logarithmic K axis.")
    if max_k is not None and max_k < 1:
        raise ValueError("max_k must be at least 1 for a logarithmic K axis.")
    if min_k is not None and max_k is not None and min_k > max_k:
        raise ValueError("min_k cannot be greater than max_k.")
    if len(y_limits) != 2 or not 0 <= y_limits[0] < y_limits[1] <= 1:
        raise ValueError("y_limits must satisfy 0 <= lower < upper <= 1.")
    if endpoint_size <= 0:
        raise ValueError("endpoint_size must be greater than 0.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    os.makedirs(out_dir, exist_ok=True)
    all_tissue_curves = []
    gt_counts_by_tissue = {}
    missing_inputs = []
    expanded_inputs = _expand_top_k_manifest(
        manifest,
        require_complete_grid=require_complete_grid,
    )

    for table_input in expanded_inputs:
        config = table_input['config']
        model_name = table_input['model']
        tissue = table_input['tissue']
        csv_path = table_input['path']
        score_col, combined_score = _resolve_top_k_score_request(
            config=config,
            trace_score_col=trace_score_col,
            trace_combined_score=trace_combined_score,
            trace_model_name=trace_model_name,
        )
        total_gt_override = _resolve_total_gt_override(config, tissue)
        if not os.path.exists(csv_path):
            missing_inputs.append(f"{tissue} / {model_name}: {csv_path}")
            continue

        source_df = pd.read_csv(csv_path)
        source_df = _filter_top_k_table_for_tissue(
            source_df, config, tissue
        )
        if source_df.empty:
            missing_inputs.append(
                f"{tissue} / {model_name}: no rows after Cell_Type filtering"
            )
            continue
        is_precomputed = (
            'K' in source_df.columns
            and (
                'Recall' in source_df.columns
                or 'Recall_at_K' in source_df.columns
            )
        )
        if is_precomputed and (
                combined_score is not None
                or (
                    trace_score_col is not None
                    and model_name.casefold()
                    == str(trace_model_name).casefold()
                )
        ):
            raise ValueError(
                f"Model '{model_name}' tissue '{tissue}' uses a precomputed "
                "Recall@K table, "
                "which cannot be reranked by a new score or combination. Use "
                "unified_evaluation_table.csv or regenerate the Top-K table "
                "with top_k_combined_score."
            )
        if is_precomputed:
            score_label = (
                str(source_df['Score_Type'].dropna().iloc[0])
                if 'Score_Type' in source_df.columns
                and not source_df['Score_Type'].dropna().empty
                else 'precomputed'
            )
        else:
            source_df, score_col, score_label = prepare_evaluation_score(
                source_df,
                score_col=score_col,
                combined_score=combined_score,
            )
        recall_df, total_gt = _extract_top_k_recall_curve(
            source_df,
            score_col=score_col,
            total_gt_override=total_gt_override,
        )
        if recall_df.empty:
            missing_inputs.append(
                f"{tissue} / {model_name}: empty Recall@K curve"
            )
            continue
        recall_df['Model'] = model_name
        recall_df['Tissue'] = tissue
        recall_df['Score_Type'] = score_label
        all_tissue_curves.append(recall_df)
        if total_gt is not None:
            gt_counts_by_tissue.setdefault(tissue, {})[model_name] = int(
                total_gt
            )

    if missing_inputs and require_complete_grid:
        raise ValueError(
            "The tissue-by-model Recall@K grid is incomplete: "
            + '; '.join(missing_inputs)
        )
    for entry in missing_inputs:
        warnings.warn(f"Skipping incomplete Recall@K input: {entry}")
    if not all_tissue_curves:
        raise ValueError("No valid Top-K recall data processed.")
    if require_same_total_gt:
        mismatches = []
        for tissue, model_counts in gt_counts_by_tissue.items():
            if len(set(model_counts.values())) > 1:
                details = ', '.join(
                    f"{model}={count}"
                    for model, count in model_counts.items()
                )
                mismatches.append(f"{tissue}: {details}")
        if mismatches:
            raise ValueError(
                "Models use different callable GT denominators within the "
                "same tissue, so Recall@K is not directly comparable: "
                + '; '.join(mismatches)
            )

    tissue_curve_df = pd.concat(all_tissue_curves, ignore_index=True)
    if min_k is not None:
        tissue_curve_df = tissue_curve_df[tissue_curve_df['K'] >= min_k]
    if max_k is not None:
        tissue_curve_df = tissue_curve_df[tissue_curve_df['K'] <= max_k]
    if tissue_curve_df.empty:
        raise ValueError("No Recall@K observations remain in the requested K range.")
    plot_df = _summarize_top_k_tissue_curves(
        tissue_curve_df=tissue_curve_df,
        value_col='Recall',
        confidence_level=confidence_level,
        common_k_only=common_k_only,
    )
    endpoint_df = (
        _extract_incomplete_curve_endpoints(plot_df, max_k)
        if mark_incomplete_endpoints
        else plot_df.iloc[0:0].copy()
    )

    plot_df = _downsample_top_k_curves(plot_df)
    actual_models = plot_df['Model'].drop_duplicates().tolist()
    valid_order = [
        model for model in GLOBAL_MODEL_ORDER if model in actual_models
    ]
    valid_order.extend(
        model for model in actual_models if model not in valid_order
    )
    plot_df['Model'] = pd.Categorical(
        plot_df['Model'], categories=valid_order, ordered=True
    )
    endpoint_df['Model'] = pd.Categorical(
        endpoint_df['Model'], categories=valid_order, ordered=True
    )
    color_mapping = {
        model: GLOBAL_MODEL_COLORS.get(model, "#C0C0C0")
        for model in valid_order
    }
    linetype_mapping = {
        model: "solid" if "TRACE" in model else "dashed"
        for model in valid_order
    }
    ci_df = plot_df[plot_df['Has_CI']].copy()

    if min_k is not None and max_k is not None:
        title_suffix = f"(K: {min_k} to {max_k})"
        file_suffix = f"{suffix}_{min_k}_to_{max_k}"
    elif min_k is not None:
        title_suffix = f"(K >= {min_k})"
        file_suffix = f"{suffix}_{min_k}_to_All"
    elif max_k is not None:
        title_suffix = f"(Top {max_k})"
        file_suffix = f"{suffix}_1_to_{max_k}"
    else:
        title_suffix = "(All Predictions)"
        file_suffix = f"{suffix}_All"

    plot = (
        ggplot(plot_df, aes(x='K', y='Recall', color='Model'))
    )
    if not ci_df.empty:
        plot += geom_ribbon(
            data=ci_df,
            mapping=aes(
                x='K', ymin='CI_Lower', ymax='CI_Upper',
                fill='Model', group='Model',
            ),
            alpha=0.18,
            color=None,
            inherit_aes=False,
        )
    plot += (
        geom_line(aes(linetype='Model'), size=1.5, alpha=0.85)
        + scale_color_manual(values=color_mapping)
        + scale_fill_manual(values=color_mapping, guide=None)
        + scale_linetype_manual(values=linetype_mapping, guide=None)
        + scale_y_continuous(limits=y_limits)
        + scale_x_log10()
        + theme_classic()
        + labs(
            title=f"Recall@K Benchmark {title_suffix}",
            x="Top K Predicted ORFs (Log Scale, Ranked by Score)",
            y="Recall (Fraction of Unique GT ORFs Recovered)",
        )
        + theme(
            figure_size=(7, 5),
            axis_title=element_text(size=12, face="bold"),
            axis_text=element_text(size=10),
            legend_position="right",
            legend_text=element_text(size=10),
            legend_title=element_blank(),
        )
    )
    if not endpoint_df.empty:
        plot += geom_point(
            data=endpoint_df,
            mapping=aes(x='K', y='Recall', color='Model'),
            shape='o',
            size=endpoint_size,
            fill='white',
            stroke=1.2,
            alpha=1.0,
            show_legend=False,
        )
    save_path = os.path.join(
        out_dir,
        f"Benchmark_TopK_Recall_Curve_{file_suffix}.pdf",
    )
    plot.save(save_path, dpi=300, verbose=False)
    return plot_df, save_path


def plot_top_k_precision_bar(
        manifest: list, target_k: int,
        out_dir: str = "./results/benchmark", 
        suffix: str = ""
):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    
    for cfg in manifest:
        model_name, csv_path = cfg['model'], cfg['path']
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        prec_val = np.nan
        
        if 'Precision' in df.columns and 'K' in df.columns:
            if target_k in df['K'].values: prec_val = df.loc[df['K'] == target_k, 'Precision'].values[0]
            else:
                max_k_avail = df['K'].max()
                if 'TP_Count' in df.columns: prec_val = df.loc[df['K'] == max_k_avail, 'TP_Count'].values[0] / target_k
                else: prec_val = (df['Precision'].iloc[-1] * max_k_avail) / target_k
        elif 'y_true' in df.columns and cfg.get('score_col', 'score') in df.columns:
            df_sorted = df.sort_values(by=cfg.get('score_col', 'score'), ascending=False)
            df_sorted = df_sorted[df_sorted[cfg.get('score_col', 'score')] >= 0].copy()
            if df_sorted.empty: prec_val = 0.0
            else: prec_val = df_sorted['y_true'].iloc[:target_k].sum() / target_k if len(df_sorted) >= target_k else df_sorted['y_true'].sum() / target_k
        else: continue
            
        records.append({'Model': model_name, 'Dataset': cfg.get('dataset', 'Unknown'), 'Cell_type': cfg.get('cell_type', 'Unknown'), 'Precision': prec_val})
        
    if not records: raise ValueError("No valid Top-K data processed.")
    plot_df = pd.DataFrame(records)

    summary_df = plot_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Precision', 'mean'),
        SEM=('Precision', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']

    # =================================================================
    # [MODIFIED] 动态过滤类别顺序
    # =================================================================
    actual_models = plot_df['Model'].unique().tolist()
    valid_order = [m for m in GLOBAL_MODEL_ORDER if m in actual_models]
    for m in actual_models:
        if m not in valid_order: valid_order.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=valid_order, ordered=True)
    summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=valid_order, ordered=True)
    model_colors = {m: GLOBAL_MODEL_COLORS.get(m, "#C0C0C0") for m in valid_order}
            
    unique_cells = plot_df['Cell_type'].unique().tolist()
    unseen_cells = [c for c in unique_cells if 'unseen' in str(c).lower()]
    ordered_cells = [c for c in unique_cells if c not in unseen_cells] + unseen_cells
    plot_df['Cell_type'] = pd.Categorical(plot_df['Cell_type'], categories=ordered_cells, ordered=True)
    cell_colors = {ct: "#D6715E" if ct in unseen_cells else "#202020" for ct in ordered_cells}

    unique_datasets = plot_df['Dataset'].unique().tolist()
    plot_df['Dataset'] = pd.Categorical(plot_df['Dataset'], categories=unique_datasets, ordered=True)
    dataset_shapes = {ds: ['o', '^', 's', 'D', 'v', 'p', 'h', '8'][i % 8] for i, ds in enumerate(unique_datasets)}

    p = (
        ggplot()
        + geom_col(data=summary_df, mapping=aes(x='Model', y='Overall_Mean', fill='Model'), width=0.7)
        + geom_errorbar(data=summary_df, mapping=aes(x='Model', ymin='ymin', ymax='ymax'), width=0.2, size=0.8)
        + geom_jitter(data=plot_df, mapping=aes(x='Model', y='Precision', shape='Dataset', color='Cell_type'), width=0.15, size=3.0)
        + scale_fill_manual(values=model_colors, guide=None) 
        + scale_shape_manual(values=dataset_shapes, name="Dataset") 
        + scale_color_manual(values=cell_colors, name="Cell type")
        + theme_bw() 
        + labs(x="", y=f"Precision @ K={target_k}")
        + theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="right")
    )
    p.save(os.path.join(out_dir, f"precision_at_{target_k}_bar{suffix}.pdf"), dpi=300, verbose=False)
    return summary_df, plot_df


# ==============================================================================
# 核心绘图函数
# ==============================================================================
def plot_multicell_performance(
        agg_df: pd.DataFrame, 
        metric_name: str, 
        target_features: dict,               
        default_feature: str = "Final Score", 
        cell_types: list = None, 
        out_dir: str = "./results/benchmark_plots",
        w: float = 5.5, # [MODIFIED] Increased width slightly to accommodate the legend on the right
        h: float = 5
):
    """
    Plot Bar (Mean) + Errorbar (SEM) + Jitter Points for multi-model comparison.
    - Custom target_features per model.
    - Differentiates cell types using point shapes.
    """
    if agg_df.empty:
        print(f"No data to plot for {metric_name}.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    raw_df = agg_df.copy()

    # 1. Data Cleaning
    if 'Cell_type' in raw_df.columns:
        raw_df.rename(columns={'Cell_type': 'Cell_Type'}, inplace=True)
        
    raw_df['Cell_Type'] = raw_df['Cell_Type'].replace({'SW480': 'SW480 (Unseen)'})
    
    # =========================================================
    # Custom Feature Extraction
    # =========================================================
    filtered_dfs = []
    
    for model in raw_df['Model'].unique():
        model_df = raw_df[raw_df['Model'] == model].copy()
        available_features = model_df['Feature'].unique().tolist()
        
        expected_feature = target_features.get(model, default_feature)
            
        # Independent Fallback mechanism
        if expected_feature not in available_features:
            if "Final Score" in available_features:
                actual_feature = "Final Score"
                print(f"  [Fallback] Model '{model}' missing '{expected_feature}', defaulting to 'Final Score'.")
            elif len(available_features) > 0:
                actual_feature = available_features[0]
                print(f"  [Fallback] Model '{model}' missing '{expected_feature}', defaulting to '{actual_feature}'.")
            else:
                continue 
        else:
            actual_feature = expected_feature
            
        filtered_dfs.append(model_df[model_df['Feature'] == actual_feature])
        
    if not filtered_dfs:
        print(f"  [Warning] No valid features found for any models. Skipping {metric_name} plot.")
        return
        
    raw_df = pd.concat(filtered_dfs, ignore_index=True)
    raw_df = raw_df[raw_df['Cell_Type'] != 'Overall']

    # 3. Filter Cell Types
    if cell_types:
        cell_types = ['SW480 (Unseen)' if ct == 'SW480' else ct for ct in cell_types]
        raw_df = raw_df[raw_df["Cell_Type"].isin(cell_types)]

    # 4. Categorical Sorting
    available_models = [m for m in GLOBAL_MODEL_ORDER if m in raw_df['Model'].unique()]
    raw_df['Model'] = pd.Categorical(raw_df['Model'], categories=available_models, ordered=True)
    
    # 5. Calculate Mean and SEM
    summary_df = raw_df.groupby('Model', observed=True)[metric_name].agg(['mean', 'sem']).reset_index()
    summary_df['sem'] = summary_df['sem'].fillna(0)

    # =========================================================
    # 6. Plotnine Layer Assembly
    # =========================================================
    
    # [NEW] Define an explicit shape palette to ensure distinct points for different cell types
    shape_palette = ['o', '^', 's', 'D', 'v', 'p', '*', 'h']
    
    p = (
        ggplot() 
        + geom_col(
            summary_df, 
            aes(x='Model', y='mean', fill='Model'), 
            width=0.8
        )
        + geom_errorbar(
            summary_df, 
            aes(x='Model', ymin='mean - sem', ymax='mean + sem'), 
            width=0.25, size=0.8, color='black'
        )
        # [MODIFIED] Added shape='Cell_Type' to aesthetics
        + geom_jitter(
            raw_df, 
            aes(x='Model', y=metric_name, shape='Cell_Type'), 
            fill='#202020', color='#202020', 
            size=3.5, width=0.2, alpha=0.8, stroke=0
        )
        + scale_fill_manual(values=GLOBAL_MODEL_COLORS)
        # [NEW] Apply manual shapes for clarity
        + scale_shape_manual(values=shape_palette)
        # [NEW] Hide the fill legend (Model names) but keep the shape legend (Cell types)
        + guides(fill=None, shape=guide_legend(title="Cell type"))
        + theme_bw()
        + labs(
            x="", 
            y=metric_name
        )
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color="black"), 
            axis_title_x=element_blank(),
            panel_grid_major_x=element_blank(),
            legend_position='right',
            legend_title=element_text(fontweight='bold')
        )
    )
    
    # 7. Save the figure
    main_feat = target_features.get("TRACE", "Custom")
    safe_feature_name = main_feat.replace(" ", "_").replace("*", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(out_dir, f"Benchmark_{metric_name}_{safe_feature_name}.pdf")
    
    p.save(save_path, width=w, height=h, verbose=False)
    print(f"✅ Saved plot: {save_path}")


# ==============================================================================
# 流水线引擎
# ==============================================================================
def run_benchmark_pipeline(
    model_csv_dict: dict, 
    target_features: dict = None,                # [MODIFIED] 变更为字典
    default_feature: str = "Final Score",        # [NEW] 默认特征
    out_dir: str = "./results/benchmark_plots"
):
    """
    接收多个模型的 CSV 文件路径字典，合并后自动输出评估图。
    通过 target_features 为特定模型指定特征，未指定的自动回退到 default_feature。
    """
    # 默认兜底：如果没有传入字典，自动初始化给 TRACE 设置 Expression Score
    if target_features is None:
        target_features = {"TRACE": "Expression Score (TPM*Signal)"}
        
    print("--- Assembling Multi-Model Benchmark Dataset ---")
    df_list = []
    
    for model_name, csv_path in model_csv_dict.items():
        if not os.path.exists(csv_path):
            print(f"  [Warning] File not found for {model_name}: {csv_path}")
            continue
            
        temp_df = pd.read_csv(csv_path)
        temp_df['Model'] = model_name
        df_list.append(temp_df)
        print(f"  -> Loaded {model_name}: {len(temp_df)} records.")
        
    if not df_list:
        print("No valid data loaded. Aborting.")
        return
        
    agg_df = pd.concat(df_list, ignore_index=True)
    
    os.makedirs(out_dir, exist_ok=True)
    agg_df.to_csv(os.path.join(out_dir, "master_benchmark_table.csv"), index=False)
    
    metrics_to_plot = ['ROC-AUC', 'PR-AUC', 'Best_F1']
    print(f"\n--- Generating Benchmark Plots ---")
    
    for metric in metrics_to_plot:
        plot_multicell_performance(
            agg_df=agg_df,
            metric_name=metric,
            target_features=target_features, # 传入字典
            default_feature=default_feature, # 传入默认值
            out_dir=out_dir
        )
        
    print("\n🎉 Pipeline Complete!")

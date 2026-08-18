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
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc, average_precision_score

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
        if tissue not in path_by_tissue:
            raise ValueError(
                f"Model '{config.get('model')}' has no path for tissue "
                f"'{tissue}'."
            )
        path_value = str(path_by_tissue[tissue])
        result_dir = os.path.dirname(path_value)
    elif tissue_result_dirs is not None:
        if not isinstance(tissue_result_dirs, Mapping):
            raise TypeError(
                "tissue_result_dirs must be a tissue-to-directory mapping."
            )
        if tissue not in tissue_result_dirs:
            raise ValueError(
                f"Model '{config.get('model')}' has no result directory for "
                f"tissue '{tissue}'."
            )
        result_dir = str(tissue_result_dirs[tissue])
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


def _collect_multi_tissue_pr_auc(
        manifest: list,
        trace_score_col: Optional[str] = None,
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        trace_model_name: str = 'TRACE',
        require_same_total_gt: bool = True,
        require_complete_grid: bool = True) -> pd.DataFrame:
    """Calculate candidate-level Average Precision for each tissue and model."""
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
                table_tissue = cell_type_by_tissue.get(
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
        warnings.warn(f"Skipping incomplete PR-AUC entry: {entry}")
    if not records:
        raise ValueError("No valid tissue-by-model PR-AUC values were found.")

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
                "same tissue, so their PR-AUC values are not directly "
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
        filename: str = 'Benchmark_Multi_Model_PR_AUC_by_Tissue.pdf',
        w: float = 6.0,
        h: float = 5.0):
    """Plot mean PR-AUC bars with one point per tissue for every model.

    Every manifest item defines its own ``tissue_result_dirs`` mapping. Its
    relative ``path`` is resolved inside each tissue directory. TRACE can be
    reranked with one existing ``trace_score_col`` or a dynamic
    ``trace_combined_score`` definition. Other models use ``score_col`` or
    ``combined_score`` from their own manifest item. ``cell_type_by_tissue``
    optionally maps display tissue labels to table ``Cell_Type`` values.
    """
    if len(y_limits) != 2 or not 0 <= y_limits[0] < y_limits[1] <= 1:
        raise ValueError("y_limits must satisfy 0 <= lower < upper <= 1.")
    if w <= 0 or h <= 0:
        raise ValueError("w and h must be greater than 0.")

    plot_df = _collect_multi_tissue_pr_auc(
        manifest=manifest,
        trace_score_col=trace_score_col,
        trace_combined_score=trace_combined_score,
        trace_model_name=trace_model_name,
        require_same_total_gt=require_same_total_gt,
        require_complete_grid=require_complete_grid,
    )
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
        plot_df.groupby('Model', observed=True)['PR_AUC']
        .agg(['mean', 'sem', 'count'])
        .reset_index()
        .rename(columns={'count': 'Tissue_Count'})
    )
    summary_df['sem'] = summary_df['sem'].fillna(0.0)
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
            color='black',
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
            mapping=aes(x='Model', y='PR_AUC', shape='Tissue'),
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
        + labs(x="", y="PR-AUC", title=title)
        + theme(
            figure_size=(w, h),
            axis_text_x=element_text(angle=45, hjust=1, color="black"),
            axis_title_x=element_blank(),
            panel_grid_major_x=element_blank(),
            legend_position='right',
            legend_title=element_text(fontweight='bold'),
        )
    )

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


def plot_multi_model_top_k_precision(
        manifest: list, 
        out_dir: str = "./results/benchmark", 
        min_k: Optional[int] = None, 
        max_k: Optional[int] = None, 
        suffix: str = "",
        y_limits: tuple = (0, 1),
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        trace_model_name: str = 'TRACE',
        mark_incomplete_endpoints: bool = True,
        endpoint_size: float = 3.2,
):
    """Plot Precision@K with optional TRACE feature-combination ranking."""
    if len(y_limits) != 2 or not 0 <= y_limits[0] < y_limits[1] <= 1:
        raise ValueError("y_limits must satisfy 0 <= lower < upper <= 1.")
    if endpoint_size <= 0:
        raise ValueError("endpoint_size must be greater than 0.")

    os.makedirs(out_dir, exist_ok=True)
    all_pk_data = []
    
    for cfg in manifest:
        model_name = cfg['model']
        csv_path = cfg['path']
        score_col, combined_score = resolve_manifest_score_request(
            cfg,
            trace_combined_score=trace_combined_score,
            trace_model_name=trace_model_name,
        )
        
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        
        if 'Precision' in df.columns and 'K' in df.columns:
            if combined_score is not None:
                raise ValueError(
                    f"Model '{model_name}' uses a precomputed Precision@K "
                    "table, which cannot be reranked by a new combination. "
                    "Use unified_evaluation_table.csv or regenerate the "
                    "Top-K table with top_k_combined_score."
                )
            pk_df = df[['K', 'Precision']].copy()
            score_label = (
                str(df['Score_Type'].dropna().iloc[0])
                if 'Score_Type' in df.columns
                and not df['Score_Type'].dropna().empty
                else 'precomputed'
            )
        elif 'y_true' in df.columns:
            had_record_type = 'Record_Type' in df.columns
            df_sorted, score_col, score_label = prepare_evaluation_score(
                df,
                score_col=score_col,
                combined_score=combined_score,
            )
            df_sorted[score_col] = pd.to_numeric(
                df_sorted[score_col], errors='coerce'
            )
            df_sorted['y_true'] = pd.to_numeric(
                df_sorted['y_true'], errors='coerce'
            )
            df_sorted = df_sorted.replace(
                [np.inf, -np.inf], np.nan
            ).dropna(subset=[score_col, 'y_true'])
            if not had_record_type:
                df_sorted = df_sorted[df_sorted[score_col] >= 0].copy()
            df_sorted = df_sorted.sort_values(
                by=score_col,
                ascending=False,
                kind='mergesort',
            ).reset_index(drop=True)
            if df_sorted.empty: continue
            k_array = np.arange(1, len(df_sorted) + 1)
            tp_cumsum = df_sorted['y_true'].cumsum()
            pk_df = pd.DataFrame({'K': k_array, 'Precision': tp_cumsum / k_array})
        else: continue

        pk_df['K'] = pd.to_numeric(pk_df['K'], errors='coerce')
        pk_df['Precision'] = pd.to_numeric(
            pk_df['Precision'], errors='coerce'
        )
        pk_df = pk_df.replace([np.inf, -np.inf], np.nan).dropna()
        pk_df = pk_df[pk_df['K'] > 0].sort_values('K')
        pk_df = pk_df.drop_duplicates('K', keep='last')
        if pk_df.empty:
            continue
        if not pk_df['Precision'].between(0, 1).all():
            raise ValueError(
                f"Model '{model_name}' has Precision values outside [0, 1]."
            )

        pk_df['Model'] = model_name
        pk_df['Score_Type'] = score_label
        all_pk_data.append(pk_df)
        
    if not all_pk_data: raise ValueError("No valid Top-K data processed.")
    plot_df = pd.concat(all_pk_data, ignore_index=True)
        
    def apply_smoothing(group):
        group['Precision_Smooth'] = group['Precision'].rolling(window=20, min_periods=1).mean()
        return group
        
    plot_df = plot_df.groupby('Model', group_keys=False).apply(apply_smoothing)

    if min_k is not None: plot_df = plot_df[plot_df['K'] >= min_k]
    if max_k is not None: plot_df = plot_df[plot_df['K'] <= max_k]
    if plot_df.empty:
        raise ValueError(
            "No Precision@K observations remain in the requested K range."
        )
    endpoint_df = (
        _extract_incomplete_curve_endpoints(plot_df, max_k)
        if mark_incomplete_endpoints
        else plot_df.iloc[0:0].copy()
    )

    def downsample(group, max_pts=3000):
        if len(group) > max_pts:
            indices = np.linspace(0, len(group) - 1, max_pts).astype(int)
            return group.iloc[indices]
        return group
    plot_df = plot_df.groupby('Model', group_keys=False).apply(downsample)

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
        + geom_line(aes(linetype='Model'), size=1.5, alpha=0.85)
        + scale_color_manual(values=color_mapping)
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


def plot_multi_model_top_k_recall(
        manifest: list,
        out_dir: str = "./results/benchmark",
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
        suffix: str = "",
        require_same_total_gt: bool = True,
        y_limits: tuple = (0, 1),
        trace_combined_score: Optional[
            Union[str, Mapping[str, object], pd.Series]
        ] = None,
        trace_model_name: str = 'TRACE',
        mark_incomplete_endpoints: bool = True,
        endpoint_size: float = 3.2):
    """Plot Recall@K with optional TRACE feature-combination ranking."""
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

    os.makedirs(out_dir, exist_ok=True)
    all_recall_data = []
    model_gt_counts = {}

    for cfg in manifest:
        model_name = cfg['model']
        csv_path = cfg['path']
        score_col, combined_score = resolve_manifest_score_request(
            cfg,
            trace_combined_score=trace_combined_score,
            trace_model_name=trace_model_name,
        )
        total_gt_override = cfg.get('total_gt')
        if not os.path.exists(csv_path):
            warnings.warn(f"Top-K input was not found and will be skipped: {csv_path}")
            continue

        source_df = pd.read_csv(csv_path)
        is_precomputed = (
            'K' in source_df.columns
            and (
                'Recall' in source_df.columns
                or 'Recall_at_K' in source_df.columns
            )
        )
        if is_precomputed and combined_score is not None:
            raise ValueError(
                f"Model '{model_name}' uses a precomputed Recall@K table, "
                "which cannot be reranked by a new combination. Use "
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
            warnings.warn(f"No valid Recall@K data for model '{model_name}'.")
            continue
        recall_df['Model'] = model_name
        recall_df['Score_Type'] = score_label
        all_recall_data.append(recall_df)
        if total_gt is not None:
            model_gt_counts[model_name] = int(total_gt)

    if not all_recall_data:
        raise ValueError("No valid Top-K recall data processed.")
    if require_same_total_gt and len(set(model_gt_counts.values())) > 1:
        details = ', '.join(
            f"{model}={count}" for model, count in model_gt_counts.items()
        )
        raise ValueError(
            "Models use different callable GT denominators, so Recall@K is "
            f"not directly comparable: {details}"
        )

    plot_df = pd.concat(all_recall_data, ignore_index=True)
    if min_k is not None:
        plot_df = plot_df[plot_df['K'] >= min_k]
    if max_k is not None:
        plot_df = plot_df[plot_df['K'] <= max_k]
    if plot_df.empty:
        raise ValueError("No Recall@K observations remain in the requested K range.")
    endpoint_df = (
        _extract_incomplete_curve_endpoints(plot_df, max_k)
        if mark_incomplete_endpoints
        else plot_df.iloc[0:0].copy()
    )

    def downsample(group, max_pts=3000):
        if len(group) <= max_pts:
            return group
        indices = np.unique(
            np.linspace(0, len(group) - 1, max_pts).astype(int)
        )
        return group.iloc[indices]

    plot_df = (
        plot_df.groupby('Model', group_keys=False, observed=False)
        .apply(downsample)
        .reset_index(drop=True)
    )
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
        + geom_line(aes(linetype='Model'), size=1.5, alpha=0.85)
        + scale_color_manual(values=color_mapping)
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

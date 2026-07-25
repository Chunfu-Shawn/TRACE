"""Position-wise P-site correlation and depth-stratified plotting utilities."""

import os
import pickle
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotnine import *
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

try:
    from .evaluation_utils import (
        cds_with_stop_slice,
        get_prediction,
        load_prediction_input,
        to_1d_signal,
        transcript_id_from_uuid,
    )
except ImportError:
    from evaluation_utils import (
        cds_with_stop_slice,
        get_prediction,
        load_prediction_input,
        to_1d_signal,
        transcript_id_from_uuid,
    )


CORRELATION_COLUMNS = [
    "Cell_type",
    "Tid",
    "Length",
    "Reads_DS1",
    "Reads_DS2",
    "Depth",
    "Pearson_R",
    "Pearson_P_value",
    "Spearman_R",
    "Spearman_P_value",
]


def load_pickle(path):
    """Load a pickle file after validating its path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def flatten_counts(counts_dict, length):
    """Convert ``{1-based position: {read_length: count}}`` to a dense array."""
    arr = np.zeros(length, dtype=np.float32)
    total_reads = 0.0
    if counts_dict is None:
        return arr, 0

    for raw_position, length_counts in counts_dict.items():
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            continue
        if not 1 <= position <= length:
            continue
        if isinstance(length_counts, dict):
            count_sum = float(sum(length_counts.values()))
        else:
            count_sum = float(length_counts)
        arr[position - 1] = count_sum
        total_reads += count_sum
    return arr, total_reads


def _correlation_pair(first, second):
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return np.nan, np.nan, np.nan, np.nan
    pearson = pearsonr(first, second)
    spearman = spearmanr(first, second)
    return (
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spearman.statistic),
        float(spearman.pvalue),
    )


def calculate_psite_metrics(data_paths_dict, seq_pkl_path, out_dir, suffix=""):
    """Calculate cross-dataset positional correlations for multiple cell types."""
    os.makedirs(out_dir, exist_ok=True)
    seq_data = load_pickle(seq_pkl_path)
    sequence_keys = set(seq_data.keys())
    all_results = []

    for cell_type, paths in data_paths_dict.items():
        if len(paths) != 2:
            raise ValueError(
                f"Expected exactly 2 paths for {cell_type}, but got {len(paths)}."
            )
        print(f"\nLoading data for {cell_type}...")
        data1 = load_pickle(paths[0])
        data2 = load_pickle(paths[1])
        common_tids = sorted(set(data1) & set(data2) & sequence_keys)
        print(f"[{cell_type}] Dataset 1: {len(data1)} transcripts")
        print(f"[{cell_type}] Dataset 2: {len(data2)} transcripts")
        print(
            f"[{cell_type}] Intersection (Analyzable): {len(common_tids)} transcripts"
        )

        for tid in tqdm(common_tids, desc=f"Comparing {cell_type} transcripts"):
            seq_len = len(seq_data[tid])
            if seq_len == 0:
                continue
            vec1, total1 = flatten_counts(data1[tid], seq_len)
            vec2, total2 = flatten_counts(data2[tid], seq_len)
            depth = (total1 + total2) / 2 / seq_len
            p_r, p_p, s_r, s_p = _correlation_pair(vec1, vec2)
            all_results.append(
                {
                    "Cell_type": cell_type,
                    "Tid": tid,
                    "Length": seq_len,
                    "Reads_DS1": total1,
                    "Reads_DS2": total2,
                    "Depth": depth,
                    "Pearson_R": p_r,
                    "Pearson_P_value": p_p,
                    "Spearman_R": s_r,
                    "Spearman_P_value": s_p,
                }
            )

    df = pd.DataFrame(all_results, columns=CORRELATION_COLUMNS)
    df = df.dropna(subset=["Pearson_R", "Spearman_R"])
    csv_path = os.path.join(out_dir, f"correlation_results.{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")
    return df


def plot_correlation_by_depth(df, out_dir, prefix="comparison", bins=5):
    """Bin sequencing depth and plot Pearson/Spearman distributions.

    ``bins`` may be an integer for quantile bins or a sequence of absolute edges.
    """
    df_plot = df.copy()
    if df_plot.empty:
        print("No correlation data to plot.")
        return

    try:
        if isinstance(bins, int):
            print(f"Binning by {bins} quantiles...")
            df_plot["Depth_Bin_Label"] = pd.qcut(
                df_plot["Depth"], q=bins, duplicates="drop"
            )
            df_plot["Depth_Bin_Code"] = pd.qcut(
                df_plot["Depth"], q=bins, labels=False, duplicates="drop"
            )
            xlabel_text = "Depth Quantile (Low -> High)"
        elif isinstance(bins, (list, tuple, np.ndarray)):
            print(f"Binning by absolute values: {bins} ...")
            df_plot["Depth_Bin_Label"] = pd.cut(
                df_plot["Depth"], bins=bins, include_lowest=True
            )
            df_plot["Depth_Bin_Code"] = pd.cut(
                df_plot["Depth"], bins=bins, labels=False, include_lowest=True
            )
            xlabel_text = "Read Depth Range (RPKM/Density)"
        else:
            raise TypeError("bins argument must be int or list.")
    except ValueError as error:
        print(f"Binning failed: {error}. Usually there are too few unique values.")
        return

    df_plot = df_plot.dropna(subset=["Depth_Bin_Label"])
    if df_plot.empty:
        print("No data remained after depth binning.")
        return

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(
        x="Depth_Bin_Label",
        y="Pearson_R",
        data=df_plot,
        showfliers=False,
        hue="Depth_Bin_Label",
        palette="Blues",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Pearson Correlation vs Read Depth")
    axes[0].set_xlabel(xlabel_text)
    axes[0].set_ylabel("Pearson R")
    axes[0].set_ylim(-0.2, 1.1)

    sns.boxplot(
        x="Depth_Bin_Label",
        y="Spearman_R",
        data=df_plot,
        showfliers=False,
        hue="Depth_Bin_Label",
        palette="Greens",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Spearman Correlation vs Read Depth")
    axes[1].set_xlabel(xlabel_text)
    axes[1].set_ylabel("Spearman R")
    axes[1].set_ylim(-0.2, 1.1)
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.grid(axis="y", linestyle="--", alpha=0.5)

    figure.tight_layout()
    out_path = os.path.join(out_dir, f"psite_depth_correlation.{prefix}.pdf")
    figure.savefig(out_path)
    plt.close(figure)
    print(f"Plot saved to {out_path}")

    print("\n=== Summary stats by Depth Bin ===")
    summary = (
        df_plot.groupby("Depth_Bin_Label", observed=True)[
            ["Pearson_R", "Spearman_R", "Depth", "Reads_DS1"]
        ]
        .agg(
            {
                "Pearson_R": "median",
                "Spearman_R": "median",
                "Depth": "mean",
                "Reads_DS1": "count",
            }
        )
        .rename(columns={"Reads_DS1": "Transcript_Count"})
    )
    print(summary)


def calculate_correlations_multitissue(
    dataset,
    pkl_input: Union[Dict[str, str], str],
    output_dir: str = ".",
    suffix: str = "",
    for_cds: bool = False,
):
    """Calculate per-transcript prediction/observation positional correlations."""
    print(">>> Loading prediction files...")
    all_predictions = load_prediction_input(pkl_input)
    results = []

    print(f"\n>>> Evaluating transcripts in the Dataset (CDS Only: {for_cds})...")
    for index in tqdm(range(len(dataset))):
        uuid, _, cell_type, _, meta_info, _, count_emb = dataset[index]
        tid = transcript_id_from_uuid(uuid)
        cell_type = str(cell_type)
        pred_signal = get_prediction(all_predictions, cell_type, tid)
        if pred_signal is None:
            continue
        gt_signal = to_1d_signal(count_emb)
        pred_len = len(pred_signal)
        gt_len = len(gt_signal)

        bounds = cds_with_stop_slice(meta_info, gt_len)
        has_cds = bounds is not None
        if has_cds:
            start_idx, end_idx = bounds
            cds_len = end_idx - start_idx
            is_pred_full = abs(pred_len - gt_len) <= abs(pred_len - cds_len)
        else:
            start_idx, end_idx = 0, gt_len
            cds_len = gt_len
            is_pred_full = True

        if for_cds:
            if not has_cds or cds_len < 6:
                continue
            gt_target = gt_signal[start_idx:end_idx]
            if is_pred_full:
                safe_end = min(end_idx, pred_len)
                pred_target = pred_signal[start_idx:safe_end]
            else:
                pred_target = pred_signal
        elif is_pred_full:
            gt_target = gt_signal
            pred_target = pred_signal
        else:
            if not has_cds:
                continue
            gt_target = gt_signal[start_idx:end_idx]
            pred_target = pred_signal

        min_len = min(len(pred_target), len(gt_target))
        if min_len < 2:
            continue
        pred_aligned = np.asarray(pred_target[:min_len], dtype=np.float32)
        gt_aligned = np.asarray(gt_target[:min_len], dtype=np.float32)
        p_r, p_p, s_r, s_p = _correlation_pair(pred_aligned, gt_aligned)
        results.append(
            {
                "Tid": tid,
                "Cell_type": cell_type,
                "Depth": float(meta_info.get("rpf_depth", np.nan)),
                "Pearson_R": p_r,
                "Pearson_P_Value": p_p,
                "Spearman_R": s_r,
                "Spearman_P_Value": s_p,
            }
        )

    columns = [
        "Tid",
        "Cell_type",
        "Depth",
        "Pearson_R",
        "Pearson_P_Value",
        "Spearman_R",
        "Spearman_P_Value",
    ]
    df = pd.DataFrame(results, columns=columns)
    cds_tag = "cds_only" if for_cds else "evaluation_results"
    save_filename = (
        f"psite_corr_{cds_tag}.{suffix}.csv"
        if suffix
        else f"psite_corr_{cds_tag}.csv"
    )
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_filename)
    df.to_csv(save_path, sep=",", index=False, float_format="%.6g")
    print(
        f"\n>>> Evaluation complete! Successfully matched and calculated {len(df)} transcripts."
    )
    print(f">>> Results saved to: {save_path}")
    return df


def plot_correlation_by_cell_type(
    df,
    out_dir,
    suffix="",
    metric="Spearman_R",
    max_points_per_cell=300,
):
    """Plot mean transcript-level correlation by cell type with 95% CIs.

    This is an additive API used by the unified evaluation runner. Existing public
    functions and their return values are unchanged.
    """
    required = {"Cell_type", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing correlation columns: {sorted(missing)}")
    os.makedirs(out_dir, exist_ok=True)
    valid = df[["Cell_type", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    summary_columns = [
        "Cell_type",
        "N",
        f"Mean_{metric}",
        f"Median_{metric}",
        "SEM",
        "CI95_low",
        "CI95_high",
    ]
    if valid.empty:
        summary = pd.DataFrame(columns=summary_columns)
        return summary

    rows = []
    for cell_type, group in valid.groupby("Cell_type", sort=True):
        values = group[metric].to_numpy(dtype=float)
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        mean = float(np.mean(values))
        rows.append(
            {
                "Cell_type": cell_type,
                "N": int(len(values)),
                f"Mean_{metric}": mean,
                f"Median_{metric}": float(np.median(values)),
                "SEM": sem,
                "CI95_low": mean - 1.96 * sem,
                "CI95_high": mean + 1.96 * sem,
            }
        )
    summary = pd.DataFrame(rows, columns=summary_columns).sort_values(
        f"Mean_{metric}", ascending=False
    )
    file_suffix = f".{suffix}" if suffix else ""
    summary.to_csv(
        os.path.join(out_dir, f"correlation_by_cell_type{file_suffix}.csv"),
        index=False,
    )

    rng = np.random.default_rng(42)
    figure_width = max(6, 1.0 * len(summary))
    figure, axis = plt.subplots(figsize=(figure_width, 4.5))
    x = np.arange(len(summary))
    means = summary[f"Mean_{metric}"].to_numpy(dtype=float)
    axis.bar(x, means, width=0.72, color="#176D9C", zorder=2)
    axis.errorbar(
        x,
        means,
        yerr=1.96 * summary["SEM"].to_numpy(dtype=float),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
        zorder=4,
    )
    for cell_index, cell_type in enumerate(summary["Cell_type"]):
        values = valid.loc[valid["Cell_type"] == cell_type, metric].to_numpy(dtype=float)
        if len(values) > max_points_per_cell:
            values = rng.choice(values, max_points_per_cell, replace=False)
        jitter = rng.uniform(-0.25, 0.25, size=len(values))
        axis.scatter(
            np.full(len(values), cell_index) + jitter,
            values,
            s=5,
            alpha=0.1,
            color="black",
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
    axis.set_xticks(x, summary["Cell_type"], rotation=35, ha="right")
    axis.set_ylim(-0.05, 1.0)
    axis.set_ylabel(f"Position-wise {metric.replace('_', ' ')}")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    plot_path = os.path.join(
        out_dir, f"correlation_by_cell_type{file_suffix}.pdf"
    )
    figure.savefig(plot_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Cell-type correlation plot saved to {plot_path}")
    return summary


def plot_scatter_depth_vs_correlation(
    df,
    out_dir,
    x_col="Depth",
    y_col="Pearson_R",
    suffix=".",
    max_points=10000,
):
    """Plot sequencing depth versus correlation on a log10 x-axis."""
    plot_name = (
        f"depth_vs_correlation.{suffix}.pdf" if suffix else "depth_vs_correlation.pdf"
    )
    plot_path = os.path.join(out_dir, plot_name)
    df_plot = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    df_plot = df_plot[df_plot[x_col] > 0].copy()
    if df_plot.empty:
        print("No positive depth data to plot.")
        return

    float16_cols = df_plot.select_dtypes(include=["float16"]).columns
    if len(float16_cols) > 0:
        df_plot[float16_cols] = df_plot[float16_cols].astype("float32")
    df_plot[x_col] = df_plot[x_col].astype("float32")
    df_plot[y_col] = df_plot[y_col].astype("float32")
    if len(df_plot) > 1 and np.ptp(df_plot[x_col]) > 0 and np.ptp(df_plot[y_col]) > 0:
        correlation = pearsonr(df_plot[x_col], df_plot[y_col])
        stats_label = (
            f"Pearson R = {correlation.statistic:.3f} (P={correlation.pvalue:.2e})"
        )
    else:
        stats_label = "Pearson R = NA"

    if len(df_plot) > max_points:
        print(f"Downsampling plot data from {len(df_plot)} to {max_points} points...")
        df_plot = df_plot.sample(n=max_points, random_state=42)

    try:
        plot = (
            ggplot(df_plot, aes(x=x_col, y=y_col))
            + geom_point(alpha=0.3, color="gray", size=2, stroke=0)
            + geom_smooth(method="lm", color="#005b96", size=1.5)
            + annotate(
                "text",
                x=df_plot[x_col].min(),
                y=df_plot[y_col].max() * 0.95,
                label=stats_label,
                ha="left",
                va="top",
                size=10,
            )
            + scale_x_log10()
            + theme_bw()
            + theme(text=element_text(size=12), plot_title=element_text(size=14))
            + labs(
                title=f"Correlation vs. Sequencing Depth (n={len(df_plot)})",
                x="Sequencing Depth (Log10 Scale)",
                y=f"{y_col} Coefficient",
            )
        )
        plot.save(plot_path, width=5, height=5, dpi=300, verbose=False)
        print(f"Plot saved to {plot_path}")
    except Exception as error:
        print(f"Error saving plot: {error}")
        print("Data types for debugging:")
        print(df_plot.dtypes)


def load_and_process_comparison_data(
    file1,
    name1,
    file2,
    name2,
    metric="Pearson_R",
    target_ratio=None,
):
    """Load two correlation CSV files and assign log10 sequencing-depth bins."""
    bins = [-np.inf, -1, -0.301, 0, 0.699, 1, np.inf]
    labels = ["<0.1", "0.1 - 0.5", "0.5 - 1", "1 - 5", "5 - 10", ">10"]

    def process_file(path, label):
        if not os.path.exists(path):
            print(f"[Error] File not found: {path}")
            return None
        try:
            frame = pd.read_csv(path)
        except Exception as error:
            print(f"[Error] Reading {path}: {error}")
            return None
        if target_ratio is not None and "Mask_Ratio" in frame.columns:
            frame = frame[frame["Mask_Ratio"] == target_ratio]
        if metric not in frame.columns or "Depth" not in frame.columns:
            print(f"[Warning] Missing '{metric}' or 'Depth' in {label}")
            return None
        frame = frame[[metric, "Depth"]].replace([np.inf, -np.inf], np.nan).dropna()
        frame = frame[frame["Depth"] > 0].copy()
        frame["log_depth"] = np.log10(frame["Depth"])
        frame["Depth_Group"] = pd.cut(frame["log_depth"], bins=bins, labels=labels)
        frame["Source"] = label
        return frame

    df1 = process_file(file1, name1)
    df2 = process_file(file2, name2)
    if df1 is None or df2 is None:
        return None
    combined_df = pd.concat([df1, df2], ignore_index=True).dropna(
        subset=["Depth_Group"]
    )
    combined_df["Depth_Group"] = pd.Categorical(
        combined_df["Depth_Group"], categories=list(reversed(labels)), ordered=True
    )
    return combined_df


def plot_ridge_density_comparison(df, metric="Pearson_R", out_dir="./results"):
    """Plot ridge-style correlation densities stratified by sequencing depth."""
    if df is None or df.empty:
        print("No data to plot.")
        return
    plot_df = df.copy()
    plot_df["Source"] = pd.Categorical(
        plot_df["Source"],
        categories=["base_model (Pred. vs Obs.)", "Cross-experiment (Obs.)"],
        ordered=True,
    )
    custom_colors = ["#3498db", "#95a5a6"]
    plot = (
        ggplot(plot_df, aes(x=metric, fill="Source", color="Source"))
        + geom_density(alpha=0.3, size=0.3)
        + facet_grid("Depth_Group ~ .", scales="free_y")
        + scale_fill_manual(values=custom_colors)
        + scale_color_manual(values=custom_colors)
        + scale_x_continuous(
            limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0], expand=[0, 0.01]
        )
        + theme_classic()
        + theme(
            panel_spacing=0,
            strip_background=element_blank(),
            axis_text_y=element_blank(),
            panel_grid_major_x=element_line(linetype="dashed", color="lightgray"),
            axis_line_x=element_line(color="black"),
            legend_position="top",
            legend_title=element_blank(),
        )
        + labs(
            x=f"Position-wise correlation per transcript ({metric})",
            y="Sequencing Depth (Log10 Bins)",
        )
    )
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"ridge_plot_depth_comparison_{metric}.pdf")
    plot.save(save_path, width=5, height=5, dpi=300, verbose=False)
    print(f"Plot saved to: {save_path}")

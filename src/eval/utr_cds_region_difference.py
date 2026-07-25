"""Compare observed and predicted translation signals across UTR/CDS regions."""

import os
from typing import Dict, Union

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_boxplot,
    ggplot,
    position_dodge,
    scale_color_manual,
    theme,
    theme_classic,
)
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


def calculate_region_metrics(
    signal_array,
    global_start_idx,
    region_start,
    region_end,
    total_transcript_sum,
    threshold=0.01,
):
    """Calculate signal proportion and frame-0 fraction for one region.

    ``threshold`` is retained for backward API compatibility.
    """
    del threshold
    if region_end <= region_start:
        return None

    region_data = np.asarray(signal_array)[region_start:region_end]
    if len(region_data) < 3:
        return None

    region_sum = float(np.sum(region_data))
    if total_transcript_sum < 1e-6:
        proportion = 0.0
    else:
        proportion = region_sum / total_transcript_sum

    global_indices = np.arange(region_start, region_end)
    frames = (global_indices - global_start_idx) % 3
    f0_sum = float(np.sum(region_data[frames == 0]))
    periodicity = np.nan if region_sum < 1e-6 else f0_sum / region_sum

    return {"Proportion": proportion, "Periodicity": periodicity}


def evaluate_region_specificity(
    truth_dataset,
    pkl_input: Union[Dict[str, str], str],
    out_dir: str = "./results/plots",
    suffix: str = "",
    width: float = 5,
    height: float = 5,
):
    """Compare P-site proportion and periodicity in 5'UTR, CDS, and 3'UTR."""
    print(">>> Loading prediction files...")
    all_predictions = load_prediction_input(pkl_input)
    os.makedirs(out_dir, exist_ok=True)
    metrics_data = []

    print("\n>>> Analyzing region specificity...")
    for index in tqdm(range(len(truth_dataset))):
        uuid, _, cell_type, _, meta_info, _, count_emb = truth_dataset[index]
        uuid_str = str(uuid)
        tid = transcript_id_from_uuid(uuid_str)
        cell_type = str(cell_type)

        pred_signal = get_prediction(all_predictions, cell_type, tid)
        if pred_signal is None:
            continue
        gt_signal = to_1d_signal(count_emb)

        pred_linear = np.expm1(pred_signal.astype(np.float32, copy=False))
        truth_linear = np.expm1(gt_signal.astype(np.float32, copy=False))
        pred_len = len(pred_linear)
        gt_len = len(truth_linear)
        aligned_len = min(pred_len, gt_len)
        if aligned_len < 2:
            continue

        bounds = cds_with_stop_slice(meta_info, gt_len)
        if bounds is None:
            continue
        start_idx, end_idx = bounds
        cds_len = end_idx - start_idx

        is_pred_full = abs(pred_len - gt_len) <= abs(pred_len - cds_len)
        if not is_pred_full:
            continue
        if end_idx > aligned_len:
            continue

        pred_aligned = pred_linear[:aligned_len]
        truth_aligned = truth_linear[:aligned_len]
        total_sum_gt = float(np.sum(truth_aligned))
        total_sum_pred = float(np.sum(pred_aligned))
        regions = {
            "5'UTR": (0, start_idx),
            "CDS": (start_idx, end_idx),
            "3'UTR": (end_idx, aligned_len),
        }

        for region_name, (region_start, region_end) in regions.items():
            observed = calculate_region_metrics(
                truth_aligned,
                start_idx,
                region_start,
                region_end,
                total_sum_gt,
            )
            if observed:
                observed["Condition"] = "Observation"
                observed["Region"] = region_name
                observed["UUID"] = uuid_str
                metrics_data.append(observed)

            predicted = calculate_region_metrics(
                pred_aligned,
                start_idx,
                region_start,
                region_end,
                total_sum_pred,
            )
            if predicted:
                predicted["Condition"] = "Prediction"
                predicted["Region"] = region_name
                predicted["UUID"] = uuid_str
                metrics_data.append(predicted)

    df = pd.DataFrame(metrics_data)
    if df.empty:
        print("Warning: No valid transcripts found for region evaluation.")
        return df

    csv_name = (
        f"region_specificity_stats.{suffix}.csv"
        if suffix
        else "region_specificity_stats.csv"
    )
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Stats saved to {csv_path}")
    plot_region_comparison(df, out_dir, suffix, width, height)
    return df


def plot_region_comparison(df, out_dir, suffix, w=4, h=5):
    """Plot side-by-side region boxplots using the historical plotnine style."""
    print("\n>>> Generating L-shaped Boxplots (ggplot style)...")
    plot_df = df.melt(
        id_vars=["UUID", "Condition", "Region"],
        value_vars=["Proportion", "Periodicity"],
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=["Value"])

    plot_df["Region"] = pd.Categorical(
        plot_df["Region"], categories=["5'UTR", "CDS", "3'UTR"], ordered=True
    )
    plot_df["Condition"] = pd.Categorical(
        plot_df["Condition"],
        categories=["Observation", "Prediction"],
        ordered=True,
    )
    plot_df["Metric"] = pd.Categorical(
        plot_df["Metric"],
        categories=["Proportion", "Periodicity"],
        ordered=True,
    )
    colors = {"Observation": "#A0A0A0", "Prediction": "#2C6B9A"}

    plot = (
        ggplot(plot_df, aes(x="Region", y="Value", color="Condition"))
        + geom_boxplot(
            fill="white",
            size=0.8,
            outlier_shape=None,
            outlier_alpha=0,
            width=0.7,
            position=position_dodge(width=0.8),
        )
        + facet_wrap("~Metric", scales="free_y", nrow=2)
        + scale_color_manual(values=colors)
        + theme_classic()
        + theme(
            legend_position="top",
            legend_title=element_blank(),
            axis_title_x=element_blank(),
            axis_title_y=element_blank(),
            strip_background=element_blank(),
            strip_text=element_text(weight="bold", size=12),
            axis_text=element_text(size=11, color="black"),
            axis_line=element_line(color="black", size=1),
        )
    )

    save_path = os.path.join(out_dir, f"region_specificity_comparison.{suffix}.pdf")
    plot.save(filename=save_path, width=w, height=h, verbose=False)
    print(f"Comparison plot saved to {save_path}")

"""Evaluate observed and predicted three-nucleotide periodicity."""

import os
import pickle
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm

try:
    from .evaluation_utils import cds_slice, get_prediction, to_1d_signal, transcript_id_from_uuid
except ImportError:
    from evaluation_utils import cds_slice, get_prediction, to_1d_signal, transcript_id_from_uuid


def calculate_periodicity(signal_array, cds_start, cds_end):
    """Return the frame-0 signal fraction for CDS or the maximum frame fraction.

    When valid half-open CDS coordinates are supplied, frame zero is anchored at
    ``cds_start``. Without valid CDS coordinates, all three full-length frames are
    evaluated and the largest fraction is returned.
    """
    signal_array = to_1d_signal(signal_array)
    if cds_start != -1 and cds_end != -1 and cds_end > cds_start:
        region_data = signal_array[max(0, cds_start) : min(len(signal_array), cds_end)]
        total_sum = float(np.sum(region_data))
        if total_sum < 1e-6 or len(region_data) < 3:
            return np.nan
        frames = np.arange(len(region_data)) % 3
        return float(np.sum(region_data[frames == 0]) / total_sum)

    region_data = signal_array
    total_sum = float(np.sum(region_data))
    if total_sum < 1e-6 or len(region_data) < 3:
        return np.nan
    frames = np.arange(len(region_data)) % 3
    frame_sums = [float(np.sum(region_data[frames == frame])) for frame in range(3)]
    return max(frame_sums) / total_sum


def summarize_periodicity_results(df):
    """Return sample count, Spearman correlation, and MAE for periodicity pairs."""
    required = {"GT_Periodicity", "Pred_Periodicity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing periodicity columns: {sorted(missing)}")
    pairs = df[["GT_Periodicity", "Pred_Periodicity"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if pairs.empty:
        return {"n": 0, "spearman": None, "mae": None}
    observed = pairs["GT_Periodicity"].to_numpy(dtype=float)
    predicted = pairs["Pred_Periodicity"].to_numpy(dtype=float)
    if len(pairs) < 2 or np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        correlation = None
    else:
        correlation = float(spearmanr(observed, predicted).statistic)
    return {
        "n": int(len(pairs)),
        "spearman": correlation,
        "mae": float(np.mean(np.abs(observed - predicted))),
    }


def plot_periodicity_scatter(df, out_dir="./results/periodicity", suffix=""):
    """Plot predicted versus observed periodicity with marginal distributions."""
    if df.empty:
        return
    required = {"GT_Periodicity", "Pred_Periodicity", "Gene_Type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing periodicity plot columns: {sorted(missing)}")
    os.makedirs(out_dir, exist_ok=True)

    plot_df = df.dropna(subset=["GT_Periodicity", "Pred_Periodicity"]).copy()
    if plot_df.empty:
        print("No finite periodicity pairs available for plotting.")
        return

    unique_types = plot_df["Gene_Type"].astype(str).unique().tolist()
    nc_types = sorted(
        gene_type
        for gene_type in unique_types
        if gene_type not in {"Other", "Housekeeping"}
    )
    category_order = [
        category
        for category in ["Other", *nc_types, "Housekeeping"]
        if category in unique_types
    ]
    plot_df["Gene_Type"] = pd.Categorical(
        plot_df["Gene_Type"], categories=category_order, ordered=True
    )
    plot_df = plot_df.sort_values("Gene_Type")

    color_map = {"Other": "#B0B0B0", "Housekeeping": "#E74C3C"}
    distinct_colors = [
        "#3498DB",
        "#2ECC71",
        "#9B59B6",
        "#F1C40F",
        "#E67E22",
        "#1ABC9C",
        "#34495E",
    ]
    for index, nc_type in enumerate(nc_types):
        color_map[nc_type] = distinct_colors[index % len(distinct_colors)]
    color_map = {key: value for key, value in color_map.items() if key in category_order}

    print("Applied Color Mapping for scatter plot:")
    for key, value in color_map.items():
        print(f"  {key}: {value}")

    if len(plot_df) > 1:
        correlation = spearmanr(
            plot_df["GT_Periodicity"], plot_df["Pred_Periodicity"]
        )
        r_val = float(correlation.statistic)
        p_val = float(correlation.pvalue)
    else:
        r_val, p_val = np.nan, np.nan
    p_text = (
        f"{p_val:.1e}"
        if np.isfinite(p_val) and p_val < 0.001
        else f"{p_val:.3f}"
    )
    label_text = f"Spearman $R$ = {r_val:.3f}\n$P$ = {p_text}"

    print("\n>>> Generating Scatter Plot with Marginal Densities...")
    grid = sns.JointGrid(
        data=plot_df,
        x="GT_Periodicity",
        y="Pred_Periodicity",
        hue="Gene_Type",
        palette=color_map,
        height=6,
        ratio=6,
        space=0,
    )
    grid.plot_joint(sns.scatterplot, alpha=0.4, s=15, edgecolor="none")
    try:
        grid.plot_marginals(
            sns.kdeplot,
            fill=True,
            alpha=0.3,
            linewidth=1.2,
            common_norm=False,
            warn_singular=False,
        )
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        print(f"KDE marginals unavailable ({error}); using histograms instead.")
        grid.ax_marg_x.clear()
        grid.ax_marg_y.clear()
        sns.histplot(
            data=plot_df,
            x="GT_Periodicity",
            hue="Gene_Type",
            palette=color_map,
            element="step",
            fill=False,
            legend=False,
            ax=grid.ax_marg_x,
        )
        sns.histplot(
            data=plot_df,
            y="Pred_Periodicity",
            hue="Gene_Type",
            palette=color_map,
            element="step",
            fill=False,
            legend=False,
            ax=grid.ax_marg_y,
        )

    grid.ax_joint.grid(
        True, color="lightgray", linestyle="-", linewidth=1.0, alpha=0.8
    )
    grid.ax_joint.set_axisbelow(True)
    grid.ax_joint.axvline(
        x=0.5,
        color="#6b6b6b",
        linestyle="--",
        linewidth=1.5,
        alpha=1.0,
        zorder=10,
    )
    grid.ax_joint.axhline(
        y=0.5,
        color="#6b6b6b",
        linestyle="--",
        linewidth=1.5,
        alpha=1.0,
        zorder=10,
    )
    grid.ax_joint.set_xlim(0.3, None)
    grid.ax_joint.set_ylim(0.3, None)
    grid.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    grid.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    grid.ax_joint.set_xlabel("Observed periodicity", fontsize=14, color="black")
    grid.ax_joint.set_ylabel("Predicted periodicity", fontsize=14, color="black")
    grid.ax_joint.text(
        0.95,
        0.05,
        label_text,
        transform=grid.ax_joint.transAxes,
        fontsize=12,
        ha="right",
        va="bottom",
        color="black",
    )
    grid.ax_joint.legend(
        title="Transcript Type",
        title_fontsize=14,
        fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.2, 1),
        frameon=False,
    )
    grid.ax_joint.tick_params(
        axis="both",
        which="major",
        labelsize=12,
        direction="out",
        length=6,
        width=1.5,
        colors="black",
    )
    for spine in grid.ax_joint.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")
    for marginal_axis in (grid.ax_marg_x, grid.ax_marg_y):
        marginal_axis.tick_params(
            axis="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        marginal_axis.grid(False)
        for spine in marginal_axis.spines.values():
            spine.set_visible(False)

    plot_name = (
        f"periodicity_scatter_density_{suffix}.pdf"
        if suffix
        else "periodicity_scatter_density.pdf"
    )
    plot_path = os.path.join(out_dir, plot_name)
    grid.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(grid.figure)
    print(f"Saved scatter plot with marginal densities to {plot_path}")


def evaluate_periodicity_correlation(
    dataset,
    pkl_path: str,
    hk_genes_path: str = "/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/other_gene_list/housekeeping_genes.tsv",
    gtf_path: str = None,
    out_dir="./results/periodicity",
    suffix="",
):
    """Evaluate periodicity recovery and optionally label transcript classes."""
    os.makedirs(out_dir, exist_ok=True)
    print("--- Evaluating Tri-nucleotide Periodicity ---")

    hk_transcript_set = set()
    if hk_genes_path and os.path.exists(hk_genes_path):
        print(f"Loading housekeeping genes from {hk_genes_path}...")
        separator = "\t" if str(hk_genes_path).endswith(".tsv") else ","
        hk_df = pd.read_csv(hk_genes_path, sep=separator)
        if "Transcript ID" in hk_df.columns:
            hk_transcript_set = set(
                hk_df["Transcript ID"].dropna().astype(str).str.split(".").str[0]
            )
            print(
                f"  -> Found {len(hk_transcript_set)} unique housekeeping transcripts."
            )
        else:
            print("  -> Missing 'Transcript ID' column; skipping housekeeping labels.")
    else:
        print("  -> Housekeeping-gene file not provided or not found; labels skipped.")

    nc_tid_to_type = {}
    if gtf_path and os.path.exists(gtf_path):
        print(f"Parsing GTF transcript types from {gtf_path}...")
        tid_re = re.compile(r'transcript_id "([^"]+)"')
        btype_re = re.compile(r'transcript_(?:bio)?type "([^"]+)"')
        with open(gtf_path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                columns = line.split("\t")
                if len(columns) <= 8 or columns[2] != "transcript":
                    continue
                biotype_match = btype_re.search(columns[8])
                if not biotype_match or biotype_match.group(1) != "lncRNA":
                    continue
                tid_match = tid_re.search(columns[8])
                if tid_match:
                    clean_tid = tid_match.group(1).split(".", 1)[0]
                    nc_tid_to_type[clean_tid] = biotype_match.group(1)
        print(
            f"  -> Extracted {len(nc_tid_to_type)} negative-control transcripts "
            f"across {len(set(nc_tid_to_type.values()))} biotypes."
        )
    else:
        print("  -> GTF path not provided or not found; transcript-type labels skipped.")

    print("Extracting ground-truth periodicity from dataset...")
    gt_records = {}
    for index in tqdm(range(len(dataset))):
        uuid, _, cell_type, _, meta_info, _, count_emb = dataset[index]
        uuid_str = str(uuid)
        tid = transcript_id_from_uuid(uuid_str)
        clean_tid = tid.split(".", 1)[0]
        cell_type = str(cell_type)
        gt_signal = np.expm1(to_1d_signal(count_emb))
        # Restrict periodicity to the elongating CDS. The separately annotated
        # stop codon is evaluated by the region and termination-profile metrics.
        bounds = cds_slice(meta_info, len(gt_signal))
        cds_start, cds_end = bounds if bounds is not None else (-1, -1)
        gt_periodicity = calculate_periodicity(gt_signal, cds_start, cds_end)
        if not np.isfinite(gt_periodicity):
            continue
        if clean_tid in hk_transcript_set:
            gene_type = "Housekeeping"
        elif clean_tid in nc_tid_to_type:
            gene_type = nc_tid_to_type[clean_tid]
        else:
            gene_type = "Other"
        gt_records[uuid_str] = {
            "Tid": tid,
            "Tid_clean": clean_tid,
            "Cell_Type": cell_type,
            "GT_Periodicity": gt_periodicity,
            "Gene_Type": gene_type,
            "CDS_Start": cds_start,
            "CDS_End": cds_end,
        }

    print(f"\nProcessing {pkl_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
    with open(pkl_path, "rb") as handle:
        preds_dict = pickle.load(handle)
    if not isinstance(preds_dict, dict):
        raise ValueError("Prediction pickle must contain a dictionary.")

    all_results = []
    for record in tqdm(gt_records.values(), desc="Matching Predictions"):
        pred_raw = get_prediction(
            preds_dict, record["Cell_Type"], record["Tid"]
        )
        if pred_raw is None:
            continue
        pred_signal = np.expm1(pred_raw.astype(np.float32, copy=False))
        if record["CDS_End"] != -1 and len(pred_signal) < record["CDS_End"]:
            continue
        pred_periodicity = calculate_periodicity(
            pred_signal, record["CDS_Start"], record["CDS_End"]
        )
        if not np.isfinite(pred_periodicity):
            continue
        all_results.append(
            {
                "Tid": record["Tid"],
                "Cell_Type": record["Cell_Type"],
                "GT_Periodicity": record["GT_Periodicity"],
                "Pred_Periodicity": pred_periodicity,
                "Gene_Type": record["Gene_Type"],
            }
        )

    columns = [
        "Tid",
        "Cell_Type",
        "GT_Periodicity",
        "Pred_Periodicity",
        "Gene_Type",
    ]
    df_final = pd.DataFrame(all_results, columns=columns)
    output_name = (
        f"periodicity_eval_results_{suffix}.csv"
        if suffix
        else "periodicity_eval_results.csv"
    )
    save_path = os.path.join(out_dir, output_name)
    df_final.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    plot_periodicity_scatter(df_final, out_dir, suffix)
    return df_final

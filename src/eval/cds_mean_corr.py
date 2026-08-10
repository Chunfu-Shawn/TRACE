"""Evaluate transcript-level translation-amplitude recovery by biotype."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

try:
    from .evaluation_utils import (
        cds_slice,
        get_prediction,
        load_prediction_input,
        to_1d_signal,
        transcript_id_from_uuid,
    )
except ImportError:
    from evaluation_utils import (
        cds_slice,
        get_prediction,
        load_prediction_input,
        to_1d_signal,
        transcript_id_from_uuid,
    )


BASES = np.asarray(list("ACGT"))
STOP_CODONS = {"TAA", "TAG", "TGA"}
DEFAULT_BIOTYPE_COLORS = {
    "protein_coding": "#176D9C",
    "lncRNA": "#C47A42",
    "nonsense_mediated_decay": "#7B6DA8",
    "retained_intron": "#5A9372",
    "processed_transcript": "#B18B49",
    "Unknown": "#8C8C8C",
    "Other": "#B5B5B5",
}


def load_transcript_biotypes(gtf_path: Optional[Union[str, os.PathLike]]) -> Dict[str, str]:
    """Parse version-free transcript IDs and biotypes from a GTF file."""
    if gtf_path is None:
        return {}
    path = Path(gtf_path)
    if not path.is_file():
        raise FileNotFoundError(f"GTF file not found: {path}")

    attribute_pattern = re.compile(r'([A-Za-z0-9_]+)\s+"([^"]*)"')
    biotypes = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 9 or columns[2] != "transcript":
                continue
            attributes = dict(attribute_pattern.findall(columns[8]))
            transcript_id = attributes.get("transcript_id")
            if not transcript_id:
                continue
            biotype = next(
                (
                    attributes[key]
                    for key in (
                        "transcript_type",
                        "transcript_biotype",
                        "gene_type",
                        "gene_biotype",
                    )
                    if attributes.get(key)
                ),
                "Unknown",
            )
            biotypes[transcript_id.split(".", 1)[0]] = biotype
    return biotypes


def decode_one_hot_sequence(sequence_embedding: object) -> str:
    """Decode an A/C/G/T one-hot matrix and represent invalid rows as N."""
    if hasattr(sequence_embedding, "detach"):
        values = sequence_embedding.detach().cpu().numpy()
    else:
        values = np.asarray(sequence_embedding)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Sequence embedding must be a two-dimensional matrix.")
    if values.shape[1] != 4 and values.shape[0] == 4:
        values = values.T
    if values.shape[1] != 4:
        raise ValueError("Sequence embedding must have four nucleotide channels.")

    valid = np.isfinite(values).all(axis=1) & (values.sum(axis=1) > 0)
    sequence = np.full(len(values), "N", dtype="<U1")
    sequence[valid] = BASES[np.argmax(values[valid], axis=1)]
    return "".join(sequence.tolist())


def find_longest_complete_orf(
    sequence: str,
    min_orf_codons: int = 10,
) -> Optional[Tuple[int, int]]:
    """Return the longest ATG-to-stop ORF as zero-based half-open coordinates."""
    if min_orf_codons < 2:
        raise ValueError("min_orf_codons must include at least start and stop codons.")
    sequence = str(sequence).upper().replace("U", "T")
    best = None
    for frame in range(3):
        active_starts = []
        for position in range(frame, len(sequence) - 2, 3):
            codon = sequence[position : position + 3]
            if codon == "ATG":
                active_starts.append(position)
            if codon not in STOP_CODONS:
                continue
            stop_end = position + 3
            for start in active_starts:
                codon_count = (stop_end - start) // 3
                if codon_count < min_orf_codons:
                    continue
                candidate = (start, stop_end)
                if best is None or (candidate[1] - candidate[0]) > (best[1] - best[0]):
                    best = candidate
            active_starts = []
    return best


def select_amplitude_region(
    meta_info: dict,
    sequence: str,
    aligned_length: int,
    min_orf_codons: int = 10,
) -> Tuple[int, int, str]:
    """Select annotated CDS, longest complete ORF, or full transcript region."""
    bounds = cds_slice(meta_info, aligned_length)
    if bounds is not None and bounds[1] - bounds[0] >= 3:
        return bounds[0], bounds[1], "annotated_CDS"

    longest_orf = find_longest_complete_orf(
        sequence[:aligned_length], min_orf_codons=min_orf_codons
    )
    if longest_orf is not None:
        return longest_orf[0], longest_orf[1], "longest_complete_ORF"
    if aligned_length < 3:
        raise ValueError("At least three aligned positions are required.")
    return 0, aligned_length, "transcript_wide"


def _to_linear_signal(signal: object, unlog_data: bool) -> np.ndarray:
    """Convert a signal to finite linear density without changing its length."""
    values = np.asarray(to_1d_signal(signal), dtype=np.float64)
    if unlog_data:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.expm1(values)
    values[~np.isfinite(values)] = np.nan
    values[values < 0] = 0.0
    return values


def _mean_signal(signal: np.ndarray, start: int, end: int) -> float:
    """Return the finite mean signal in a half-open region."""
    region = np.asarray(signal[start:end], dtype=np.float64)
    region = region[np.isfinite(region)]
    return float(np.mean(region)) if region.size else float("nan")


def _safe_correlations(observed: Sequence[float], predicted: Sequence[float]) -> dict:
    """Return Pearson and Spearman statistics for finite nonconstant pairs."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    result = {
        "N": int(len(observed)),
        "Pearson_R": float("nan"),
        "Pearson_P": float("nan"),
        "Spearman_R": float("nan"),
        "Spearman_P": float("nan"),
        "MAE_Log1p": float("nan"),
    }
    if len(observed):
        result["MAE_Log1p"] = float(np.mean(np.abs(predicted - observed)))
    if len(observed) < 2 or np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        return result
    pearson = pearsonr(observed, predicted)
    spearman = spearmanr(observed, predicted)
    result.update(
        {
            "Pearson_R": float(pearson.statistic),
            "Pearson_P": float(pearson.pvalue),
            "Spearman_R": float(spearman.statistic),
            "Spearman_P": float(spearman.pvalue),
        }
    )
    return result


def _bootstrap_spearman_ci(
    observed: Sequence[float],
    predicted: Sequence[float],
    n_bootstrap: int,
    random_state: int,
) -> Tuple[float, float]:
    """Return a transcript-bootstrap 95% CI for Spearman correlation."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if len(observed) < 3 or n_bootstrap <= 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    estimates = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(observed), size=len(observed))
        sampled_observed = observed[indices]
        sampled_predicted = predicted[indices]
        if np.ptp(sampled_observed) == 0 or np.ptp(sampled_predicted) == 0:
            continue
        estimate = spearmanr(sampled_observed, sampled_predicted).statistic
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if len(estimates) < max(20, int(0.1 * n_bootstrap)):
        return float("nan"), float("nan")
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def aggregate_by_transcript(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Average cell-type observations so each transcript contributes once."""
    required = {
        "Tid_Clean",
        "Biotype",
        "Region_Source",
        "Observed_Mean_Linear",
        "Predicted_Mean_Linear",
        "Cell_Type",
    }
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"Missing sample-level columns: {sorted(missing)}")
    if sample_df.empty:
        return pd.DataFrame()

    transcript_df = (
        sample_df.groupby(
            ["Tid_Clean", "Biotype", "Region_Source"],
            as_index=False,
            observed=True,
        )
        .agg(
            Tid=("Tid", "first"),
            Cell_Type_Count=("Cell_Type", "nunique"),
            Sample_Count=("Cell_Type", "size"),
            Region_Length=("Region_Length", "first"),
            Observed_Mean_Linear=("Observed_Mean_Linear", "mean"),
            Predicted_Mean_Linear=("Predicted_Mean_Linear", "mean"),
        )
    )
    transcript_df["Observed_Mean_Log1p"] = np.log1p(
        transcript_df["Observed_Mean_Linear"].clip(lower=0)
    )
    transcript_df["Predicted_Mean_Log1p"] = np.log1p(
        transcript_df["Predicted_Mean_Linear"].clip(lower=0)
    )
    transcript_df["Absolute_Error_Log1p"] = np.abs(
        transcript_df["Predicted_Mean_Log1p"]
        - transcript_df["Observed_Mean_Log1p"]
    )
    return transcript_df


def summarize_by_biotype(
    transcript_df: pd.DataFrame,
    min_biotype_n: int = 20,
    max_biotypes: int = 8,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse rare biotypes for plotting and calculate correlation summaries."""
    if transcript_df.empty:
        return transcript_df.copy(), pd.DataFrame()
    if min_biotype_n < 2:
        raise ValueError("min_biotype_n must be at least 2.")
    if max_biotypes < 3:
        raise ValueError("max_biotypes must be at least 3.")

    plot_df = transcript_df.copy()
    counts = plot_df["Biotype"].value_counts()
    eligible = counts[counts >= min_biotype_n].index.tolist()
    priority = []
    for priority_type in ("protein_coding", "lncRNA"):
        if counts.get(priority_type, 0) >= 3:
            priority.append(priority_type)
    ordered_candidates = priority + [
        biotype for biotype in eligible if biotype not in priority
    ]
    if len(ordered_candidates) > max_biotypes:
        ordered_candidates = ordered_candidates[: max_biotypes - 1]
    keep = set(ordered_candidates)
    plot_df["Plot_Biotype"] = plot_df["Biotype"].where(
        plot_df["Biotype"].isin(keep), "Other"
    )

    rows = []
    for index, (biotype, group) in enumerate(
        plot_df.groupby("Plot_Biotype", sort=False, observed=True)
    ):
        observed = group["Observed_Mean_Log1p"].to_numpy(dtype=float)
        predicted = group["Predicted_Mean_Log1p"].to_numpy(dtype=float)
        stats = _safe_correlations(observed, predicted)
        ci_low, ci_high = _bootstrap_spearman_ci(
            observed,
            predicted,
            n_bootstrap=n_bootstrap,
            random_state=random_state + index,
        )
        rows.append(
            {
                "Biotype": biotype,
                **stats,
                "Spearman_CI95_Low": ci_low,
                "Spearman_CI95_High": ci_high,
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["N", "Spearman_R"], ascending=[False, False], na_position="last"
    )
    return plot_df, summary


def _biotype_colors(biotypes: Sequence[str]) -> Dict[str, str]:
    """Return a stable restrained color map for displayed biotypes."""
    fallback = plt.get_cmap("tab20").colors
    colors = {}
    for index, biotype in enumerate(biotypes):
        colors[str(biotype)] = DEFAULT_BIOTYPE_COLORS.get(
            str(biotype), mpl.colors.to_hex(fallback[index % len(fallback)])
        )
    return colors


def _display_biotype(biotype: object) -> str:
    """Convert machine-readable GTF biotypes to compact figure labels."""
    return str(biotype).replace("_", " ")


def plot_amplitude_correlation_by_biotype(
    transcript_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: Union[str, os.PathLike],
    suffix: str = "",
) -> Optional[Path]:
    """Plot transcript amplitude agreement and biotype-stratified correlations."""
    required = {
        "Plot_Biotype",
        "Observed_Mean_Log1p",
        "Predicted_Mean_Log1p",
    }
    missing = required - set(transcript_df.columns)
    if missing:
        raise ValueError(f"Missing plotting columns: {sorted(missing)}")
    input_count = len(transcript_df)
    plot_df = transcript_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Observed_Mean_Log1p", "Predicted_Mean_Log1p"]
    )
    print(
        f"Amplitude plotting pairs: {len(plot_df):,}/{input_count:,} transcripts "
        "after finite-value filtering."
    )
    if plot_df.empty or summary_df.empty:
        print("No finite amplitude pairs available for plotting.")
        return None

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    category_order = summary_df["Biotype"].astype(str).tolist()
    colors = _biotype_colors(category_order)

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure, (scatter_axis, summary_axis) = plt.subplots(
            1,
            2,
            figsize=(7.0, 3.25),
            gridspec_kw={"width_ratios": [1.25, 1.0]},
        )

        for biotype in category_order:
            group = plot_df[plot_df["Plot_Biotype"].astype(str) == biotype]
            if group.empty:
                continue
            scatter_axis.scatter(
                group["Observed_Mean_Log1p"],
                group["Predicted_Mean_Log1p"],
                s=8,
                alpha=0.35,
                color=colors[biotype],
                edgecolors="none",
                label=f"{_display_biotype(biotype)} (n={len(group):,})",
                rasterized=True,
            )
        lower = float(
            min(plot_df["Observed_Mean_Log1p"].min(), plot_df["Predicted_Mean_Log1p"].min())
        )
        upper = float(
            max(plot_df["Observed_Mean_Log1p"].max(), plot_df["Predicted_Mean_Log1p"].max())
        )
        padding = max(0.05 * (upper - lower), 0.05)
        limits = (lower - padding, upper + padding)
        scatter_axis.plot(limits, limits, color="#555555", linewidth=0.9, linestyle="--")
        scatter_axis.set_xlim(limits)
        scatter_axis.set_ylim(limits)
        scatter_axis.set_aspect("equal", adjustable="box")
        scatter_axis.set_xlabel("Observed mean translation signal, log1p")
        scatter_axis.set_ylabel("Predicted mean translation signal, log1p")
        scatter_axis.set_title("a  Transcript-level signal amplitude", loc="left", fontweight="bold")
        scatter_axis.grid(color="#E5E5E5", linewidth=0.6, zorder=0)
        overall = _safe_correlations(
            plot_df["Observed_Mean_Log1p"], plot_df["Predicted_Mean_Log1p"]
        )
        scatter_axis.text(
            0.04,
            0.96,
            (
                f"Spearman $R$ = {overall['Spearman_R']:.3f}\n"
                f"Pearson $R$ = {overall['Pearson_R']:.3f}\n"
                f"n = {overall['N']:,} transcripts"
            ),
            transform=scatter_axis.transAxes,
            ha="left",
            va="top",
        )
        scatter_axis.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.20),
            ncol=2,
            handletextpad=0.4,
            columnspacing=0.8,
        )

        summary_plot = summary_df.iloc[::-1].reset_index(drop=True)
        y_positions = np.arange(len(summary_plot))
        for position, row in summary_plot.iterrows():
            biotype = str(row["Biotype"])
            correlation = float(row["Spearman_R"])
            if not np.isfinite(correlation):
                continue
            ci_low = float(row["Spearman_CI95_Low"])
            ci_high = float(row["Spearman_CI95_High"])
            xerr = None
            if np.isfinite(ci_low) and np.isfinite(ci_high):
                xerr = np.asarray(
                    [
                        [max(correlation - ci_low, 0.0)],
                        [max(ci_high - correlation, 0.0)],
                    ]
                )
            summary_axis.errorbar(
                correlation,
                position,
                xerr=xerr,
                fmt="o",
                color=colors.get(biotype, "#777777"),
                markersize=5,
                capsize=2.5,
                linewidth=1.1,
                zorder=3,
            )
            summary_axis.text(
                1.02,
                position,
                f"n={int(row['N']):,}",
                va="center",
                fontsize=6.5,
                clip_on=False,
            )
        summary_axis.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        summary_axis.set_yticks(
            y_positions,
            [_display_biotype(value) for value in summary_plot["Biotype"]],
        )
        summary_axis.set_xlim(-1.05, 1.15)
        summary_axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        summary_axis.set_xlabel("Spearman correlation")
        summary_axis.set_title("b  Correlation by transcript biotype", loc="left", fontweight="bold")
        summary_axis.grid(axis="x", color="#E5E5E5", linewidth=0.6, zorder=0)

        figure.subplots_adjust(left=0.09, right=0.94, bottom=0.28, top=0.90, wspace=0.42)
        stem = "cds_mean_amplitude_by_biotype"
        if suffix:
            stem += f".{suffix}"
        pdf_path = output_dir / f"{stem}.pdf"
        figure.savefig(pdf_path, bbox_inches="tight")
        plt.close(figure)
    print(
        "Amplitude correlation figure saved to "
        f"{pdf_path}"
    )
    return pdf_path


def evaluate_cds_mean_correlation(
    dataset,
    pkl_input,
    gtf_path: Optional[Union[str, os.PathLike]] = None,
    out_dir: Union[str, os.PathLike] = "./results/cds_mean_correlation",
    suffix: str = "",
    unlog_data: bool = True,
    min_orf_codons: int = 10,
    min_biotype_n: int = 20,
    max_biotypes: int = 8,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate mean translation-signal amplitude across transcripts and biotypes.

    Annotated transcripts use their CDS. Transcripts without CDS annotations use
    the longest complete ORF when available and otherwise use the full transcript.
    The latter values are equivalent translation-amplitude proxies, not canonical
    RNA-normalized translation efficiencies.
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(pkl_input, dict) and all(
        isinstance(value, dict) for value in pkl_input.values()
    ):
        predictions = pkl_input
    else:
        predictions = load_prediction_input(pkl_input)
    biotype_map = load_transcript_biotypes(gtf_path)
    records = []

    print("--- Evaluating transcript-level translation amplitude ---")
    for index in tqdm(range(len(dataset)), desc="Matching observations and predictions"):
        uuid, _, cell_type, _, meta_info, seq_emb, count_emb = dataset[index]
        tid = transcript_id_from_uuid(uuid)
        clean_tid = tid.split(".", 1)[0]
        cell_type = str(cell_type)
        prediction = get_prediction(predictions, cell_type, tid)
        if prediction is None:
            continue

        observed = _to_linear_signal(count_emb, unlog_data=unlog_data)
        predicted = _to_linear_signal(prediction, unlog_data=unlog_data)
        aligned_length = min(len(observed), len(predicted))
        if aligned_length < 3:
            continue
        observed = observed[:aligned_length]
        predicted = predicted[:aligned_length]
        try:
            sequence = decode_one_hot_sequence(seq_emb)[:aligned_length]
            start, end, region_source = select_amplitude_region(
                meta_info,
                sequence,
                aligned_length,
                min_orf_codons=min_orf_codons,
            )
        except ValueError:
            continue

        observed_mean = _mean_signal(observed, start, end)
        predicted_mean = _mean_signal(predicted, start, end)
        if not np.isfinite(observed_mean) or not np.isfinite(predicted_mean):
            continue
        records.append(
            {
                "UUID": str(uuid),
                "Tid": tid,
                "Tid_Clean": clean_tid,
                "Cell_Type": cell_type,
                "Biotype": biotype_map.get(clean_tid, "Unknown"),
                "Region_Source": region_source,
                "Region_Start_0based": start,
                "Region_End_0based_Exclusive": end,
                "Region_Length": end - start,
                "Observed_Mean_Linear": observed_mean,
                "Predicted_Mean_Linear": predicted_mean,
                "Observed_Mean_Log1p": float(np.log1p(max(observed_mean, 0.0))),
                "Predicted_Mean_Log1p": float(np.log1p(max(predicted_mean, 0.0))),
                "Absolute_Error_Log1p": float(
                    abs(np.log1p(max(predicted_mean, 0.0)) - np.log1p(max(observed_mean, 0.0)))
                ),
            }
        )

    sample_df = pd.DataFrame.from_records(records)
    if sample_df.empty:
        print("No matched finite amplitude pairs were found.")
        return sample_df, pd.DataFrame(), pd.DataFrame()

    transcript_df = aggregate_by_transcript(sample_df)
    plot_df, summary_df = summarize_by_biotype(
        transcript_df,
        min_biotype_n=min_biotype_n,
        max_biotypes=max_biotypes,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    tag = f".{suffix}" if suffix else ""
    sample_path = output_dir / f"cds_mean_amplitude_samples{tag}.csv"
    transcript_path = output_dir / f"cds_mean_amplitude_transcripts{tag}.csv"
    summary_path = output_dir / f"cds_mean_amplitude_by_biotype{tag}.csv"
    sample_df.to_csv(sample_path, index=False)
    plot_df.to_csv(transcript_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Sample-level source data saved to {sample_path}")
    print(f"Transcript-level source data saved to {transcript_path}")
    print(f"Biotype summary saved to {summary_path}")
    plot_amplitude_correlation_by_biotype(
        plot_df,
        summary_df,
        out_dir=output_dir,
        suffix=suffix,
    )
    return sample_df, plot_df, summary_df

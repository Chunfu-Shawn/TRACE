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
import seaborn as sns
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
GENE_TYPE_ORDER = ["Other", "lncRNA", "Housekeeping"]
GENE_TYPE_COLORS = {
    "Other": "#B0B0B0",
    "lncRNA": "#3498DB",
    "Housekeeping": "#E74C3C",
}
DEFAULT_HOUSEKEEPING_PATH = (
    "/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/"
    "other_gene_list/housekeeping_genes.tsv"
)


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


def load_housekeeping_transcripts(
    hk_genes_path: Optional[Union[str, os.PathLike]],
) -> set:
    """Load version-free housekeeping transcript IDs from a CSV or TSV file."""
    if hk_genes_path is None:
        return set()
    path = Path(hk_genes_path)
    if not path.is_file():
        print(f"Housekeeping transcript file not found; labels skipped: {path}")
        return set()
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=separator)
    transcript_column = next(
        (
            column
            for column in ("Transcript ID", "Transcript_ID", "transcript_id", "Tid")
            if column in frame.columns
        ),
        None,
    )
    if transcript_column is None:
        raise ValueError(
            "Housekeeping file must contain Transcript ID, Transcript_ID, "
            "transcript_id, or Tid."
        )
    return set(
        frame[transcript_column]
        .dropna()
        .astype(str)
        .str.split(".", regex=False)
        .str[0]
    )


def classify_gene_type(
    transcript_id: str,
    biotype: str,
    housekeeping_transcripts: set,
) -> str:
    """Map a transcript to Housekeeping, lncRNA, or Other."""
    clean_tid = str(transcript_id).split(".", 1)[0]
    if clean_tid in housekeeping_transcripts:
        return "Housekeeping"
    if str(biotype) == "lncRNA":
        return "lncRNA"
    return "Other"


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


def _natural_log_positive(values: object) -> np.ndarray:
    """Return natural logs for strictly positive finite values and NaN otherwise."""
    values = np.asarray(values, dtype=np.float64)
    logged = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0)
    logged[valid] = np.log(values[valid])
    return logged


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
        "MAE_Log": float("nan"),
    }
    if len(observed):
        result["MAE_Log"] = float(np.mean(np.abs(predicted - observed)))
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
        "Gene_Type",
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
            ["Tid_Clean", "Biotype", "Gene_Type"],
            as_index=False,
            observed=True,
        )
        .agg(
            Tid=("Tid", "first"),
            Cell_Type_Count=("Cell_Type", "nunique"),
            Sample_Count=("Cell_Type", "size"),
            Region_Source=("Region_Source", "first"),
            Region_Length=("Region_Length", "first"),
            Observed_Mean_Linear=("Observed_Mean_Linear", "mean"),
            Predicted_Mean_Linear=("Predicted_Mean_Linear", "mean"),
        )
    )
    transcript_df["Observed_Mean_Log"] = _natural_log_positive(
        transcript_df["Observed_Mean_Linear"]
    )
    transcript_df["Predicted_Mean_Log"] = _natural_log_positive(
        transcript_df["Predicted_Mean_Linear"]
    )
    transcript_df["Absolute_Error_Log"] = np.abs(
        transcript_df["Predicted_Mean_Log"]
        - transcript_df["Observed_Mean_Log"]
    )
    return transcript_df


def summarize_by_biotype(
    transcript_df: pd.DataFrame,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate correlation and amplitude summaries for three transcript types."""
    if transcript_df.empty:
        return transcript_df.copy(), pd.DataFrame()

    plot_df = transcript_df.copy()
    plot_df["Gene_Type"] = pd.Categorical(
        plot_df["Gene_Type"], categories=GENE_TYPE_ORDER, ordered=True
    )

    rows = []
    for index, (gene_type, group) in enumerate(
        plot_df.groupby("Gene_Type", sort=True, observed=True)
    ):
        observed = group["Observed_Mean_Log"].to_numpy(dtype=float)
        predicted = group["Predicted_Mean_Log"].to_numpy(dtype=float)
        valid = np.isfinite(observed) & np.isfinite(predicted)
        observed = observed[valid]
        predicted = predicted[valid]
        stats = _safe_correlations(observed, predicted)
        ci_low, ci_high = _bootstrap_spearman_ci(
            observed,
            predicted,
            n_bootstrap=n_bootstrap,
            random_state=random_state + index,
        )
        rows.append(
            {
                "Gene_Type": str(gene_type),
                **stats,
                "Observed_Median_Log": (
                    float(np.median(observed)) if observed.size else float("nan")
                ),
                "Predicted_Median_Log": (
                    float(np.median(predicted)) if predicted.size else float("nan")
                ),
                "Spearman_CI95_Low": ci_low,
                "Spearman_CI95_High": ci_high,
            }
        )
    summary = pd.DataFrame(rows)
    return plot_df, summary


def plot_amplitude_scatter(
    transcript_df: pd.DataFrame,
    out_dir: Union[str, os.PathLike],
    suffix: str = "",
) -> Optional[Path]:
    """Plot observed/predicted amplitude with transcript-type marginal densities."""
    required = {
        "Gene_Type",
        "Observed_Mean_Log",
        "Predicted_Mean_Log",
    }
    missing = required - set(transcript_df.columns)
    if missing:
        raise ValueError(f"Missing plotting columns: {sorted(missing)}")
    input_count = len(transcript_df)
    plot_df = transcript_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Observed_Mean_Log", "Predicted_Mean_Log"]
    )
    print(
        f"Amplitude plotting pairs: {len(plot_df):,}/{input_count:,} transcripts "
        "after finite-value filtering."
    )
    if plot_df.empty:
        print("No finite amplitude pairs available for plotting.")
        return None

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    present_types = set(plot_df["Gene_Type"].dropna().astype(str))
    category_order = [value for value in GENE_TYPE_ORDER if value in present_types]
    plot_df["Gene_Type"] = pd.Categorical(
        plot_df["Gene_Type"], categories=category_order, ordered=True
    )
    plot_df = plot_df.sort_values("Gene_Type")
    color_map = {key: GENE_TYPE_COLORS[key] for key in category_order}

    overall = _safe_correlations(
        plot_df["Observed_Mean_Log"], plot_df["Predicted_Mean_Log"]
    )
    p_value = overall["Spearman_P"]
    p_text = (
        f"{p_value:.1e}"
        if np.isfinite(p_value) and p_value < 0.001
        else f"{p_value:.3f}"
    )
    stats_text = (
        f"Spearman $R$ = {overall['Spearman_R']:.3f}\n"
        f"$P$ = {p_text}\n"
        f"$n$ = {overall['N']:,} transcripts"
    )

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        grid = sns.JointGrid(
            data=plot_df,
            x="Observed_Mean_Log",
            y="Predicted_Mean_Log",
            hue="Gene_Type",
            palette=color_map,
            height=6,
            ratio=6,
            space=0,
        )
        grid.plot_joint(
            sns.scatterplot,
            alpha=0.4,
            s=15,
            edgecolor="none",
            rasterized=True,
        )
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
                x="Observed_Mean_Log",
                hue="Gene_Type",
                palette=color_map,
                element="step",
                fill=False,
                legend=False,
                ax=grid.ax_marg_x,
            )
            sns.histplot(
                data=plot_df,
                y="Predicted_Mean_Log",
                hue="Gene_Type",
                palette=color_map,
                element="step",
                fill=False,
                legend=False,
                ax=grid.ax_marg_y,
            )

        lower = float(
            min(
                plot_df["Observed_Mean_Log"].min(),
                plot_df["Predicted_Mean_Log"].min(),
            )
        )
        upper = float(
            max(
                plot_df["Observed_Mean_Log"].max(),
                plot_df["Predicted_Mean_Log"].max(),
            )
        )
        padding = max(0.05 * (upper - lower), 0.05)
        limits = (lower - padding, upper + padding)
        grid.ax_joint.plot(
            limits,
            limits,
            color="#6B6B6B",
            linestyle="--",
            linewidth=1.5,
            zorder=10,
        )
        grid.ax_joint.set_xlim(limits)
        grid.ax_joint.set_ylim(limits)
        grid.ax_joint.set_aspect("equal", adjustable="box")
        grid.ax_joint.grid(
            True,
            color="lightgray",
            linestyle="-",
            linewidth=1.0,
            alpha=0.8,
        )
        grid.ax_joint.set_axisbelow(True)
        grid.ax_joint.set_xlabel("Observed TE proxy (ln)", fontsize=14)
        grid.ax_joint.set_ylabel("Predicted TE proxy (ln)", fontsize=14)
        grid.ax_joint.text(
            0.95,
            0.05,
            stats_text,
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

        stem = "cds_mean_amplitude_scatter_density"
        if suffix:
            stem += f".{suffix}"
        pdf_path = output_dir / f"{stem}.pdf"
        svg_path = output_dir / f"{stem}.svg"
        png_path = output_dir / f"{stem}.png"
        tiff_path = output_dir / f"{stem}.tiff"
        grid.savefig(pdf_path, bbox_inches="tight")
        grid.figure.savefig(svg_path, bbox_inches="tight")
        grid.figure.savefig(png_path, dpi=300, bbox_inches="tight")
        grid.figure.savefig(tiff_path, dpi=600, bbox_inches="tight")
        plt.close(grid.figure)
    print(
        "Amplitude correlation figure saved to "
        f"{pdf_path}, {svg_path}, {png_path}, and {tiff_path}"
    )
    return pdf_path


def evaluate_cds_mean_correlation(
    dataset,
    pkl_input,
    hk_genes_path: Optional[Union[str, os.PathLike]] = DEFAULT_HOUSEKEEPING_PATH,
    gtf_path: Optional[Union[str, os.PathLike]] = None,
    out_dir: Union[str, os.PathLike] = "./results/cds_mean_correlation",
    suffix: str = "",
    unlog_data: bool = True,
    min_orf_codons: int = 10,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate mean translation-signal amplitude across transcripts and biotypes.

    Annotated transcripts use their CDS. Transcripts without CDS annotations use
    the longest complete ORF when available and otherwise use the full transcript.
    The latter values are equivalent translation-amplitude proxies, not canonical
    RNA-normalized translation efficiencies. Plot groups follow the periodicity
    evaluation convention: Other, lncRNA, and Housekeeping.
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
    housekeeping_transcripts = load_housekeeping_transcripts(hk_genes_path)
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
        biotype = biotype_map.get(clean_tid, "Unknown")
        gene_type = classify_gene_type(
            clean_tid,
            biotype,
            housekeeping_transcripts,
        )
        observed_log = float(_natural_log_positive(observed_mean))
        predicted_log = float(_natural_log_positive(predicted_mean))
        records.append(
            {
                "UUID": str(uuid),
                "Tid": tid,
                "Tid_Clean": clean_tid,
                "Cell_Type": cell_type,
                "Biotype": biotype,
                "Gene_Type": gene_type,
                "Region_Source": region_source,
                "Region_Start_0based": start,
                "Region_End_0based_Exclusive": end,
                "Region_Length": end - start,
                "Observed_Mean_Linear": observed_mean,
                "Predicted_Mean_Linear": predicted_mean,
                "Observed_Mean_Log": observed_log,
                "Predicted_Mean_Log": predicted_log,
                "Absolute_Error_Log": (
                    abs(predicted_log - observed_log)
                    if np.isfinite(observed_log) and np.isfinite(predicted_log)
                    else float("nan")
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
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    tag = f".{suffix}" if suffix else ""
    sample_path = output_dir / f"cds_mean_amplitude_samples{tag}.csv"
    transcript_path = output_dir / f"cds_mean_amplitude_transcripts{tag}.csv"
    summary_path = output_dir / f"cds_mean_amplitude_by_gene_type{tag}.csv"
    sample_df.to_csv(sample_path, index=False)
    plot_df.to_csv(transcript_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Sample-level source data saved to {sample_path}")
    print(f"Transcript-level source data saved to {transcript_path}")
    print(f"Transcript-type summary saved to {summary_path}")
    plot_amplitude_scatter(
        plot_df,
        out_dir=output_dir,
        suffix=suffix,
    )
    return sample_df, plot_df, summary_df

import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, hypergeom, mannwhitneyu


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


DEFAULT_RBP_ALIASES = {
    "HUD": "ELAVL4",
    "TTP": "ZFP36",
    "BRF1": "ZFP36L1",
    "IMP1": "IGF2BP1",
    "ZBP1": "IGF2BP1",
}


def _load_table(table_or_path):
    """Load a tabular input while preserving DataFrame inputs."""
    if isinstance(table_or_path, pd.DataFrame):
        return table_or_path.copy()
    path = os.fspath(table_or_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input table not found: {path}")
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def _normalize_rbp_symbol(value, aliases=None):
    """Normalize an RBP symbol without guessing gene-family membership."""
    if pd.isna(value):
        return None
    alias_mapping = dict(DEFAULT_RBP_ALIASES)
    if aliases:
        alias_mapping.update({
            str(key).upper(): str(mapped).upper()
            for key, mapped in aliases.items()
        })
    symbol = str(value).strip().upper()
    symbol = re.sub(r"\s+", " ", symbol)
    symbol = symbol.split("(", 1)[0].strip()
    symbol = alias_mapping.get(symbol, symbol)
    return symbol or None


def _expand_annotation_name(value, aliases=None):
    """Expand compact symbols such as PUM1/2 into explicit gene symbols."""
    normalized = _normalize_rbp_symbol(value, aliases=aliases)
    if normalized is None:
        return []
    compact_match = re.fullmatch(r"([A-Z]+)(\d+)/(\d+)", normalized)
    if compact_match:
        prefix, first_number, second_number = compact_match.groups()
        return [f"{prefix}{first_number}", f"{prefix}{second_number}"]
    return [normalized]


def _normalize_direction(value):
    """Map literature direction labels to three reproducible categories."""
    if pd.isna(value):
        return "Unknown"
    label = str(value).strip().lower()
    if "bidirectional" in label or "context" in label:
        return "Context-dependent"
    if "positive" in label:
        return "Positive"
    if "negative" in label:
        return "Negative"
    return "Unknown"


def _benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p values with NaN preservation."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    valid_values = values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    n_tests = len(ranked)
    ranked_adjusted = ranked * n_tests / np.arange(1, n_tests + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)
    restored = np.empty(n_tests, dtype=float)
    restored[order] = ranked_adjusted
    adjusted[valid] = restored
    return adjusted


def _odds_ratio_interval(a, b, c, d, alpha=0.05):
    """Calculate a continuity-corrected log-odds confidence interval."""
    cells = np.asarray([a, b, c, d], dtype=float)
    if np.any(cells == 0):
        cells = cells + 0.5
    a, b, c, d = cells
    odds_ratio = (a * d) / (b * c)
    standard_error = np.sqrt(np.sum(1.0 / cells))
    z_value = 1.959963984540054
    lower = np.exp(np.log(odds_ratio) - z_value * standard_error)
    upper = np.exp(np.log(odds_ratio) + z_value * standard_error)
    return float(odds_ratio), float(lower), float(upper)


def _wilson_interval(successes, total, alpha=0.05):
    """Calculate a two-sided Wilson interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    z_value = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z_value ** 2 / total
    center = (
        proportion + z_value ** 2 / (2.0 * total)
    ) / denominator
    half_width = z_value * np.sqrt(
        proportion * (1.0 - proportion) / total
        + z_value ** 2 / (4.0 * total ** 2)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _running_enrichment(labels):
    """Calculate an unweighted preranked running enrichment curve."""
    labels = np.asarray(labels, dtype=bool)
    n_positive = int(labels.sum())
    n_negative = int((~labels).sum())
    if n_positive == 0 or n_negative == 0:
        return np.full(len(labels), np.nan), np.nan, None
    increments = np.where(
        labels,
        1.0 / n_positive,
        -1.0 / n_negative,
    )
    running = np.cumsum(increments)
    peak_index = int(np.nanargmax(running))
    return running, float(running[peak_index]), peak_index


def _prepare_literature_annotations(
        annotation_table,
        name_col="Name",
        direction_col="Translation_direction",
        aliases=None):
    """Expand aliases and collapse duplicate literature annotations."""
    required = {name_col, direction_col}
    missing = required.difference(annotation_table.columns)
    if missing:
        raise ValueError(
            f"Literature annotation table is missing columns: {sorted(missing)}"
        )
    records = []
    for row in annotation_table.itertuples(index=False):
        raw_name = getattr(row, name_col)
        raw_direction = getattr(row, direction_col)
        for symbol in _expand_annotation_name(raw_name, aliases=aliases):
            records.append({
                "RBP_Name": symbol,
                "Annotation_Name": str(raw_name),
                "Literature_Direction": _normalize_direction(raw_direction),
            })
    expanded = pd.DataFrame(records)
    if expanded.empty:
        raise ValueError("No valid RBP symbols were found in the annotation table.")

    collapsed_records = []
    for rbp_name, group in expanded.groupby("RBP_Name", sort=True):
        directions = set(group["Literature_Direction"])
        directional = directions.intersection({"Positive", "Negative"})
        if "Context-dependent" in directions or len(directional) > 1:
            consensus_direction = "Context-dependent"
        elif len(directional) == 1:
            consensus_direction = next(iter(directional))
        else:
            consensus_direction = "Unknown"
        collapsed_records.append({
            "RBP_Name": rbp_name,
            "Literature_Direction": consensus_direction,
            "Annotation_Names": "; ".join(
                sorted(set(group["Annotation_Name"]))
            ),
            "N_Annotation_Rows": int(len(group)),
        })
    return pd.DataFrame(collapsed_records)


def evaluate_translation_rbp_rank_enrichment(
        model_summary,
        translation_rbp_table,
        score_col="Median_Delta_Log2_TE",
        fdr_col="FDR_BH",
        n_col="N_Transcripts",
        rbp_col="RBP_Name",
        region_col="Region",
        annotation_name_col="Name",
        annotation_direction_col="Translation_direction",
        min_transcripts=5,
        fdr_threshold=0.05,
        rank_score="absolute_effect",
        region_selection="max_abs_effect",
        top_ks=(10, 20, 30, 50),
        n_permutations=10000,
        random_state=42,
        aliases=None):
    """Evaluate whether known translation RBPs concentrate near model ranks.

    Literature-annotated RBPs are treated as a positive-unlabeled reference
    set, not as a complete binary ground truth. Model rows are collapsed to one
    row per RBP before any test to avoid counting multiple regions as
    independent observations.
    """
    model_df = _load_table(model_summary)
    annotation_df = _load_table(translation_rbp_table)
    model_required = {rbp_col, region_col, score_col, fdr_col, n_col}
    model_missing = model_required.difference(model_df.columns)
    if model_missing:
        raise ValueError(
            f"Model summary is missing columns: {sorted(model_missing)}"
        )
    if int(min_transcripts) < 1:
        raise ValueError("min_transcripts must be at least 1.")
    if not 0 < float(fdr_threshold) <= 1:
        raise ValueError("fdr_threshold must be within (0, 1].")
    if int(n_permutations) < 0:
        raise ValueError("n_permutations cannot be negative.")

    rank_score = str(rank_score).lower()
    valid_rank_scores = {
        "absolute_effect": "Absolute_Effect",
        "evidence": "Evidence_Score",
        "significance": "Significance_Score",
    }
    if rank_score not in valid_rank_scores:
        raise ValueError(
            "rank_score must be 'absolute_effect', 'evidence', or "
            "'significance'."
        )
    region_selection = str(region_selection).lower()
    if region_selection not in {"max_abs_effect", "max_evidence"}:
        raise ValueError(
            "region_selection must be 'max_abs_effect' or 'max_evidence'."
        )

    working = model_df[
        [rbp_col, region_col, score_col, fdr_col, n_col]
    ].copy()
    working["RBP_Name"] = working[rbp_col].map(
        lambda value: _normalize_rbp_symbol(value, aliases=aliases)
    )
    working["Effect"] = pd.to_numeric(working[score_col], errors="coerce")
    working["FDR"] = pd.to_numeric(working[fdr_col], errors="coerce")
    working["N_Transcripts"] = pd.to_numeric(
        working[n_col], errors="coerce"
    )
    before_filter = len(working)
    working = working.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["RBP_Name", "Effect", "FDR", "N_Transcripts"]
    )
    working = working[working["N_Transcripts"] >= int(min_transcripts)].copy()
    if working.empty:
        raise ValueError("No model RBP rows passed the requested filters.")
    working["Absolute_Effect"] = working["Effect"].abs()
    working["Significance_Score"] = -np.log10(
        working["FDR"].clip(lower=np.finfo(float).tiny, upper=1.0)
    )
    working["Evidence_Score"] = (
        working["Absolute_Effect"] * working["Significance_Score"]
    )

    rbp_aggregates = working.groupby("RBP_Name", as_index=False).agg(
        Min_FDR_All_Regions=("FDR", "min"),
        N_Regions_Tested=(region_col, "nunique"),
    )
    rbp_aggregates["Any_Region_Significant"] = (
        rbp_aggregates["Min_FDR_All_Regions"] <= float(fdr_threshold)
    )
    selection_column = (
        "Absolute_Effect"
        if region_selection == "max_abs_effect"
        else "Evidence_Score"
    )
    selected = (
        working.sort_values(
            ["RBP_Name", selection_column, "FDR"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("RBP_Name", keep="first")
        .rename(columns={region_col: "Selected_Region"})
    )
    selected = selected.merge(rbp_aggregates, on="RBP_Name", how="left")

    annotations = _prepare_literature_annotations(
        annotation_df,
        name_col=annotation_name_col,
        direction_col=annotation_direction_col,
        aliases=aliases,
    )
    tested_names = set(selected["RBP_Name"])
    annotations["Tested_By_Model"] = annotations["RBP_Name"].isin(tested_names)
    unmatched_annotations = annotations[
        ~annotations["Tested_By_Model"]
    ].copy()
    matched_annotations = annotations[
        annotations["Tested_By_Model"]
    ].drop(columns="Tested_By_Model")

    ranked = selected.merge(
        matched_annotations,
        on="RBP_Name",
        how="left",
    )
    ranked["Literature_Annotated"] = ranked[
        "Literature_Direction"
    ].notna()
    rank_column = valid_rank_scores[rank_score]
    ranked = ranked.sort_values(
        [rank_column, "Absolute_Effect", "RBP_Name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    ranked["Rank_Percentile"] = ranked["Rank"] / len(ranked)
    ranked["Model_Direction"] = np.where(
        ranked["Effect"] > 0,
        "Positive",
        np.where(ranked["Effect"] < 0, "Negative", "Neutral"),
    )
    directional = ranked["Literature_Direction"].isin(
        ["Positive", "Negative"]
    )
    ranked["Direction_Concordant"] = pd.Series(pd.NA, index=ranked.index)
    ranked.loc[directional, "Direction_Concordant"] = (
        ranked.loc[directional, "Model_Direction"]
        == ranked.loc[directional, "Literature_Direction"]
    )

    universe_size = len(ranked)
    annotated_count = int(ranked["Literature_Annotated"].sum())
    if annotated_count == 0:
        raise ValueError(
            "None of the literature-annotated RBPs were tested by the model."
        )
    unlabeled_count = universe_size - annotated_count
    if unlabeled_count == 0:
        raise ValueError(
            "All tested RBPs are literature annotated; enrichment requires "
            "an unlabeled comparison set."
        )

    top_k_records = []
    cleaned_top_ks = sorted({
        min(max(1, int(k)), universe_size)
        for k in top_ks
    })
    for k in cleaned_top_ks:
        observed_hits = int(ranked.iloc[:k]["Literature_Annotated"].sum())
        expected_hits = k * annotated_count / universe_size
        fold_enrichment = (
            (observed_hits / k) / (annotated_count / universe_size)
        )
        p_value = float(
            hypergeom.sf(
                observed_hits - 1,
                universe_size,
                annotated_count,
                k,
            )
        )
        odds_ratio, ci_lower, ci_upper = _odds_ratio_interval(
            observed_hits,
            k - observed_hits,
            annotated_count - observed_hits,
            universe_size - k - annotated_count + observed_hits,
        )
        top_k_records.append({
            "Top_K": k,
            "Observed_Annotated": observed_hits,
            "Expected_Annotated": expected_hits,
            "Fold_Enrichment": fold_enrichment,
            "Odds_Ratio": odds_ratio,
            "OR_CI_Lower": ci_lower,
            "OR_CI_Upper": ci_upper,
            "P_Value_Hypergeometric": p_value,
        })
    top_k_df = pd.DataFrame(top_k_records)
    top_k_df["FDR_BH"] = _benjamini_hochberg(
        top_k_df["P_Value_Hypergeometric"]
    )

    annotated_scores = ranked.loc[
        ranked["Literature_Annotated"], rank_column
    ].to_numpy(float)
    unlabeled_scores = ranked.loc[
        ~ranked["Literature_Annotated"], rank_column
    ].to_numpy(float)
    rank_sum = mannwhitneyu(
        annotated_scores,
        unlabeled_scores,
        alternative="greater",
        method="auto",
    )
    auroc_pu = float(
        rank_sum.statistic / (annotated_count * unlabeled_count)
    )
    sorted_labels = ranked["Literature_Annotated"].to_numpy(bool)
    cumulative_hits = np.cumsum(sorted_labels)
    positive_ranks = np.flatnonzero(sorted_labels) + 1
    average_precision_pu = float(np.mean(
        cumulative_hits[positive_ranks - 1] / positive_ranks
    ))

    running_curve, enrichment_score, peak_index = _running_enrichment(
        sorted_labels
    )
    rng = np.random.default_rng(random_state)
    empirical_p = np.nan
    if int(n_permutations) > 0:
        permuted_scores = np.empty(int(n_permutations), dtype=float)
        for index in range(int(n_permutations)):
            permuted = np.zeros(universe_size, dtype=bool)
            permuted[rng.choice(
                universe_size,
                size=annotated_count,
                replace=False,
            )] = True
            _, permuted_scores[index], _ = _running_enrichment(permuted)
        empirical_p = float(
            (1 + np.sum(permuted_scores >= enrichment_score))
            / (int(n_permutations) + 1)
        )

    annotated_significant = int(
        (
            ranked["Literature_Annotated"]
            & ranked["Any_Region_Significant"]
        ).sum()
    )
    annotated_not_significant = annotated_count - annotated_significant
    unlabeled_significant = int(
        (
            ~ranked["Literature_Annotated"]
            & ranked["Any_Region_Significant"]
        ).sum()
    )
    unlabeled_not_significant = unlabeled_count - unlabeled_significant
    fisher_table = np.asarray([
        [annotated_significant, annotated_not_significant],
        [unlabeled_significant, unlabeled_not_significant],
    ])
    fisher_result = fisher_exact(fisher_table, alternative="greater")
    fisher_or, fisher_lower, fisher_upper = _odds_ratio_interval(
        annotated_significant,
        annotated_not_significant,
        unlabeled_significant,
        unlabeled_not_significant,
    )

    directional_rows = ranked[directional].copy()
    direction_total = len(directional_rows)
    direction_matches = int(
        (
            directional_rows["Model_Direction"]
            == directional_rows["Literature_Direction"]
        ).sum()
    )
    direction_fraction = (
        direction_matches / direction_total if direction_total else np.nan
    )
    direction_lower, direction_upper = _wilson_interval(
        direction_matches,
        direction_total,
    )
    direction_p = (
        float(binomtest(
            direction_matches,
            direction_total,
            p=0.5,
            alternative="greater",
        ).pvalue)
        if direction_total
        else np.nan
    )

    summary = pd.DataFrame([{
        "Universe_RBPs": universe_size,
        "Literature_Unique_RBPs": len(annotations),
        "Literature_RBPs_Tested": annotated_count,
        "Literature_Coverage": annotated_count / len(annotations),
        "Rank_Score": rank_score,
        "Region_Selection": region_selection,
        "Rows_Before_Filter": before_filter,
        "Rows_After_Filter": len(working),
        "PU_AUROC": auroc_pu,
        "PU_Average_Precision": average_precision_pu,
        "Mann_Whitney_P": float(rank_sum.pvalue),
        "Preranked_Enrichment_Score": enrichment_score,
        "Preranked_Enrichment_Permutation_P": empirical_p,
        "Peak_Enrichment_Rank": (
            int(peak_index + 1) if peak_index is not None else np.nan
        ),
        "Significant_Overlap_Odds_Ratio": fisher_or,
        "Significant_Overlap_OR_CI_Lower": fisher_lower,
        "Significant_Overlap_OR_CI_Upper": fisher_upper,
        "Significant_Overlap_Fisher_P": float(fisher_result.pvalue),
        "Direction_Concordant": direction_matches,
        "Direction_Evaluable": direction_total,
        "Direction_Concordance": direction_fraction,
        "Direction_CI_Lower": direction_lower,
        "Direction_CI_Upper": direction_upper,
        "Direction_Binomial_P": direction_p,
    }])

    return {
        "summary": summary,
        "ranked_rbps": ranked,
        "top_k_enrichment": top_k_df,
        "annotations": annotations,
        "unmatched_annotations": unmatched_annotations,
        "running_enrichment": running_curve,
        "rank_column": rank_column,
    }


def plot_translation_rbp_rank_enrichment(
        results,
        out_path,
        top_label_count=12,
        w=7.2,
        h=3.2):
    """Plot preranked enrichment and cumulative recovery as a PDF figure."""
    ranked = results["ranked_rbps"].copy()
    summary = results["summary"].iloc[0]
    running = np.asarray(results["running_enrichment"], dtype=float)
    labels = ranked["Literature_Annotated"].to_numpy(bool)
    universe_size = len(ranked)
    annotated_count = int(labels.sum())
    ranks = np.arange(1, universe_size + 1)
    cumulative = np.cumsum(labels)
    expected = ranks * annotated_count / universe_size
    lower = hypergeom.ppf(0.025, universe_size, annotated_count, ranks)
    upper = hypergeom.ppf(0.975, universe_size, annotated_count, ranks)

    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    enrichment_ax, recovery_ax = axes
    signal_color = "#2C6B9A"
    accent_color = "#C44E52"
    neutral_color = "#A8A8A8"

    enrichment_ax.axhline(0, color="#D9D9D9", linewidth=0.8)
    enrichment_ax.plot(ranks, running, color=signal_color, linewidth=1.6)
    hit_ranks = ranks[labels]
    enrichment_ax.vlines(
        hit_ranks,
        ymin=-0.035,
        ymax=0.0,
        color=accent_color,
        linewidth=0.7,
        alpha=0.75,
    )
    peak_rank = int(summary["Peak_Enrichment_Rank"])
    enrichment_ax.axvline(
        peak_rank,
        color=neutral_color,
        linestyle="--",
        linewidth=0.8,
    )
    enrichment_ax.set_xlabel("Model rank")
    enrichment_ax.set_ylabel("Running enrichment score")
    enrichment_ax.set_title("a  Preranked enrichment", loc="left", weight="bold")
    enrichment_ax.text(
        0.98,
        0.96,
        (
            f"ES={summary['Preranked_Enrichment_Score']:.2f}\n"
            f"permutation P={summary['Preranked_Enrichment_Permutation_P']:.2g}"
        ),
        transform=enrichment_ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )

    recovery_ax.fill_between(
        ranks,
        lower,
        upper,
        color="#DCE6EF",
        alpha=0.85,
        linewidth=0,
        label="Random 95% interval",
    )
    recovery_ax.plot(
        ranks,
        expected,
        color=neutral_color,
        linestyle="--",
        linewidth=1.0,
        label="Random expectation",
    )
    recovery_ax.plot(
        ranks,
        cumulative,
        color=signal_color,
        linewidth=1.6,
        label="Observed",
    )
    recovery_ax.set_xlabel("Top K model-ranked RBPs")
    recovery_ax.set_ylabel("Cumulative annotated RBPs")
    recovery_ax.set_title("b  Recovery of known translation RBPs", loc="left", weight="bold")
    recovery_ax.legend(loc="lower right", fontsize=6.5)

    annotated_ranked = ranked[ranked["Literature_Annotated"]].head(
        int(top_label_count)
    )
    label_text = ", ".join(
        f"{row.RBP_Name} ({int(row.Rank)})"
        for row in annotated_ranked.itertuples()
    )
    fig.text(
        0.5,
        -0.02,
        f"Top annotated hits: {label_text}",
        ha="center",
        va="top",
        fontsize=6.5,
        wrap=True,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.88, wspace=0.34)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_translation_rbp_validation(
        model_summary,
        translation_rbp_table,
        out_dir,
        prefix="translation_rbp_validation",
        **evaluation_kwargs):
    """Run the complete validation workflow and save reproducibility tables."""
    results = evaluate_translation_rbp_rank_enrichment(
        model_summary=model_summary,
        translation_rbp_table=translation_rbp_table,
        **evaluation_kwargs,
    )
    os.makedirs(out_dir, exist_ok=True)
    output_paths = {
        "summary_csv": os.path.join(out_dir, f"{prefix}.summary.csv"),
        "ranked_csv": os.path.join(out_dir, f"{prefix}.ranked_rbps.csv"),
        "top_k_csv": os.path.join(out_dir, f"{prefix}.top_k_enrichment.csv"),
        "unmatched_csv": os.path.join(
            out_dir,
            f"{prefix}.unmatched_annotations.csv",
        ),
        "pdf": os.path.join(out_dir, f"{prefix}.rank_enrichment.pdf"),
    }
    results["summary"].to_csv(output_paths["summary_csv"], index=False)
    results["ranked_rbps"].to_csv(output_paths["ranked_csv"], index=False)
    results["top_k_enrichment"].to_csv(output_paths["top_k_csv"], index=False)
    results["unmatched_annotations"].to_csv(
        output_paths["unmatched_csv"],
        index=False,
    )
    plot_translation_rbp_rank_enrichment(
        results,
        out_path=output_paths["pdf"],
    )
    results["output_paths"] = output_paths
    return results

"""Evaluate whether curated translation-regulatory RBPs rank near the top.

The primary analysis ranks every scanned RBP by the largest absolute median
in-silico translation effect observed across the requested transcript regions.
The curated literature table is used only to define external positive labels;
it is never used to construct the primary model score.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, hypergeom, mannwhitneyu


DEFAULT_TRANSLATION_RBP_ALIASES = {
    "ELAVL4 (HuD)": ("ELAVL4",),
    "ZFP36 (TTP)": ("ZFP36",),
    "ZFP36L1 (BRF1)": ("ZFP36L1",),
    "PUM1/2": ("PUM1", "PUM2"),
    "TTP (via GIGYF1/2)": ("ZFP36",),
    "CPEB (via Maskin)": ("CPEB1",),
    "IGF2BP1 (IMP1, ZBP1)": ("IGF2BP1",),
}

_DIRECTION_LABELS = {
    "positive": "Positive",
    "negative": "Negative",
    "bidirectional/context-dependent": "Bidirectional/context-dependent",
    "bidirectional": "Bidirectional/context-dependent",
    "context-dependent": "Bidirectional/context-dependent",
    "context dependent": "Bidirectional/context-dependent",
}

_REGION_LABELS = {
    "5utr": "5UTR",
    "5'utr": "5UTR",
    "5′utr": "5UTR",
    "cds": "CDS",
    "3utr": "3UTR",
    "3'utr": "3UTR",
    "3′utr": "3UTR",
}

_RANK_TYPE_LABELS = {
    "Overall": "All curated RBPs",
    "Positive": "Known positive regulators",
    "Negative": "Known negative regulators",
}

_RANK_TYPE_COLORS = {
    "Overall": "#4A4A4A",
    "Positive": "#D95F59",
    "Negative": "#3B75AF",
}

_RANK_TYPE_MARKERS = {
    "Overall": "o",
    "Positive": "^",
    "Negative": "s",
}


def _read_dataframe(table, table_name):
    """Read a DataFrame or a CSV/TSV/XLSX path."""
    if isinstance(table, pd.DataFrame):
        return table.copy()
    table_path = os.fspath(table)
    if not os.path.isfile(table_path):
        raise FileNotFoundError(f"{table_name} was not found: {table_path}")
    extension = os.path.splitext(table_path)[1].lower()
    if extension in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(table_path)
        except ImportError as exc:
            raise ImportError(
                "Reading the curated Excel table requires openpyxl. "
                "Install openpyxl or provide the same table as CSV."
            ) from exc
    separator = "\t" if extension in {".tsv", ".txt"} else ","
    return pd.read_csv(table_path, sep=separator)


def _normalize_direction(value):
    """Normalize literature direction labels to three supported categories."""
    key = str(value).strip().lower()
    if key not in _DIRECTION_LABELS:
        raise ValueError(
            f"Unsupported translation direction '{value}'. Expected "
            "Positive, Negative, or Bidirectional/context-dependent."
        )
    return _DIRECTION_LABELS[key]


def _normalize_region_list(value):
    """Normalize an optional delimited region annotation."""
    if pd.isna(value) or not str(value).strip():
        return []
    tokens = re.split(r"[;,|/]", str(value))
    regions = []
    for token in tokens:
        key = re.sub(r"\s+", "", token).lower()
        if key not in _REGION_LABELS:
            raise ValueError(
                f"Unsupported curated region '{token}'. Use 5UTR, CDS, or 3UTR."
            )
        region = _REGION_LABELS[key]
        if region not in regions:
            regions.append(region)
    return regions


def _collapse_curated_direction(values):
    """Collapse duplicate literature rows without double counting an RBP."""
    directions = set(values)
    if (
            "Bidirectional/context-dependent" in directions
            or len(directions) > 1):
        return "Bidirectional/context-dependent"
    return next(iter(directions))


def normalize_curated_translation_rbps(
        curated_table,
        name_col="Name",
        direction_col="Translation_direction",
        region_col=None,
        alias_map=None):
    """Normalize aliases and collapse the literature table to unique genes.

    Parameters
    ----------
    region_col : str or None
        Optional explicit literature-supported region column. Free-text evidence
        is intentionally not parsed because that would make the evaluation
        difficult to audit.
    """
    curated = _read_dataframe(curated_table, "Curated translation-RBP table")
    required = {name_col, direction_col}
    if region_col is not None:
        required.add(region_col)
    missing = required.difference(curated.columns)
    if missing:
        raise ValueError(
            "Curated translation-RBP table is missing columns: "
            f"{sorted(missing)}."
        )

    aliases = dict(DEFAULT_TRANSLATION_RBP_ALIASES)
    if alias_map is not None:
        aliases.update(dict(alias_map))

    expanded_rows = []
    for row_index, row in curated.iterrows():
        raw_name = str(row[name_col]).strip()
        if not raw_name or raw_name.lower() == "nan":
            continue
        mapped_names = aliases.get(raw_name, (raw_name,))
        if isinstance(mapped_names, str):
            mapped_names = (mapped_names,)
        direction = _normalize_direction(row[direction_col])
        regions = (
            _normalize_region_list(row[region_col])
            if region_col is not None else []
        )
        for mapped_name in mapped_names:
            normalized_name = str(mapped_name).strip().upper()
            if not normalized_name:
                continue
            expanded_rows.append({
                "RBP_Name": normalized_name,
                "Curated_Direction_Row": direction,
                "Curated_Regions_Row": tuple(regions),
                "Curated_Source_Name": raw_name,
                "Curated_Source_Row": int(row_index) + 2,
            })
    if not expanded_rows:
        raise ValueError("No valid RBP rows were found in the curated table.")

    expanded = pd.DataFrame(expanded_rows)
    collapsed_rows = []
    for rbp_name, group in expanded.groupby("RBP_Name", sort=True):
        regions = []
        for region_tuple in group["Curated_Regions_Row"]:
            for region in region_tuple:
                if region not in regions:
                    regions.append(region)
        collapsed_rows.append({
            "RBP_Name": rbp_name,
            "Curated_Direction": _collapse_curated_direction(
                group["Curated_Direction_Row"]
            ),
            "Curated_Regions": ";".join(regions),
            "Curated_Source_Names": "; ".join(dict.fromkeys(
                group["Curated_Source_Name"].astype(str)
            )),
            "Curated_Source_Rows": ";".join(
                group["Curated_Source_Row"].astype(str)
            ),
            "N_Curated_Rows": int(len(group)),
        })
    return pd.DataFrame(collapsed_rows)


def _benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p values."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    finite_values = values[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n_values = len(ranked)
    corrected = ranked * n_values / np.arange(1, n_values + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[finite] = restored
    return adjusted


def _aggregate_model_scores(summary, effect_col, rank_type):
    """Create one pre-specified score per model-scanned RBP."""
    selected_rows = []
    for _, group in summary.groupby("RBP_Name", sort=False):
        effects = group[effect_col].to_numpy(float)
        if rank_type == "Overall":
            local_index = int(np.nanargmax(np.abs(effects)))
            score = abs(effects[local_index])
        elif rank_type == "Positive":
            local_index = int(np.nanargmax(effects))
            score = effects[local_index]
        elif rank_type == "Negative":
            local_index = int(np.nanargmin(effects))
            score = -effects[local_index]
        else:
            raise ValueError(f"Unsupported rank type: {rank_type}")
        selected = group.iloc[local_index].to_dict()
        selected.update({
            "Rank_Type": rank_type,
            "Rank_Score": float(score),
            "Selected_Effect": float(effects[local_index]),
        })
        selected_rows.append(selected)
    ranked = pd.DataFrame(selected_rows).sort_values(
        ["Rank_Score", "RBP_Name"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    ranked["Rank_Fraction"] = ranked["Rank"] / len(ranked)
    return ranked


def _running_enrichment(hit_labels):
    """Calculate a one-sided GSEA-like running enrichment score."""
    hits = np.asarray(hit_labels, dtype=int)
    n_hits = int(hits.sum())
    n_misses = int(len(hits) - n_hits)
    if n_hits == 0 or n_misses == 0:
        return np.full(len(hits), np.nan), np.nan
    increments = np.where(hits == 1, 1.0 / n_hits, -1.0 / n_misses)
    running = np.cumsum(increments)
    return running, float(np.nanmax(running))


def _make_strata(values, n_bins):
    """Create abundance strata for label permutation."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().sum() < 4 or int(n_bins) < 2:
        return np.zeros(len(numeric), dtype=int)
    numeric = numeric.fillna(numeric.median())
    try:
        strata = pd.qcut(
            numeric.rank(method="average"),
            q=min(int(n_bins), int(numeric.nunique())),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return np.zeros(len(numeric), dtype=int)
    return np.asarray(strata, dtype=int)


def _average_precision(hit_labels):
    """Calculate average precision from an ordered binary label vector."""
    hits = np.asarray(hit_labels, dtype=int)
    n_hits = int(hits.sum())
    if n_hits == 0:
        return np.nan
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(np.sum(precision * hits) / n_hits)


def _evaluate_ranked_table(
        ranked,
        known_rbps,
        top_ks,
        permutation_iterations,
        stratify_bins,
        random_state):
    """Evaluate one ranked RBP table against an external known set."""
    ranked = ranked.copy()
    ranked["Is_Curated_Translation_RBP"] = ranked["RBP_Name"].isin(
        known_rbps
    )
    hit_labels = ranked["Is_Curated_Translation_RBP"].to_numpy(int)
    n_total = len(ranked)
    n_known = int(hit_labels.sum())
    if n_known == 0:
        raise ValueError(
            f"No curated RBPs matched rank type {ranked['Rank_Type'].iloc[0]}."
        )
    if n_known == n_total:
        raise ValueError("Every scanned RBP is curated; enrichment is undefined.")

    running, enrichment_score = _running_enrichment(hit_labels)
    ranked["Cumulative_Curated_Fraction"] = (
        np.cumsum(hit_labels) / n_known
    )
    ranked["Running_Enrichment"] = running
    baseline_prevalence = n_known / n_total
    average_precision = _average_precision(hit_labels)

    positive_scores = ranked.loc[
        ranked["Is_Curated_Translation_RBP"], "Rank_Score"
    ]
    background_scores = ranked.loc[
        ~ranked["Is_Curated_Translation_RBP"], "Rank_Score"
    ]
    u_statistic, rank_sum_p = mannwhitneyu(
        positive_scores,
        background_scores,
        alternative="greater",
    )
    roc_auc = float(
        u_statistic / (len(positive_scores) * len(background_scores))
    )

    top_ks = sorted(set(
        int(k) for k in top_ks if 1 <= int(k) <= n_total
    ))
    if not top_ks:
        raise ValueError("No top-K value falls within the ranked universe.")

    strata = _make_strata(ranked["N_Transcripts"], stratify_bins)
    strata_indices = [
        np.flatnonzero(strata == stratum)
        for stratum in np.unique(strata)
    ]
    rng = np.random.default_rng(random_state)
    permutation_iterations = int(permutation_iterations)
    if permutation_iterations < 100:
        raise ValueError("permutation_iterations must be at least 100.")
    null_top_counts = np.zeros(
        (permutation_iterations, len(top_ks)), dtype=float
    )
    null_enrichment = np.zeros(permutation_iterations, dtype=float)
    null_average_precision = np.zeros(permutation_iterations, dtype=float)
    for permutation_index in range(permutation_iterations):
        permuted = hit_labels.copy()
        for indices in strata_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        cumulative = np.cumsum(permuted)
        null_top_counts[permutation_index] = [
            cumulative[k - 1] for k in top_ks
        ]
        _, null_enrichment[permutation_index] = _running_enrichment(permuted)
        null_average_precision[permutation_index] = _average_precision(permuted)

    top_k_records = []
    for k_index, k_value in enumerate(top_ks):
        observed_hits = int(np.cumsum(hit_labels)[k_value - 1])
        expected_hits = k_value * baseline_prevalence
        enrichment_fold = (
            (observed_hits / k_value) / baseline_prevalence
        )
        hypergeometric_p = float(hypergeom.sf(
            observed_hits - 1,
            n_total,
            n_known,
            k_value,
        ))
        null_counts = null_top_counts[:, k_index]
        null_folds = (
            (null_counts / k_value) / baseline_prevalence
        )
        empirical_p = float(
            (1 + np.sum(null_counts >= observed_hits))
            / (permutation_iterations + 1)
        )
        top_k_records.append({
            "Rank_Type": ranked["Rank_Type"].iloc[0],
            "K": k_value,
            "Observed_Curated_RBPs": observed_hits,
            "Expected_Curated_RBPs": expected_hits,
            "Enrichment_Fold": enrichment_fold,
            "Hypergeometric_P": hypergeometric_p,
            "Stratified_Permutation_P": empirical_p,
            "Null_Fold_CI_Lower": float(np.quantile(null_folds, 0.025)),
            "Null_Fold_CI_Upper": float(np.quantile(null_folds, 0.975)),
        })
    top_k = pd.DataFrame(top_k_records)
    top_k["Hypergeometric_FDR_BH"] = _benjamini_hochberg(
        top_k["Hypergeometric_P"]
    )
    top_k["Stratified_Permutation_FDR_BH"] = _benjamini_hochberg(
        top_k["Stratified_Permutation_P"]
    )

    enrichment_p = float(
        (1 + np.sum(null_enrichment >= enrichment_score))
        / (permutation_iterations + 1)
    )
    average_precision_p = float(
        (1 + np.sum(null_average_precision >= average_precision))
        / (permutation_iterations + 1)
    )
    metrics = {
        "Rank_Type": ranked["Rank_Type"].iloc[0],
        "N_Scanned_RBPs": n_total,
        "N_Curated_Matched": n_known,
        "Curated_Prevalence": baseline_prevalence,
        "Average_Precision": average_precision,
        "Average_Precision_Fold_Over_Baseline": (
            average_precision / baseline_prevalence
        ),
        "Average_Precision_Permutation_P": average_precision_p,
        "ROC_AUC": roc_auc,
        "Mann_Whitney_One_Sided_P": float(rank_sum_p),
        "Running_Enrichment_Score": enrichment_score,
        "Running_Enrichment_Permutation_P": enrichment_p,
        "Permutation_Iterations": permutation_iterations,
        "Permutation_Strata": int(len(strata_indices)),
    }
    return ranked, top_k, metrics


def _evaluate_direction_concordance(curated, summary, effect_col):
    """Compare literature and model directions for non-contextual RBPs."""
    fixed = curated[curated["Curated_Direction"].isin(
        ["Positive", "Negative"]
    )].copy()
    records = []
    for curated_row in fixed.itertuples(index=False):
        candidates = summary[summary["RBP_Name"].eq(curated_row.RBP_Name)]
        expected_regions = [
            region for region in str(curated_row.Curated_Regions).split(";")
            if region
        ]
        if expected_regions:
            candidates = candidates[candidates["Region"].isin(expected_regions)]
        if candidates.empty:
            continue
        local_index = int(np.nanargmax(
            np.abs(candidates[effect_col].to_numpy(float))
        ))
        selected = candidates.iloc[local_index]
        selected_effect = float(selected[effect_col])
        model_direction = "Positive" if selected_effect > 0 else "Negative"
        records.append({
            "RBP_Name": curated_row.RBP_Name,
            "Curated_Direction": curated_row.Curated_Direction,
            "Curated_Regions": curated_row.Curated_Regions,
            "Model_Selected_Region": selected["Region"],
            "Model_Selected_Effect": selected_effect,
            "Model_Direction": model_direction,
            "Direction_Concordant": (
                model_direction == curated_row.Curated_Direction
            ),
        })
    concordance = pd.DataFrame(records)
    if concordance.empty:
        return concordance, {
            "Direction_N": 0,
            "Direction_Concordant_N": 0,
            "Direction_Concordance": np.nan,
            "Direction_Binomial_P": np.nan,
            "Direction_CI_Lower": np.nan,
            "Direction_CI_Upper": np.nan,
        }
    n_concordant = int(concordance["Direction_Concordant"].sum())
    n_total = len(concordance)
    binomial = binomtest(
        n_concordant,
        n_total,
        p=0.5,
        alternative="greater",
    )
    confidence_interval = binomial.proportion_ci(
        confidence_level=0.95,
        method="exact",
    )
    return concordance, {
        "Direction_N": n_total,
        "Direction_Concordant_N": n_concordant,
        "Direction_Concordance": n_concordant / n_total,
        "Direction_Binomial_P": float(binomial.pvalue),
        "Direction_CI_Lower": float(confidence_interval.low),
        "Direction_CI_Upper": float(confidence_interval.high),
    }


def evaluate_translation_rbp_rank_enrichment(
        curated_table,
        model_effect_summary,
        out_dir=None,
        name_col="Name",
        direction_col="Translation_direction",
        curated_region_col=None,
        alias_map=None,
        effect_col="Median_Delta_Log2_TE",
        regions=("5UTR", "3UTR"),
        min_transcripts=5,
        fdr_threshold=None,
        top_ks=(10, 20, 30, 50, 75, 100),
        permutation_iterations=10000,
        stratify_bins=5,
        random_state=42,
        save_tables=True):
    """Evaluate rank enrichment of literature-curated translation RBPs.

    Three pre-specified rankings are evaluated:

    - Overall: maximum absolute effect across requested regions.
    - Positive: maximum signed effect for known positive regulators.
    - Negative: maximum negative effect magnitude for known negative regulators.

    Label permutations are performed within transcript-count strata to reduce
    bias from unequal motif-hit abundance and effect-estimate precision.
    """
    curated = normalize_curated_translation_rbps(
        curated_table,
        name_col=name_col,
        direction_col=direction_col,
        region_col=curated_region_col,
        alias_map=alias_map,
    )
    summary = _read_dataframe(model_effect_summary, "Model RBP effect summary")
    n_summary_rows_input = len(summary)
    required = {
        "RBP_Name", "Region", "N_Transcripts", effect_col,
    }
    if fdr_threshold is not None:
        required.add("FDR_BH")
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            f"Model effect summary is missing columns: {sorted(missing)}."
        )
    summary = summary.copy()
    summary["RBP_Name"] = (
        summary["RBP_Name"].astype(str).str.strip().str.upper()
    )
    summary[effect_col] = pd.to_numeric(summary[effect_col], errors="coerce")
    summary["N_Transcripts"] = pd.to_numeric(
        summary["N_Transcripts"], errors="coerce"
    )
    summary = summary.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["RBP_Name", "Region", "N_Transcripts", effect_col]
    )
    regions = tuple(dict.fromkeys(str(region) for region in regions))
    unsupported_regions = set(regions).difference({"5UTR", "CDS", "3UTR"})
    if unsupported_regions:
        raise ValueError(f"Unsupported regions: {sorted(unsupported_regions)}")
    summary = summary[
        summary["Region"].isin(regions)
        & summary["N_Transcripts"].ge(int(min_transcripts))
    ].copy()
    if fdr_threshold is not None:
        summary["FDR_BH"] = pd.to_numeric(
            summary["FDR_BH"], errors="coerce"
        )
        summary = summary[
            summary["FDR_BH"].le(float(fdr_threshold))
        ].copy()
    if summary.empty:
        raise ValueError("No model RBP effects passed the requested filters.")
    n_summary_rows_filtered = len(summary)

    scanned_rbps = set(summary["RBP_Name"])
    curated["Matched_In_Model"] = curated["RBP_Name"].isin(scanned_rbps)
    matched_curated = curated[curated["Matched_In_Model"]].copy()
    if matched_curated.empty:
        raise ValueError("No curated RBP matched the model-scanned RBP universe.")

    rank_specs = {
        "Overall": set(matched_curated["RBP_Name"]),
        "Positive": set(matched_curated.loc[
            matched_curated["Curated_Direction"].eq("Positive"),
            "RBP_Name",
        ]),
        "Negative": set(matched_curated.loc[
            matched_curated["Curated_Direction"].eq("Negative"),
            "RBP_Name",
        ]),
    }
    ranked_tables = []
    top_k_tables = []
    metric_records = []
    for rank_index, (rank_type, known_rbps) in enumerate(rank_specs.items()):
        if not known_rbps:
            continue
        ranked = _aggregate_model_scores(summary, effect_col, rank_type)
        ranked, top_k, metrics = _evaluate_ranked_table(
            ranked,
            known_rbps,
            top_ks=top_ks,
            permutation_iterations=permutation_iterations,
            stratify_bins=stratify_bins,
            random_state=int(random_state) + rank_index,
        )
        ranked_tables.append(ranked)
        top_k_tables.append(top_k)
        metric_records.append(metrics)

    ranked = pd.concat(ranked_tables, ignore_index=True)
    top_k = pd.concat(top_k_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_records)
    for p_value_column in (
            "Average_Precision_Permutation_P",
            "Mann_Whitney_One_Sided_P",
            "Running_Enrichment_Permutation_P"):
        metrics[f"{p_value_column}_FDR_BH"] = _benjamini_hochberg(
            metrics[p_value_column]
        )
    top_k["Hypergeometric_Global_FDR_BH"] = _benjamini_hochberg(
        top_k["Hypergeometric_P"]
    )
    top_k["Stratified_Permutation_Global_FDR_BH"] = _benjamini_hochberg(
        top_k["Stratified_Permutation_P"]
    )
    metrics["N_Curated_Unique"] = len(curated)
    metrics["N_Curated_Matched_Any_Direction"] = len(matched_curated)
    metrics["N_Model_Summary_Rows_Input"] = n_summary_rows_input
    metrics["N_Model_Summary_Rows_Filtered"] = n_summary_rows_filtered
    metrics["N_Model_Summary_Rows_Excluded"] = (
        n_summary_rows_input - n_summary_rows_filtered
    )
    metrics["Effect_Column"] = effect_col
    metrics["Regions"] = ";".join(regions)
    metrics["Min_Transcripts"] = int(min_transcripts)
    metrics["FDR_Threshold"] = (
        np.nan if fdr_threshold is None else float(fdr_threshold)
    )

    concordance, concordance_metrics = _evaluate_direction_concordance(
        matched_curated,
        summary,
        effect_col,
    )
    for key, value in concordance_metrics.items():
        metrics[key] = value

    results = {
        "curated_normalized": curated,
        "model_summary_filtered": summary,
        "ranked": ranked,
        "top_k": top_k,
        "metrics": metrics,
        "direction_concordance": concordance,
    }
    if out_dir is not None and save_tables:
        os.makedirs(out_dir, exist_ok=True)
        table_mapping = {
            "curated_normalized": "translation_rbp_curated_normalized.csv",
            "ranked": "translation_rbp_ranked_table.csv",
            "top_k": "translation_rbp_topk_enrichment.csv",
            "metrics": "translation_rbp_enrichment_metrics.csv",
            "direction_concordance": (
                "translation_rbp_direction_concordance.csv"
            ),
        }
        for result_key, file_name in table_mapping.items():
            results[result_key].to_csv(
                os.path.join(out_dir, file_name),
                index=False,
            )
    return results


def plot_translation_rbp_rank_enrichment(
        results,
        out_dir,
        suffix="",
        rank_types=("Overall", "Positive", "Negative"),
        w=7.2,
        h=3.1):
    """Plot cumulative rank enrichment and Top-K enrichment as a PDF."""
    import sys

    import matplotlib as mpl
    if "matplotlib.pyplot" not in sys.modules:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "DejaVu Sans", "sans-serif"
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    if not isinstance(results, Mapping):
        raise TypeError("results must be returned by the evaluation function.")
    for key in ("ranked", "top_k", "metrics"):
        if key not in results:
            raise ValueError(f"results is missing '{key}'.")
    ranked = results["ranked"].copy()
    top_k = results["top_k"].copy()
    metrics = results["metrics"].copy()
    rank_types = [
        rank_type for rank_type in rank_types
        if rank_type in set(ranked["Rank_Type"])
    ]
    if not rank_types:
        raise ValueError("None of the requested rank_types is available.")

    fig, axes = plt.subplots(1, 2, figsize=(float(w), float(h)))
    for rank_type in rank_types:
        color = _RANK_TYPE_COLORS.get(rank_type, "#777777")
        marker = _RANK_TYPE_MARKERS.get(rank_type, "o")
        label = _RANK_TYPE_LABELS.get(rank_type, rank_type)
        rank_group = ranked[ranked["Rank_Type"].eq(rank_type)].sort_values(
            "Rank"
        )
        metric_row = metrics[metrics["Rank_Type"].eq(rank_type)].iloc[0]
        axes[0].plot(
            rank_group["Rank_Fraction"],
            rank_group["Cumulative_Curated_Fraction"],
            color=color,
            linewidth=1.6,
            label=(
                f"{label} (AP={metric_row['Average_Precision']:.2f}, "
                f"ES={metric_row['Running_Enrichment_Score']:.2f})"
            ),
        )

        top_group = top_k[top_k["Rank_Type"].eq(rank_type)].sort_values("K")
        axes[1].fill_between(
            top_group["K"].to_numpy(float),
            top_group["Null_Fold_CI_Lower"].to_numpy(float),
            top_group["Null_Fold_CI_Upper"].to_numpy(float),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        axes[1].plot(
            top_group["K"],
            top_group["Enrichment_Fold"],
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.3,
            label=label,
        )

    axes[0].plot(
        [0, 1], [0, 1],
        color="#BDBDBD",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )
    axes[0].set(
        xlabel="Fraction of ranked RBP list",
        ylabel="Fraction of curated RBPs recovered",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Cumulative rank enrichment",
    )
    axes[0].legend(loc="lower right", fontsize=6.3)

    axes[1].axhline(
        1.0,
        color="#BDBDBD",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )
    axes[1].set(
        xlabel="Top K model-ranked RBPs",
        ylabel="Fold enrichment over random",
        title="Top-K enrichment",
    )
    axes[1].legend(loc="best", fontsize=6.3)
    for axis in axes:
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        axis.set_axisbelow(True)

    fig.suptitle(
        "Enrichment of literature-curated translation-regulatory RBPs",
        y=1.01,
        fontsize=8.5,
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    file_suffix = f".{suffix}" if suffix else ""
    pdf_path = os.path.join(
        out_dir,
        f"translation_rbp_rank_enrichment{file_suffix}.pdf",
    )
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path

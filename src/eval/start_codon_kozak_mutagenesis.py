"""Matched in silico mutagenesis of CDS start codons and Kozak contexts.

This module tests whether a translation-profile model has learned causal
sequence sensitivity to the annotated CDS initiation context. For every
transcript, it first constructs the same strong-ATG reference and then changes
only the start codon, position -3, and position +4. Consequently, every mutant
is compared with a sequence-matched reference from the same transcript.
"""

from __future__ import annotations

import itertools
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eval.save_prediction_results import (
    _autocast_context,
    _extract_head_tensor,
    _model_device,
)
from utils import unwrap_model


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
START_CODON_ORDER = ["ATG", "CTG", "GTG", "TTG"]
START_CODON_WEIGHTS = {
    "ATG": 1.0,
    "CTG": 0.3,
    "GTG": 0.2,
    "TTG": 0.05,
}

# Representative bases make the four contexts differ only at -3 and +4.
# Position +4 is the first nucleotide after the three-base start codon.
KOZAK_CONTEXTS = OrderedDict([
    ("Strong (+4G, -3R)", {"minus3": "A", "plus4": "G"}),
    ("Moderate (-3R)", {"minus3": "A", "plus4": "C"}),
    ("Moderate (+4G)", {"minus3": "C", "plus4": "G"}),
    ("Weak", {"minus3": "C", "plus4": "C"}),
])
KOZAK_CONTEXT_ORDER = list(KOZAK_CONTEXTS)
KOZAK_CONTEXT_COLORS = {
    "Strong (+4G, -3R)": "#08306B",
    "Moderate (-3R)": "#2171B5",
    "Moderate (+4G)": "#6BAED6",
    "Weak": "#B0B0B0",
}
KOZAK_CRITICAL_WEIGHTS = {
    "minus3": {"A": 0.14, "C": -0.08, "G": 0.14, "T": -0.29},
    "plus4": {"A": -0.02, "C": -0.09, "G": 0.12, "T": -0.02},
}


def _one_hot(base: str) -> np.ndarray:
    vector = np.zeros(4, dtype=np.float32)
    vector[BASE_TO_INDEX[base]] = 1.0
    return vector


def _meta_value(meta_info, keys: Sequence[str], default=None):
    for key in keys:
        if isinstance(meta_info, Mapping) and key in meta_info:
            return meta_info[key]
        if hasattr(meta_info, key):
            return getattr(meta_info, key)
    return default


def _normalize_tid(value: object) -> str:
    tid = str(value)
    if tid.startswith("ENST"):
        return tid.split(".", 1)[0]
    return tid


def _extract_tid(uuid: object, meta_info) -> str:
    tid = _meta_value(
        meta_info,
        ("Tid", "tid", "transcript_id", "transcript", "tx_id"),
        default=None,
    )
    if tid is None:
        tid = str(uuid).split("-", 1)[0]
    return _normalize_tid(tid)


def _resolve_target_tids(
    target_transcript_ids: Optional[
        Union[Iterable[str], Mapping[str, Iterable[str]]]
    ],
    cell_type: str,
) -> Optional[set]:
    if target_transcript_ids is None:
        return None
    if isinstance(target_transcript_ids, Mapping):
        values = target_transcript_ids.get(cell_type, [])
    else:
        values = target_transcript_ids
    return {_normalize_tid(value) for value in values}


def _as_sequence_embedding(seq_emb) -> np.ndarray:
    if isinstance(seq_emb, torch.Tensor):
        array = seq_emb.detach().cpu().numpy()
    else:
        array = np.asarray(seq_emb)
    if array.ndim != 2:
        raise ValueError(f"Sequence embedding must be two-dimensional, got {array.shape}.")
    if array.shape[1] != 4 and array.shape[0] == 4:
        array = array.T
    if array.shape[1] != 4:
        raise ValueError(f"Sequence embedding must have four channels, got {array.shape}.")
    return np.array(array, dtype=np.float32, copy=True)


def _as_expression_vector(expr_vector) -> np.ndarray:
    if expr_vector is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(expr_vector, torch.Tensor):
        array = expr_vector.detach().cpu().numpy()
    else:
        array = np.asarray(expr_vector)
    return np.asarray(array, dtype=np.float32).reshape(-1)


def designed_kozak_score(start_codon: str, kozak_class: str) -> float:
    """Score only the three experimentally manipulated motif components."""
    context = KOZAK_CONTEXTS[kozak_class]
    return float(
        START_CODON_WEIGHTS[start_codon]
        + KOZAK_CRITICAL_WEIGHTS["minus3"][context["minus3"]]
        + KOZAK_CRITICAL_WEIGHTS["plus4"][context["plus4"]]
    )


def mutate_cds_start_context(
    seq_emb: np.ndarray,
    cds_start: int,
    start_codon: str,
    kozak_class: str,
) -> np.ndarray:
    """Return a copy carrying one controlled start-codon/Kozak variant."""
    start_codon = str(start_codon).upper()
    if start_codon not in START_CODON_ORDER:
        raise ValueError(
            f"start_codon must be one of {START_CODON_ORDER}, got '{start_codon}'."
        )
    if kozak_class not in KOZAK_CONTEXTS:
        raise ValueError(
            f"kozak_class must be one of {KOZAK_CONTEXT_ORDER}, got '{kozak_class}'."
        )

    mutated = _as_sequence_embedding(seq_emb)
    if cds_start < 3 or cds_start + 3 >= len(mutated):
        raise ValueError(
            "The annotated CDS start requires at least 3 upstream nucleotides "
            "and one downstream nucleotide."
        )

    context = KOZAK_CONTEXTS[kozak_class]
    mutated[cds_start - 3] = _one_hot(context["minus3"])
    for offset, base in enumerate(start_codon):
        mutated[cds_start + offset] = _one_hot(base)
    mutated[cds_start + 3] = _one_hot(context["plus4"])
    return mutated


def collect_kozak_mutagenesis_samples(
    dataset,
    target_cell_type: Optional[Union[str, Sequence[str]]] = None,
    target_transcript_ids: Optional[
        Union[Iterable[str], Mapping[str, Iterable[str]]]
    ] = None,
    min_cds_nt: int = 21,
    max_cds_nt: Optional[int] = None,
    num_samples: Optional[int] = None,
    random_state: int = 42,
) -> List[Dict]:
    """Collect annotated CDS samples that can support controlled mutation."""
    if min_cds_nt < 3:
        raise ValueError("min_cds_nt must be at least 3.")
    if max_cds_nt is not None and max_cds_nt < min_cds_nt:
        raise ValueError("max_cds_nt must be greater than or equal to min_cds_nt.")

    if target_cell_type is None:
        target_cells = None
    elif isinstance(target_cell_type, str):
        target_cells = {target_cell_type}
    else:
        target_cells = {str(value) for value in target_cell_type}

    samples: List[Dict] = []
    exclusion_counts: Dict[str, int] = {}

    def exclude(reason: str):
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    for dataset_index in tqdm(range(len(dataset)), desc="Collect Kozak samples"):
        try:
            item = dataset[dataset_index]
            if len(item) < 6:
                exclude("missing_required_fields")
                continue
            uuid, species, cell_type, expr_vector, meta_info, seq_emb = item[:6]
            cell_type = str(cell_type)
            if target_cells is not None and cell_type not in target_cells:
                exclude("cell_type_filter")
                continue

            tid = _extract_tid(uuid, meta_info)
            allowed_tids = _resolve_target_tids(target_transcript_ids, cell_type)
            if allowed_tids is not None and tid not in allowed_tids:
                exclude("transcript_filter")
                continue

            cds_start_pos = _meta_value(
                meta_info, ("cds_start_pos", "CDS_Start", "cds_start"), -1
            )
            cds_end_pos = _meta_value(
                meta_info, ("cds_end_pos", "CDS_End", "cds_end"), -1
            )
            cds_start = int(cds_start_pos) - 1
            cds_end = int(cds_end_pos)
            sequence = _as_sequence_embedding(seq_emb)
            cds_end = min(cds_end, len(sequence))
            cds_length = cds_end - cds_start

            if cds_start < 3 or cds_start + 3 >= len(sequence):
                exclude("insufficient_kozak_flank")
                continue
            if cds_end <= cds_start or cds_length < min_cds_nt:
                exclude("short_or_invalid_cds")
                continue
            if max_cds_nt is not None and cds_length > max_cds_nt:
                exclude("long_cds")
                continue

            samples.append({
                "sample_id": f"{cell_type}::{tid}::{dataset_index}",
                "uuid": str(uuid),
                "tid": tid,
                "species": species,
                "cell_type": cell_type,
                "expr_vector": _as_expression_vector(expr_vector),
                "seq_emb": sequence,
                "cds_start": cds_start,
                "cds_end": cds_end,
                "cds_length": cds_length,
            })
        except (TypeError, ValueError, IndexError):
            exclude("malformed_sample")

    before_sampling = len(samples)
    if num_samples is not None and before_sampling > num_samples:
        rng = np.random.default_rng(random_state)
        selected = np.sort(rng.choice(before_sampling, num_samples, replace=False))
        samples = [samples[index] for index in selected]

    print(
        f"Collected {len(samples)} of {len(dataset)} dataset entries "
        f"({before_sampling} before optional sampling)."
    )
    if exclusion_counts:
        print("Exclusions: " + ", ".join(
            f"{key}={value}" for key, value in sorted(exclusion_counts.items())
        ))
    return samples


class _KozakVariantDataset(Dataset):
    def __init__(self, records: Sequence[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def _collate_kozak_variants(batch: Sequence[Dict]) -> Dict:
    sequences = [torch.from_numpy(record["seq_emb"]) for record in batch]
    lengths = [int(sequence.shape[0]) for sequence in sequences]
    expression_vectors = [
        torch.from_numpy(record["expr_vector"]) for record in batch
    ]
    expression_widths = {int(vector.numel()) for vector in expression_vectors}
    if len(expression_widths) != 1:
        raise ValueError("All expression vectors in a batch must have the same length.")
    expression_batch = torch.stack(expression_vectors)
    return {
        "records": list(batch),
        "seq_batch": pad_sequence(sequences, batch_first=True, padding_value=-1),
        "expr_batch": expression_batch,
        "lengths": lengths,
    }


def _mean_frame_signal(
    profile: np.ndarray,
    start: int,
    end: int,
    skip_codons: int = 0,
    eps: float = 1e-8,
) -> float:
    first_position = start + 3 * skip_codons
    valid_end = min(int(end), len(profile))
    if first_position >= valid_end:
        return np.nan
    values = np.asarray(profile[first_position:valid_end:3], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.mean(finite) + eps)


class KozakMutagenesisEvaluator:
    """Run matched Kozak/start-codon perturbations with a compatible model."""

    def __init__(
        self,
        model,
        out_dir: str = ".",
        prediction_scale: str = "log1p",
    ):
        prediction_model = unwrap_model(model)
        if not callable(getattr(prediction_model, "predict", None)):
            raise TypeError(
                "KozakMutagenesisEvaluator requires a model with a callable "
                "predict() method."
            )
        if prediction_scale not in {"log1p", "linear"}:
            raise ValueError("prediction_scale must be 'log1p' or 'linear'.")
        self.model = prediction_model
        self.device = _model_device(prediction_model)
        self.out_dir = out_dir
        self.prediction_scale = prediction_scale

    def _predict_batch(
        self,
        seq_batch: torch.Tensor,
        expr_batch: torch.Tensor,
        species: Sequence,
        lengths: Sequence[int],
    ) -> np.ndarray:
        seq_batch = seq_batch.to(self.device)
        expr_batch = expr_batch.float().to(self.device)
        positions = torch.arange(seq_batch.shape[1], device=self.device).unsqueeze(0)
        src_mask = positions < torch.tensor(
            lengths, device=self.device
        ).unsqueeze(1)

        self.model.eval()
        with torch.inference_mode(), _autocast_context(self.device):
            output = self.model.predict(
                seq_batch=seq_batch,
                species=list(species),
                expr_vector=expr_batch,
                src_mask=src_mask,
                head_names=["count"],
            )
            profile = _extract_head_tensor(output, "count")
        if profile.ndim != 3 or profile.shape[-1] != 1:
            raise ValueError(
                "The model count head must return (batch, length, 1), "
                f"got {tuple(profile.shape)}."
            )
        profile = profile.squeeze(-1).float()
        if self.prediction_scale == "log1p":
            profile = torch.expm1(profile)
        return profile.cpu().numpy()

    @staticmethod
    def _build_variant_records(samples: Sequence[Dict]) -> List[Dict]:
        records = []
        for sample in samples:
            for start_codon, kozak_class in itertools.product(
                START_CODON_ORDER, KOZAK_CONTEXT_ORDER
            ):
                record = dict(sample)
                record.update({
                    "start_codon": start_codon,
                    "kozak_class": kozak_class,
                    "designed_kozak_score": designed_kozak_score(
                        start_codon, kozak_class
                    ),
                    "is_wt": (
                        start_codon == "ATG"
                        and kozak_class == "Strong (+4G, -3R)"
                    ),
                    "seq_emb": mutate_cds_start_context(
                        sample["seq_emb"],
                        sample["cds_start"],
                        start_codon,
                        kozak_class,
                    ),
                })
                records.append(record)
        return records

    def evaluate(
        self,
        samples: Sequence[Dict],
        batch_size: int = 32,
        num_workers: int = 0,
        cds_skip_codons: int = 5,
        start_window_codons: int = 10,
        suffix: str = "",
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """Predict all matched variants and quantify downstream CDS effects."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if cds_skip_codons < 0 or start_window_codons < 1:
            raise ValueError("Invalid CDS signal window parameters.")
        if not samples:
            return pd.DataFrame()

        records = self._build_variant_records(samples)
        loader = DataLoader(
            _KozakVariantDataset(records),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_kozak_variants,
        )

        rows = []
        for batch in tqdm(loader, desc="Kozak mutagenesis inference"):
            batch_records = batch["records"]
            profiles = self._predict_batch(
                batch["seq_batch"],
                batch["expr_batch"],
                [record["species"] for record in batch_records],
                batch["lengths"],
            )
            for index, record in enumerate(batch_records):
                profile = profiles[index, :batch["lengths"][index]]
                cds_signal = _mean_frame_signal(
                    profile,
                    record["cds_start"],
                    record["cds_end"],
                    skip_codons=cds_skip_codons,
                )
                proximal_end = min(
                    record["cds_end"],
                    record["cds_start"] + 3 * start_window_codons,
                )
                proximal_signal = _mean_frame_signal(
                    profile,
                    record["cds_start"],
                    proximal_end,
                    skip_codons=0,
                )
                rows.append({
                    "Sample_ID": record["sample_id"],
                    "UUID": record["uuid"],
                    "Tid": record["tid"],
                    "Cell_Type": record["cell_type"],
                    "Start_Codon": record["start_codon"],
                    "Kozak_Class": record["kozak_class"],
                    "Designed_Kozak_Score": record["designed_kozak_score"],
                    "Is_WT": record["is_wt"],
                    "CDS_Start_0based": record["cds_start"],
                    "CDS_End_exclusive": record["cds_end"],
                    "CDS_Skip_Codons": cds_skip_codons,
                    "CDS_Mean_Signal": cds_signal,
                    "Start_Proximal_Mean_Signal": proximal_signal,
                })

        results = pd.DataFrame(rows)
        wt = results.loc[
            results["Is_WT"],
            ["Sample_ID", "CDS_Mean_Signal", "Start_Proximal_Mean_Signal"],
        ].rename(columns={
            "CDS_Mean_Signal": "WT_CDS_Mean_Signal",
            "Start_Proximal_Mean_Signal": "WT_Start_Proximal_Mean_Signal",
        })
        if wt["Sample_ID"].duplicated().any():
            raise RuntimeError("Each sample must have exactly one strong-ATG WT row.")
        results = results.merge(wt, on="Sample_ID", how="left", validate="many_to_one")
        results["Relative_CDS_Translation"] = (
            results["CDS_Mean_Signal"] / results["WT_CDS_Mean_Signal"]
        )
        results["Relative_Start_Proximal_Translation"] = (
            results["Start_Proximal_Mean_Signal"]
            / results["WT_Start_Proximal_Mean_Signal"]
        )
        results["Log2_Relative_CDS_Translation"] = np.log2(
            results["Relative_CDS_Translation"].clip(lower=1e-12)
        )
        results["Variant"] = (
            results["Start_Codon"].astype(str)
            + " | "
            + results["Kozak_Class"].astype(str)
        )

        if save_csv:
            os.makedirs(self.out_dir, exist_ok=True)
            tag = f".{suffix}" if suffix else ""
            raw_path = os.path.join(
                self.out_dir, f"kozak_mutagenesis_raw{tag}.csv"
            )
            summary_path = os.path.join(
                self.out_dir, f"kozak_mutagenesis_summary{tag}.csv"
            )
            stats_path = os.path.join(
                self.out_dir, f"kozak_mutagenesis_paired_stats{tag}.csv"
            )
            results.to_csv(raw_path, index=False)
            summarize_kozak_mutagenesis(results).to_csv(summary_path, index=False)
            calculate_paired_variant_statistics(results).to_csv(stats_path, index=False)
            print(f"Saved mutagenesis results to {raw_path}")
        return results


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * valid.size / np.arange(1, valid.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def summarize_kozak_mutagenesis(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize each designed variant without discarding transcript rows."""
    required = {
        "Start_Codon", "Kozak_Class", "Designed_Kozak_Score",
        "Relative_CDS_Translation", "Log2_Relative_CDS_Translation",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    summary = (
        results.groupby(
            ["Start_Codon", "Kozak_Class", "Designed_Kozak_Score"],
            observed=True,
        )
        .agg(
            N=("Sample_ID", "nunique"),
            Mean_Relative_CDS_Translation=("Relative_CDS_Translation", "mean"),
            Median_Relative_CDS_Translation=("Relative_CDS_Translation", "median"),
            Mean_Log2_Fold_Change=("Log2_Relative_CDS_Translation", "mean"),
            SD_Log2_Fold_Change=("Log2_Relative_CDS_Translation", "std"),
        )
        .reset_index()
    )
    return summary


def calculate_paired_variant_statistics(results: pd.DataFrame) -> pd.DataFrame:
    """Compare every variant with its matched strong-ATG reference."""
    required = {"Sample_ID", "Variant", "Is_WT", "Log2_Relative_CDS_Translation"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")

    rows = []
    for variant, group in results.groupby("Variant", observed=True):
        values = group["Log2_Relative_CDS_Translation"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        is_wt = bool(group["Is_WT"].all())
        if is_wt:
            statistic, p_value = 0.0, 1.0
        elif len(values) >= 2 and np.any(values.to_numpy() != 0):
            statistic, p_value = wilcoxon(values.to_numpy(), alternative="two-sided")
        else:
            statistic, p_value = np.nan, np.nan
        rows.append({
            "Variant": variant,
            "N": len(values),
            "Median_Log2_Fold_Change": values.median() if len(values) else np.nan,
            "Wilcoxon_Statistic": statistic,
            "P_Value": p_value,
            "Is_WT": is_wt,
        })
    stats = pd.DataFrame(rows)
    stats["FDR_BH"] = _benjamini_hochberg(stats["P_Value"])
    return stats


def _correlation_summary(
    results: pd.DataFrame,
    effect_col: str,
) -> Tuple[float, float, float, int]:
    valid = results[[
        "Sample_ID", "Designed_Kozak_Score", effect_col
    ]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3:
        return np.nan, np.nan, np.nan, 0
    if (
        valid["Designed_Kozak_Score"].nunique() < 2
        or valid[effect_col].nunique() < 2
    ):
        pooled_r, pooled_p = np.nan, np.nan
    else:
        pooled_r, pooled_p = spearmanr(
            valid["Designed_Kozak_Score"], valid[effect_col]
        )
    within = []
    for _, group in valid.groupby("Sample_ID", observed=True):
        if (
            len(group) >= 3
            and group["Designed_Kozak_Score"].nunique() >= 2
            and group[effect_col].nunique() >= 2
        ):
            value, _ = spearmanr(group["Designed_Kozak_Score"], group[effect_col])
            if np.isfinite(value):
                within.append(value)
    median_within = float(np.median(within)) if within else np.nan
    return float(pooled_r), float(pooled_p), median_within, len(within)


def _suffix_tag(suffix: str) -> str:
    return f".{suffix}" if suffix else ""


def _effect_axis_label(effect_col: str) -> str:
    labels = {
        "Relative_CDS_Translation": "Relative downstream CDS translation",
        "Relative_Start_Proximal_Translation": (
            "Relative start-proximal CDS translation"
        ),
        "CDS_Mean_Signal": "Predicted downstream CDS translation",
        "Start_Proximal_Mean_Signal": (
            "Predicted start-proximal CDS translation"
        ),
    }
    return labels.get(effect_col, effect_col.replace("_", " "))


def plot_kozak_mutagenesis_boxplot(
    results: pd.DataFrame,
    out_dir: str,
    suffix: str = "",
    effect_col: str = "Relative_CDS_Translation",
    y_label: Optional[str] = None,
    width: float = 5.5,
    height: float = 5.0,
    y_limits: Optional[Tuple[float, float]] = None,
) -> str:
    """Draw the codon-by-context comparison corresponding to the motif plot."""
    if effect_col not in results:
        raise ValueError(f"effect_col '{effect_col}' is not present in results.")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(width, height))
    base_positions = np.arange(len(START_CODON_ORDER), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(KOZAK_CONTEXT_ORDER))
    box_width = 0.15

    for context_index, context in enumerate(KOZAK_CONTEXT_ORDER):
        data = []
        positions = []
        for codon_index, codon in enumerate(START_CODON_ORDER):
            values = results.loc[
                (results["Start_Codon"] == codon)
                & (results["Kozak_Class"] == context),
                effect_col,
            ].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(values):
                data.append(values)
                positions.append(base_positions[codon_index] + offsets[context_index])
        if not data:
            continue
        box = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": KOZAK_CONTEXT_COLORS[context], "linewidth": 1.2},
            whiskerprops={"color": KOZAK_CONTEXT_COLORS[context], "linewidth": 0.8},
            capprops={"color": KOZAK_CONTEXT_COLORS[context], "linewidth": 0.8},
            boxprops={"color": KOZAK_CONTEXT_COLORS[context], "linewidth": 1.0},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("white")
        ax.plot([], [], color=KOZAK_CONTEXT_COLORS[context], lw=2, label=context)

    medians = (
        results.groupby("Start_Codon", observed=True)[effect_col]
        .median()
        .reindex(START_CODON_ORDER)
    )
    valid_trend = medians.dropna()
    if len(valid_trend) >= 2:
        trend_x = [START_CODON_ORDER.index(codon) for codon in valid_trend.index]
        ax.plot(
            trend_x,
            valid_trend.to_numpy(),
            color="#E64B35",
            linestyle="--",
            linewidth=1.4,
            marker="o",
            markersize=3,
            zorder=0,
        )

    pooled_r, pooled_p, median_within, n_within = _correlation_summary(
        results, effect_col
    )
    label = (
        f"Pooled Spearman ρ = {pooled_r:.3f}\n"
        f"Median within-transcript ρ = {median_within:.3f} (n={n_within})\n"
        f"P = {pooled_p:.2e}"
    )
    ax.text(0.02, 0.03, label, transform=ax.transAxes, ha="left", va="bottom")
    ax.axhline(1.0, color="#808080", linestyle=":", linewidth=0.9)
    ax.set_xticks(base_positions)
    ax.set_xticklabels(START_CODON_ORDER)
    ax.set_xlabel("Start codon")
    ax.set_ylabel(y_label or _effect_axis_label(effect_col))
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=7,
    )
    fig.tight_layout()
    path = os.path.join(
        out_dir, f"start_codon_kozak_mutagenesis_boxplot{_suffix_tag(suffix)}.pdf"
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_score_panel(
    ax,
    data: pd.DataFrame,
    effect_col: str,
    title: Optional[str] = None,
    point_color: str = "#2C3E50",
):
    valid = data[["Designed_Kozak_Score", effect_col]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    ax.scatter(
        valid["Designed_Kozak_Score"],
        valid[effect_col],
        s=6,
        alpha=0.12,
        color=point_color,
        linewidths=0,
        rasterized=True,
    )
    if len(valid) >= 2 and valid["Designed_Kozak_Score"].nunique() >= 2:
        grouped = valid.groupby("Designed_Kozak_Score", observed=True)[effect_col]
        centers = grouped.median().sort_index()
        ax.plot(
            centers.index,
            centers.values,
            color="#E67E22",
            linestyle="--",
            marker="o",
            markersize=3,
            linewidth=1.2,
        )
        if valid[effect_col].nunique() >= 2:
            rho, p_value = spearmanr(
                valid["Designed_Kozak_Score"], valid[effect_col]
            )
        else:
            rho, p_value = np.nan, np.nan
        ax.text(
            0.04,
            0.96,
            f"ρ = {rho:.3f}\nP = {p_value:.2e}",
            transform=ax.transAxes,
            va="top",
        )
    if title:
        ax.set_title(title)


def plot_kozak_mutagenesis_score_scatter(
    results: pd.DataFrame,
    out_dir: str,
    suffix: str = "",
    effect_col: str = "Relative_CDS_Translation",
    width: float = 12.0,
    height: float = 3.3,
) -> str:
    """Draw score-effect relationships separately for each start codon."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(width, height), sharex=False, sharey=True)
    for ax, codon in zip(axes, START_CODON_ORDER):
        _draw_score_panel(
            ax,
            results.loc[results["Start_Codon"] == codon],
            effect_col,
            title=codon,
        )
        ax.set_xlabel("Designed Kozak score")
    axes[0].set_ylabel(_effect_axis_label(effect_col))
    fig.tight_layout()
    path = os.path.join(
        out_dir, f"kozak_mutagenesis_score_scatter{_suffix_tag(suffix)}.pdf"
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_global_kozak_mutagenesis_correlation(
    results: pd.DataFrame,
    out_dir: str,
    suffix: str = "",
    effect_col: str = "Relative_CDS_Translation",
    width: float = 5.5,
    height: float = 4.8,
) -> str:
    """Draw the global designed-strength versus matched-effect relationship."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(width, height))
    _draw_score_panel(ax, results, effect_col)
    pooled_r, pooled_p, median_within, n_within = _correlation_summary(
        results, effect_col
    )
    ax.text(
        0.97,
        0.04,
        (
            f"Pooled ρ = {pooled_r:.3f}, P = {pooled_p:.2e}\n"
            f"Median within-transcript ρ = {median_within:.3f} (n={n_within})"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    ax.axhline(1.0, color="#808080", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Designed Kozak score (start codon + positions -3 and +4)")
    ax.set_ylabel(_effect_axis_label(effect_col))
    fig.tight_layout()
    path = os.path.join(
        out_dir, f"global_kozak_mutagenesis_correlation{_suffix_tag(suffix)}.pdf"
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_kozak_mutagenesis_results(
    results: Union[pd.DataFrame, str, os.PathLike],
    out_dir: str,
    suffix: str = "",
    effect_col: str = "Relative_CDS_Translation",
    y_limits: Optional[Tuple[float, float]] = None,
) -> Dict[str, str]:
    """Create all PDF panels from a result table or its CSV path."""
    if not isinstance(results, pd.DataFrame):
        results = pd.read_csv(results)
    if results.empty:
        raise ValueError("The mutagenesis result table is empty.")
    return {
        "boxplot": plot_kozak_mutagenesis_boxplot(
            results,
            out_dir,
            suffix=suffix,
            effect_col=effect_col,
            y_limits=y_limits,
        ),
        "per_codon_scatter": plot_kozak_mutagenesis_score_scatter(
            results, out_dir, suffix=suffix, effect_col=effect_col
        ),
        "global_scatter": plot_global_kozak_mutagenesis_correlation(
            results, out_dir, suffix=suffix, effect_col=effect_col
        ),
    }


def evaluate_start_codon_kozak_mutagenesis(
    model,
    dataset,
    out_dir: str = "./results/kozak_mutagenesis",
    target_cell_type: Optional[Union[str, Sequence[str]]] = None,
    target_transcript_ids: Optional[
        Union[Iterable[str], Mapping[str, Iterable[str]]]
    ] = None,
    min_cds_nt: int = 21,
    max_cds_nt: Optional[int] = None,
    num_samples: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    cds_skip_codons: int = 5,
    start_window_codons: int = 10,
    prediction_scale: str = "log1p",
    suffix: str = "",
    y_limits: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Collect samples, evaluate all variants, and create the three PDF plots."""
    samples = collect_kozak_mutagenesis_samples(
        dataset=dataset,
        target_cell_type=target_cell_type,
        target_transcript_ids=target_transcript_ids,
        min_cds_nt=min_cds_nt,
        max_cds_nt=max_cds_nt,
        num_samples=num_samples,
    )
    if not samples:
        raise ValueError("No eligible annotated CDS samples were found.")
    evaluator = KozakMutagenesisEvaluator(
        model=model,
        out_dir=out_dir,
        prediction_scale=prediction_scale,
    )
    results = evaluator.evaluate(
        samples=samples,
        batch_size=batch_size,
        num_workers=num_workers,
        cds_skip_codons=cds_skip_codons,
        start_window_codons=start_window_codons,
        suffix=suffix,
        save_csv=True,
    )
    paths = plot_kozak_mutagenesis_results(
        results=results,
        out_dir=out_dir,
        suffix=suffix,
        y_limits=y_limits,
    )
    return results, paths

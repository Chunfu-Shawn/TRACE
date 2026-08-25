"""Matched RBP-motif perturbation and de novo translation-motif discovery.

The primary effect estimate is paired within the same transcript and model
environment. A positive delta means that disrupting the native motif lowers
predicted CDS translation; a negative delta means that disruption raises it.
These quantities describe model sensitivity and should not be interpreted as
biological causality without orthogonal RBP-binding and perturbation evidence.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact, wilcoxon
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from eval.save_prediction_results import (
    _autocast_context,
    _extract_head_tensor,
    _model_device,
)
from model.base_model import BaseModel
from utils import unwrap_model


BASES = np.asarray(list("ACGT"))
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}
def _normalize_tid(value: object) -> str:
    tid = str(value)
    if tid.startswith("ENST"):
        return tid.split(".", 1)[0]
    return tid


def _meta_value(meta_info, names: Sequence[str], default=None):
    for name in names:
        if isinstance(meta_info, Mapping) and name in meta_info:
            return meta_info[name]
        if hasattr(meta_info, name):
            return getattr(meta_info, name)
    return default


def _extract_tid(uuid: object, meta_info) -> str:
    value = _meta_value(
        meta_info,
        ("Tid", "tid", "transcript_id", "transcript", "tx_id"),
        default=None,
    )
    if value is None:
        uuid_text = str(uuid)
        value = uuid_text.rsplit("-", 2)[0] if "-" in uuid_text else uuid_text
    return _normalize_tid(value)


def _as_sequence_embedding(value) -> np.ndarray:
    array = (
        value.detach().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )
    if array.ndim != 2:
        raise ValueError(
            f"Sequence embedding must be two-dimensional, got {array.shape}."
        )
    if array.shape[1] != 4 and array.shape[0] == 4:
        array = array.T
    if array.shape[1] != 4:
        raise ValueError(
            f"Sequence embedding must have four channels, got {array.shape}."
        )
    return np.asarray(array, dtype=np.float32)


def _as_expression_vector(value) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    array = (
        value.detach().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _embedding_to_sequence(sequence_embedding: np.ndarray) -> str:
    valid = np.asarray(sequence_embedding).sum(axis=1) > 0
    indices = np.asarray(sequence_embedding).argmax(axis=1)
    return "".join(
        BASES[index] if is_valid else "N"
        for index, is_valid in zip(indices, valid)
    )


def normalize_pwm(pwm: np.ndarray, pseudocount: float = 1e-4) -> np.ndarray:
    """Normalize an A/C/G/T PWM or count matrix to row probabilities."""
    matrix = np.asarray(pwm, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError(f"PWM must have shape (length, 4), got {matrix.shape}.")
    if len(matrix) == 0 or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("PWM values must be finite, non-negative, and non-empty.")
    matrix = matrix + float(pseudocount)
    return matrix / matrix.sum(axis=1, keepdims=True)


def validate_rbp_pwm_library(
    pwm_library: Mapping[str, np.ndarray],
    metadata: Optional[pd.DataFrame] = None,
    target_rbps: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Validate relevant PWM matrices and return an audit table.

    Matrices that cannot be interpreted as finite, non-negative A/C/G/T
    probabilities or counts are excluded instead of terminating a full RBP
    analysis. Negative PSSM/log-odds matrices are deliberately not transformed
    because doing so without their original background model changes meaning.
    """
    normalized_library = {
        str(matrix_id).strip(): pwm
        for matrix_id, pwm in pwm_library.items()
    }
    target_set = (
        None if target_rbps is None else {str(value) for value in target_rbps}
    )

    rbp_names_by_matrix = defaultdict(set)
    if metadata is None:
        relevant_matrix_ids = list(normalized_library)
    else:
        required = {"Matrix_id", "Gene_name"}
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"RBP metadata is missing columns: {sorted(missing)}")
        motif_rows = metadata.dropna(subset=["Matrix_id", "Gene_name"]).copy()
        motif_rows["Matrix_id"] = motif_rows["Matrix_id"].astype(str).str.strip()
        motif_rows["Gene_name"] = motif_rows["Gene_name"].astype(str)
        if target_set is not None:
            motif_rows = motif_rows[motif_rows["Gene_name"].isin(target_set)]
        for matrix_id, rbp_name in motif_rows[["Matrix_id", "Gene_name"]].itertuples(
            index=False, name=None
        ):
            rbp_names_by_matrix[matrix_id].add(rbp_name)
        relevant_matrix_ids = list(dict.fromkeys(motif_rows["Matrix_id"]))

    valid_pwms = {}
    audit_records = []
    for matrix_id in relevant_matrix_ids:
        rbp_names = ";".join(sorted(rbp_names_by_matrix.get(matrix_id, [])))
        if matrix_id not in normalized_library:
            audit_records.append({
                "Matrix_ID": matrix_id,
                "RBP_Names": rbp_names,
                "Status": "Missing",
                "Reason": "Matrix ID is present in metadata but absent from pwm_library.",
                "Shape": "",
                "Nonfinite_Count": np.nan,
                "Negative_Count": np.nan,
                "Minimum": np.nan,
                "Maximum": np.nan,
            })
            continue

        raw_pwm = normalized_library[matrix_id]
        try:
            matrix = np.asarray(raw_pwm, dtype=float)
            shape = str(tuple(matrix.shape))
            finite_values = matrix[np.isfinite(matrix)]
            nonfinite_count = int(np.size(matrix) - finite_values.size)
            negative_count = int((finite_values < 0).sum())
            minimum = float(finite_values.min()) if finite_values.size else np.nan
            maximum = float(finite_values.max()) if finite_values.size else np.nan
            valid_pwms[matrix_id] = normalize_pwm(matrix)
            status = "Valid"
            reason = ""
        except (TypeError, ValueError) as error:
            matrix = np.asarray(raw_pwm)
            shape = str(tuple(matrix.shape))
            try:
                numeric = np.asarray(raw_pwm, dtype=float)
                finite_values = numeric[np.isfinite(numeric)]
                nonfinite_count = int(np.size(numeric) - finite_values.size)
                negative_count = int((finite_values < 0).sum())
                minimum = (
                    float(finite_values.min()) if finite_values.size else np.nan
                )
                maximum = (
                    float(finite_values.max()) if finite_values.size else np.nan
                )
            except (TypeError, ValueError):
                nonfinite_count = np.nan
                negative_count = np.nan
                minimum = np.nan
                maximum = np.nan
            status = "Invalid"
            reason = str(error)
        audit_records.append({
            "Matrix_ID": matrix_id,
            "RBP_Names": rbp_names,
            "Status": status,
            "Reason": reason,
            "Shape": shape,
            "Nonfinite_Count": nonfinite_count,
            "Negative_Count": negative_count,
            "Minimum": minimum,
            "Maximum": maximum,
        })

    audit = pd.DataFrame(audit_records)
    if not audit.empty:
        counts = audit["Status"].value_counts().to_dict()
        print(
            "PWM validation: "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
        rejected = audit[audit["Status"].ne("Valid")]
        if not rejected.empty:
            preview = ", ".join(rejected["Matrix_ID"].astype(str).head(10))
            suffix = " ..." if len(rejected) > 10 else ""
            print(
                f"Skipped {len(rejected)} invalid/missing PWM matrices: "
                f"{preview}{suffix}"
            )
    return valid_pwms, audit


def pwm_consensus(pwm: np.ndarray) -> str:
    """Return the maximum-probability consensus sequence of a PWM."""
    matrix = normalize_pwm(pwm)
    return "".join(BASES[matrix.argmax(axis=1)])


def scan_pwm_hits(
    sequence: str,
    pwm: np.ndarray,
    score_threshold: float = 0.80,
    background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> pd.DataFrame:
    """Vectorize forward-strand PWM scanning and return normalized hits.

    The normalized score maps the theoretical minimum and maximum log-odds
    scores of the PWM to 0 and 1. RBP motifs are scanned in transcript
    orientation; reverse-complement scanning is intentionally not performed.
    """
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be within [0, 1].")
    matrix = normalize_pwm(pwm)
    bg = np.asarray(background, dtype=float)
    if bg.shape != (4,) or (bg <= 0).any() or not np.isfinite(bg).all():
        raise ValueError("background must contain four positive finite values.")
    bg = bg / bg.sum()
    log_odds = np.log2(matrix / bg[None, :])
    minimum = float(log_odds.min(axis=1).sum())
    maximum = float(log_odds.max(axis=1).sum())
    denominator = maximum - minimum
    motif_length = len(matrix)
    sequence = str(sequence).upper().replace("U", "T")
    if len(sequence) < motif_length or denominator <= 0:
        return pd.DataFrame(columns=[
            "Start", "End", "PWM_Score", "Raw_Log_Odds", "Sequence"
        ])

    indices = np.fromiter(
        (BASE_TO_INDEX.get(base, -1) for base in sequence),
        dtype=np.int8,
        count=len(sequence),
    )
    windows = np.lib.stride_tricks.sliding_window_view(indices, motif_length)
    valid = (windows >= 0).all(axis=1)
    raw_scores = np.full(len(windows), np.nan, dtype=float)
    if valid.any():
        valid_windows = windows[valid]
        raw_scores[valid] = log_odds[
            np.arange(motif_length)[None, :], valid_windows
        ].sum(axis=1)
    normalized = (raw_scores - minimum) / denominator
    selected = np.flatnonzero(valid & (normalized >= score_threshold))
    return pd.DataFrame({
        "Start": selected.astype(int),
        "End": (selected + motif_length).astype(int),
        "PWM_Score": normalized[selected],
        "Raw_Log_Odds": raw_scores[selected],
        "Sequence": [sequence[start:start + motif_length] for start in selected],
    })


def collect_unique_transcript_samples(
    dataset,
    target_transcript_ids: Optional[Iterable[str]] = None,
    num_transcripts: Optional[int] = None,
    min_length: int = 0,
    max_length: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Dict]:
    """Collect one representative dataset row per transcript."""
    if min_length < 0 or (
        max_length is not None and max_length < min_length
    ):
        raise ValueError("Invalid transcript-length bounds.")
    allowed = (
        None
        if target_transcript_ids is None
        else {_normalize_tid(value) for value in target_transcript_ids}
    )
    rng = np.random.default_rng(random_state)
    representatives: Dict[str, Dict] = {}
    occurrence_counts: Dict[str, int] = defaultdict(int)
    exclusions: Counter = Counter()

    for dataset_index in tqdm(
        range(len(dataset)), desc="Collect unique transcript samples"
    ):
        try:
            item = dataset[dataset_index]
            if len(item) < 6:
                exclusions["missing_fields"] += 1
                continue
            uuid, species, cell_type, expr_vector, meta_info, seq_emb = item[:6]
            tid = _extract_tid(uuid, meta_info)
            if allowed is not None and tid not in allowed:
                exclusions["target_filter"] += 1
                continue
            sequence_embedding = _as_sequence_embedding(seq_emb)
            transcript_length = len(sequence_embedding)
            if transcript_length < min_length or (
                max_length is not None and transcript_length > max_length
            ):
                exclusions["length_filter"] += 1
                continue
            cds_start = int(_meta_value(
                meta_info,
                ("cds_start_pos", "CDS_Start", "cds_start"),
                -1,
            )) - 1
            cds_end = int(_meta_value(
                meta_info,
                ("cds_end_pos", "CDS_End", "cds_end"),
                -1,
            ))
            cds_end = min(cds_end, transcript_length)
            if cds_start < 0 or cds_end <= cds_start:
                exclusions["invalid_cds"] += 1
                continue
            occurrence_counts[tid] += 1
            if (
                tid in representatives
                and rng.random() >= 1.0 / occurrence_counts[tid]
            ):
                continue
            representatives[tid] = {
                "Sample_ID": tid,
                "Dataset_Index": dataset_index,
                "UUID": str(uuid),
                "Tid": tid,
                "Species": species,
                "Cell_Type": str(cell_type),
                "Expr_Vector": _as_expression_vector(expr_vector),
                "Seq_Emb": np.array(sequence_embedding, copy=True),
                "Sequence": _embedding_to_sequence(sequence_embedding),
                "Transcript_Length": transcript_length,
                "CDS_Start_0based": cds_start,
                "CDS_End_exclusive": cds_end,
            }
        except (TypeError, ValueError, IndexError):
            exclusions["malformed_sample"] += 1

    tids = np.asarray(list(representatives), dtype=object)
    if num_transcripts is not None and len(tids) > num_transcripts:
        selected = set(rng.choice(tids, num_transcripts, replace=False))
        representatives = {
            tid: sample for tid, sample in representatives.items()
            if tid in selected
        }
    print(
        f"Collected {len(representatives)} unique transcripts from "
        f"{len(dataset)} dataset rows."
    )
    if exclusions:
        print("Exclusions: " + ", ".join(
            f"{key}={value}" for key, value in sorted(exclusions.items())
        ))
    return representatives


def _collect_selected_hit_samples(
    dataset,
    selected_hits: pd.DataFrame,
) -> Dict[str, Dict]:
    """Collect dataset samples matching selected transcript and cell pairs."""
    requested_cells = defaultdict(set)
    has_cell_type = "Cell_Type" in selected_hits.columns
    for row in selected_hits.itertuples(index=False):
        tid = _normalize_tid(row.Tid)
        if has_cell_type and pd.notna(row.Cell_Type):
            requested_cells[tid].add(str(row.Cell_Type))
        else:
            requested_cells[tid]

    representatives: Dict[str, Dict] = {}
    for dataset_index in tqdm(
        range(len(dataset)), desc="Collect missing targeted samples"
    ):
        try:
            item = dataset[dataset_index]
            if len(item) < 6:
                continue
            uuid, species, cell_type, expr_vector, meta_info, seq_emb = item[:6]
            tid = _extract_tid(uuid, meta_info)
            if tid not in requested_cells:
                continue
            required_cells = requested_cells[tid]
            if required_cells and str(cell_type) not in required_cells:
                continue
            sequence_embedding = _as_sequence_embedding(seq_emb)
            transcript_length = len(sequence_embedding)
            cds_start = int(_meta_value(
                meta_info,
                ("cds_start_pos", "CDS_Start", "cds_start"),
                -1,
            )) - 1
            cds_end = int(_meta_value(
                meta_info,
                ("cds_end_pos", "CDS_End", "cds_end"),
                -1,
            ))
            cds_end = min(cds_end, transcript_length)
            if cds_start < 0 or cds_end <= cds_start:
                continue
            representatives[tid] = {
                "Sample_ID": tid,
                "Dataset_Index": dataset_index,
                "UUID": str(uuid),
                "Tid": tid,
                "Species": species,
                "Cell_Type": str(cell_type),
                "Expr_Vector": _as_expression_vector(expr_vector),
                "Seq_Emb": np.array(sequence_embedding, copy=True),
                "Sequence": _embedding_to_sequence(sequence_embedding),
                "Transcript_Length": transcript_length,
                "CDS_Start_0based": cds_start,
                "CDS_End_exclusive": cds_end,
            }
            if len(representatives) == len(requested_cells):
                break
        except (TypeError, ValueError, IndexError):
            continue
    return representatives


def _region_bounds(sample: Mapping, region: str) -> Tuple[int, int]:
    start = int(sample["CDS_Start_0based"])
    end = int(sample["CDS_End_exclusive"])
    length = int(sample["Transcript_Length"])
    bounds = {
        "5UTR": (0, start),
        "CDS": (start, end),
        "3UTR": (end, length),
    }
    if region not in bounds:
        raise ValueError(f"Unknown transcript region '{region}'.")
    return bounds[region]


def _non_overlapping_top_hits(
    candidates: Sequence[Dict],
    maximum_hits: int,
) -> Sequence[Dict]:
    selected = []
    for candidate in sorted(
        candidates, key=lambda row: row["PWM_Score"], reverse=True
    ):
        overlaps = any(
            candidate["Start"] < previous["End"]
            and previous["Start"] < candidate["End"]
            for previous in selected
        )
        if not overlaps:
            selected.append(candidate)
        if len(selected) >= maximum_hits:
            break
    return selected


_RBP_SCAN_PROCESS_STATE = {}


def _scan_one_rbp_transcript(
    item,
    motif_groups,
    prepared_pwms,
    pwm_consensuses,
    regions,
    score_threshold,
    max_hits_per_rbp_transcript_region,
    context_flank,
):
    """Scan one compact transcript record for known RBP motifs."""
    tid, sample = item
    transcript_records = []
    sequence = str(sample["Sequence"])
    for region in regions:
        region_start, region_end = _region_bounds(sample, region)
        region_sequence = sequence[region_start:region_end]
        if not region_sequence:
            continue
        for rbp_name, matrix_ids in motif_groups:
            candidates = []
            for matrix_id in matrix_ids:
                pwm = prepared_pwms[matrix_id]
                hits = scan_pwm_hits(
                    region_sequence,
                    pwm,
                    score_threshold=score_threshold,
                )
                for hit in hits.to_dict("records"):
                    absolute_start = region_start + int(hit["Start"])
                    absolute_end = region_start + int(hit["End"])
                    context_start = max(0, absolute_start - context_flank)
                    context_end = min(
                        len(sequence), absolute_end + context_flank
                    )
                    candidates.append({
                        "Tid": tid,
                        "RBP_Name": rbp_name,
                        "Matrix_ID": matrix_id,
                        "Region": region,
                        "Start": absolute_start,
                        "End": absolute_end,
                        "PWM_Length": len(pwm),
                        "PWM_Score": float(hit["PWM_Score"]),
                        "Raw_Log_Odds": float(hit["Raw_Log_Odds"]),
                        "Motif_Sequence": hit["Sequence"],
                        "PWM_Consensus": pwm_consensuses[matrix_id],
                        "Context_Start": context_start,
                        "Context_End": context_end,
                        "Context_Sequence": sequence[
                            context_start:context_end
                        ],
                    })
            selected = _non_overlapping_top_hits(
                candidates,
                max_hits_per_rbp_transcript_region,
            )
            transcript_records.extend(selected)
    return transcript_records


def _initialize_rbp_scan_process(
    motif_groups,
    prepared_pwms,
    pwm_consensuses,
    regions,
    score_threshold,
    max_hits_per_rbp_transcript_region,
    context_flank,
):
    """Initialize immutable scan state once in each worker process."""
    global _RBP_SCAN_PROCESS_STATE
    _RBP_SCAN_PROCESS_STATE = {
        "motif_groups": motif_groups,
        "prepared_pwms": prepared_pwms,
        "pwm_consensuses": pwm_consensuses,
        "regions": regions,
        "score_threshold": score_threshold,
        "max_hits_per_rbp_transcript_region": (
            max_hits_per_rbp_transcript_region
        ),
        "context_flank": context_flank,
    }


def _scan_rbp_transcript_chunk(sample_chunk):
    """Scan a transcript chunk using process-local immutable state."""
    state = _RBP_SCAN_PROCESS_STATE
    chunk_records = []
    for item in sample_chunk:
        chunk_records.extend(_scan_one_rbp_transcript(item, **state))
    return chunk_records


def collect_rbp_motif_hits(
    samples: Mapping[str, Mapping],
    pwm_library: Mapping[str, np.ndarray],
    metadata: pd.DataFrame,
    target_rbps: Optional[Iterable[str]] = None,
    regions: Sequence[str] = ("5UTR", "CDS", "3UTR"),
    score_threshold: float = 0.85,
    max_hits_per_rbp_transcript_region: int = 1,
    context_flank: int = 12,
    num_workers: int = 1,
    scan_backend: str = "thread",
    scan_chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Scan known RBP PWMs and retain non-overlapping top hits.

    Transcripts are independent scan units. The process backend initializes
    shared scan configuration once per worker and submits compact transcript
    chunks to reduce inter-process communication overhead.
    """
    required = {"Matrix_id", "Gene_name"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"RBP metadata is missing columns: {sorted(missing)}")
    if max_hits_per_rbp_transcript_region < 1 or context_flank < 0:
        raise ValueError("Hit limit must be positive and context_flank non-negative.")
    if int(num_workers) < 1:
        raise ValueError("num_workers must be at least 1.")
    scan_backend = str(scan_backend).lower()
    if scan_backend not in {"thread", "process"}:
        raise ValueError("scan_backend must be 'thread' or 'process'.")
    if scan_chunk_size is not None and int(scan_chunk_size) < 1:
        raise ValueError("scan_chunk_size must be at least 1 when provided.")
    regions = tuple(str(region) for region in regions)
    for region in regions:
        if region not in {"5UTR", "CDS", "3UTR"}:
            raise ValueError(f"Unsupported region '{region}'.")
    target_set = (
        None if target_rbps is None else {str(value) for value in target_rbps}
    )
    normalized_library = {
        str(matrix_id): pwm for matrix_id, pwm in pwm_library.items()
    }
    motif_rows = metadata.dropna(subset=["Matrix_id", "Gene_name"]).copy()
    motif_rows["Matrix_id"] = motif_rows["Matrix_id"].astype(str)
    motif_rows["Gene_name"] = motif_rows["Gene_name"].astype(str)
    motif_rows = motif_rows[motif_rows["Matrix_id"].isin(normalized_library)]
    if target_set is not None:
        motif_rows = motif_rows[motif_rows["Gene_name"].isin(target_set)]
    motif_rows = motif_rows.drop_duplicates(["Gene_name", "Matrix_id"])
    if motif_rows.empty:
        return pd.DataFrame()

    prepared_pwms = {}
    rejected_matrix_ids = []
    for matrix_id in motif_rows["Matrix_id"].unique():
        try:
            prepared_pwms[matrix_id] = normalize_pwm(
                normalized_library[matrix_id]
            )
        except (TypeError, ValueError):
            rejected_matrix_ids.append(matrix_id)
    if rejected_matrix_ids:
        print(
            f"Skipped {len(rejected_matrix_ids)} invalid PWM matrices during "
            "motif scanning. Use validate_rbp_pwm_library() for details."
        )
        motif_rows = motif_rows[
            motif_rows["Matrix_id"].isin(prepared_pwms)
        ]
    if motif_rows.empty:
        return pd.DataFrame()
    motif_by_rbp = motif_rows.groupby("Gene_name", observed=True)[
        "Matrix_id"
    ].apply(list)
    motif_groups = list(motif_by_rbp.items())
    pwm_consensuses = {
        matrix_id: pwm_consensus(pwm)
        for matrix_id, pwm in prepared_pwms.items()
    }

    sample_items = [
        (
            tid,
            {
                "Sequence": str(sample["Sequence"]),
                "Transcript_Length": int(sample["Transcript_Length"]),
                "CDS_Start_0based": int(sample["CDS_Start_0based"]),
                "CDS_End_exclusive": int(sample["CDS_End_exclusive"]),
            },
        )
        for tid, sample in samples.items()
    ]
    if not sample_items:
        return pd.DataFrame()
    scan_kwargs = {
        "motif_groups": motif_groups,
        "prepared_pwms": prepared_pwms,
        "pwm_consensuses": pwm_consensuses,
        "regions": regions,
        "score_threshold": score_threshold,
        "max_hits_per_rbp_transcript_region": (
            max_hits_per_rbp_transcript_region
        ),
        "context_flank": context_flank,
    }
    records = []
    if int(num_workers) == 1:
        for item in tqdm(sample_items, desc="Scan known RBP motifs"):
            records.extend(_scan_one_rbp_transcript(item, **scan_kwargs))
    elif scan_backend == "thread":
        print(
            f"Scanning known RBP motifs with {int(num_workers)} worker threads."
        )
        with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
            iterator = executor.map(
                lambda item: _scan_one_rbp_transcript(item, **scan_kwargs),
                sample_items,
            )
            for transcript_records in tqdm(
                iterator,
                total=len(sample_items),
                desc="Scan known RBP motifs",
            ):
                records.extend(transcript_records)
    else:
        chunk_size = (
            int(scan_chunk_size)
            if scan_chunk_size is not None
            else max(1, math.ceil(len(sample_items) / (int(num_workers) * 4)))
        )
        sample_chunks = [
            sample_items[start:start + chunk_size]
            for start in range(0, len(sample_items), chunk_size)
        ]
        effective_workers = min(int(num_workers), len(sample_chunks))
        print(
            f"Scanning known RBP motifs with {effective_workers} worker "
            f"processes in {len(sample_chunks)} chunks "
            f"({chunk_size} transcripts/chunk)."
        )
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_rbp_scan_process,
            initargs=(
                motif_groups,
                prepared_pwms,
                pwm_consensuses,
                regions,
                score_threshold,
                max_hits_per_rbp_transcript_region,
                context_flank,
            ),
        ) as executor:
            futures = {
                executor.submit(_scan_rbp_transcript_chunk, sample_chunk): index
                for index, sample_chunk in enumerate(sample_chunks)
            }
            completed_chunks = {}
            for future in tqdm(
                as_completed(futures),
                total=len(sample_chunks),
                desc="Scan known RBP motif chunks",
            ):
                completed_chunks[futures[future]] = future.result()
            for index in range(len(sample_chunks)):
                records.extend(completed_chunks[index])
    hits = pd.DataFrame(records)
    if hits.empty:
        return hits
    hits = hits.sort_values(
        ["RBP_Name", "Tid", "Region", "PWM_Score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    hits.insert(0, "Hit_ID", [f"RBP_HIT_{i:07d}" for i in range(len(hits))])
    return hits


def _fixed_metagene_position_bin(
    position: float,
    region_start: int,
    region_end: int,
    region: str,
    bin_size: int,
    utr5_length: int,
    cds_length: int,
    utr3_length: int,
) -> Tuple[Optional[int], Optional[float]]:
    """Map one position using exact UTR distance and scaled CDS distance."""
    region_length = int(region_end) - int(region_start)
    if region_length <= 0:
        return None, None
    if region == "5UTR":
        metagene_position = float(position) - float(region_end)
        if metagene_position < -utr5_length or metagene_position >= 0:
            return None, None
        local_bin = int((metagene_position + utr5_length) // bin_size)
        return local_bin, metagene_position
    if region == "CDS":
        relative = (float(position) - float(region_start)) / region_length
        metagene_position = min(max(relative * cds_length, 0.0), cds_length - 1e-9)
        local_bin = int(metagene_position // bin_size)
        return local_bin, metagene_position
    if region == "3UTR":
        distance_from_tts = float(position) - float(region_start)
        if distance_from_tts < 0 or distance_from_tts >= utr3_length:
            return None, None
        metagene_position = cds_length + distance_from_tts
        local_bin = int(distance_from_tts // bin_size)
        return local_bin, metagene_position
    raise ValueError(f"Unknown transcript region '{region}'.")


def _overlapping_exact_matches(sequence: str, motif: str) -> Iterable[int]:
    """Yield overlapping exact motif starts in transcript orientation."""
    start = 0
    while True:
        start = sequence.find(motif, start)
        if start < 0:
            return
        yield start
        start += 1


def build_motif_position_profiles(
    samples: Mapping[str, Mapping],
    known_hits: Optional[pd.DataFrame] = None,
    de_novo_motifs: Optional[pd.DataFrame] = None,
    bin_size: int = 20,
    utr5_length: int = 300,
    cds_length: int = 600,
    utr3_length: int = 300,
    bins_per_region: Optional[int] = None,
    known_rbp_names: Optional[Iterable[str]] = None,
    pseudocount: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """Build fixed-coordinate metagene profiles for known and de novo motifs.

    UTR coordinates retain exact nucleotide distance from the TIS/TTS and are
    cropped to user-defined windows. CDS coordinates are proportionally scaled
    to ``cds_length``. ``Spatial_Probability`` is the fraction of a feature's
    displayed hits in each bin. ``Log2_Positional_Enrichment`` compares the
    opportunity-adjusted bin hit rate against the full-transcript background
    hit rate. Known-RBP positions use retained PWM hits; de novo k-mers are
    rescanned across all sampled transcripts.
    """
    if bin_size < 1:
        raise ValueError("bin_size must be positive.")
    if bins_per_region is not None:
        if bins_per_region < 3:
            raise ValueError("bins_per_region must be at least 3.")
        utr5_length = bins_per_region * bin_size
        cds_length = bins_per_region * bin_size
        utr3_length = bins_per_region * bin_size
    fixed_lengths = {
        "5UTR": int(utr5_length),
        "CDS": int(cds_length),
        "3UTR": int(utr3_length),
    }
    for region, length in fixed_lengths.items():
        if length < bin_size or length % bin_size != 0:
            raise ValueError(
                f"{region} fixed length ({length}) must be a positive "
                f"multiple of bin_size ({bin_size})."
            )
    if pseudocount <= 0 or not np.isfinite(pseudocount):
        raise ValueError("pseudocount must be positive and finite.")

    regions = ("5UTR", "CDS", "3UTR")
    region_bin_counts = {
        region: fixed_lengths[region] // bin_size for region in regions
    }
    region_offsets = {
        "5UTR": 0,
        "CDS": region_bin_counts["5UTR"],
        "3UTR": region_bin_counts["5UTR"] + region_bin_counts["CDS"],
    }
    def metagene_position(region: str, bin_index: int) -> float:
        if region == "5UTR":
            return -fixed_lengths["5UTR"] + (bin_index + 0.5) * bin_size
        if region == "CDS":
            return (bin_index + 0.5) * bin_size
        return fixed_lengths["CDS"] + (bin_index + 0.5) * bin_size
    allowed_rbps = (
        None
        if known_rbp_names is None
        else {str(value) for value in known_rbp_names}
    )

    def region_lengths() -> Dict[str, np.ndarray]:
        lengths = {region: [] for region in regions}
        for sample in samples.values():
            for region in regions:
                start, end = _region_bounds(sample, region)
                lengths[region].append(max(int(end) - int(start), 0))
        return {
            region: np.asarray(values, dtype=int)
            for region, values in lengths.items()
        }

    sample_region_lengths = region_lengths()
    opportunity_cache = {}

    def opportunity_by_bin(motif_length: int, region: str) -> np.ndarray:
        """Count scannable motif starts in each fixed metagene bin."""
        cache_key = (int(motif_length), region)
        if cache_key in opportunity_cache:
            return opportunity_cache[cache_key]
        number_of_bins = region_bin_counts[region]
        opportunities = np.zeros(number_of_bins, dtype=float)
        length_frequencies = Counter(sample_region_lengths[region].tolist())
        for region_length, frequency in length_frequencies.items():
            number_of_starts = max(int(region_length) - motif_length + 1, 0)
            if number_of_starts == 0:
                continue
            centers = np.arange(number_of_starts, dtype=float) + motif_length / 2
            if region == "5UTR":
                positions = centers - region_length
                valid = positions >= -fixed_lengths["5UTR"]
                local_bins = np.floor(
                    (positions[valid] + fixed_lengths["5UTR"]) / bin_size
                ).astype(int)
            elif region == "CDS":
                positions = centers / region_length * fixed_lengths["CDS"]
                local_bins = np.floor(positions / bin_size).astype(int)
            else:
                positions = centers
                valid = positions < fixed_lengths["3UTR"]
                local_bins = np.floor(positions[valid] / bin_size).astype(int)
            local_bins = local_bins[
                (local_bins >= 0) & (local_bins < number_of_bins)
            ]
            opportunities += np.bincount(
                local_bins, minlength=number_of_bins
            ) * frequency
        opportunity_cache[cache_key] = opportunities
        return opportunities

    def assemble_profile(
        feature_type: str,
        feature_lengths: Mapping[str, int],
        counts: Mapping[Tuple[str, str, int], int],
        background_hit_totals: Optional[Mapping[str, int]] = None,
        annotations: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> pd.DataFrame:
        records = []
        annotations = annotations or {}
        background_hit_totals = background_hit_totals or {}
        for feature, motif_length_value in feature_lengths.items():
            motif_length = max(int(motif_length_value), 1)
            displayed_hits = sum(
                counts.get((feature, region, bin_index), 0)
                for region in regions
                for bin_index in range(region_bin_counts[region])
            )
            total_hits = int(background_hit_totals.get(
                feature, displayed_hits
            ))
            feature_opportunities = {
                region: opportunity_by_bin(motif_length, region)
                for region in regions
            }
            full_transcript_opportunity = float(sum(
                np.maximum(lengths - motif_length + 1, 0).sum()
                for lengths in sample_region_lengths.values()
            ))
            smoothed_global_rate = (
                (total_hits + pseudocount)
                / (full_transcript_opportunity + 1.0)
            )
            for region in regions:
                number_of_bins = region_bin_counts[region]
                region_hits = sum(
                    counts.get((feature, region, index), 0)
                    for index in range(number_of_bins)
                )
                for bin_index in range(number_of_bins):
                    hits = int(counts.get((feature, region, bin_index), 0))
                    bin_opportunity = float(
                        feature_opportunities[region][bin_index]
                    )
                    smoothed_bin_rate = (
                        (hits + pseudocount) / (bin_opportunity + 1.0)
                    )
                    enrichment = np.log2(
                        smoothed_bin_rate / smoothed_global_rate
                    )
                    record = {
                        "Feature_Type": feature_type,
                        "Feature": feature,
                        "Region": region,
                        "Region_Bin": bin_index,
                        "Region_Relative_Position": (
                            bin_index + 0.5
                        ) / number_of_bins,
                        "Global_Bin": region_offsets[region] + bin_index,
                        "Metagene_Position": metagene_position(
                            region, bin_index
                        ),
                        "Bin_Size": bin_size,
                        "Fixed_5UTR_Length": fixed_lengths["5UTR"],
                        "Fixed_CDS_Length": fixed_lengths["CDS"],
                        "Fixed_3UTR_Length": fixed_lengths["3UTR"],
                        "Hits": hits,
                        "Region_Hits": region_hits,
                        "Total_Hits": total_hits,
                        "Displayed_Hits": displayed_hits,
                        "Spatial_Probability": (
                            hits / displayed_hits
                            if displayed_hits > 0 else 0.0
                        ),
                        "Motif_Length": motif_length,
                        "Opportunity": bin_opportunity,
                        "Hit_Rate_Per_Kb": (
                            1000.0 * hits / bin_opportunity
                            if bin_opportunity > 0 else np.nan
                        ),
                        "Full_Transcript_Opportunity": (
                            full_transcript_opportunity
                        ),
                        "Full_Transcript_Hit_Rate_Per_Kb": (
                            1000.0 * total_hits / full_transcript_opportunity
                            if full_transcript_opportunity > 0 else np.nan
                        ),
                        "Log2_Positional_Enrichment": float(enrichment),
                        "Normalization": (
                            "opportunity-adjusted bin hit rate relative to "
                            "the full-transcript background hit rate"
                        ),
                    }
                    record.update(annotations.get(feature, {}))
                    records.append(record)
        return pd.DataFrame(records)

    known_counts: Counter = Counter()
    known_background_counts: Counter = Counter()
    known_lengths = {}
    known_annotations = {}
    if known_hits is not None and not known_hits.empty:
        required = {"Tid", "RBP_Name", "Start", "End", "Region"}
        missing = required.difference(known_hits.columns)
        if missing:
            raise ValueError(
                f"Known-hit table is missing columns: {sorted(missing)}"
            )
        working_hits = known_hits.copy()
        working_hits["RBP_Name"] = working_hits["RBP_Name"].astype(str)
        if allowed_rbps is not None:
            working_hits = working_hits[
                working_hits["RBP_Name"].isin(allowed_rbps)
            ]
        for rbp_name, group in working_hits.groupby("RBP_Name", observed=True):
            lengths = (
                group["PWM_Length"].to_numpy(float)
                if "PWM_Length" in group.columns
                else (group["End"] - group["Start"]).to_numpy(float)
            )
            finite_lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
            if finite_lengths.size:
                known_lengths[str(rbp_name)] = int(round(np.median(finite_lengths)))
        for hit in working_hits.itertuples(index=False):
            tid = str(hit.Tid)
            region = str(hit.Region)
            if tid not in samples or region not in region_offsets:
                continue
            known_background_counts[str(hit.RBP_Name)] += 1
            region_start, region_end = _region_bounds(samples[tid], region)
            center = (float(hit.Start) + float(hit.End)) / 2
            bin_index, _ = _fixed_metagene_position_bin(
                center,
                region_start,
                region_end,
                region,
                bin_size,
                fixed_lengths["5UTR"],
                fixed_lengths["CDS"],
                fixed_lengths["3UTR"],
            )
            if bin_index is not None:
                known_counts[(str(hit.RBP_Name), region, bin_index)] += 1

    de_novo_counts: Counter = Counter()
    de_novo_background_counts: Counter = Counter()
    de_novo_lengths = {}
    de_novo_annotations = {}
    if de_novo_motifs is not None and not de_novo_motifs.empty:
        required = {"Direction", "Kmer", "Region"}
        missing = required.difference(de_novo_motifs.columns)
        if missing:
            raise ValueError(
                f"De novo motif table is missing columns: {sorted(missing)}"
            )
        motif_records = []
        motif_table = de_novo_motifs.copy()
        if "Is_Cluster_Representative" in motif_table.columns:
            motif_table = motif_table[
                motif_table["Is_Cluster_Representative"].astype(bool)
            ]
        for row in motif_table.drop_duplicates(
            ["Region", "Direction", "Kmer"]
        ).itertuples(index=False):
            discovery_region = str(row.Region)
            direction = str(row.Direction)
            kmer = str(row.Kmer).upper().replace("U", "T")
            if not kmer or not set(kmer).issubset(BASE_TO_INDEX):
                continue
            feature = (
                f"{kmer} ({discovery_region}, {direction.lower()})"
            )
            de_novo_lengths[feature] = len(kmer)
            de_novo_annotations[feature] = {
                "Discovery_Region": discovery_region,
                "Direction": direction,
                "Kmer": kmer,
            }
            motif_records.append((feature, kmer, discovery_region))
        for sample in tqdm(
            samples.values(), desc="Scan de novo motif positions"
        ):
            sequence = str(sample["Sequence"]).upper().replace("U", "T")
            for region in regions:
                region_start, region_end = _region_bounds(sample, region)
                region_sequence = sequence[region_start:region_end]
                if not region_sequence:
                    continue
                for feature, kmer, discovery_region in motif_records:
                    if region != discovery_region:
                        continue
                    for local_start in _overlapping_exact_matches(
                        region_sequence, kmer
                    ):
                        de_novo_background_counts[feature] += 1
                        center = region_start + local_start + len(kmer) / 2
                        bin_index, _ = _fixed_metagene_position_bin(
                            center,
                            region_start,
                            region_end,
                            region,
                            bin_size,
                            fixed_lengths["5UTR"],
                            fixed_lengths["CDS"],
                            fixed_lengths["3UTR"],
                        )
                        if bin_index is not None:
                            de_novo_counts[(feature, region, bin_index)] += 1

    return {
        "known_rbp": assemble_profile(
            "Known RBP",
            known_lengths,
            known_counts,
            known_background_counts,
            known_annotations,
        ),
        "de_novo": assemble_profile(
            "De novo",
            de_novo_lengths,
            de_novo_counts,
            de_novo_background_counts,
            de_novo_annotations,
        ),
    }


def _least_preferred_alternative(
    pwm_row: np.ndarray,
    native_index: int,
) -> int:
    for index in np.argsort(pwm_row):
        if int(index) != native_index:
            return int(index)
    return int((native_index + 1) % 4)


def disrupt_pwm_hit(
    sequence_embedding: np.ndarray,
    start: int,
    pwm: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Replace each motif base by a low-probability alternative base."""
    mutated = np.array(sequence_embedding, dtype=np.float32, copy=True)
    matrix = normalize_pwm(pwm)
    if start < 0 or start + len(matrix) > len(mutated):
        raise ValueError("PWM hit lies outside the sequence embedding.")
    native = mutated[start:start + len(matrix)].argmax(axis=1)
    changes = 0
    for offset, native_index in enumerate(native):
        replacement = _least_preferred_alternative(
            matrix[offset], int(native_index)
        )
        mutated[start + offset] = 0
        mutated[start + offset, replacement] = 1
        changes += int(replacement != native_index)
    return mutated, changes


def _mean_cds_signal(
    profile: np.ndarray,
    cds_start: int,
    cds_end: int,
    skip_codons: int = 0,
) -> float:
    """Return the mean predicted signal across every retained CDS nucleotide."""
    start = int(cds_start) + 3 * int(skip_codons)
    end = min(int(cds_end), len(profile))
    if start >= end:
        return np.nan
    values = np.asarray(profile[start:end], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(values.mean())


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


class RBPMotifMutagenesisEvaluator:
    """Evaluate paired effects of disrupting known RBP motif instances."""

    def __init__(
        self,
        model,
        pwm_library: Mapping[str, np.ndarray],
        prediction_scale: str = "log1p",
    ):
        base_model = unwrap_model(model)
        if not isinstance(base_model, BaseModel):
            raise TypeError(
                "RBPMotifMutagenesisEvaluator requires a BaseModel instance."
            )
        if prediction_scale not in {"log1p", "linear"}:
            raise ValueError("prediction_scale must be 'log1p' or 'linear'.")
        self.model = base_model
        self.device = _model_device(base_model)
        self.pwm_library = {
            str(key): normalize_pwm(value)
            for key, value in pwm_library.items()
        }
        self.prediction_scale = prediction_scale

    def _predict_records(
        self,
        records: Sequence[Mapping],
        batch_size: int,
    ) -> Dict[str, np.ndarray]:
        predictions = {}
        self.model.eval()
        for batch_start in tqdm(
            range(0, len(records), batch_size), desc="RBP motif inference"
        ):
            batch = records[batch_start:batch_start + batch_size]
            sequences = [
                torch.from_numpy(np.asarray(row["Seq_Emb"], dtype=np.float32))
                for row in batch
            ]
            lengths = [len(sequence) for sequence in sequences]
            seq_batch = pad_sequence(
                sequences, batch_first=True, padding_value=-1
            ).to(self.device)
            expression_vectors = [
                torch.from_numpy(np.asarray(
                    row["Expr_Vector"], dtype=np.float32
                ))
                for row in batch
            ]
            widths = {int(vector.numel()) for vector in expression_vectors}
            if len(widths) != 1:
                raise ValueError(
                    "Expression-vector widths differ within an inference batch."
                )
            expr_batch = (
                None
                if next(iter(widths)) == 0
                else torch.stack(expression_vectors).to(self.device)
            )
            positions = torch.arange(
                seq_batch.shape[1], device=self.device
            ).unsqueeze(0)
            src_mask = positions < torch.tensor(
                lengths, device=self.device
            ).unsqueeze(1)
            with torch.inference_mode(), _autocast_context(self.device):
                output = self.model.predict(
                    seq_batch=seq_batch,
                    species=[row["Species"] for row in batch],
                    expr_vector=expr_batch,
                    src_mask=src_mask,
                    head_names=["count"],
                )
                profiles = _extract_head_tensor(output, "count")
            if profiles.ndim != 3 or profiles.shape[-1] != 1:
                raise ValueError(
                    "The count head must return shape (batch, length, 1)."
                )
            profiles = profiles.squeeze(-1).float()
            if self.prediction_scale == "log1p":
                profiles = torch.expm1(profiles)
            profiles = profiles.cpu().numpy()
            for index, row in enumerate(batch):
                predictions[str(row["Variant_ID"])] = profiles[
                    index, :lengths[index]
                ]
        return predictions

    def evaluate_hits(
        self,
        hits: pd.DataFrame,
        samples: Mapping[str, Mapping],
        batch_size: int = 32,
        cds_skip_codons: int = 0,
        eps: float = 1e-8,
    ) -> pd.DataFrame:
        """Disrupt every hit and calculate paired CDS translation effects."""
        if hits.empty:
            return pd.DataFrame()
        required = {
            "Hit_ID", "Tid", "RBP_Name", "Matrix_ID", "Region",
            "Start", "End", "PWM_Score", "Motif_Sequence",
        }
        missing = required.difference(hits.columns)
        if missing:
            raise ValueError(f"Hit table is missing columns: {sorted(missing)}")
        unknown_tids = set(hits["Tid"]) - set(samples)
        if unknown_tids:
            raise ValueError(
                f"Samples are missing {len(unknown_tids)} hit transcripts."
            )

        wt_records = []
        for tid in hits["Tid"].drop_duplicates():
            sample = samples[tid]
            wt_records.append({
                "Variant_ID": f"WT::{tid}",
                "Seq_Emb": sample["Seq_Emb"],
                "Expr_Vector": sample["Expr_Vector"],
                "Species": sample["Species"],
            })
        disrupted_records = []
        mutation_counts = {}
        for hit in hits.itertuples(index=False):
            matrix_id = str(hit.Matrix_ID)
            if matrix_id not in self.pwm_library:
                raise KeyError(f"PWM '{matrix_id}' is unavailable.")
            sample = samples[hit.Tid]
            disrupted, changes = disrupt_pwm_hit(
                sample["Seq_Emb"], int(hit.Start), self.pwm_library[matrix_id]
            )
            variant_id = f"DISRUPTED::{hit.Hit_ID}"
            mutation_counts[hit.Hit_ID] = changes
            disrupted_records.append({
                "Variant_ID": variant_id,
                "Seq_Emb": disrupted,
                "Expr_Vector": sample["Expr_Vector"],
                "Species": sample["Species"],
            })
        profiles = self._predict_records(
            wt_records + disrupted_records,
            batch_size=batch_size,
        )

        rows = []
        for hit in hits.itertuples(index=False):
            sample = samples[hit.Tid]
            wt_signal = _mean_cds_signal(
                profiles[f"WT::{hit.Tid}"],
                sample["CDS_Start_0based"],
                sample["CDS_End_exclusive"],
                skip_codons=cds_skip_codons,
            )
            disrupted_signal = _mean_cds_signal(
                profiles[f"DISRUPTED::{hit.Hit_ID}"],
                sample["CDS_Start_0based"],
                sample["CDS_End_exclusive"],
                skip_codons=cds_skip_codons,
            )
            effect = np.log2((wt_signal + eps) / (disrupted_signal + eps))
            rows.append({
                **hit._asdict(),
                "Cell_Type": sample["Cell_Type"],
                "CDS_Start_0based": sample["CDS_Start_0based"],
                "CDS_End_exclusive": sample["CDS_End_exclusive"],
                "Transcript_Length": sample["Transcript_Length"],
                "Mutation_Count": mutation_counts[hit.Hit_ID],
                "Disruption_Strategy": "least_preferred_all_positions",
                "CDS_Signal_Aggregation": "full_cds_nucleotide_mean",
                "WT_CDS_Mean_Signal": wt_signal,
                "Disrupted_CDS_Mean_Signal": disrupted_signal,
                "Delta_Log2_TE": effect,
                "Delta_Log2_TE_Per_Mutation": (
                    effect / max(mutation_counts[hit.Hit_ID], 1)
                ),
                "Direction": (
                    "Positive" if effect > 0
                    else "Negative" if effect < 0
                    else "Neutral"
                ),
            })
        return pd.DataFrame(rows)

    def compute_nucleotide_contributions(
        self,
        hit_effects: pd.DataFrame,
        samples: Mapping[str, Mapping],
        n_cases_per_direction: int = 3,
        min_case_transcripts: int = 5,
        context_flank: int = 12,
        batch_size: int = 64,
        cds_skip_codons: int = 0,
        case_regions: Sequence[str] = ("5UTR", "3UTR"),
        target_rbps: Optional[Iterable[str]] = None,
        target_regions: Optional[Iterable[str]] = None,
        target_hit_ids: Optional[Iterable[str]] = None,
        target_transcript_ids: Optional[Iterable[str]] = None,
        target_motif_starts: Optional[Iterable[int]] = None,
        targeted_cases_per_rbp: int = 3,
        selection_mode: str = "global",
        eps: float = 1e-8,
    ) -> pd.DataFrame:
        """Run saturation mutagenesis around representative UTR motif hits."""
        if hit_effects.empty:
            return pd.DataFrame()
        if (
            n_cases_per_direction < 1
            or min_case_transcripts < 2
            or context_flank < 0
        ):
            raise ValueError("Invalid case count or context flank.")
        allowed_regions = tuple(str(region) for region in case_regions)
        invalid_regions = set(allowed_regions).difference(
            {"5UTR", "CDS", "3UTR"}
        )
        if not allowed_regions or invalid_regions:
            raise ValueError(
                "case_regions must contain one or more of 5UTR, CDS, 3UTR."
            )
        target_rbps = (
            None if target_rbps is None
            else {str(target_rbps)} if isinstance(target_rbps, str)
            else {str(value) for value in target_rbps}
        )
        target_regions = (
            None if target_regions is None
            else {str(target_regions)} if isinstance(target_regions, str)
            else {str(value) for value in target_regions}
        )
        if target_regions is not None and target_regions.difference(
            {"5UTR", "CDS", "3UTR"}
        ):
            raise ValueError(
                "target_regions must contain only 5UTR, CDS, or 3UTR."
            )
        target_hit_ids = (
            None if target_hit_ids is None
            else [str(target_hit_ids)] if isinstance(target_hit_ids, str)
            else [str(value) for value in target_hit_ids]
        )
        target_transcript_ids = (
            None if target_transcript_ids is None
            else {str(target_transcript_ids)}
            if isinstance(target_transcript_ids, str)
            else {str(value) for value in target_transcript_ids}
        )
        target_motif_starts = (
            None if target_motif_starts is None
            else {int(target_motif_starts)}
            if np.isscalar(target_motif_starts)
            else {int(value) for value in target_motif_starts}
        )
        region_filter = (
            target_regions
            if target_regions is not None
            else None if target_hit_ids is not None
            else set(allowed_regions)
        )
        if region_filter is not None:
            hit_effects = hit_effects[
                hit_effects["Region"].astype(str).isin(region_filter)
            ].copy()
        if hit_effects.empty:
            return pd.DataFrame()
        transcript_group_effects = (
            hit_effects.groupby(
                ["RBP_Name", "Region", "Tid"], observed=True
            )["Delta_Log2_TE"]
            .median()
            .reset_index()
        )
        group_summary = (
            transcript_group_effects.groupby(
                ["RBP_Name", "Region"], observed=True
            )
            .agg(
                Group_Median_Delta_Log2_TE=("Delta_Log2_TE", "median"),
                Group_N_Transcripts=("Tid", "nunique"),
            )
            .reset_index()
        )
        all_group_summary = group_summary.copy()
        group_summary = group_summary[
            group_summary["Group_N_Transcripts"] >= min_case_transcripts
        ]
        group_summary["Direction"] = np.where(
            group_summary["Group_Median_Delta_Log2_TE"] > 0,
            "Positive",
            np.where(
                group_summary["Group_Median_Delta_Log2_TE"] < 0,
                "Negative",
                "Neutral",
            ),
        )
        if selection_mode not in {"global", "per_rbp"}:
            raise ValueError("selection_mode must be 'global' or 'per_rbp'.")
        targeted = selection_mode == "per_rbp" or any(
            value is not None for value in (
                target_rbps,
                target_regions,
                target_hit_ids,
                target_transcript_ids,
                target_motif_starts,
            )
        )
        if targeted_cases_per_rbp < 1:
            raise ValueError("targeted_cases_per_rbp must be positive.")

        if targeted:
            candidates = hit_effects.copy()
            if target_rbps is not None:
                candidates = candidates[
                    candidates["RBP_Name"].astype(str).isin(target_rbps)
                ]
            if target_regions is not None:
                candidates = candidates[
                    candidates["Region"].astype(str).isin(target_regions)
                ]
            if target_transcript_ids is not None:
                candidates = candidates[
                    candidates["Tid"].astype(str).isin(target_transcript_ids)
                ]
            if target_motif_starts is not None:
                candidates = candidates[
                    candidates["Start"].astype(int).isin(target_motif_starts)
                ]
            candidates = candidates.merge(
                all_group_summary[
                    [
                        "RBP_Name", "Region", "Group_Median_Delta_Log2_TE",
                        "Group_N_Transcripts",
                    ]
                ],
                on=["RBP_Name", "Region"],
                how="left",
            )
            if target_hit_ids is not None:
                order = {
                    hit_id: index
                    for index, hit_id in enumerate(target_hit_ids)
                }
                candidates = candidates[
                    candidates["Hit_ID"].astype(str).isin(order)
                ].copy()
                candidates["_Selection_Order"] = candidates[
                    "Hit_ID"
                ].astype(str).map(order)
                selected = candidates.sort_values("_Selection_Order")
            else:
                effect_sign = np.sign(candidates["Delta_Log2_TE"])
                group_sign = np.sign(
                    candidates["Group_Median_Delta_Log2_TE"]
                )
                candidates = candidates[
                    (effect_sign == group_sign) | (group_sign == 0)
                ].copy()
                candidates["_Absolute_Effect"] = candidates[
                    "Delta_Log2_TE"
                ].abs()
                selected = (
                    candidates.sort_values(
                        ["RBP_Name", "_Absolute_Effect", "Hit_ID"],
                        ascending=[True, False, True],
                    )
                    .groupby("RBP_Name", observed=True, group_keys=False)
                    .head(int(targeted_cases_per_rbp))
                )
        else:
            selected_rows = []
            for direction, ascending in (
                ("Positive", False), ("Negative", True)
            ):
                groups = group_summary[
                    group_summary["Direction"] == direction
                ]
                groups = groups.sort_values(
                    "Group_Median_Delta_Log2_TE", ascending=ascending
                ).drop_duplicates("RBP_Name").head(n_cases_per_direction)
                for group in groups.itertuples(index=False):
                    candidates = hit_effects[
                        (hit_effects["RBP_Name"] == group.RBP_Name)
                        & (hit_effects["Region"] == group.Region)
                    ].copy()
                    distance = (
                        candidates["Delta_Log2_TE"]
                        - group.Group_Median_Delta_Log2_TE
                    ).abs()
                    representative = candidates.loc[
                        distance.idxmin()
                    ].to_dict()
                    representative.update({
                        "Group_Median_Delta_Log2_TE": (
                            group.Group_Median_Delta_Log2_TE
                        ),
                        "Group_N_Transcripts": group.Group_N_Transcripts,
                    })
                    selected_rows.append(representative)
            selected = pd.DataFrame(selected_rows)
        if selected.empty:
            return pd.DataFrame()

        variant_records = []
        position_variants = defaultdict(list)
        alternative_variants = {}
        for hit in selected.itertuples(index=False):
            sample = samples[hit.Tid]
            sequence = sample["Sequence"]
            context_start = max(0, int(hit.Start) - context_flank)
            context_end = min(len(sequence), int(hit.End) + context_flank)
            for position in range(context_start, context_end):
                native_base = sequence[position]
                if native_base not in BASE_TO_INDEX:
                    continue
                for alternative in BASES:
                    if alternative == native_base:
                        continue
                    mutated = np.array(sample["Seq_Emb"], copy=True)
                    mutated[position] = 0
                    mutated[position, BASE_TO_INDEX[str(alternative)]] = 1
                    variant_id = (
                        f"ISM::{hit.Hit_ID}::{position}::{alternative}"
                    )
                    variant_records.append({
                        "Variant_ID": variant_id,
                        "Seq_Emb": mutated,
                        "Expr_Vector": sample["Expr_Vector"],
                        "Species": sample["Species"],
                    })
                    position_variants[(hit.Hit_ID, position)].append(variant_id)
                    alternative_variants[
                        (hit.Hit_ID, position, str(alternative))
                    ] = variant_id
        predictions = self._predict_records(
            variant_records,
            batch_size=batch_size,
        )

        rows = []
        for hit in selected.itertuples(index=False):
            sample = samples[hit.Tid]
            wt_signal = float(hit.WT_CDS_Mean_Signal)
            context_start = max(0, int(hit.Start) - context_flank)
            context_end = min(
                sample["Transcript_Length"], int(hit.End) + context_flank
            )
            for position in range(context_start, context_end):
                variant_ids = position_variants.get((hit.Hit_ID, position), [])
                if not variant_ids:
                    continue
                mutant_signals = [
                    _mean_cds_signal(
                        predictions[variant_id],
                        sample["CDS_Start_0based"],
                        sample["CDS_End_exclusive"],
                        skip_codons=cds_skip_codons,
                    )
                    for variant_id in variant_ids
                ]
                mean_log_mutant = np.mean(np.log2(
                    np.asarray(mutant_signals, dtype=float) + eps
                ))
                mean_alternative_contribution = (
                    np.log2(wt_signal + eps) - mean_log_mutant
                )
                is_motif = int(hit.Start) <= position < int(hit.End)
                least_preferred_base = None
                least_preferred_contribution = np.nan
                if is_motif:
                    matrix_id = str(hit.Matrix_ID)
                    if matrix_id not in self.pwm_library:
                        raise KeyError(f"PWM '{matrix_id}' is unavailable.")
                    motif_offset = position - int(hit.Start)
                    native_index = BASE_TO_INDEX[sample["Sequence"][position]]
                    least_preferred_index = _least_preferred_alternative(
                        self.pwm_library[matrix_id][motif_offset],
                        native_index,
                    )
                    least_preferred_base = str(BASES[least_preferred_index])
                    least_variant_id = alternative_variants[
                        (hit.Hit_ID, position, least_preferred_base)
                    ]
                    least_signal = _mean_cds_signal(
                        predictions[least_variant_id],
                        sample["CDS_Start_0based"],
                        sample["CDS_End_exclusive"],
                        skip_codons=cds_skip_codons,
                    )
                    least_preferred_contribution = np.log2(
                        (wt_signal + eps) / (least_signal + eps)
                    )
                rows.append({
                    "Hit_ID": hit.Hit_ID,
                    "Tid": hit.Tid,
                    "RBP_Name": hit.RBP_Name,
                    "Matrix_ID": hit.Matrix_ID,
                    "Region": hit.Region,
                    "Motif_Start": int(hit.Start),
                    "Motif_End": int(hit.End),
                    "Absolute_Position": position,
                    "Relative_Position": position - int(hit.Start),
                    "Base": sample["Sequence"][position],
                    "Is_Motif": is_motif,
                    "Base_Contribution_Log2_TE": (
                        mean_alternative_contribution
                    ),
                    "Base_Contribution_Mean_Alternatives_Log2_TE": (
                        mean_alternative_contribution
                    ),
                    "Base_Contribution_PWM_Least_Preferred_Log2_TE": (
                        least_preferred_contribution
                    ),
                    "PWM_Least_Preferred_Alternative": least_preferred_base,
                    "Motif_Delta_Log2_TE": float(hit.Delta_Log2_TE),
                    "Group_Median_Delta_Log2_TE": float(
                        hit.Group_Median_Delta_Log2_TE
                    ),
                    "Group_N_Transcripts": int(hit.Group_N_Transcripts),
                    "PWM_Score": float(hit.PWM_Score),
                    "CDS_Start_0based": sample["CDS_Start_0based"],
                    "CDS_End_exclusive": sample["CDS_End_exclusive"],
                    "Transcript_Length": sample["Transcript_Length"],
                })
        return pd.DataFrame(rows)


def run_targeted_rbp_saturation_mutagenesis(
    model,
    hit_effects: pd.DataFrame,
    samples: Mapping[str, Mapping],
    pwm_library: Mapping[str, np.ndarray],
    output_csv: Optional[str] = None,
    target_hit_ids: Optional[Iterable[str]] = None,
    target_rbps: Optional[Iterable[str]] = None,
    target_regions: Optional[Iterable[str]] = None,
    target_transcript_ids: Optional[Iterable[str]] = None,
    target_motif_starts: Optional[Iterable[int]] = None,
    context_flank: int = 12,
    prediction_scale: str = "log1p",
    batch_size: int = 64,
    cds_skip_codons: int = 0,
    max_hits: Optional[int] = None,
    hit_chunk_size: Optional[int] = 8,
    force_zero_expression: Optional[bool] = None,
    dataset=None,
) -> pd.DataFrame:
    """Run standalone saturation mutagenesis for explicitly selected hits.

    Existing ``output_csv`` files are authoritative and loaded directly. If
    selected transcripts are absent from ``samples``, ``dataset`` can recover
    the matching transcript/cell-type rows. Hits are processed in bounded
    chunks to control memory. Every selected position is substituted with each
    of the three non-native bases. The reported native-base contribution is the
    WT log2 CDS-mean signal minus the mean log2 CDS-mean signal across those
    three substitutions.
    """
    if output_csv is not None:
        output_csv = os.path.abspath(os.path.expanduser(output_csv))
        if os.path.isfile(output_csv):
            try:
                cached = pd.read_csv(output_csv)
            except pd.errors.EmptyDataError:
                cached = pd.DataFrame()
            print(f"[SKIP] targeted saturation: loaded {output_csv}")
            return cached
    if context_flank < 0:
        raise ValueError("context_flank must be non-negative.")
    if max_hits is not None and int(max_hits) < 1:
        raise ValueError("max_hits must be positive or None.")

    required = {
        "Hit_ID", "Tid", "RBP_Name", "Region", "Start", "End",
        "Matrix_ID", "WT_CDS_Mean_Signal", "Delta_Log2_TE", "PWM_Score",
    }
    missing = required.difference(hit_effects.columns)
    if missing:
        raise ValueError(
            f"Hit-effect table is missing columns: {sorted(missing)}"
        )

    def normalize_strings(values):
        if values is None:
            return None
        if isinstance(values, str):
            return [values]
        return list(dict.fromkeys(str(value) for value in values))

    hit_ids = normalize_strings(target_hit_ids)
    rbps = normalize_strings(target_rbps)
    regions = normalize_strings(target_regions)
    transcript_ids = normalize_strings(target_transcript_ids)
    motif_starts = (
        None if target_motif_starts is None
        else {int(target_motif_starts)}
        if np.isscalar(target_motif_starts)
        else {int(value) for value in target_motif_starts}
    )
    if all(
        value is None
        for value in (hit_ids, rbps, regions, transcript_ids, motif_starts)
    ):
        raise ValueError(
            "Specify at least one hit, RBP, region, transcript, or motif start."
        )
    if regions is not None:
        invalid_regions = set(regions).difference({"5UTR", "CDS", "3UTR"})
        if invalid_regions:
            raise ValueError(
                "target_regions contains unsupported values: "
                + ", ".join(sorted(invalid_regions))
            )

    selected = hit_effects.copy()
    selected["Tid"] = selected["Tid"].astype(str)
    if hit_ids is not None:
        hit_order = {hit_id: index for index, hit_id in enumerate(hit_ids)}
        selected = selected[
            selected["Hit_ID"].astype(str).isin(hit_order)
        ].copy()
        selected["_Selection_Order"] = selected["Hit_ID"].astype(str).map(
            hit_order
        )
        selected = selected.sort_values("_Selection_Order")
    if rbps is not None:
        selected = selected[selected["RBP_Name"].astype(str).isin(rbps)]
    if regions is not None:
        selected = selected[selected["Region"].astype(str).isin(regions)]
    if transcript_ids is not None:
        normalized_targets = {_normalize_tid(tid) for tid in transcript_ids}
        selected = selected[
            selected["Tid"].map(_normalize_tid).isin(normalized_targets)
        ]
    if motif_starts is not None:
        selected = selected[selected["Start"].astype(int).isin(motif_starts)]
    selected = selected.drop_duplicates("Hit_ID")
    if max_hits is not None:
        selected = selected.head(int(max_hits))
    if selected.empty:
        raise ValueError("No hit-effect row satisfies the requested filters.")
    if hit_chunk_size is not None and int(hit_chunk_size) < 1:
        raise ValueError("hit_chunk_size must be positive or None.")

    normalized_samples = {}
    for sample_key, sample in samples.items():
        aliases = {
            _normalize_tid(sample_key),
            _normalize_tid(sample.get("Tid", sample_key)),
            _normalize_tid(sample.get("Sample_ID", sample_key)),
        }
        for alias in aliases:
            normalized_samples.setdefault(alias, sample)

    resolved_samples = {}
    for tid in selected["Tid"].drop_duplicates():
        if tid in samples:
            resolved_samples[tid] = samples[tid]
        elif _normalize_tid(tid) in normalized_samples:
            resolved_samples[tid] = normalized_samples[_normalize_tid(tid)]

    missing_samples = sorted(set(selected["Tid"]) - set(resolved_samples))
    if missing_samples and dataset is not None:
        missing_hits = selected[selected["Tid"].isin(missing_samples)]
        supplemented = _collect_selected_hit_samples(dataset, missing_hits)
        for tid in missing_samples:
            sample = supplemented.get(_normalize_tid(tid))
            if sample is not None:
                resolved_samples[tid] = sample
        recovered = len(missing_samples) - len(
            set(missing_samples) - set(resolved_samples)
        )
        print(
            f"Recovered {recovered}/{len(missing_samples)} missing targeted "
            "transcripts from dataset."
        )
        missing_samples = sorted(
            set(selected["Tid"]) - set(resolved_samples)
        )
    if missing_samples:
        preview = ", ".join(map(str, missing_samples[:5]))
        dataset_hint = (
            " The supplied dataset does not contain matching transcript/cell "
            "pairs."
            if dataset is not None
            else " Pass dataset=<original dataset> to recover them."
        )
        raise ValueError(
            f"Samples are missing {len(missing_samples)} selected transcripts: "
            f"{preview}.{dataset_hint}"
        )

    if force_zero_expression is None:
        expression_vectors = [
            _as_expression_vector(sample.get("Expr_Vector"))
            for sample in resolved_samples.values()
        ]
        force_zero_expression = bool(expression_vectors) and all(
            vector.size == 0 or np.allclose(vector, 0)
            for vector in expression_vectors
        )
    if force_zero_expression:
        resolved_samples = {
            tid: {
                **sample,
                "Expr_Vector": np.zeros_like(
                    _as_expression_vector(sample.get("Expr_Vector"))
                ),
            }
            for tid, sample in resolved_samples.items()
        }
        print("Using zero expression conditioning for targeted saturation.")

    evaluator = RBPMotifMutagenesisEvaluator(
        model,
        pwm_library,
        prediction_scale=prediction_scale,
    )
    selected_hit_ids = selected["Hit_ID"].astype(str).tolist()
    chunk_size = len(selected_hit_ids) if hit_chunk_size is None else int(
        hit_chunk_size
    )
    contribution_chunks = []
    total_chunks = math.ceil(len(selected_hit_ids) / chunk_size)
    for chunk_index, start in enumerate(
        range(0, len(selected_hit_ids), chunk_size), start=1
    ):
        chunk_hit_ids = selected_hit_ids[start:start + chunk_size]
        print(
            f"[RUN] targeted saturation chunk {chunk_index}/{total_chunks}: "
            f"{len(chunk_hit_ids)} hits"
        )
        chunk = evaluator.compute_nucleotide_contributions(
            hit_effects,
            resolved_samples,
            n_cases_per_direction=1,
            min_case_transcripts=2,
            context_flank=context_flank,
            batch_size=batch_size,
            cds_skip_codons=cds_skip_codons,
            case_regions=("5UTR", "CDS", "3UTR"),
            target_hit_ids=chunk_hit_ids,
            targeted_cases_per_rbp=max(len(chunk_hit_ids), 1),
            selection_mode="per_rbp",
        )
        if not chunk.empty:
            contribution_chunks.append(chunk)
    contributions = (
        pd.concat(contribution_chunks, ignore_index=True)
        if contribution_chunks else pd.DataFrame()
    )
    if contributions.empty:
        raise ValueError("Targeted saturation mutagenesis produced no rows.")
    contributions["Alternative_Mutations_Per_Position"] = 3
    contributions["Contribution_Definition"] = (
        "log2(WT_CDS_mean)-mean(log2(single_base_mutant_CDS_mean))"
    )
    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        temporary_csv = f"{output_csv}.tmp.{os.getpid()}"
        contributions.to_csv(temporary_csv, index=False)
        os.replace(temporary_csv, output_csv)
        print(f"[DONE] targeted saturation: saved {output_csv}")
    return contributions


def summarize_rbp_motif_effects(
    hit_effects: pd.DataFrame,
    min_transcripts: int = 5,
    bootstrap_iterations: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Summarize motif effects at the transcript level with paired statistics."""
    required = {"RBP_Name", "Region", "Tid", "Delta_Log2_TE"}
    missing = required.difference(hit_effects.columns)
    if missing:
        raise ValueError(f"Effect table is missing columns: {sorted(missing)}")
    if min_transcripts < 2 or bootstrap_iterations < 100:
        raise ValueError("Use at least two transcripts and 100 bootstrap draws.")
    working_effects = hit_effects.copy()
    if "Delta_Log2_TE_Per_Mutation" not in working_effects:
        if "Mutation_Count" in working_effects:
            working_effects["Delta_Log2_TE_Per_Mutation"] = (
                working_effects["Delta_Log2_TE"]
                / working_effects["Mutation_Count"].clip(lower=1)
            )
        else:
            working_effects["Delta_Log2_TE_Per_Mutation"] = np.nan
    transcript_effects = (
        working_effects.groupby(
            ["RBP_Name", "Region", "Tid"], observed=True
        )[["Delta_Log2_TE", "Delta_Log2_TE_Per_Mutation"]]
        .median()
        .reset_index()
    )
    rng = np.random.default_rng(random_state)
    alpha = 1 - confidence_level
    rows = []
    for (rbp_name, region), group in transcript_effects.groupby(
        ["RBP_Name", "Region"], observed=True
    ):
        values = group["Delta_Log2_TE"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna().to_numpy()
        normalized_values = group["Delta_Log2_TE_Per_Mutation"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna().to_numpy()
        if len(values) < min_transcripts:
            continue
        boot = np.median(
            rng.choice(values, size=(bootstrap_iterations, len(values))),
            axis=1,
        )
        if np.any(values != 0):
            _, p_value = wilcoxon(values, alternative="two-sided")
        else:
            p_value = 1.0
        median = float(np.median(values))
        rows.append({
            "RBP_Name": rbp_name,
            "Region": region,
            "N_Transcripts": len(values),
            "Median_Delta_Log2_TE": median,
            "Mean_Delta_Log2_TE": float(np.mean(values)),
            "Median_Delta_Log2_TE_Per_Mutation": float(
                np.median(normalized_values)
            ) if len(normalized_values) else np.nan,
            "CI_Lower": float(np.quantile(boot, alpha / 2)),
            "CI_Upper": float(np.quantile(boot, 1 - alpha / 2)),
            "P_Value": float(p_value),
            "Direction": (
                "Positive" if median > 0
                else "Negative" if median < 0
                else "Neutral"
            ),
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["FDR_BH"] = _benjamini_hochberg(summary["P_Value"])
    return summary.sort_values(
        "Median_Delta_Log2_TE", ascending=False
    ).reset_index(drop=True)


def _peak_overlapping_kmers(sequence: str, k: int, peak_offset: int) -> set:
    """Return valid k-mers whose span contains the nominated peak base."""
    sequence = str(sequence).upper().replace("U", "T")
    peak_offset = int(peak_offset)
    first_start = max(0, peak_offset - k + 1)
    last_start = min(peak_offset, len(sequence) - k)
    if first_start > last_start:
        return set()
    return {
        sequence[start:start + k]
        for start in range(first_start, last_start + 1)
        if set(sequence[start:start + k]).issubset(BASE_TO_INDEX)
    }


def _position_region(position: int, sample: Mapping) -> str:
    if position < int(sample["CDS_Start_0based"]):
        return "5UTR"
    if position < int(sample["CDS_End_exclusive"]):
        return "CDS"
    return "3UTR"


def _select_signed_peaks(
    scores: np.ndarray,
    valid_positions: np.ndarray,
    direction: str,
    number_of_peaks: int,
    minimum_separation: int,
) -> Sequence[int]:
    candidates = valid_positions[np.isfinite(scores[valid_positions])]
    if direction == "Positive":
        candidates = candidates[scores[candidates] > 0]
        order = candidates[np.argsort(scores[candidates])[::-1]]
    elif direction == "Negative":
        candidates = candidates[scores[candidates] < 0]
        order = candidates[np.argsort(scores[candidates])]
    else:
        raise ValueError("direction must be 'Positive' or 'Negative'.")
    selected = []
    for position in order:
        if all(
            abs(int(position) - previous) >= minimum_separation
            for previous in selected
        ):
            selected.append(int(position))
        if len(selected) >= number_of_peaks:
            break
    return selected


def extract_signed_translation_attribution_windows(
    model,
    samples: Mapping[str, Mapping],
    prediction_scale: str = "log1p",
    num_transcripts: Optional[int] = 500,
    peaks_per_direction: int = 1,
    window_radius: int = 10,
    target_regions: Sequence[str] = ("5UTR", "3UTR"),
    cds_skip_codons: int = 0,
    random_state: int = 42,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Extract signed UTR input-gradient peaks for predicted CDS translation.

    The target is the log full-nucleotide CDS mean. Positive native-base
    gradients nominate sequence positions that locally support the target;
    negative gradients nominate positions that locally suppress it. This
    first-order attribution is intended for candidate generation, not proof.
    """
    base_model = unwrap_model(model)
    if not isinstance(base_model, BaseModel):
        raise TypeError(
            "Signed translation attribution requires a BaseModel instance."
        )
    if prediction_scale not in {"log1p", "linear"}:
        raise ValueError("prediction_scale must be 'log1p' or 'linear'.")
    if peaks_per_direction < 1 or window_radius < 1 or cds_skip_codons < 0:
        raise ValueError("Invalid attribution-window parameters.")
    target_regions = tuple(str(region) for region in target_regions)
    invalid_regions = set(target_regions).difference(
        {"5UTR", "CDS", "3UTR"}
    )
    if not target_regions or invalid_regions:
        raise ValueError(
            "target_regions must contain one or more of 5UTR, CDS, 3UTR."
        )
    device = _model_device(base_model)
    rng = np.random.default_rng(random_state)
    tids = np.asarray(list(samples), dtype=object)
    if num_transcripts is not None and len(tids) > num_transcripts:
        tids = rng.choice(tids, num_transcripts, replace=False)
    else:
        rng.shuffle(tids)
    base_model.eval()
    records = []
    for tid in tqdm(tids, desc="Signed translation attribution"):
        sample = samples[str(tid)]
        sequence = str(sample["Sequence"])
        sequence_tensor = torch.from_numpy(
            np.asarray(sample["Seq_Emb"], dtype=np.float32)
        ).unsqueeze(0).to(device).requires_grad_(True)
        expression = np.asarray(sample["Expr_Vector"], dtype=np.float32)
        expression_tensor = (
            None
            if expression.size == 0
            else torch.from_numpy(expression).unsqueeze(0).to(device)
        )
        mask = torch.ones(
            (1, sequence_tensor.shape[1]), dtype=torch.bool, device=device
        )
        with torch.enable_grad():
            output = base_model.forward(
                seq_batch=sequence_tensor,
                species=sample["Species"],
                cell_type=None,
                expr_vector=expression_tensor,
                src_mask=mask,
                head_names=["count"],
            )
            profile = _extract_head_tensor(output, "count")[0, :, 0]
            if prediction_scale == "log1p":
                profile = torch.expm1(profile)
            cds_start = int(sample["CDS_Start_0based"]) + 3 * cds_skip_codons
            cds_end = min(
                int(sample["CDS_End_exclusive"]), len(profile)
            )
            if cds_start >= cds_end:
                continue
            target = torch.log(
                torch.clamp(profile[cds_start:cds_end], min=0).mean() + eps
            )
            base_model.zero_grad(set_to_none=True)
            target.backward()
        gradient = sequence_tensor.grad[0].detach().cpu().numpy()
        native_indices = np.asarray(sample["Seq_Emb"]).argmax(axis=1)
        native_scores = gradient[
            np.arange(len(native_indices)), native_indices
        ]
        valid_positions = []
        for region in target_regions:
            region_start, region_end = _region_bounds(sample, region)
            first = max(region_start + window_radius, window_radius)
            last = min(
                region_end - window_radius,
                len(sequence) - window_radius,
            )
            for position in range(first, last):
                context = sequence[
                    position - window_radius:position + window_radius + 1
                ]
                if "N" not in context:
                    valid_positions.append(position)
        valid = np.asarray(sorted(set(valid_positions)), dtype=int)
        for direction in ("Positive", "Negative"):
            peak_positions = _select_signed_peaks(
                native_scores,
                valid,
                direction=direction,
                number_of_peaks=peaks_per_direction,
                minimum_separation=2 * window_radius + 1,
            )
            for position in peak_positions:
                context_start = position - window_radius
                context_end = position + window_radius + 1
                records.append({
                    "Tid": str(tid),
                    "Cell_Type": sample["Cell_Type"],
                    "Region": _position_region(position, sample),
                    "Absolute_Position": position,
                    "Attribution_Direction": direction,
                    "Signed_Attribution": float(native_scores[position]),
                    "Context_Start": context_start,
                    "Context_End": context_end,
                    "Context_Sequence": sequence[context_start:context_end],
                    "Peak_Offset": position - context_start,
                    "Native_Base": sequence[position],
                })
        sequence_tensor.grad = None
    return pd.DataFrame(records, columns=[
        "Tid", "Cell_Type", "Region", "Absolute_Position",
        "Attribution_Direction", "Signed_Attribution", "Context_Start",
        "Context_End", "Context_Sequence", "Peak_Offset", "Native_Base",
    ])


def discover_de_novo_translation_motifs(
    sequence_effects: pd.DataFrame,
    sequence_col: str = "Context_Sequence",
    effect_col: str = "Delta_Log2_TE",
    unit_col: str = "Tid",
    region_col: str = "Region",
    peak_offset_col: str = "Peak_Offset",
    discovery_regions: Sequence[str] = ("5UTR", "3UTR"),
    k_values: Sequence[int] = (5, 6, 7, 8),
    extreme_quantile: float = 0.75,
    neutral_quantile: float = 0.40,
    min_foreground_occurrences: int = 5,
    top_n_per_direction: int = 10,
    logo_flank: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Sequence[str]]]:
    """Discover non-redundant UTR motifs against region-matched backgrounds.

    Positive and negative foregrounds are contrasted only with neutral windows
    from the same RNA region. Nested exact k-mers are connected into clusters;
    one statistically strongest representative per cluster is retained. Logo
    alignments remain centered on the attribution peak rather than on the first
    occurrence of the representative k-mer.
    """
    required = {sequence_col, effect_col, unit_col, region_col}
    missing = required.difference(sequence_effects.columns)
    if missing:
        raise ValueError(
            f"Sequence-effect table is missing columns: {sorted(missing)}"
        )
    if not 0.5 <= extreme_quantile < 1:
        raise ValueError("extreme_quantile must be within [0.5, 1).")
    if not 0 < neutral_quantile < 1:
        raise ValueError("neutral_quantile must be within (0, 1).")
    if logo_flank < 1:
        raise ValueError("logo_flank must be positive.")
    discovery_regions = tuple(str(region) for region in discovery_regions)
    invalid_regions = set(discovery_regions).difference(
        {"5UTR", "CDS", "3UTR"}
    )
    if not discovery_regions or invalid_regions:
        raise ValueError(
            "discovery_regions must contain one or more of 5UTR, CDS, 3UTR."
        )
    selected_columns = [unit_col, sequence_col, effect_col, region_col]
    if peak_offset_col in sequence_effects.columns:
        selected_columns.append(peak_offset_col)
    working = sequence_effects[selected_columns].copy()
    working = working.replace([np.inf, -np.inf], np.nan).dropna()
    working = working[
        working[region_col].astype(str).isin(discovery_regions)
    ].copy()
    working = working.loc[
        working.groupby([region_col, unit_col], observed=True)[effect_col]
        .transform(lambda values: values.abs() == values.abs().max())
    ].drop_duplicates([region_col, unit_col])
    if len(working) < 2 * min_foreground_occurrences:
        return pd.DataFrame(), {}

    records = []
    foreground_tables = {}
    for region, region_df in working.groupby(region_col, observed=True):
        neutral_cutoff = region_df[effect_col].abs().quantile(neutral_quantile)
        background = region_df[region_df[effect_col].abs() <= neutral_cutoff]
        if len(background) < min_foreground_occurrences:
            continue
        direction_sets = {}
        positive_values = region_df.loc[region_df[effect_col] > 0, effect_col]
        negative_values = region_df.loc[region_df[effect_col] < 0, effect_col]
        if len(positive_values) >= min_foreground_occurrences:
            cutoff = positive_values.quantile(extreme_quantile)
            direction_sets["Positive"] = region_df[
                region_df[effect_col] >= cutoff
            ]
        if len(negative_values) >= min_foreground_occurrences:
            cutoff = negative_values.quantile(1 - extreme_quantile)
            direction_sets["Negative"] = region_df[
                region_df[effect_col] <= cutoff
            ]
        for direction, foreground in direction_sets.items():
            foreground_tables[(str(region), direction)] = foreground.copy()
            for k in k_values:
                if k < 3:
                    raise ValueError("All k-mer lengths must be at least 3.")
                fg_sets = [
                    _peak_overlapping_kmers(
                        row[sequence_col],
                        k,
                        row.get(
                            peak_offset_col,
                            len(str(row[sequence_col])) // 2,
                        ),
                    )
                    for _, row in foreground.iterrows()
                ]
                bg_sets = [
                    _peak_overlapping_kmers(
                        row[sequence_col],
                        k,
                        row.get(
                            peak_offset_col,
                            len(str(row[sequence_col])) // 2,
                        ),
                    )
                    for _, row in background.iterrows()
                ]
                candidates = set().union(*fg_sets) if fg_sets else set()
                for kmer in candidates:
                    fg_hit = sum(kmer in values for values in fg_sets)
                    if fg_hit < min_foreground_occurrences:
                        continue
                    bg_hit = sum(kmer in values for values in bg_sets)
                    table = [
                        [fg_hit, len(fg_sets) - fg_hit],
                        [bg_hit, len(bg_sets) - bg_hit],
                    ]
                    odds_ratio, p_value = fisher_exact(
                        table, alternative="greater"
                    )
                    fg_rate = fg_hit / len(fg_sets)
                    bg_rate = bg_hit / len(bg_sets)
                    records.append({
                        "Region": str(region),
                        "Direction": direction,
                        "Kmer": kmer,
                        "K": k,
                        "Foreground_Hits": fg_hit,
                        "Foreground_N": len(fg_sets),
                        "Background_Hits": bg_hit,
                        "Background_N": len(bg_sets),
                        "Foreground_Rate": fg_rate,
                        "Background_Rate": bg_rate,
                        "Log2_Enrichment": float(np.log2(
                            (fg_rate + 0.5 / len(fg_sets))
                            / (bg_rate + 0.5 / len(bg_sets))
                        )),
                        "Odds_Ratio": float(odds_ratio),
                        "P_Value": float(p_value),
                        "Background_Matching": "same_region_neutral_windows",
                    })
    results = pd.DataFrame(records)
    if results.empty:
        return results, {}
    results["FDR_BH"] = results.groupby(
        ["Region", "Direction"], observed=True
    )["P_Value"].transform(_benjamini_hochberg)

    cluster_ids = pd.Series(index=results.index, dtype=object)
    cluster_sizes = pd.Series(index=results.index, dtype=int)
    representative_flags = pd.Series(False, index=results.index, dtype=bool)
    parent_motifs = pd.Series(index=results.index, dtype=object)
    cluster_members = pd.Series(index=results.index, dtype=object)
    next_cluster = 1
    for (region, direction), group in results.groupby(
        ["Region", "Direction"], observed=True
    ):
        indices = list(group.index)
        parents = {index: index for index in indices}

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left_pos, left_index in enumerate(indices):
            left_kmer = str(results.at[left_index, "Kmer"])
            for right_index in indices[left_pos + 1:]:
                right_kmer = str(results.at[right_index, "Kmer"])
                if left_kmer in right_kmer or right_kmer in left_kmer:
                    union(left_index, right_index)
        components = defaultdict(list)
        for index in indices:
            components[find(index)].append(index)
        for component in components.values():
            ranked = results.loc[component].sort_values(
                ["FDR_BH", "Log2_Enrichment", "Foreground_Hits", "K"],
                ascending=[True, False, False, False],
            )
            representative = ranked.index[0]
            cluster_label = f"M{next_cluster:04d}"
            next_cluster += 1
            cluster_ids.loc[component] = cluster_label
            cluster_sizes.loc[component] = len(component)
            representative_flags.loc[representative] = True
            parent_motifs.loc[component] = results.at[representative, "Kmer"]
            member_text = ",".join(sorted(
                results.loc[component, "Kmer"].astype(str).unique(),
                key=lambda value: (len(value), value),
            ))
            cluster_members.loc[component] = member_text
    results["Motif_Cluster"] = cluster_ids
    results["Cluster_Size"] = cluster_sizes.astype(int)
    results["Is_Cluster_Representative"] = representative_flags
    results["Cluster_Representative_Kmer"] = parent_motifs
    results["Cluster_Members"] = cluster_members
    results = results.sort_values(
        ["Region", "Direction", "FDR_BH", "Log2_Enrichment"],
        ascending=[True, True, True, False],
    )
    results = results[results["Is_Cluster_Representative"]].groupby(
        ["Region", "Direction"], observed=True
    ).head(top_n_per_direction).reset_index(drop=True)

    alignments: Dict[str, Sequence[str]] = {}
    for row in results.itertuples(index=False):
        aligned = []
        foreground = foreground_tables[(row.Region, row.Direction)]
        for foreground_row in foreground.itertuples(index=False):
            sequence = str(getattr(foreground_row, sequence_col))
            if row.Kmer not in sequence:
                continue
            peak_offset = (
                int(getattr(foreground_row, peak_offset_col))
                if peak_offset_col in foreground.columns
                else len(sequence) // 2
            )
            start = max(0, peak_offset - logo_flank)
            end = min(len(sequence), peak_offset + logo_flank + 1)
            if end - start == 2 * logo_flank + 1:
                aligned.append(sequence[start:end])
        alignments[f"{row.Region}|{row.Direction}|{row.Kmer}"] = aligned
        results.loc[
            (results["Region"] == row.Region)
            & (results["Direction"] == row.Direction)
            & (results["Kmer"] == row.Kmer),
            "Logo_Center_Offset",
        ] = logo_flank
        results.loc[
            (results["Region"] == row.Region)
            & (results["Direction"] == row.Direction)
            & (results["Kmer"] == row.Kmer),
            "Logo_Alignment",
        ] = "attribution_peak_centered"
    return results, alignments


def run_rbp_translation_effect_analysis(
    model,
    dataset,
    pwm_library: Mapping[str, np.ndarray],
    metadata: pd.DataFrame,
    out_dir: str,
    target_rbps: Optional[Iterable[str]] = None,
    target_transcript_ids: Optional[Iterable[str]] = None,
    regions: Sequence[str] = ("5UTR", "CDS", "3UTR"),
    num_transcripts: Optional[int] = None,
    score_threshold: float = 0.85,
    max_hits_per_rbp_transcript_region: int = 1,
    context_flank: int = 12,
    known_motif_scan_workers: int = 1,
    scan_backend: str = "thread",
    scan_chunk_size: Optional[int] = None,
    prediction_scale: str = "log1p",
    force_zero_expression: bool = True,
    batch_size: int = 32,
    min_transcripts: int = 5,
    n_cases_per_direction: int = 3,
    case_regions: Sequence[str] = ("5UTR", "3UTR"),
    case_target_rbps: Optional[Iterable[str]] = None,
    case_target_regions: Optional[Iterable[str]] = None,
    case_target_hit_ids: Optional[Iterable[str]] = None,
    case_target_transcript_ids: Optional[Iterable[str]] = None,
    case_target_motif_starts: Optional[Iterable[int]] = None,
    targeted_cases_per_rbp: int = 3,
    case_selection_mode: str = "global",
    de_novo_source: str = "signed_attribution",
    de_novo_num_transcripts: Optional[int] = 500,
    de_novo_peaks_per_direction: int = 1,
    de_novo_regions: Sequence[str] = ("5UTR", "3UTR"),
    position_bin_size: int = 20,
    position_utr5_length: int = 300,
    position_cds_length: int = 600,
    position_utr3_length: int = 300,
    position_bins_per_region: Optional[int] = None,
    position_known_rbp_scope: str = "all",
    position_pseudocount: float = 0.5,
    random_state: int = 42,
) -> Dict[str, object]:
    """Run the analysis while treating canonical result files as complete."""
    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.abspath(os.path.expanduser(out_dir))

    def result_path(filename: str) -> str:
        return os.path.join(out_dir, filename)

    def load_csv_result(filename: str, stage: str) -> pd.DataFrame:
        path = result_path(filename)
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        print(f"[SKIP] {stage}: loaded {filename}")
        return table

    def save_csv_result(table: pd.DataFrame, filename: str, stage: str) -> None:
        table.to_csv(result_path(filename), index=False)
        print(f"[DONE] {stage}: {filename} saved")

    target_rbps = (
        None if target_rbps is None
        else tuple(str(value) for value in target_rbps)
    )
    regions = tuple(str(region) for region in regions)

    samples_path = result_path("unique_transcript_samples.pkl")
    if os.path.isfile(samples_path):
        with open(samples_path, "rb") as handle:
            samples = pickle.load(handle)
        print("[SKIP] samples: loaded unique_transcript_samples.pkl")
    else:
        samples = collect_unique_transcript_samples(
            dataset,
            target_transcript_ids=target_transcript_ids,
            num_transcripts=num_transcripts,
            random_state=random_state,
        )
        if force_zero_expression:
            for sample in samples.values():
                sample["Expr_Vector"] = np.zeros_like(sample["Expr_Vector"])
            print(
                "Using zero expression conditioning for sequence-focused RBP "
                "motif analysis."
            )
        with open(samples_path, "wb") as handle:
            pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("[DONE] samples: unique_transcript_samples.pkl saved")

    pwm_validation_path = result_path("rbp_pwm_validation.csv")
    valid_pwms_path = result_path("validated_rbp_pwms.pkl")
    if os.path.isfile(pwm_validation_path) and os.path.isfile(valid_pwms_path):
        pwm_validation = load_csv_result(
            "rbp_pwm_validation.csv", "validate_pwms"
        )
        with open(valid_pwms_path, "rb") as handle:
            valid_pwm_library = pickle.load(handle)
        print("[SKIP] validate_pwms: loaded validated_rbp_pwms.pkl")
    else:
        valid_pwm_library, pwm_validation = validate_rbp_pwm_library(
            pwm_library,
            metadata=metadata,
            target_rbps=target_rbps,
        )
        save_csv_result(
            pwm_validation, "rbp_pwm_validation.csv", "validate_pwms"
        )
        with open(valid_pwms_path, "wb") as handle:
            pickle.dump(
                valid_pwm_library,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print("[DONE] validate_pwms: validated_rbp_pwms.pkl saved")

    hits_path = result_path("rbp_motif_hits.csv")
    if os.path.isfile(hits_path):
        hits = load_csv_result("rbp_motif_hits.csv", "hits")
    else:
        hits = collect_rbp_motif_hits(
            samples,
            valid_pwm_library,
            metadata,
            target_rbps=target_rbps,
            regions=regions,
            score_threshold=score_threshold,
            max_hits_per_rbp_transcript_region=(
                max_hits_per_rbp_transcript_region
            ),
            context_flank=context_flank,
            num_workers=known_motif_scan_workers,
            scan_backend=scan_backend,
            scan_chunk_size=scan_chunk_size,
        )
        save_csv_result(hits, "rbp_motif_hits.csv", "hits")

    effects_path = result_path("rbp_motif_hit_effects.csv")
    if os.path.isfile(effects_path):
        effects = load_csv_result("rbp_motif_hit_effects.csv", "effects")
    else:
        if hits.empty:
            effects = pd.DataFrame()
        else:
            evaluator = RBPMotifMutagenesisEvaluator(
                model,
                valid_pwm_library,
                prediction_scale=prediction_scale,
            )
            effects = evaluator.evaluate_hits(
                hits,
                samples,
                batch_size=batch_size,
            )
            effects["Expression_Conditioning"] = (
                "zero" if force_zero_expression else "dataset"
            )
        save_csv_result(effects, "rbp_motif_hit_effects.csv", "effects")

    summary_path = result_path("rbp_motif_effect_summary.csv")
    if os.path.isfile(summary_path):
        summary = load_csv_result("rbp_motif_effect_summary.csv", "summary")
    else:
        summary = (
            pd.DataFrame()
            if effects.empty
            else summarize_rbp_motif_effects(
                effects,
                min_transcripts=min_transcripts,
                random_state=random_state,
            )
        )
        save_csv_result(summary, "rbp_motif_effect_summary.csv", "summary")

    contributions_path = result_path("rbp_nucleotide_contributions.csv")
    if os.path.isfile(contributions_path):
        contributions = load_csv_result(
            "rbp_nucleotide_contributions.csv", "cases"
        )
    elif effects.empty:
        contributions = pd.DataFrame()
        save_csv_result(
            contributions, "rbp_nucleotide_contributions.csv", "cases"
        )
    else:
        evaluator = RBPMotifMutagenesisEvaluator(
            model,
            valid_pwm_library,
            prediction_scale=prediction_scale,
        )
        contributions = evaluator.compute_nucleotide_contributions(
            effects,
            samples,
            n_cases_per_direction=n_cases_per_direction,
            min_case_transcripts=min_transcripts,
            context_flank=context_flank,
            batch_size=batch_size,
            case_regions=case_regions,
            target_rbps=case_target_rbps,
            target_regions=case_target_regions,
            target_hit_ids=case_target_hit_ids,
            target_transcript_ids=case_target_transcript_ids,
            target_motif_starts=case_target_motif_starts,
            targeted_cases_per_rbp=targeted_cases_per_rbp,
            selection_mode=case_selection_mode,
        )
        save_csv_result(
            contributions, "rbp_nucleotide_contributions.csv", "cases"
        )

    attribution_path = result_path(
        "signed_translation_attribution_windows.csv"
    )
    if os.path.isfile(attribution_path):
        attribution_windows = load_csv_result(
            "signed_translation_attribution_windows.csv", "attribution"
        )
    elif de_novo_source == "signed_attribution":
        attribution_windows = extract_signed_translation_attribution_windows(
            model,
            samples,
            prediction_scale=prediction_scale,
            num_transcripts=de_novo_num_transcripts,
            peaks_per_direction=de_novo_peaks_per_direction,
            window_radius=context_flank,
            target_regions=de_novo_regions,
            random_state=random_state,
        )
        save_csv_result(
            attribution_windows,
            "signed_translation_attribution_windows.csv",
            "attribution",
        )
    else:
        attribution_windows = pd.DataFrame()
        save_csv_result(
            attribution_windows,
            "signed_translation_attribution_windows.csv",
            "attribution",
        )

    de_novo_path = result_path("de_novo_translation_motifs.csv")
    alignments_path = result_path("de_novo_motif_alignments.json")
    if os.path.isfile(de_novo_path) and os.path.isfile(alignments_path):
        de_novo = load_csv_result(
            "de_novo_translation_motifs.csv", "de_novo"
        )
        with open(alignments_path, "r", encoding="utf-8") as handle:
            alignments = json.load(handle)
        print("[SKIP] de_novo: loaded de_novo_motif_alignments.json")
    else:
        if de_novo_source == "signed_attribution":
            source_table = attribution_windows
            effect_column = "Signed_Attribution"
        elif de_novo_source == "known_hit_context":
            source_table = effects
            effect_column = "Delta_Log2_TE"
        else:
            raise ValueError(
                "de_novo_source must be 'signed_attribution' or "
                "'known_hit_context'."
            )
        if source_table.empty:
            de_novo, alignments = pd.DataFrame(), {}
        else:
            de_novo, alignments = discover_de_novo_translation_motifs(
                source_table,
                sequence_col="Context_Sequence",
                effect_col=effect_column,
                unit_col="Tid",
                region_col="Region",
                peak_offset_col="Peak_Offset",
                discovery_regions=de_novo_regions,
            )
        save_csv_result(
            de_novo, "de_novo_translation_motifs.csv", "de_novo"
        )
        with open(alignments_path, "w", encoding="utf-8") as handle:
            json.dump(alignments, handle, indent=2)
        print("[DONE] de_novo: de_novo_motif_alignments.json saved")

    if position_known_rbp_scope not in {"summary", "all"}:
        raise ValueError(
            "position_known_rbp_scope must be 'summary' or 'all'."
        )
    known_positions_path = result_path("known_rbp_position_profiles.csv")
    de_novo_positions_path = result_path(
        "de_novo_motif_position_profiles.csv"
    )
    if os.path.isfile(known_positions_path) and os.path.isfile(
        de_novo_positions_path
    ):
        known_position_profiles = load_csv_result(
            "known_rbp_position_profiles.csv", "positions"
        )
        de_novo_position_profiles = load_csv_result(
            "de_novo_motif_position_profiles.csv", "positions"
        )
    else:
        known_rbp_names = None
        if position_known_rbp_scope == "summary" and not summary.empty:
            known_rbp_names = summary[
                "RBP_Name"
            ].dropna().astype(str).unique()
        position_profiles = build_motif_position_profiles(
            samples,
            known_hits=hits,
            de_novo_motifs=de_novo,
            bin_size=position_bin_size,
            utr5_length=position_utr5_length,
            cds_length=position_cds_length,
            utr3_length=position_utr3_length,
            bins_per_region=position_bins_per_region,
            known_rbp_names=known_rbp_names,
            pseudocount=position_pseudocount,
        )
        known_position_profiles = position_profiles["known_rbp"]
        de_novo_position_profiles = position_profiles["de_novo"]
        save_csv_result(
            known_position_profiles,
            "known_rbp_position_profiles.csv",
            "positions",
        )
        save_csv_result(
            de_novo_position_profiles,
            "de_novo_motif_position_profiles.csv",
            "positions",
        )
    return {
        "samples": samples,
        "pwm_validation": pwm_validation,
        "hits": hits,
        "hit_effects": effects,
        "summary": summary,
        "nucleotide_contributions": contributions,
        "signed_attribution_windows": attribution_windows,
        "de_novo_motifs": de_novo,
        "de_novo_alignments": alignments,
        "known_rbp_position_profiles": known_position_profiles,
        "de_novo_position_profiles": de_novo_position_profiles,
        "result_directory": out_dir,
    }

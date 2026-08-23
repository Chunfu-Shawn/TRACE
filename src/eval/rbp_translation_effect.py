"""Matched RBP-motif perturbation and de novo translation-motif discovery.

The primary effect estimate is paired within the same transcript and model
environment. A positive delta means that disrupting the native motif lowers
predicted CDS translation; a negative delta means that disruption raises it.
These quantities describe model sensitivity and should not be interpreted as
biological causality without orthogonal RBP-binding and perturbation evidence.
"""

from __future__ import annotations

import hashlib
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
KNOWN_MOTIF_SCAN_CACHE_VERSION = 1


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


def compute_known_motif_scan_signature(
    samples: Mapping[str, Mapping],
    pwm_library: Mapping[str, np.ndarray],
    metadata: pd.DataFrame,
    target_rbps: Optional[Iterable[str]],
    regions: Sequence[str],
    score_threshold: float,
    max_hits_per_rbp_transcript_region: int,
    context_flank: int,
) -> str:
    """Fingerprint every scientific input affecting known-motif positions."""
    digest = hashlib.sha256()
    target_values = (
        None if target_rbps is None
        else sorted({str(value) for value in target_rbps})
    )
    parameters = {
        "cache_version": KNOWN_MOTIF_SCAN_CACHE_VERSION,
        "target_rbps": target_values,
        "regions": list(regions),
        "score_threshold": float(score_threshold),
        "max_hits_per_rbp_transcript_region": int(
            max_hits_per_rbp_transcript_region
        ),
        "context_flank": int(context_flank),
    }
    digest.update(json.dumps(parameters, sort_keys=True).encode("utf-8"))
    for tid in sorted(samples):
        sample = samples[tid]
        digest.update(str(tid).encode("utf-8"))
        digest.update(str(sample["Sequence"]).encode("ascii", errors="replace"))
        digest.update(str(int(sample["CDS_Start_0based"])).encode("ascii"))
        digest.update(str(int(sample["CDS_End_exclusive"])).encode("ascii"))
    target_set = None if target_values is None else set(target_values)
    relevant_metadata = metadata.dropna(
        subset=["Matrix_id", "Gene_name"]
    ).copy()
    relevant_metadata["Matrix_id"] = (
        relevant_metadata["Matrix_id"].astype(str)
    )
    relevant_metadata["Gene_name"] = (
        relevant_metadata["Gene_name"].astype(str)
    )
    normalized_library = {
        str(matrix_id): pwm for matrix_id, pwm in pwm_library.items()
    }
    relevant_metadata = relevant_metadata[
        relevant_metadata["Matrix_id"].isin(
            set(normalized_library)
        )
    ]
    if target_set is not None:
        relevant_metadata = relevant_metadata[
            relevant_metadata["Gene_name"].isin(target_set)
        ]
    relevant_matrix_ids = set(relevant_metadata["Matrix_id"])
    for matrix_id in sorted(relevant_matrix_ids):
        matrix = np.ascontiguousarray(
            np.asarray(normalized_library[matrix_id], dtype=np.float32)
        )
        digest.update(str(matrix_id).encode("utf-8"))
        digest.update(str(matrix.shape).encode("ascii"))
        digest.update(matrix.tobytes())
    metadata_columns = ["Matrix_id", "Gene_name"]
    if not relevant_metadata.empty:
        metadata_text = (
            relevant_metadata[metadata_columns]
            .fillna("")
            .astype(str)
            .drop_duplicates()
            .sort_values(metadata_columns)
            .to_csv(index=False)
        )
        digest.update(metadata_text.encode("utf-8"))
    return digest.hexdigest()


def load_known_motif_scan_cache(
    cache_path: str,
    expected_signature: str,
) -> Optional[pd.DataFrame]:
    """Load a completed known-motif positional scan when its signature matches."""
    pickle_path = os.path.abspath(os.path.expanduser(cache_path))
    manifest_path = f"{pickle_path}.manifest.json"
    if not os.path.isfile(pickle_path) or not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("signature") != expected_signature:
            print("Known-RBP scan cache is stale; rescanning motifs.")
            return None
        with open(pickle_path, "rb") as handle:
            hits = pickle.load(handle)
        if not isinstance(hits, pd.DataFrame):
            raise TypeError("Cached known-RBP hits are not a pandas DataFrame.")
        print(f"Loaded {len(hits):,} known-RBP motif hits from {pickle_path}")
        return hits
    except (OSError, ValueError, TypeError, pickle.UnpicklingError) as error:
        print(f"Known-RBP scan cache could not be loaded ({error}); rescanning.")
        return None


def save_known_motif_scan_cache(
    hits: pd.DataFrame,
    cache_path: str,
    signature: str,
) -> None:
    """Atomically save known-motif positions and their cache manifest."""
    pickle_path = os.path.abspath(os.path.expanduser(cache_path))
    manifest_path = f"{pickle_path}.manifest.json"
    os.makedirs(os.path.dirname(pickle_path) or ".", exist_ok=True)
    pickle_tmp = f"{pickle_path}.tmp"
    manifest_tmp = f"{manifest_path}.tmp"
    with open(pickle_tmp, "wb") as handle:
        pickle.dump(hits, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(manifest_tmp, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_version": KNOWN_MOTIF_SCAN_CACHE_VERSION,
                "signature": signature,
                "n_hits": int(len(hits)),
                "columns": list(hits.columns),
            },
            handle,
            indent=2,
        )
    os.replace(pickle_tmp, pickle_path)
    os.replace(manifest_tmp, manifest_path)
    print(f"Saved {len(hits):,} known-RBP motif hits to {pickle_path}")


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
        required = {"Direction", "Kmer"}
        missing = required.difference(de_novo_motifs.columns)
        if missing:
            raise ValueError(
                f"De novo motif table is missing columns: {sorted(missing)}"
            )
        motif_records = []
        for row in de_novo_motifs.drop_duplicates(
            ["Direction", "Kmer"]
        ).itertuples(index=False):
            direction = str(row.Direction)
            kmer = str(row.Kmer).upper().replace("U", "T")
            if not kmer or not set(kmer).issubset(BASE_TO_INDEX):
                continue
            feature = f"{kmer} ({direction.lower()})"
            de_novo_lengths[feature] = len(kmer)
            de_novo_annotations[feature] = {
                "Direction": direction,
                "Kmer": kmer,
            }
            motif_records.append((feature, kmer))
        for sample in tqdm(
            samples.values(), desc="Scan de novo motif positions"
        ):
            sequence = str(sample["Sequence"]).upper().replace("U", "T")
            for region in regions:
                region_start, region_end = _region_bounds(sample, region)
                region_sequence = sequence[region_start:region_end]
                if not region_sequence:
                    continue
                for feature, kmer in motif_records:
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
        eps: float = 1e-8,
    ) -> pd.DataFrame:
        """Run saturation mutagenesis around representative signed motif hits."""
        if hit_effects.empty:
            return pd.DataFrame()
        if (
            n_cases_per_direction < 1
            or min_case_transcripts < 2
            or context_flank < 0
        ):
            raise ValueError("Invalid case count or context flank.")
        group_summary = (
            hit_effects.groupby(["RBP_Name", "Region"], observed=True)
            .agg(
                Group_Median_Delta_Log2_TE=("Delta_Log2_TE", "median"),
                Group_N_Transcripts=("Tid", "nunique"),
            )
            .reset_index()
        )
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
        selected_rows = []
        for direction, ascending in (("Positive", False), ("Negative", True)):
            groups = group_summary[group_summary["Direction"] == direction]
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
                representative = candidates.loc[distance.idxmin()].to_dict()
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
                contribution = np.log2(wt_signal + eps) - mean_log_mutant
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
                    "Is_Motif": int(hit.Start) <= position < int(hit.End),
                    "Base_Contribution_Log2_TE": contribution,
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


def _window_kmers(sequence: str, k: int) -> set:
    sequence = str(sequence).upper().replace("U", "T")
    return {
        sequence[index:index + k]
        for index in range(len(sequence) - k + 1)
        if set(sequence[index:index + k]).issubset(BASE_TO_INDEX)
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
    cds_skip_codons: int = 5,
    random_state: int = 42,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Extract signed input-gradient peaks for predicted CDS translation.

    The target is log mean frame-0 CDS signal. Positive native-base gradients
    nominate sequence positions that locally support the target; negative
    gradients nominate positions that locally suppress it. This first-order
    attribution is intended for de novo candidate generation, not causal proof.
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
                torch.clamp(profile[cds_start:cds_end:3], min=0).mean()
                + eps
            )
            base_model.zero_grad(set_to_none=True)
            target.backward()
        gradient = sequence_tensor.grad[0].detach().cpu().numpy()
        native_indices = np.asarray(sample["Seq_Emb"]).argmax(axis=1)
        native_scores = gradient[
            np.arange(len(native_indices)), native_indices
        ]
        valid = np.asarray([
            position
            for position in range(window_radius, len(sequence) - window_radius)
            if "N" not in sequence[
                position - window_radius:position + window_radius + 1
            ]
        ], dtype=int)
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
                    "Native_Base": sequence[position],
                })
        sequence_tensor.grad = None
    return pd.DataFrame(records, columns=[
        "Tid", "Cell_Type", "Region", "Absolute_Position",
        "Attribution_Direction", "Signed_Attribution", "Context_Start",
        "Context_End", "Context_Sequence", "Native_Base",
    ])


def discover_de_novo_translation_motifs(
    sequence_effects: pd.DataFrame,
    sequence_col: str = "Context_Sequence",
    effect_col: str = "Delta_Log2_TE",
    unit_col: str = "Tid",
    k_values: Sequence[int] = (5, 6, 7, 8),
    extreme_quantile: float = 0.75,
    neutral_quantile: float = 0.40,
    min_foreground_occurrences: int = 5,
    top_n_per_direction: int = 10,
    logo_flank: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Sequence[str]]]:
    """Discover k-mers enriched in signed-effect contexts versus neutral ones."""
    required = {sequence_col, effect_col, unit_col}
    missing = required.difference(sequence_effects.columns)
    if missing:
        raise ValueError(
            f"Sequence-effect table is missing columns: {sorted(missing)}"
        )
    if not 0.5 <= extreme_quantile < 1:
        raise ValueError("extreme_quantile must be within [0.5, 1).")
    if not 0 < neutral_quantile < 1:
        raise ValueError("neutral_quantile must be within (0, 1).")
    working = sequence_effects[[unit_col, sequence_col, effect_col]].copy()
    working = working.replace([np.inf, -np.inf], np.nan).dropna()
    working = working.loc[
        working.groupby(unit_col, observed=True)[effect_col]
        .transform(lambda values: values.abs() == values.abs().max())
    ].drop_duplicates(unit_col)
    if len(working) < 2 * min_foreground_occurrences:
        return pd.DataFrame(), {}
    positive_values = working.loc[working[effect_col] > 0, effect_col]
    negative_values = working.loc[working[effect_col] < 0, effect_col]
    neutral_cutoff = working[effect_col].abs().quantile(neutral_quantile)
    background = working[working[effect_col].abs() <= neutral_cutoff]
    direction_sets = {}
    if len(positive_values) >= min_foreground_occurrences:
        cutoff = positive_values.quantile(extreme_quantile)
        direction_sets["Positive"] = working[working[effect_col] >= cutoff]
    if len(negative_values) >= min_foreground_occurrences:
        cutoff = negative_values.quantile(1 - extreme_quantile)
        direction_sets["Negative"] = working[working[effect_col] <= cutoff]
    if len(background) < min_foreground_occurrences or not direction_sets:
        return pd.DataFrame(), {}

    records = []
    foreground_sequences = {}
    for direction, foreground in direction_sets.items():
        foreground_sequences[direction] = foreground[sequence_col].astype(str).tolist()
        for k in k_values:
            if k < 3:
                raise ValueError("All k-mer lengths must be at least 3.")
            fg_sets = [_window_kmers(sequence, k) for sequence in foreground_sequences[direction]]
            bg_sets = [
                _window_kmers(sequence, k)
                for sequence in background[sequence_col].astype(str)
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
                odds_ratio, p_value = fisher_exact(table, alternative="greater")
                fg_rate = fg_hit / len(fg_sets)
                bg_rate = bg_hit / len(bg_sets)
                records.append({
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
                })
    results = pd.DataFrame(records)
    if results.empty:
        return results, {}
    results["FDR_BH"] = _benjamini_hochberg(results["P_Value"])
    results = results.sort_values(
        ["Direction", "FDR_BH", "Log2_Enrichment"],
        ascending=[True, True, False],
    )
    results = results.groupby("Direction", observed=True).head(
        top_n_per_direction
    ).reset_index(drop=True)

    alignments: Dict[str, Sequence[str]] = {}
    for row in results.itertuples(index=False):
        aligned = []
        for sequence in foreground_sequences[row.Direction]:
            position = sequence.find(row.Kmer)
            if position < logo_flank:
                continue
            end = position + len(row.Kmer) + logo_flank
            if end > len(sequence):
                continue
            aligned.append(sequence[position - logo_flank:end])
        alignments[f"{row.Direction}|{row.Kmer}"] = aligned
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
    reuse_known_motif_scan: bool = True,
    known_motif_scan_cache_path: Optional[str] = None,
    prediction_scale: str = "log1p",
    force_zero_expression: bool = True,
    batch_size: int = 32,
    min_transcripts: int = 5,
    n_cases_per_direction: int = 3,
    de_novo_source: str = "signed_attribution",
    de_novo_num_transcripts: Optional[int] = 500,
    de_novo_peaks_per_direction: int = 1,
    position_bin_size: int = 20,
    position_utr5_length: int = 300,
    position_cds_length: int = 600,
    position_utr3_length: int = 300,
    position_bins_per_region: Optional[int] = None,
    position_known_rbp_scope: str = "all",
    position_pseudocount: float = 0.5,
    random_state: int = 42,
) -> Dict[str, object]:
    """Run known-PWM perturbation, case attribution, and de novo discovery."""
    os.makedirs(out_dir, exist_ok=True)
    target_rbps = (
        None if target_rbps is None
        else tuple(str(value) for value in target_rbps)
    )
    regions = tuple(str(region) for region in regions)
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
    valid_pwm_library, pwm_validation = validate_rbp_pwm_library(
        pwm_library,
        metadata=metadata,
        target_rbps=target_rbps,
    )
    scan_cache_path = known_motif_scan_cache_path or os.path.join(
        out_dir, "known_rbp_motif_hits.pkl"
    )
    scan_signature = compute_known_motif_scan_signature(
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
    )
    hits = None
    if reuse_known_motif_scan:
        hits = load_known_motif_scan_cache(
            scan_cache_path,
            expected_signature=scan_signature,
        )
    if hits is None:
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
        save_known_motif_scan_cache(
            hits,
            scan_cache_path,
            signature=scan_signature,
        )
        # Save positions immediately so downstream failures do not lose a scan.
        hits.to_csv(
            os.path.join(out_dir, "rbp_motif_hits.csv"),
            index=False,
        )
    if hits.empty:
        print("No known RBP motif hits passed the requested filters.")
        effects = pd.DataFrame()
        summary = pd.DataFrame()
        contributions = pd.DataFrame()
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
        summary = summarize_rbp_motif_effects(
            effects,
            min_transcripts=min_transcripts,
            random_state=random_state,
        )
        contributions = evaluator.compute_nucleotide_contributions(
            effects,
            samples,
            n_cases_per_direction=n_cases_per_direction,
            min_case_transcripts=min_transcripts,
            context_flank=context_flank,
            batch_size=batch_size,
        )
    if de_novo_source == "signed_attribution":
        attribution_windows = extract_signed_translation_attribution_windows(
            model,
            samples,
            prediction_scale=prediction_scale,
            num_transcripts=de_novo_num_transcripts,
            peaks_per_direction=de_novo_peaks_per_direction,
            window_radius=context_flank,
            random_state=random_state,
        )
        de_novo, alignments = discover_de_novo_translation_motifs(
            attribution_windows,
            sequence_col="Context_Sequence",
            effect_col="Signed_Attribution",
            unit_col="Tid",
        )
    elif de_novo_source == "known_hit_context":
        attribution_windows = pd.DataFrame()
        if effects.empty:
            de_novo, alignments = pd.DataFrame(), {}
        else:
            de_novo, alignments = discover_de_novo_translation_motifs(
                effects,
                sequence_col="Context_Sequence",
                effect_col="Delta_Log2_TE",
                unit_col="Tid",
            )
    else:
        raise ValueError(
            "de_novo_source must be 'signed_attribution' or "
            "'known_hit_context'."
        )

    if position_known_rbp_scope not in {"summary", "all"}:
        raise ValueError(
            "position_known_rbp_scope must be 'summary' or 'all'."
        )
    known_rbp_names = None
    if position_known_rbp_scope == "summary" and not summary.empty:
        known_rbp_names = summary["RBP_Name"].dropna().astype(str).unique()
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

    outputs = {
        "rbp_pwm_validation.csv": pwm_validation,
        "rbp_motif_hits.csv": hits,
        "rbp_motif_hit_effects.csv": effects,
        "rbp_motif_effect_summary.csv": summary,
        "rbp_nucleotide_contributions.csv": contributions,
        "signed_translation_attribution_windows.csv": attribution_windows,
        "de_novo_translation_motifs.csv": de_novo,
        "known_rbp_position_profiles.csv": position_profiles["known_rbp"],
        "de_novo_motif_position_profiles.csv": position_profiles["de_novo"],
    }
    for filename, table in outputs.items():
        table.to_csv(os.path.join(out_dir, filename), index=False)
    with open(
        os.path.join(out_dir, "de_novo_motif_alignments.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(alignments, handle, indent=2)
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
        "known_rbp_position_profiles": position_profiles["known_rbp"],
        "de_novo_position_profiles": position_profiles["de_novo"],
        "known_motif_scan_cache_path": os.path.abspath(
            os.path.expanduser(scan_cache_path)
        ),
    }

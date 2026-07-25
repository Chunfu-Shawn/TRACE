"""Shared compatibility helpers for TRACE evaluation modules."""

from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


def transcript_id_from_uuid(uuid: object) -> str:
    """Extract the transcript identifier from the dataset UUID contract."""
    return str(uuid).split("-", 1)[0]


def to_1d_signal(signal: object) -> np.ndarray:
    """Convert tensor or array signals with one or more channels to float32 1D."""
    if isinstance(signal, torch.Tensor):
        values = signal.detach().cpu().numpy()
    else:
        values = np.asarray(signal)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    return values.reshape(values.shape[0], -1).sum(axis=1)


def get_prediction(
    predictions: Dict[str, Dict[str, object]],
    cell_type: object,
    transcript_id: str,
) -> Optional[np.ndarray]:
    """Look up a nested prediction using versioned and unversioned IDs."""
    cell_predictions = predictions.get(str(cell_type))
    if not isinstance(cell_predictions, dict):
        return None
    if transcript_id in cell_predictions:
        return to_1d_signal(cell_predictions[transcript_id])
    clean_id = transcript_id.split(".", 1)[0]
    if clean_id in cell_predictions:
        return to_1d_signal(cell_predictions[clean_id])
    return None


def cds_slice(meta_info: dict, length: int) -> Optional[Tuple[int, int]]:
    """Convert 1-based inclusive CDS coordinates to a clipped half-open slice."""
    start_1based = int(meta_info.get("cds_start_pos", -1))
    end_1based = int(meta_info.get("cds_end_pos", -1))
    if start_1based < 1 or end_1based < start_1based:
        return None
    start = min(max(start_1based - 1, 0), length)
    end = min(max(end_1based, 0), length)
    if end <= start:
        return None
    return start, end


def cds_with_stop_slice(meta_info: dict, length: int) -> Optional[Tuple[int, int]]:
    """Return the CDS plus its separately annotated three-nucleotide stop codon."""
    bounds = cds_slice(meta_info, length)
    if bounds is None:
        return None
    start, cds_end = bounds
    end = min(cds_end + 3, length)
    return start, end


def load_prediction_input(
    pkl_input: Union[Dict[str, str], str]
) -> Dict[str, Dict[str, object]]:
    """Load either one combined prediction PKL or per-cell-type PKL files."""
    if isinstance(pkl_input, (str, os.PathLike)):
        with open(pkl_input, "rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict):
            raise ValueError("The prediction pickle does not contain a dictionary.")
        return data

    if isinstance(pkl_input, dict):
        combined = {}
        for cell_type, pkl_path in pkl_input.items():
            with open(pkl_path, "rb") as handle:
                data = pickle.load(handle)
            if not isinstance(data, dict):
                raise ValueError(f"Prediction pickle for {cell_type} is not a dictionary.")
            if cell_type in data and isinstance(data[cell_type], dict):
                combined[cell_type] = data[cell_type]
            else:
                combined[cell_type] = data
        return combined

    raise TypeError("pkl_input must be a path string or a cell-type-to-path dictionary.")

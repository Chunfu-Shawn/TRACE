#!/usr/bin/env python3
"""Rank all 4,096 RNA 6-mers by TRACE attention and saliency.

This module is intentionally independent of the existing de novo motif and RBP
perturbation workflows. It loads a BaseModel checkpoint, selects one dataset row
per transcript, uses explicit zero-expression conditioning by default, scans
every valid 6-nt window, and aggregates model-derived scores by sequence and
transcript region. Portable stage manifests support validated cross-server
reuse without repeating completed workflow stages.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        """Fall back to a plain iterator when tqdm is unavailable."""
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


BASES = np.asarray(list("ACGT"))
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}
REGIONS = ("5UTR", "CDS", "3UTR")
REGION_TO_INDEX = {region: index for index, region in enumerate(REGIONS)}
SCORE_COLUMNS = (
    "Attention_Mean",
    "Attention_Max",
    "Saliency_L1_Mean",
    "Native_Abs_Gradient_Mean",
    "InputXGradient_Mean",
)
POSITIVE_SCORE_COLUMNS = (
    "Attention_Mean",
    "Saliency_L1_Mean",
)
SCAN_FIELDS = (
    "Tid", "Cell_Type", "Transcript_Length",
    "CDS_Start_0based", "CDS_End_exclusive",
    "Window_Start_0based", "Window_End_exclusive",
    "Region", "CDS_Overlap_nt", "CDS_Frame",
    "Is_InFrame_Codon_Pair", "Sixmer",
    "Attention_Mean", "Attention_Max", "Saliency_L1_Mean",
    "Native_Abs_Gradient_Mean", "InputXGradient_Mean",
    "CDS_Mean_Prediction",
)
SCAN_CACHE_VERSION = 1


def _json_default(value):
    """Convert common scientific values to JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _atomic_json(value, path):
    """Write JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(table, path):
    """Write a CSV table atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(payload, path):
    """Write a compressed NumPy archive atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _sha256_file(path, block_size=8 * 1024 * 1024):
    """Calculate a portable content checksum for a result file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value):
    """Hash a JSON-compatible object independently of filesystem paths."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path):
    """Read a JSON object and return None for invalid files."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _pdf_is_valid(path):
    """Check the PDF header, trailer, and a conservative size floor."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                return False
            handle.seek(max(0, path.stat().st_size - 2048))
            return b"%%EOF" in handle.read()
    except OSError:
        return False


def _commit_stage(manifest_path, stage_signature, files):
    """Commit a lightweight derived stage with portable file checksums."""
    file_records = {}
    for label, path in files.items():
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot commit missing stage file: {path}")
        file_records[label] = {
            "Name": path.name,
            "Size": int(path.stat().st_size),
            "SHA256": _sha256_file(path),
        }
    _atomic_json({
        "Status": "complete",
        "Stage_Signature": stage_signature,
        "Files": file_records,
    }, manifest_path)


def _stage_cache_is_valid(manifest_path, stage_signature, files):
    """Validate a derived stage after copying its output directory."""
    manifest = _read_json(manifest_path)
    if (
            manifest is None
            or manifest.get("Status") != "complete"
            or manifest.get("Stage_Signature") != stage_signature):
        return False
    records = manifest.get("Files", {})
    try:
        for label, path in files.items():
            path = Path(path)
            record = records[label]
            if (
                    not path.is_file()
                    or path.name != record["Name"]
                    or path.stat().st_size != int(record["Size"])
                    or _sha256_file(path) != record["SHA256"]):
                return False
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def _normalize_tid(value):
    """Remove versions only from ENST transcript identifiers."""
    transcript_id = str(value).strip()
    if transcript_id.startswith("ENST"):
        return transcript_id.split(".", 1)[0]
    return transcript_id


def _meta_value(meta_info, names, default=None):
    """Read a metadata value from a mapping or object."""
    for name in names:
        if isinstance(meta_info, Mapping) and name in meta_info:
            return meta_info[name]
        if hasattr(meta_info, name):
            return getattr(meta_info, name)
    return default


def _extract_tid(uuid, meta_info):
    """Resolve a transcript identifier from metadata or dataset UUID."""
    transcript_id = _meta_value(
        meta_info,
        ("Tid", "tid", "transcript_id", "transcript", "tx_id"),
        default=None,
    )
    if transcript_id is None:
        uuid_text = str(uuid)
        transcript_id = (
            uuid_text.rsplit("-", 2)[0]
            if "-" in uuid_text else uuid_text
        )
    return _normalize_tid(transcript_id)


def _as_numpy(value, dtype=np.float32):
    """Convert a tensor or array-like value to NumPy."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _extract_dataset_sample(dataset, index):
    """Extract the six BaseModel inputs and transcript coordinates."""
    item = dataset[index]
    if len(item) < 6:
        raise ValueError(
            "Dataset rows must contain uuid, species, cell type, expression, "
            "metadata, and sequence embedding."
        )
    uuid, species, cell_type, expression, meta_info, sequence_embedding = item[:6]
    sequence_embedding = _as_numpy(sequence_embedding)
    if sequence_embedding.ndim != 2:
        raise ValueError(
            "Sequence embedding must be two-dimensional, got "
            f"{sequence_embedding.shape}."
        )
    if sequence_embedding.shape[1] != 4 and sequence_embedding.shape[0] == 4:
        sequence_embedding = sequence_embedding.T
    if sequence_embedding.shape[1] != 4:
        raise ValueError(
            "Sequence embedding must have four nucleotide channels, got "
            f"{sequence_embedding.shape}."
        )

    if expression is None:
        expression_array = np.zeros(0, dtype=np.float32)
    else:
        expression_array = _as_numpy(expression).reshape(-1)
    cds_start = int(_meta_value(
        meta_info,
        ("cds_start_pos", "CDS_Start", "cds_start"),
        default=-1,
    )) - 1
    cds_end = int(_meta_value(
        meta_info,
        ("cds_end_pos", "CDS_End", "cds_end"),
        default=-1,
    ))
    transcript_id = _extract_tid(uuid, meta_info)
    length = len(sequence_embedding)
    return {
        "Dataset_Index": int(index),
        "Tid": transcript_id,
        "Species": species,
        "Cell_Type": cell_type,
        "Expression": expression_array,
        "Meta_Info": meta_info,
        "Sequence_Embedding": sequence_embedding,
        "Transcript_Length": int(length),
        "CDS_Start_0based": int(cds_start),
        "CDS_End_exclusive": int(cds_end),
        "Valid_CDS": bool(0 <= cds_start < cds_end <= length),
    }


def _decode_sequence(sequence_embedding):
    """Decode a one-hot nucleotide array and preserve invalid bases as N."""
    matrix = np.asarray(sequence_embedding)
    row_sums = matrix.sum(axis=1)
    maxima = matrix.max(axis=1)
    valid = np.isfinite(matrix).all(axis=1) & (row_sums > 0) & (maxima >= 0.5)
    indices = matrix.argmax(axis=1)
    return "".join(
        BASES[index] if is_valid else "N"
        for index, is_valid in zip(indices, valid)
    )


def _load_dataset(paths):
    """Load one or more lazy HDF5 or pickle datasets."""
    from data.translation_dataset import TranslationDataset

    datasets = []
    for path_text in paths:
        source = Path(path_text).expanduser().resolve()
        suffix = source.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            dataset = TranslationDataset.from_h5(str(source), lazy=True)
        elif suffix in {".pkl", ".pickle"}:
            dataset = TranslationDataset.from_pickle(str(source))
        else:
            raise ValueError(
                f"Unsupported dataset format '{suffix}' for {source}."
            )
        print(f"Loaded dataset: {source} ({len(dataset):,} rows)", flush=True)
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _load_id_collection(path):
    """Load optional target transcript IDs from common file formats."""
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with source.open("rb") as handle:
            values = pickle.load(handle)
    elif suffix == ".json":
        values = json.loads(source.read_text(encoding="utf-8"))
    elif suffix in {".csv", ".tsv"}:
        table = pd.read_csv(source, sep="\t" if suffix == ".tsv" else ",")
        id_column = next(
            (
                column for column in
                ("Tid", "tid", "Transcript_ID", "transcript_id")
                if column in table.columns
            ),
            table.columns[0],
        )
        values = table[id_column].dropna().tolist()
    else:
        values = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if isinstance(values, Mapping):
        flattened = []
        for group_values in values.values():
            if isinstance(group_values, str):
                flattened.append(group_values)
            else:
                flattened.extend(group_values)
        values = flattened
    if isinstance(values, str):
        values = [values]
    return set(_normalize_tid(value) for value in values)


def _select_unique_transcripts(
        dataset,
        num_transcripts,
        min_length,
        max_length,
        target_transcript_ids,
        random_state):
    """Reservoir-sample one representative row for every transcript."""
    rng = np.random.default_rng(random_state)
    representatives = {}
    occurrence_counts = defaultdict(int)
    eligible_rows = 0
    duplicate_rows = 0
    invalid_rows = 0
    for index in tqdm(range(len(dataset)), desc="Select unique transcripts"):
        try:
            sample = _extract_dataset_sample(dataset, index)
        except (IndexError, KeyError, TypeError, ValueError):
            invalid_rows += 1
            continue
        transcript_id = sample["Tid"]
        length = sample["Transcript_Length"]
        if (
                not sample["Valid_CDS"]
                or length < min_length
                or (max_length is not None and length > max_length)
                or (
                    target_transcript_ids is not None
                    and transcript_id not in target_transcript_ids
                )):
            invalid_rows += 1
            continue
        eligible_rows += 1
        occurrence_counts[transcript_id] += 1
        if transcript_id in representatives:
            duplicate_rows += 1
        if (
                transcript_id not in representatives
                or rng.random() < 1.0 / occurrence_counts[transcript_id]):
            representatives[transcript_id] = sample

    transcript_ids = np.asarray(list(representatives), dtype=object)
    if num_transcripts is not None and len(transcript_ids) > num_transcripts:
        transcript_ids = rng.choice(
            transcript_ids,
            size=int(num_transcripts),
            replace=False,
        )
    else:
        rng.shuffle(transcript_ids)
    selected = [representatives[transcript_id] for transcript_id in transcript_ids]
    audit = {
        "Dataset_Rows": int(len(dataset)),
        "Eligible_Rows": int(eligible_rows),
        "Duplicate_Rows": int(duplicate_rows),
        "Invalid_Or_Filtered_Rows": int(invalid_rows),
        "Unique_Eligible_Transcripts": int(len(representatives)),
        "Selected_Transcripts": int(len(selected)),
    }
    print(
        "Transcript selection: "
        + ", ".join(f"{key}={value:,}" for key, value in audit.items()),
        flush=True,
    )
    return selected, audit


def _load_selected_transcripts(dataset, selected_table_path):
    """Reload a prior transcript selection using saved dataset indices."""
    table = pd.read_csv(selected_table_path)
    required = {
        "Dataset_Index", "Tid", "Transcript_Length",
        "CDS_Start_0based", "CDS_End_exclusive",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Selected transcript table is missing columns: {sorted(missing)}"
        )
    normalized_tids = table["Tid"].map(_normalize_tid)
    if table.empty or normalized_tids.duplicated().any():
        raise ValueError(
            "Selected transcript table must contain unique transcript IDs."
        )
    selected = []
    for row in tqdm(
            table.itertuples(index=False),
            total=len(table),
            desc="Reload selected transcripts"):
        sample = _extract_dataset_sample(dataset, int(row.Dataset_Index))
        if sample["Tid"] != _normalize_tid(row.Tid):
            raise ValueError(
                "Saved transcript selection does not match the current dataset "
                f"at index {row.Dataset_Index}."
            )
        selected.append(sample)
    print(
        f"[SKIP] transcript selection: loaded {len(selected):,} rows from "
        f"{Path(selected_table_path).name}",
        flush=True,
    )
    return selected


def _try_load_selected_transcripts(dataset, selected_table_path):
    """Reuse a valid transcript selection or request deterministic rebuilding."""
    path = Path(selected_table_path)
    if not path.is_file():
        return None
    try:
        return _load_selected_transcripts(dataset, path)
    except (OSError, ValueError, TypeError, IndexError, pd.errors.ParserError) as exc:
        print(
            f"[INVALID] transcript selection will be rebuilt: {exc}",
            flush=True,
        )
        return None


def _save_selected_transcripts(selected, path):
    """Save only reproducibility metadata, not model input arrays."""
    records = [{
        "Dataset_Index": sample["Dataset_Index"],
        "Tid": sample["Tid"],
        "Cell_Type": str(sample["Cell_Type"]),
        "Transcript_Length": sample["Transcript_Length"],
        "CDS_Start_0based": sample["CDS_Start_0based"],
        "CDS_End_exclusive": sample["CDS_End_exclusive"],
    } for sample in selected]
    _atomic_csv(pd.DataFrame(records), path)


def _device_from_argument(value):
    """Resolve a torch device with explicit CUDA validation."""
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _load_model(args, device):
    """Construct BaseModel, attach its count head, and restore a checkpoint."""
    from model.base_model import BaseModel
    from model.prediction_heads import PsiteDensityHead

    model = BaseModel.from_config(str(Path(args.model_config).expanduser()))
    model.add_head(
        "count",
        PsiteDensityHead.create_from_model(
            model,
            d_pred_h=args.head_hidden_dim,
        ),
        overwrite=True,
        move_to_model_device=False,
    )
    model.to(device)
    checkpoint = torch.load(
        str(Path(args.checkpoint).expanduser()),
        map_location=device,
    )
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, Mapping) and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {args.checkpoint}")
    state_dict = model._strip_head_module_prefix(state_dict)
    load_result = model.load_state_dict(
        state_dict,
        strict=not args.non_strict,
    )
    if args.non_strict:
        print(
            "Non-strict checkpoint load: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}",
            flush=True,
        )
    model.eval()
    print(f"Loaded BaseModel on {device}: {args.checkpoint}", flush=True)
    return model


def _sequence_mask(sequence_tensor):
    """Return an all-valid sequence mask for an unpadded transcript."""
    return torch.ones(
        sequence_tensor.shape[:2],
        dtype=torch.bool,
        device=sequence_tensor.device,
    )


def _expression_tensor(model, sample, device, force_zero_expression):
    """Prepare optional environmental conditioning."""
    expression = sample["Expression"]
    if force_zero_expression:
        return torch.zeros(
            (1, int(model.d_expr)),
            dtype=torch.float32,
            device=device,
        )
    if expression.size == 0:
        return None
    expression_tensor = torch.from_numpy(expression).float().unsqueeze(0).to(device)
    return expression_tensor


def _resolve_encoder_inputs(
        model,
        sample,
        sequence_tensor,
        expression_tensor):
    """Build the BaseModel encoder inputs without calling old analysis code."""
    resolved_expression = model._resolve_expr_vector(
        cell_type=sample["Cell_Type"],
        expr_vector=expression_tensor,
        batch_size=1,
    ).to(sequence_tensor.device)
    species_index = model._normalize_species(sample["Species"], 1).to(
        sequence_tensor.device
    )
    species_embedding = model.species_embedding(species_index)
    compact_style = model.expr_projector(torch.cat(
        [resolved_expression, species_embedding], dim=-1
    ))
    sequence_representations = model.seq_embedding(sequence_tensor)
    return compact_style, sequence_representations


def _adaln_parameters(sublayer, compact_style):
    """Resolve AdaLN parameters exactly as BaseModel does."""
    gamma, beta, alpha = sublayer.adaLN_modulation(compact_style).chunk(3, dim=-1)
    bounds = getattr(sublayer, "adaln_modulation_bounds", None)
    if bounds is not None:
        gamma_bound, beta_bound, alpha_bound = bounds
        gamma = sublayer._smooth_bound(gamma, gamma_bound)
        beta = sublayer._smooth_bound(beta, beta_bound)
        alpha = sublayer._smooth_bound(alpha, alpha_bound)
    return gamma, beta, alpha


def _prepare_sublayer_input(sublayer, representations, compact_style):
    """Prepare a Pre-AdaLN transformer sublayer input."""
    gamma, beta, alpha = _adaln_parameters(sublayer, compact_style)
    normalized = (
        (1 + gamma.unsqueeze(1)) * sublayer.LN(representations)
        + beta.unsqueeze(1)
    )
    return normalized, alpha


def _apply_sublayer_residual(sublayer, representations, output, gate):
    """Apply the BaseModel gated residual update."""
    return representations + gate.unsqueeze(1) * sublayer.dropout(output)


def _compute_received_attention(
        model,
        sample,
        sequence_tensor,
        expression_tensor,
        query_chunk_size):
    """Compute mean received attention across every encoder layer and head.

    Received attention is summed over query positions. Consequently, a uniform
    attention map has an expected per-position score of approximately one,
    avoiding the direct 1/L scaling of query-averaged attention.
    """
    model.eval()
    sequence_length = sequence_tensor.shape[1]
    n_heads = int(model.n_heads)
    head_dim = int(
        model.encoder.encoder_layers[0].multi_headed_attention.head_dim
    )
    attention_track = torch.zeros(
        sequence_length,
        device=sequence_tensor.device,
        dtype=torch.float32,
    )
    compact_style, representations = _resolve_encoder_inputs(
        model,
        sample,
        sequence_tensor,
        expression_tensor,
    )
    mask = _sequence_mask(sequence_tensor)

    with torch.no_grad():
        for encoder_layer in model.encoder.encoder_layers:
            attention_sublayer = encoder_layer.sublayers[0]
            normalized, attention_gate = _prepare_sublayer_input(
                attention_sublayer,
                representations,
                compact_style,
            )
            attention_module = encoder_layer.multi_headed_attention
            queries = attention_module.toqueries(normalized).view(
                1, sequence_length, n_heads, head_dim
            )
            keys = attention_module.tokeys(normalized).view(
                1, sequence_length, n_heads, head_dim
            )
            values = attention_module.tovalues(normalized).view(
                1, sequence_length, n_heads, head_dim
            )
            if hasattr(attention_module, "RoPE"):
                queries = attention_module.RoPE(
                    queries.transpose(1, 2)
                ).transpose(1, 2)
                keys = attention_module.RoPE(
                    keys.transpose(1, 2)
                ).transpose(1, 2)

            layer_received = torch.zeros_like(attention_track)
            head_outputs = []
            scale = math.sqrt(head_dim)
            key_mask = mask[0]
            for head_index in range(n_heads):
                head_queries = queries[0, :, head_index, :]
                head_keys = keys[0, :, head_index, :]
                head_values = values[0, :, head_index, :]
                head_received = torch.zeros_like(attention_track)
                head_output = torch.empty(
                    (sequence_length, head_dim),
                    device=sequence_tensor.device,
                    dtype=head_values.dtype,
                )
                for query_start in range(0, sequence_length, query_chunk_size):
                    query_end = min(
                        query_start + query_chunk_size,
                        sequence_length,
                    )
                    scores = torch.matmul(
                        head_queries[query_start:query_end],
                        head_keys.T,
                    ) / scale
                    scores.masked_fill_(~key_mask.unsqueeze(0), float("-inf"))
                    weights = torch.softmax(scores, dim=-1)
                    head_received += weights.sum(dim=0).float()
                    head_output[query_start:query_end] = torch.matmul(
                        weights,
                        head_values,
                    )
                layer_received += head_received
                head_outputs.append(head_output)

            attention_track += layer_received / n_heads
            attention_output = torch.stack(head_outputs, dim=1).reshape(
                1, sequence_length, n_heads * head_dim
            )
            attention_output = attention_module.unifyheads(attention_output)
            if hasattr(attention_module, "dropout"):
                attention_output = attention_module.dropout(attention_output)
            representations = _apply_sublayer_residual(
                attention_sublayer,
                representations,
                attention_output,
                attention_gate,
            )

            ffn_sublayer = encoder_layer.sublayers[1]
            normalized_ffn, ffn_gate = _prepare_sublayer_input(
                ffn_sublayer,
                representations,
                compact_style,
            )
            representations = _apply_sublayer_residual(
                ffn_sublayer,
                representations,
                encoder_layer.ffn(normalized_ffn),
                ffn_gate,
            )
    attention_track /= len(model.encoder.encoder_layers)
    return attention_track.detach().cpu().numpy()


def _extract_count_profile(output):
    """Extract the single-channel positional count profile."""
    if not isinstance(output, Mapping) or "count" not in output:
        raise KeyError("BaseModel output must contain a count head.")
    profile = output["count"]
    if isinstance(profile, Mapping):
        profile = profile.get("profile")
    if not torch.is_tensor(profile):
        raise TypeError("Count head output must be a tensor.")
    if profile.ndim != 3 or profile.shape[-1] != 1:
        raise ValueError(
            "Count profile must have shape (batch, length, 1), got "
            f"{tuple(profile.shape)}."
        )
    return profile


def _compute_saliency(
        model,
        sample,
        sequence_tensor,
        expression_tensor,
        output_transform):
    """Differentiate mean CDS output with respect to the nucleotide input."""
    sequence_for_gradient = sequence_tensor.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = model.forward(
        seq_batch=sequence_for_gradient,
        cell_type=sample["Cell_Type"],
        expr_vector=expression_tensor,
        species=sample["Species"],
        src_mask=_sequence_mask(sequence_for_gradient),
        head_names=["count"],
    )
    profile = _extract_count_profile(output)[0, :, 0]
    if output_transform == "expm1":
        profile_for_target = torch.expm1(profile)
    elif output_transform == "none":
        profile_for_target = profile
    else:
        raise ValueError(f"Unsupported saliency output transform: {output_transform}")
    cds_start = sample["CDS_Start_0based"]
    cds_end = sample["CDS_End_exclusive"]
    cds_mean = profile_for_target[cds_start:cds_end].mean()
    cds_mean.backward()
    gradient = sequence_for_gradient.grad[0].detach().cpu().numpy()
    one_hot = sequence_for_gradient[0].detach().cpu().numpy()
    saliency_l1 = np.abs(gradient).sum(axis=-1)
    native_gradient = (gradient * one_hot).sum(axis=-1)
    native_abs_gradient = np.abs(native_gradient)
    return {
        "Saliency_L1": saliency_l1,
        "Native_Abs_Gradient": native_abs_gradient,
        "InputXGradient": native_gradient,
        "CDS_Mean_Prediction": float(cds_mean.detach().cpu()),
    }


def _enumerate_kmers(kmer_length):
    """Return all A/C/G/T k-mers in base-4 index order."""
    return np.asarray([
        "".join(chars)
        for chars in itertools.product("ACGT", repeat=kmer_length)
    ], dtype=object)


def _sequence_codes(sequence):
    """Encode A/C/G/T and mark ambiguous positions as -1."""
    return np.fromiter(
        (BASE_TO_INDEX.get(base, -1) for base in sequence),
        dtype=np.int16,
        count=len(sequence),
    )


def _window_mean(values, kmer_length, starts):
    """Calculate fixed-length window means at selected starts."""
    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(values, dtype=float),
        kmer_length,
    )
    return windows[starts].mean(axis=1)


def _window_max(values, kmer_length, starts):
    """Calculate fixed-length window maxima at selected starts."""
    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(values, dtype=float),
        kmer_length,
    )
    return windows[starts].max(axis=1)


def _prepare_windows(
        sequence,
        cds_start,
        cds_end,
        kmer_length,
        stride,
        cds_overlap_threshold,
        all_kmers):
    """Encode valid windows and apply the explicit CDS-overlap region rule."""
    if len(sequence) < kmer_length:
        return None
    codes = _sequence_codes(sequence)
    all_windows = np.lib.stride_tricks.sliding_window_view(codes, kmer_length)
    starts = np.arange(0, len(all_windows), stride, dtype=int)
    selected_codes = all_windows[starts]
    valid = (selected_codes >= 0).all(axis=1)
    starts = starts[valid]
    selected_codes = selected_codes[valid]
    if len(starts) == 0:
        return None
    powers = (4 ** np.arange(kmer_length - 1, -1, -1)).astype(int)
    kmer_indices = selected_codes @ powers
    ends = starts + kmer_length
    cds_overlap = np.maximum(
        0,
        np.minimum(ends, cds_end) - np.maximum(starts, cds_start),
    )
    region_indices = np.where(
        cds_overlap >= cds_overlap_threshold,
        REGION_TO_INDEX["CDS"],
        np.where(
            starts < cds_start,
            REGION_TO_INDEX["5UTR"],
            REGION_TO_INDEX["3UTR"],
        ),
    ).astype(int)
    fully_inside_cds = (starts >= cds_start) & (ends <= cds_end)
    cds_frames = np.where(
        region_indices == REGION_TO_INDEX["CDS"],
        (starts - cds_start) % 3,
        -1,
    )
    in_frame_codon_pair = fully_inside_cds & (cds_frames == 0)
    return {
        "Starts": starts,
        "Ends": ends,
        "Kmer_Indices": kmer_indices.astype(int),
        "Kmers": all_kmers[kmer_indices],
        "CDS_Overlap": cds_overlap.astype(int),
        "Region_Indices": region_indices,
        "CDS_Frames": cds_frames.astype(int),
        "In_Frame_Codon_Pair": in_frame_codon_pair,
    }


class KmerScoreAccumulator:
    """Streaming region-by-k-mer sufficient statistics."""

    def __init__(self, n_kmers):
        self.n_kmers = int(n_kmers)
        self.hit_counts = np.zeros((len(REGIONS), n_kmers), dtype=np.int64)
        self.transcript_sets = [
            [set() for _ in range(n_kmers)] for _ in REGIONS
        ]
        self.in_frame_codon_pair_counts = np.zeros_like(self.hit_counts)
        self.sums = {
            score: np.zeros((len(REGIONS), n_kmers), dtype=np.float64)
            for score in SCORE_COLUMNS
        }
        self.sum_squares = {
            score: np.zeros((len(REGIONS), n_kmers), dtype=np.float64)
            for score in SCORE_COLUMNS
        }
        self.score_counts = {
            score: np.zeros((len(REGIONS), n_kmers), dtype=np.int64)
            for score in SCORE_COLUMNS
        }

    def update(self, transcript_id, windows, scores):
        """Add every valid window from one transcript."""
        flat_indices = (
            windows["Region_Indices"] * self.n_kmers
            + windows["Kmer_Indices"]
        )
        flat_size = len(REGIONS) * self.n_kmers
        counts = np.bincount(flat_indices, minlength=flat_size).reshape(
            len(REGIONS), self.n_kmers
        )
        self.hit_counts += counts
        in_frame_counts = np.bincount(
            flat_indices[windows["In_Frame_Codon_Pair"]],
            minlength=flat_size,
        ).reshape(len(REGIONS), self.n_kmers)
        self.in_frame_codon_pair_counts += in_frame_counts

        for region_index, kmer_index in np.unique(
                np.column_stack([
                    windows["Region_Indices"],
                    windows["Kmer_Indices"],
                ]),
                axis=0):
            self.transcript_sets[int(region_index)][int(kmer_index)].add(
                transcript_id
            )

        for score_name, values in scores.items():
            if score_name not in self.sums:
                continue
            values = np.asarray(values, dtype=float)
            finite = np.isfinite(values)
            if not finite.all():
                score_indices = flat_indices[finite]
                score_values = values[finite]
            else:
                score_indices = flat_indices
                score_values = values
            weighted_sum = np.bincount(
                score_indices,
                weights=score_values,
                minlength=flat_size,
            ).reshape(len(REGIONS), self.n_kmers)
            weighted_square_sum = np.bincount(
                score_indices,
                weights=score_values ** 2,
                minlength=flat_size,
            ).reshape(len(REGIONS), self.n_kmers)
            score_counts = np.bincount(
                score_indices,
                minlength=flat_size,
            ).reshape(len(REGIONS), self.n_kmers)
            self.sums[score_name] += weighted_sum
            self.sum_squares[score_name] += weighted_square_sum
            self.score_counts[score_name] += score_counts

    def to_payload(self):
        """Return compact sufficient statistics as portable arrays."""
        payload = {
            "hit_counts": self.hit_counts,
            "in_frame_codon_pair_counts": self.in_frame_codon_pair_counts,
            "transcript_counts": np.asarray([
                [len(values) for values in region_values]
                for region_values in self.transcript_sets
            ], dtype=np.int64),
        }
        for score_name in SCORE_COLUMNS:
            payload[f"sum__{score_name}"] = self.sums[score_name]
            payload[f"sum_sq__{score_name}"] = self.sum_squares[score_name]
            payload[f"count__{score_name}"] = self.score_counts[score_name]
        return payload

    def save(self, path):
        """Save compact sufficient statistics atomically."""
        _atomic_npz(self.to_payload(), path)


def _write_window_rows(
        writer,
        sample,
        windows,
        score_arrays,
        cds_mean_prediction):
    """Stream hit-level windows to a compressed CSV."""
    for window_index in range(len(windows["Starts"])):
        region_index = int(windows["Region_Indices"][window_index])
        frame = int(windows["CDS_Frames"][window_index])
        writer.writerow({
            "Tid": sample["Tid"],
            "Cell_Type": str(sample["Cell_Type"]),
            "Transcript_Length": sample["Transcript_Length"],
            "CDS_Start_0based": sample["CDS_Start_0based"],
            "CDS_End_exclusive": sample["CDS_End_exclusive"],
            "Window_Start_0based": int(windows["Starts"][window_index]),
            "Window_End_exclusive": int(windows["Ends"][window_index]),
            "Region": REGIONS[region_index],
            "CDS_Overlap_nt": int(windows["CDS_Overlap"][window_index]),
            "CDS_Frame": "" if frame < 0 else frame,
            "Is_InFrame_Codon_Pair": bool(
                windows["In_Frame_Codon_Pair"][window_index]
            ),
            "Sixmer": str(windows["Kmers"][window_index]),
            "Attention_Mean": score_arrays["Attention_Mean"][window_index],
            "Attention_Max": score_arrays["Attention_Max"][window_index],
            "Saliency_L1_Mean": score_arrays[
                "Saliency_L1_Mean"
            ][window_index],
            "Native_Abs_Gradient_Mean": score_arrays[
                "Native_Abs_Gradient_Mean"
            ][window_index],
            "InputXGradient_Mean": score_arrays[
                "InputXGradient_Mean"
            ][window_index],
            "CDS_Mean_Prediction": cds_mean_prediction,
        })


def _scan_selected_transcripts(
        model,
        selected,
        output_hits_path,
        accumulator_path,
        kmer_length,
        stride,
        cds_overlap_threshold,
        query_chunk_size,
        saliency_output_transform,
        force_zero_expression,
        empty_cache_every,
        progress_desc="6-mer attention/saliency scan"):
    """Compute model tracks and stream every valid k-mer window."""
    device = next(model.parameters()).device
    all_kmers = _enumerate_kmers(kmer_length)
    accumulator = KmerScoreAccumulator(len(all_kmers))
    output_hits_path = Path(output_hits_path)
    output_hits_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_hits = output_hits_path.with_suffix(output_hits_path.suffix + ".tmp")
    n_windows = 0
    n_failed = 0
    with gzip.open(temporary_hits, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCAN_FIELDS)
        writer.writeheader()
        for sample_index, sample in enumerate(tqdm(
                selected,
                desc=progress_desc)):
            try:
                sequence_embedding = sample["Sequence_Embedding"]
                sequence = _decode_sequence(sequence_embedding)
                sequence_tensor = torch.from_numpy(
                    sequence_embedding
                ).float().unsqueeze(0).to(device)
                expression_tensor = _expression_tensor(
                    model,
                    sample,
                    device,
                    force_zero_expression,
                )
                attention = _compute_received_attention(
                    model,
                    sample,
                    sequence_tensor,
                    expression_tensor,
                    query_chunk_size,
                )
                saliency = _compute_saliency(
                    model,
                    sample,
                    sequence_tensor,
                    expression_tensor,
                    saliency_output_transform,
                )
                windows = _prepare_windows(
                    sequence,
                    sample["CDS_Start_0based"],
                    sample["CDS_End_exclusive"],
                    kmer_length,
                    stride,
                    cds_overlap_threshold,
                    all_kmers,
                )
                if windows is None:
                    continue
                starts = windows["Starts"]
                score_arrays = {
                    "Attention_Mean": _window_mean(
                        attention, kmer_length, starts
                    ),
                    "Attention_Max": _window_max(
                        attention, kmer_length, starts
                    ),
                    "Saliency_L1_Mean": _window_mean(
                        saliency["Saliency_L1"], kmer_length, starts
                    ),
                    "Native_Abs_Gradient_Mean": _window_mean(
                        saliency["Native_Abs_Gradient"],
                        kmer_length,
                        starts,
                    ),
                    "InputXGradient_Mean": _window_mean(
                        saliency["InputXGradient"],
                        kmer_length,
                        starts,
                    ),
                }
                accumulator.update(sample["Tid"], windows, score_arrays)
                _write_window_rows(
                    writer,
                    sample,
                    windows,
                    score_arrays,
                    saliency["CDS_Mean_Prediction"],
                )
                n_windows += len(starts)
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                n_failed += 1
                print(
                    f"[WARNING] skipped {sample['Tid']}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if (
                    torch.cuda.is_available()
                    and empty_cache_every > 0
                    and (sample_index + 1) % empty_cache_every == 0):
                torch.cuda.empty_cache()
    os.replace(temporary_hits, output_hits_path)
    accumulator.save(accumulator_path)
    print(
        f"Scanned {n_windows:,} valid windows; failed transcripts={n_failed:,}.",
        flush=True,
    )
    return accumulator, {
        "Valid_Windows": int(n_windows),
        "Failed_Transcripts": int(n_failed),
    }


def _load_accumulator(path):
    """Load sufficient statistics without materializing the hit table."""
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _required_accumulator_keys():
    """Return the complete accumulator schema."""
    keys = {
        "hit_counts",
        "transcript_counts",
        "in_frame_codon_pair_counts",
    }
    for score_name in SCORE_COLUMNS:
        keys.update({
            f"sum__{score_name}",
            f"sum_sq__{score_name}",
            f"count__{score_name}",
        })
    return keys


def _accumulator_is_valid(path, n_kmers):
    """Validate archive readability, schema, shapes, and numeric content."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = _load_accumulator(path)
    except (OSError, ValueError, EOFError):
        return False
    if not _required_accumulator_keys().issubset(payload):
        return False
    expected_shape = (len(REGIONS), int(n_kmers))
    for key in _required_accumulator_keys():
        array = np.asarray(payload[key])
        if array.shape != expected_shape or not np.isfinite(array).all():
            return False
        if (
                key.endswith("counts")
                or key.startswith("count__")
        ) and (array < 0).any():
            return False
    return True


def _gzip_csv_is_valid(
        path,
        expected_sha256,
        expected_size,
        expected_rows=None):
    """Validate a compressed CSV by checksum, schema, CRC, and row count."""
    path = Path(path)
    if (
            not path.is_file()
            or path.stat().st_size != int(expected_size)
            or _sha256_file(path) != str(expected_sha256)):
        return False
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SCAN_FIELDS:
                return False
            row_count = sum(1 for _ in reader)
    except (OSError, EOFError, UnicodeDecodeError, csv.Error):
        return False
    return expected_rows is None or row_count == int(expected_rows)


def _portable_file_identity(path, content_hash=False):
    """Describe an input without embedding a server-specific absolute path."""
    source = Path(path).expanduser()
    identity = {"Name": source.name}
    if source.is_file():
        identity["Size"] = int(source.stat().st_size)
        if content_hash:
            identity["SHA256"] = _sha256_file(source)
    return identity


def _build_scan_signature(args, selected, expression_mode):
    """Build a server-independent signature for all inference-defining inputs."""
    specification = {
        "Cache_Version": SCAN_CACHE_VERSION,
        "Model_Config": _portable_file_identity(
            args.model_config,
            content_hash=True,
        ),
        "Checkpoint": _portable_file_identity(
            args.checkpoint,
            content_hash=True,
        ),
        "Datasets": [
            _portable_file_identity(path)
            for path in args.dataset
        ],
        "Head_Hidden_Dim": int(args.head_hidden_dim),
        "Non_Strict": bool(args.non_strict),
        "Kmer_Length": int(args.kmer_length),
        "Stride": int(args.stride),
        "CDS_Overlap_Threshold": int(args.cds_overlap_threshold),
        "Saliency_Output_Transform": args.saliency_output_transform,
        "Expression_Mode": str(expression_mode),
        "Selected_Transcripts": [{
            "Dataset_Index": int(sample["Dataset_Index"]),
            "Tid": str(sample["Tid"]),
            "Cell_Type": str(sample["Cell_Type"]),
            "Transcript_Length": int(sample["Transcript_Length"]),
            "CDS_Start_0based": int(sample["CDS_Start_0based"]),
            "CDS_End_exclusive": int(sample["CDS_End_exclusive"]),
        } for sample in selected],
    }
    return _stable_hash(specification), specification


def _completed_scan_is_valid(paths, scan_signature, n_kmers, n_transcripts):
    """Validate final scan outputs using the stage completion manifest."""
    manifest = _read_json(paths["scan_complete"])
    if manifest is None:
        return None
    if (
            manifest.get("Status") != "complete"
            or manifest.get("Scan_Signature") != scan_signature
            or int(manifest.get("Selected_Transcripts", -1))
            != int(n_transcripts)):
        return None
    files = manifest.get("Files", {})
    try:
        hits_info = files["Hits"]
        accumulator_info = files["Accumulator"]
        hits_ok = _gzip_csv_is_valid(
            paths["hits"],
            expected_sha256=hits_info["SHA256"],
            expected_size=hits_info["Size"],
            expected_rows=manifest["Valid_Windows"],
        )
        accumulator_ok = (
            _accumulator_is_valid(paths["accumulator"], n_kmers)
            and paths["accumulator"].stat().st_size
            == int(accumulator_info["Size"])
            and _sha256_file(paths["accumulator"])
            == accumulator_info["SHA256"]
        )
    except (KeyError, TypeError, ValueError, OSError):
        return None
    return manifest if hits_ok and accumulator_ok else None


def _commit_completed_scan(
        paths,
        scan_signature,
        selected_count,
        audit):
    """Commit the complete model-scan stage after all outputs are written."""
    if int(audit.get("Failed_Transcripts", 0)) != 0:
        raise RuntimeError(
            "The model scan contains failed transcripts and will not be "
            "marked complete."
        )
    completion = {
        "Cache_Version": SCAN_CACHE_VERSION,
        "Status": "complete",
        "Scan_Signature": scan_signature,
        "Selected_Transcripts": int(selected_count),
        "Valid_Windows": int(audit["Valid_Windows"]),
        "Failed_Transcripts": 0,
        "Files": {
            "Hits": {
                "Name": paths["hits"].name,
                "Size": int(paths["hits"].stat().st_size),
                "SHA256": _sha256_file(paths["hits"]),
            },
            "Accumulator": {
                "Name": paths["accumulator"].name,
                "Size": int(paths["accumulator"].stat().st_size),
                "SHA256": _sha256_file(paths["accumulator"]),
            },
        },
    }
    _atomic_json(completion, paths["scan_complete"])
    return completion


def _run_resumable_scan(args, selected, paths, expression_mode):
    """Reuse or run the complete model-scanning workflow stage."""
    n_kmers = 4 ** int(args.kmer_length)
    scan_signature, scan_specification = _build_scan_signature(
        args,
        selected,
        expression_mode,
    )
    completed = _completed_scan_is_valid(
        paths,
        scan_signature,
        n_kmers,
        len(selected),
    )
    if completed is not None:
        print(
            "[SKIP] model scan: final raw outputs passed checksum and schema "
            "validation.",
            flush=True,
        )
        return completed, False, scan_specification

    print(
        f"[RUN] model scan: {len(selected)} unique transcripts; "
        f"expression_mode={expression_mode}",
        flush=True,
    )
    device = _device_from_argument(args.device)
    model = _load_model(args, device)
    try:
        _, audit = _scan_selected_transcripts(
            model,
            selected,
            output_hits_path=paths["hits"],
            accumulator_path=paths["accumulator"],
            kmer_length=args.kmer_length,
            stride=args.stride,
            cds_overlap_threshold=args.cds_overlap_threshold,
            query_chunk_size=args.attention_query_chunk_size,
            saliency_output_transform=args.saliency_output_transform,
            force_zero_expression=expression_mode == "zero",
            empty_cache_every=args.empty_cache_every,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    completion = _commit_completed_scan(
        paths,
        scan_signature,
        selected_count=len(selected),
        audit=audit,
    )
    return completion, True, scan_specification


def _summary_from_accumulator(
        accumulator_payload,
        kmer_length,
        min_hits,
        bottom_quantile):
    """Build all region-by-k-mer summaries and bottom-quantile fold changes."""
    all_kmers = _enumerate_kmers(kmer_length)
    counts = accumulator_payload["hit_counts"]
    transcript_counts = accumulator_payload["transcript_counts"]
    codon_counts = accumulator_payload["in_frame_codon_pair_counts"]
    records = []
    for region_index, region in enumerate(REGIONS):
        for kmer_index, kmer in enumerate(all_kmers):
            n_hits = int(counts[region_index, kmer_index])
            record = {
                "Region": region,
                "Sixmer": str(kmer),
                "N_Hits": n_hits,
                "N_Transcripts": int(
                    transcript_counts[region_index, kmer_index]
                ),
                "N_InFrame_CodonPair_Hits": int(
                    codon_counts[region_index, kmer_index]
                ),
                "InFrame_CodonPair_Fraction": (
                    codon_counts[region_index, kmer_index] / n_hits
                    if n_hits > 0 else np.nan
                ),
                "Codon_Pair": f"{kmer[:3]}-{kmer[3:]}",
            }
            for score_name in SCORE_COLUMNS:
                score_count_key = f"count__{score_name}"
                score_count = int(
                    accumulator_payload.get(score_count_key, counts)[
                        region_index, kmer_index
                    ]
                )
                score_sum = accumulator_payload[
                    f"sum__{score_name}"
                ][region_index, kmer_index]
                score_square_sum = accumulator_payload[
                    f"sum_sq__{score_name}"
                ][region_index, kmer_index]
                if score_count > 0:
                    mean = score_sum / score_count
                    variance = max(
                        0.0,
                        score_square_sum / score_count - mean ** 2,
                    )
                    std = math.sqrt(variance)
                    sem = std / math.sqrt(score_count)
                else:
                    mean = std = sem = np.nan
                record[f"N_{score_name}"] = score_count
                record[f"Mean_{score_name}"] = mean
                record[f"SD_{score_name}"] = std
                record[f"SEM_{score_name}"] = sem
            records.append(record)
    summary = pd.DataFrame(records)

    eligible = summary["N_Hits"].ge(int(min_hits))
    for score_name in POSITIVE_SCORE_COLUMNS:
        summary[f"Rank_{score_name}"] = np.nan
        summary[f"Fold_vs_Bottom_{score_name}"] = np.nan
        summary[f"Log2_Fold_vs_Bottom_{score_name}"] = np.nan
        summary[f"Bottom_Background_{score_name}"] = np.nan
        summary[f"Bottom_Background_N_{score_name}"] = 0
        summary[f"Fold_Pseudocount_{score_name}"] = np.nan
    for region in REGIONS:
        region_eligible = eligible & summary["Region"].eq(region)
        for score_name in POSITIVE_SCORE_COLUMNS:
            mean_column = f"Mean_{score_name}"
            rank_column = f"Rank_{score_name}"
            fold_column = f"Fold_vs_Bottom_{score_name}"
            log_fold_column = f"Log2_Fold_vs_Bottom_{score_name}"
            background_column = f"Bottom_Background_{score_name}"
            background_n_column = f"Bottom_Background_N_{score_name}"
            pseudocount_column = f"Fold_Pseudocount_{score_name}"
            values = summary.loc[region_eligible, mean_column].dropna()
            if values.empty:
                continue
            threshold = values.quantile(float(bottom_quantile))
            background_values = values[values <= threshold]
            background_mean = float(background_values.mean())
            ordered_index = values.sort_values(ascending=False).index
            summary.loc[ordered_index, rank_column] = np.arange(
                1, len(ordered_index) + 1
            )
            summary.loc[region_eligible, background_column] = background_mean
            summary.loc[
                region_eligible, background_n_column
            ] = len(background_values)
            positive_values = values[values > 0]
            if np.isfinite(background_mean) and not positive_values.empty:
                pseudocount = max(
                    np.finfo(float).eps,
                    float(positive_values.median()) * 1e-6,
                )
                summary.loc[region_eligible, pseudocount_column] = pseudocount
                folds = (
                    summary.loc[region_eligible, mean_column] + pseudocount
                ) / (
                    background_mean + pseudocount
                )
                summary.loc[region_eligible, fold_column] = folds
                summary.loc[region_eligible, log_fold_column] = np.log2(folds)
    return summary


def _normalize_pwm(pwm, pseudocount=1e-4):
    """Normalize a finite non-negative A/C/G/T PWM."""
    matrix = np.asarray(pwm, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("PWM must be two-dimensional.")
    if matrix.shape[1] != 4 and matrix.shape[0] == 4:
        matrix = matrix.T
    if (
            matrix.shape[1] != 4
            or len(matrix) == 0
            or not np.isfinite(matrix).all()
            or (matrix < 0).any()):
        raise ValueError("PWM is not a valid non-negative A/C/G/T matrix.")
    matrix = matrix + float(pseudocount)
    return matrix / matrix.sum(axis=1, keepdims=True)


def _score_all_kmers_against_pwm(kmer_codes, pwm):
    """Return best normalized local PWM compatibility for every k-mer."""
    matrix = _normalize_pwm(pwm)
    kmer_length = kmer_codes.shape[1]
    pwm_length = len(matrix)
    best_scores = np.full(len(kmer_codes), -np.inf, dtype=float)
    best_offsets = np.full(len(kmer_codes), -1, dtype=int)
    if pwm_length >= kmer_length:
        for pwm_offset in range(pwm_length - kmer_length + 1):
            segment = matrix[pwm_offset:pwm_offset + kmer_length]
            log_odds = np.log2(segment / 0.25)
            minimum = log_odds.min(axis=1).sum()
            maximum = log_odds.max(axis=1).sum()
            denominator = maximum - minimum
            if denominator <= 0:
                continue
            raw = log_odds[
                np.arange(kmer_length)[None, :], kmer_codes
            ].sum(axis=1)
            scores = (raw - minimum) / denominator
            improved = scores > best_scores
            best_scores[improved] = scores[improved]
            best_offsets[improved] = pwm_offset
    else:
        log_odds = np.log2(matrix / 0.25)
        minimum = log_odds.min(axis=1).sum()
        maximum = log_odds.max(axis=1).sum()
        denominator = maximum - minimum
        if denominator <= 0:
            return best_scores, best_offsets
        for kmer_offset in range(kmer_length - pwm_length + 1):
            segment_codes = kmer_codes[:, kmer_offset:kmer_offset + pwm_length]
            raw = log_odds[
                np.arange(pwm_length)[None, :], segment_codes
            ].sum(axis=1)
            scores = (raw - minimum) / denominator
            improved = scores > best_scores
            best_scores[improved] = scores[improved]
            best_offsets[improved] = -kmer_offset - 1
    return best_scores, best_offsets


def _annotate_rbp_pwm_compatibility(
        summary,
        pwm_path,
        metadata_path,
        score_threshold,
        percentile_threshold,
        max_matches,
        output_matches_path,
        kmer_length):
    """Annotate 6-mers with local, not full-site, RBP PWM compatibility."""
    if pwm_path is None or metadata_path is None:
        return summary
    with Path(pwm_path).expanduser().open("rb") as handle:
        pwm_library = pickle.load(handle)
    metadata = pd.read_csv(metadata_path, sep="\t")
    required = {"Matrix_id", "Gene_name"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"RBP metadata is missing columns: {sorted(missing)}")
    metadata = metadata.dropna(subset=["Matrix_id", "Gene_name"]).copy()
    metadata["Matrix_id"] = metadata["Matrix_id"].astype(str).str.strip()
    metadata["Gene_name"] = metadata["Gene_name"].astype(str).str.strip()
    names_by_matrix = metadata.groupby("Matrix_id")["Gene_name"].apply(
        lambda values: ";".join(sorted(set(values)))
    ).to_dict()
    normalized_library = {
        str(matrix_id).strip(): matrix
        for matrix_id, matrix in pwm_library.items()
    }
    all_kmers = _enumerate_kmers(kmer_length)
    kmer_codes = np.asarray([
        [BASE_TO_INDEX[base] for base in kmer]
        for kmer in all_kmers
    ], dtype=int)
    matches_by_kmer = defaultdict(list)
    rejected = 0
    for matrix_id, rbp_names in tqdm(
            names_by_matrix.items(),
            desc="Annotate local RBP PWM compatibility"):
        if matrix_id not in normalized_library:
            continue
        try:
            scores, offsets = _score_all_kmers_against_pwm(
                kmer_codes,
                normalized_library[matrix_id],
            )
        except (TypeError, ValueError):
            rejected += 1
            continue
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            continue
        sorted_scores = np.sort(finite_scores)
        percentiles = np.searchsorted(
            sorted_scores,
            scores,
            side="right",
        ) / len(sorted_scores)
        selected = np.flatnonzero(
            (scores >= float(score_threshold))
            & (percentiles >= float(percentile_threshold))
        )
        for kmer_index in selected:
            matches_by_kmer[int(kmer_index)].append({
                "Sixmer": str(all_kmers[kmer_index]),
                "RBP_Names": rbp_names,
                "Matrix_ID": matrix_id,
                "Local_PWM_Compatibility": float(scores[kmer_index]),
                "Empirical_PWM_Percentile": float(percentiles[kmer_index]),
                "Alignment_Offset": int(offsets[kmer_index]),
            })
    match_records = []
    annotation_records = []
    for kmer_index, kmer in enumerate(all_kmers):
        matches = sorted(
            matches_by_kmer.get(kmer_index, []),
            key=lambda record: (
                record["Empirical_PWM_Percentile"],
                record["Local_PWM_Compatibility"],
            ),
            reverse=True,
        )[:int(max_matches)]
        match_records.extend(matches)
        annotation_records.append({
            "Sixmer": str(kmer),
            "Top_RBP_Local_Matches": ";".join(
                record["RBP_Names"] for record in matches
            ),
            "Top_RBP_Matrix_IDs": ";".join(
                record["Matrix_ID"] for record in matches
            ),
            "Top_RBP_Local_PWM_Scores": ";".join(
                f"{record['Local_PWM_Compatibility']:.4f}"
                for record in matches
            ),
            "Top_RBP_Local_PWM_Percentiles": ";".join(
                f"{record['Empirical_PWM_Percentile']:.4f}"
                for record in matches
            ),
            "N_RBP_Local_Matches": len(matches),
        })
    match_columns = [
        "Sixmer", "RBP_Names", "Matrix_ID",
        "Local_PWM_Compatibility", "Empirical_PWM_Percentile",
        "Alignment_Offset",
    ]
    _atomic_csv(
        pd.DataFrame(match_records, columns=match_columns),
        output_matches_path,
    )
    print(f"Rejected {rejected} invalid RBP PWM matrices.", flush=True)
    return summary.merge(
        pd.DataFrame(annotation_records),
        on="Sixmer",
        how="left",
        validate="many_to_one",
    )


def _annotate_codon_pair_table(summary, codon_pair_table):
    """Merge an optional external codon-pair annotation table."""
    if codon_pair_table is None:
        return summary
    source = Path(codon_pair_table).expanduser().resolve()
    separator = "\t" if source.suffix.lower() in {".tsv", ".txt"} else ","
    table = pd.read_csv(source, sep=separator)
    sixmer_column = next(
        (
            column for column in ("Sixmer", "sixmer", "Kmer", "kmer", "Motif")
            if column in table.columns
        ),
        None,
    )
    if sixmer_column is None:
        codon1_column = next(
            (column for column in ("Codon1", "codon1") if column in table),
            None,
        )
        codon2_column = next(
            (column for column in ("Codon2", "codon2") if column in table),
            None,
        )
        if codon1_column is None or codon2_column is None:
            raise ValueError(
                "Codon-pair table must contain Sixmer or Codon1/Codon2 columns."
            )
        table["Sixmer"] = (
            table[codon1_column].astype(str)
            + table[codon2_column].astype(str)
        )
    else:
        table["Sixmer"] = table[sixmer_column].astype(str)
    table["Sixmer"] = (
        table["Sixmer"].str.upper().str.replace("U", "T", regex=False)
    )
    table = table[table["Sixmer"].str.fullmatch(r"[ACGT]{6}")].copy()
    table = table.drop_duplicates("Sixmer")
    rename_mapping = {
        column: f"CodonPairRef_{column}"
        for column in table.columns
        if column != "Sixmer"
    }
    return summary.merge(
        table.rename(columns=rename_mapping),
        on="Sixmer",
        how="left",
        validate="many_to_one",
    )


def _select_top_motifs(summary, top_n):
    """Select the union of attention- and saliency-ranked motifs."""
    selected = np.zeros(len(summary), dtype=bool)
    for score_name in POSITIVE_SCORE_COLUMNS:
        rank_column = f"Rank_{score_name}"
        selected |= summary[rank_column].le(int(top_n)).fillna(False).to_numpy()
    return summary[selected].sort_values(
        ["Region", "Rank_Attention_Mean", "Rank_Saliency_L1_Mean"],
        na_position="last",
    )


def _plot_top_motifs(
        summary,
        output_pdf,
        top_n,
        width,
        height,
        bottom_quantile):
    """Create a region-by-score PDF ranking panel."""
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
    region_colors = {
        "5UTR": "#4C78A8",
        "CDS": "#D9A441",
        "3UTR": "#59A14F",
    }
    score_specs = (
        (
            "Attention_Mean",
            "Attention",
            "Log2_Fold_vs_Bottom_Attention_Mean",
        ),
        (
            "Saliency_L1_Mean",
            "Saliency",
            "Log2_Fold_vs_Bottom_Saliency_L1_Mean",
        ),
    )
    fig, axes = plt.subplots(
        len(REGIONS),
        len(score_specs),
        figsize=(float(width), float(height)),
        squeeze=False,
    )
    for region_index, region in enumerate(REGIONS):
        for score_index, (score_name, display_name, fold_column) in enumerate(
                score_specs):
            axis = axes[region_index, score_index]
            panel_letter = chr(
                ord("a") + region_index * len(score_specs) + score_index
            )
            axis.text(
                -0.18,
                1.06,
                panel_letter,
                transform=axis.transAxes,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            rank_column = f"Rank_{score_name}"
            region_data = summary[
                summary["Region"].eq(region)
                & summary[rank_column].le(int(top_n))
            ].copy()
            region_data = region_data.dropna(subset=[fold_column]).sort_values(
                fold_column,
                ascending=True,
            )
            if region_data.empty:
                axis.text(
                    0.5, 0.5, "No eligible motifs",
                    ha="center", va="center", transform=axis.transAxes,
                )
                axis.set_axis_off()
                continue
            positions = np.arange(len(region_data))
            axis.barh(
                positions,
                region_data[fold_column],
                color=region_colors[region],
                alpha=0.88,
                height=0.72,
            )
            axis.set_yticks(positions)
            axis.set_yticklabels(region_data["Sixmer"])
            axis.axvline(0, color="#BDBDBD", linewidth=0.7)
            axis.grid(axis="x", color="#E6E6E6", linewidth=0.5)
            axis.set_axisbelow(True)
            background_percent = 100 * float(bottom_quantile)
            axis.set_xlabel(
                f"log2 fold vs bottom {background_percent:g}% motifs"
            )
            axis.set_title(f"{region}: {display_name}", fontsize=7.5)
            for y_position, n_hits in zip(positions, region_data["N_Hits"]):
                axis.text(
                    axis.get_xlim()[1],
                    y_position,
                    f"  n={int(n_hits):,}",
                    ha="left",
                    va="center",
                    fontsize=5.5,
                    clip_on=False,
                )
    fig.suptitle(
        "Model-derived ranking of all observed 6-nt sequence motifs",
        y=0.995,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _load_valid_summary(path, kmer_length):
    """Load a complete region-by-k-mer summary or return None."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    required = {
        "Region", "Sixmer", "N_Hits",
        "Mean_Attention_Mean", "Mean_Saliency_L1_Mean",
        "Rank_Attention_Mean", "Rank_Saliency_L1_Mean",
    }
    expected_rows = len(REGIONS) * (4 ** int(kmer_length))
    if (
            not required.issubset(table.columns)
            or len(table) != expected_rows
            or table[["Region", "Sixmer"]].duplicated().any()
            or set(table["Region"].dropna()) != set(REGIONS)):
        return None
    return table


def _top_table_is_valid(path):
    """Check whether the selected top-motif table can be reused."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    required = {
        "Region", "Sixmer",
        "Rank_Attention_Mean", "Rank_Saliency_L1_Mean",
    }
    return not table.empty and required.issubset(table.columns)


def build_parser():
    """Build the cluster-friendly command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan every transcript 6-mer and rank motifs by BaseModel "
            "attention and CDS-output saliency."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--head-hidden-dim", type=int, default=384)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--num-transcripts", type=int, default=500)
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--target-transcript-file")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kmer-length", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cds-overlap-threshold", type=int, default=3)
    parser.add_argument("--attention-query-chunk-size", type=int, default=256)
    parser.add_argument(
        "--saliency-output-transform",
        choices=("none", "expm1"),
        default="none",
    )
    parser.add_argument(
        "--expression-mode",
        choices=("zero", "dataset"),
        default="zero",
        help=(
            "Use an all-zero expression vector to remove cell-type "
            "conditioning, or retain the dataset vector."
        ),
    )
    parser.add_argument(
        "--use-dataset-expression",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--empty-cache-every", type=int, default=20)
    parser.add_argument("--min-hits", type=int, default=20)
    parser.add_argument("--bottom-quantile", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--pwm-pkl")
    parser.add_argument("--metadata-tsv")
    parser.add_argument("--rbp-match-threshold", type=float, default=0.85)
    parser.add_argument(
        "--rbp-match-percentile",
        type=float,
        default=0.99,
    )
    parser.add_argument("--max-rbp-matches", type=int, default=5)
    parser.add_argument("--codon-pair-table")
    parser.add_argument("--plot-width", type=float, default=7.2)
    parser.add_argument("--plot-height", type=float, default=8.0)
    parser.add_argument("--skip-plot", action="store_true")
    return parser


def _validate_args(args):
    """Validate parameters before loading the checkpoint."""
    if not 1 <= args.kmer_length <= 8:
        raise ValueError("kmer_length must be within [1, 8].")
    if args.kmer_length != 6:
        print(
            "[WARNING] This workflow is designed for 6-mers; codon-pair "
            "annotations are meaningful only when kmer_length=6.",
            flush=True,
        )
    if args.stride < 1:
        raise ValueError("stride must be positive.")
    if not 1 <= args.cds_overlap_threshold <= args.kmer_length:
        raise ValueError(
            "cds_overlap_threshold must be between 1 and kmer_length."
        )
    if args.min_length < args.kmer_length:
        raise ValueError("min_length must be at least kmer_length.")
    if args.max_length is not None and args.max_length < args.min_length:
        raise ValueError("max_length must not be smaller than min_length.")
    if args.num_transcripts is not None and args.num_transcripts < 1:
        raise ValueError("num_transcripts must be positive.")
    if args.attention_query_chunk_size < 1:
        raise ValueError("attention_query_chunk_size must be positive.")
    if args.min_hits < 1:
        raise ValueError("min_hits must be positive.")
    if not 0 < args.bottom_quantile < 0.5:
        raise ValueError("bottom_quantile must be within (0, 0.5).")
    if not 0 <= args.rbp_match_threshold <= 1:
        raise ValueError("rbp_match_threshold must be within [0, 1].")
    if not 0 <= args.rbp_match_percentile <= 1:
        raise ValueError("rbp_match_percentile must be within [0, 1].")
    if (args.pwm_pkl is None) != (args.metadata_tsv is None):
        raise ValueError(
            "pwm-pkl and metadata-tsv must be supplied together."
        )


def main(argv=None):
    """Run file-reusable transcript scanning, aggregation, annotation, and plot."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected": out_dir / "evaluated_transcripts.csv",
        "scan_complete": out_dir / "sixmer_scan_complete.json",
        "hits": out_dir / "sixmer_window_hits.csv.gz",
        "accumulator": out_dir / "sixmer_scan_accumulators.npz",
        "summary": out_dir / "sixmer_attribution_summary.csv",
        "summary_complete": out_dir / "sixmer_summary_complete.json",
        "top": out_dir / "sixmer_top_motifs.csv",
        "top_complete": out_dir / "sixmer_top_complete.json",
        "rbp_matches": out_dir / "sixmer_rbp_pwm_matches.csv",
        "plot": out_dir / "sixmer_attribution_ranking.pdf",
        "plot_complete": out_dir / "sixmer_plot_complete.json",
        "manifest": out_dir / "sixmer_scan_manifest.json",
    }
    dataset = _load_dataset(args.dataset)
    target_transcript_ids = _load_id_collection(args.target_transcript_file)
    expression_mode = (
        "dataset" if args.use_dataset_expression else args.expression_mode
    )
    print(
        "Inference conditioning: "
        f"expression_mode={expression_mode}; one row per unique transcript.",
        flush=True,
    )

    selection_audit = {}
    selected = _try_load_selected_transcripts(dataset, paths["selected"])
    if selected is None:
        selected, selection_audit = _select_unique_transcripts(
            dataset,
            num_transcripts=args.num_transcripts,
            min_length=args.min_length,
            max_length=args.max_length,
            target_transcript_ids=target_transcript_ids,
            random_state=args.random_state,
        )
        _save_selected_transcripts(selected, paths["selected"])
    if not selected:
        raise ValueError("No transcripts were selected for 6-mer scanning.")
    selected_tids = [sample["Tid"] for sample in selected]
    if len(selected_tids) != len(set(selected_tids)):
        raise ValueError("Transcript selection contains duplicate transcript IDs.")

    scan_audit, scan_ran, scan_specification = _run_resumable_scan(
        args,
        selected,
        paths,
        expression_mode,
    )

    scan_signature = scan_audit["Scan_Signature"]
    summary_specification = {
        "Scan_Signature": scan_signature,
        "Kmer_Length": int(args.kmer_length),
        "Minimum_Hits": int(args.min_hits),
        "Bottom_Quantile": float(args.bottom_quantile),
        "PWM_Library": (
            _portable_file_identity(args.pwm_pkl, content_hash=True)
            if args.pwm_pkl else None
        ),
        "RBP_Metadata": (
            _portable_file_identity(args.metadata_tsv, content_hash=True)
            if args.metadata_tsv else None
        ),
        "RBP_Match_Threshold": float(args.rbp_match_threshold),
        "RBP_Match_Percentile": float(args.rbp_match_percentile),
        "Maximum_RBP_Matches": int(args.max_rbp_matches),
        "Codon_Pair_Table": (
            _portable_file_identity(args.codon_pair_table, content_hash=True)
            if args.codon_pair_table else None
        ),
    }
    summary_signature = _stable_hash(summary_specification)
    summary_files = {"Summary": paths["summary"]}
    if args.pwm_pkl:
        summary_files["RBP_Matches"] = paths["rbp_matches"]
    summary_cache_valid = (
        not scan_ran
        and _stage_cache_is_valid(
            paths["summary_complete"],
            summary_signature,
            summary_files,
        )
    )
    summary = _load_valid_summary(
        paths["summary"],
        args.kmer_length,
    ) if summary_cache_valid else None
    summary_ran = summary is None
    if not summary_ran:
        print(f"[SKIP] summary: loaded {paths['summary'].name}", flush=True)
    else:
        accumulator_payload = _load_accumulator(paths["accumulator"])
        summary = _summary_from_accumulator(
            accumulator_payload,
            kmer_length=args.kmer_length,
            min_hits=args.min_hits,
            bottom_quantile=args.bottom_quantile,
        )
        summary = _annotate_rbp_pwm_compatibility(
            summary,
            pwm_path=args.pwm_pkl,
            metadata_path=args.metadata_tsv,
            score_threshold=args.rbp_match_threshold,
            percentile_threshold=args.rbp_match_percentile,
            max_matches=args.max_rbp_matches,
            output_matches_path=paths["rbp_matches"],
            kmer_length=args.kmer_length,
        )
        summary = _annotate_codon_pair_table(
            summary,
            args.codon_pair_table,
        )
        _atomic_csv(summary, paths["summary"])
        _commit_stage(
            paths["summary_complete"],
            summary_signature,
            summary_files,
        )

    top_signature = _stable_hash({
        "Summary_Signature": summary_signature,
        "Top_N": int(args.top_n),
    })
    top_cache_valid = (
        not summary_ran
        and _stage_cache_is_valid(
            paths["top_complete"],
            top_signature,
            {"Top_Motifs": paths["top"]},
        )
        and _top_table_is_valid(paths["top"])
    )
    if top_cache_valid:
        print(f"[SKIP] top motifs: found {paths['top'].name}", flush=True)
    else:
        top_motifs = _select_top_motifs(summary, args.top_n)
        _atomic_csv(top_motifs, paths["top"])
        _commit_stage(
            paths["top_complete"],
            top_signature,
            {"Top_Motifs": paths["top"]},
        )

    plot_signature = _stable_hash({
        "Summary_Signature": summary_signature,
        "Top_N": int(args.top_n),
        "Width": float(args.plot_width),
        "Height": float(args.plot_height),
        "Bottom_Quantile": float(args.bottom_quantile),
    })
    plot_cache_valid = (
        not summary_ran
        and _stage_cache_is_valid(
            paths["plot_complete"],
            plot_signature,
            {"PDF": paths["plot"]},
        )
        and _pdf_is_valid(paths["plot"])
    )
    if args.skip_plot:
        print("[SKIP] plot: --skip-plot was requested", flush=True)
    elif plot_cache_valid:
        print(f"[SKIP] plot: found {paths['plot'].name}", flush=True)
    else:
        _plot_top_motifs(
            summary,
            output_pdf=paths["plot"],
            top_n=args.top_n,
            width=args.plot_width,
            height=args.plot_height,
            bottom_quantile=args.bottom_quantile,
        )
        _commit_stage(
            paths["plot_complete"],
            plot_signature,
            {"PDF": paths["plot"]},
        )
        print(f"Saved PDF: {paths['plot']}", flush=True)

    manifest = {
        "Arguments": vars(args),
        "Effective_Expression_Mode": expression_mode,
        "Selection_Audit": selection_audit,
        "Scan_Audit": scan_audit,
        "Scan_Specification": scan_specification,
        "Summary_Specification": summary_specification,
        "Summary_Signature": summary_signature,
        "Top_Signature": top_signature,
        "Plot_Signature": plot_signature,
        "Selected_Transcripts": int(len(selected)),
        "Summary_Rows": int(len(summary)),
        "Expected_Region_Kmer_Rows": int(
            len(REGIONS) * (4 ** args.kmer_length)
        ),
        "Outputs": {key: str(path) for key, path in paths.items()},
        "Region_Rule": (
            f"CDS when overlap >= {args.cds_overlap_threshold} nt; "
            "otherwise assigned to the flanking UTR side."
        ),
        "Attention_Definition": (
            "Received attention summed over queries and averaged across all "
            "heads and encoder layers."
        ),
        "Saliency_Definition": (
            "Gradient of mean CDS count-head output with respect to the "
            "one-hot sequence input."
        ),
        "Expression_Definition": (
            "Every selected transcript is evaluated once. In zero mode, an "
            "all-zero expression vector is supplied explicitly, so the saved "
            "cell type is not used for expression conditioning."
        ),
        "Resume_Definition": (
            "Each workflow stage is reused only after its final outputs pass "
            "parameter-signature, checksum, and schema validation. The output "
            "directory is portable across servers."
        ),
        "Fold_Change_Definition": (
            "Motif mean divided by the unweighted mean of bottom-quantile "
            "motif means within the same region. A pseudocount equal to "
            "1e-6 times the median positive motif mean is applied to both "
            "numerator and denominator."
        ),
    }
    _atomic_json(manifest, paths["manifest"])
    print("6-mer attribution scan completed.", flush=True)


if __name__ == "__main__":
    main()

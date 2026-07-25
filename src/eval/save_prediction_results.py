"""Inference helpers that save TRACE predictions in the legacy nested PKL format."""

import contextlib
import os
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from train.distributed_balanced_bucket_sampler import DistributedBucketSampler
from utils import clean_up_memory, unwrap_model


def _prepare_prediction_dataloader(
    dataset,
    collate_fn,
    num_samples: Optional[int],
    batch_size: int,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
):
    """Resolve distributed settings, select samples, and build a bucketed loader."""
    if torch.distributed.is_initialized():
        rank = rank if rank is not None else torch.distributed.get_rank()
        world_size = world_size if world_size is not None else torch.distributed.get_world_size()
    else:
        rank = rank if rank is not None else 0
        world_size = world_size if world_size is not None else 1

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_samples is not None and num_samples < 1:
        raise ValueError("num_samples must be positive or None")

    all_indices = np.arange(len(dataset))
    if num_samples is not None and len(all_indices) > num_samples:
        rng = np.random.default_rng(42)
        target_indices = rng.choice(all_indices, num_samples, replace=False)
    else:
        target_indices = all_indices

    print(f"[Rank {rank}] Selected {len(target_indices)} samples for inference.")
    subset = Subset(dataset, target_indices)

    if hasattr(dataset, "lengths"):
        subset_lengths = [int(dataset.lengths[i]) for i in target_indices]
    else:
        print(f"[Rank {rank}] Calculating sequence lengths from dataset samples...")
        subset_lengths = [int(dataset[i][5].shape[0]) for i in target_indices]

    sampler = DistributedBucketSampler(
        lengths=subset_lengths,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    num_workers = 4
    dataloader = DataLoader(
        subset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    return dataloader, rank, world_size


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _extract_head_tensor(output, head_name: str):
    head_output = output[head_name]
    if isinstance(head_output, dict):
        if "profile" not in head_output:
            raise KeyError(
                f"Head '{head_name}' returned a dictionary without a 'profile' key."
            )
        head_output = head_output["profile"]
    if not isinstance(head_output, torch.Tensor):
        raise TypeError(f"Head '{head_name}' must return a tensor or profile dictionary.")
    return head_output


def save_count_predictions(
    model,
    dataset,
    num_samples: int = 200,
    batch_size: int = 16,
    out_dir: str = "./results",
    suffix: str = "count",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
):
    """Predict positional density and save ``{cell_type: {tid: signal}}``."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    base_model = unwrap_model(model)
    device = _model_device(base_model)

    def collate_fn_count(batch):
        uuids, species, cell_types, expr_vectors, meta_infos, seq_embs, _ = zip(*batch)
        lengths = [int(sequence.shape[0]) for sequence in seq_embs]
        seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        return (
            uuids,
            list(species),
            list(cell_types),
            torch.stack(expr_vectors),
            meta_infos,
            seq_padded,
            lengths,
        )

    dataloader, run_rank, run_world_size = _prepare_prediction_dataloader(
        dataset, collate_fn_count, num_samples, batch_size, rank, world_size
    )

    model_name = getattr(base_model, "model_name", "model")
    file_name = f"predictions_count.{model_name}.{suffix}"
    file_name += f".rank{run_rank}.pkl" if run_world_size > 1 else ".pkl"
    save_path = os.path.join(out_dir, file_name)

    saved_data = {}
    duplicate_count = 0
    iterator = (
        tqdm(dataloader, desc=f"[Rank {run_rank}] Infer Count")
        if run_rank == 0 or run_world_size == 1
        else dataloader
    )

    with torch.inference_mode():
        for batch_data in iterator:
            (
                b_uuids,
                b_species,
                b_cell_types,
                b_expr_vectors,
                _,
                b_seq,
                b_lengths,
            ) = batch_data

            b_seq = b_seq.to(device, non_blocking=device.type == "cuda")
            b_expr_vectors = b_expr_vectors.to(
                device, non_blocking=device.type == "cuda"
            )
            positions = torch.arange(b_seq.shape[1]).unsqueeze(0)
            src_mask = positions < torch.tensor(b_lengths).unsqueeze(1)
            src_mask = src_mask.to(device, non_blocking=device.type == "cuda")

            with _autocast_context(device):
                output = base_model.predict(
                    seq_batch=b_seq,
                    species=b_species,
                    expr_vector=b_expr_vectors,
                    src_mask=src_mask,
                    head_names=["count"],
                )
            pred_batch = _extract_head_tensor(output, "count")

            for index, uuid in enumerate(b_uuids):
                valid_len = b_lengths[index]
                pred = (
                    pred_batch[index, :valid_len]
                    .squeeze(-1)
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                tid = str(uuid).split("-", 1)[0]
                cell_type = str(b_cell_types[index])
                if cell_type not in saved_data:
                    saved_data[cell_type] = {}
                if tid in saved_data[cell_type]:
                    duplicate_count += 1
                saved_data[cell_type][tid] = pred

    total_preds = sum(len(tids) for tids in saved_data.values())
    print(
        f"[Rank {run_rank}] Saving {total_preds} Count predictions across "
        f"{len(saved_data)} cell types to {save_path}"
    )
    if duplicate_count:
        print(
            f"[Rank {run_rank}] Warning: {duplicate_count} duplicate cell-type/TID "
            "keys were overwritten by the legacy PKL format."
        )

    with open(save_path, "wb") as handle:
        pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    clean_up_memory()
    return save_path


def save_te_predictions(
    model,
    dataset,
    num_samples: int = 200,
    batch_size: int = 16,
    out_dir: str = "./results",
    suffix: str = "te",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
):
    """Predict scalar TE values and save ``{cell_type: {tid: scalar}}``."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    base_model = unwrap_model(model)
    device = _model_device(base_model)

    def collate_fn_te(batch):
        uuids, species, cell_types, expr_vectors, meta_infos, seq_embs, _ = zip(*batch)
        lengths = [int(sequence.shape[0]) for sequence in seq_embs]
        seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        return (
            uuids,
            list(species),
            list(cell_types),
            torch.stack(expr_vectors),
            meta_infos,
            seq_padded,
            lengths,
        )

    dataloader, run_rank, run_world_size = _prepare_prediction_dataloader(
        dataset, collate_fn_te, num_samples, batch_size, rank, world_size
    )

    model_name = getattr(base_model, "model_name", "model")
    file_name = f"predictions_te.{model_name}.{suffix}"
    file_name += f".rank{run_rank}.pkl" if run_world_size > 1 else ".pkl"
    save_path = os.path.join(out_dir, file_name)

    saved_data = {}
    iterator = (
        tqdm(dataloader, desc=f"[Rank {run_rank}] Infer TE")
        if run_rank == 0 or run_world_size == 1
        else dataloader
    )

    with torch.inference_mode():
        for batch_data in iterator:
            (
                b_uuids,
                b_species,
                b_cell_types,
                b_expr_vectors,
                _,
                b_seq,
                b_lengths,
            ) = batch_data
            b_seq = b_seq.to(device, non_blocking=device.type == "cuda")
            b_expr_vectors = b_expr_vectors.to(
                device, non_blocking=device.type == "cuda"
            )
            positions = torch.arange(b_seq.shape[1]).unsqueeze(0)
            src_mask = positions < torch.tensor(b_lengths).unsqueeze(1)
            src_mask = src_mask.to(device, non_blocking=device.type == "cuda")

            with _autocast_context(device):
                output = base_model.predict(
                    seq_batch=b_seq,
                    species=b_species,
                    expr_vector=b_expr_vectors,
                    src_mask=src_mask,
                    head_names=["te"],
                )
            te_values = _extract_head_tensor(output, "te").reshape(len(b_uuids), -1)

            for index, uuid in enumerate(b_uuids):
                tid = str(uuid).split("-", 1)[0]
                cell_type = str(b_cell_types[index])
                te_scalar = float(te_values[index, 0].float().cpu().item())
                if cell_type not in saved_data:
                    saved_data[cell_type] = {}
                saved_data[cell_type][tid] = te_scalar

    total_preds = sum(len(tids) for tids in saved_data.values())
    print(
        f"[Rank {run_rank}] Saving {total_preds} TE predictions across "
        f"{len(saved_data)} cell types to {save_path}"
    )
    with open(save_path, "wb") as handle:
        pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    clean_up_memory()
    return save_path

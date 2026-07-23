#!/usr/bin/env python3
"""Compare legacy 80/10/10 RPF masking with a strict zero-RPF input."""

import contextlib
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.translation_dataset import TranslationDataset
from model.prediction_heads import PsiteDensityHead
from model.translation_base_model import TranslationBaseModel
from train.distributed_balanced_bucket_sampler import DistributedBucketSampler
from train.masking_adapter import BatchMaskingAdapter
from train.model_trainer import Trainer


# -----------------------------------------------------------------------------
# Validation configuration: edit paths here before running the script.
# -----------------------------------------------------------------------------
DATASET_DIR = Path("/public-supool/home/annie/translation_model/dataset")
VALID_DATASET_FILES = [
    "human_tissue_22c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "human_cell_line_18c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "macaque_4c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "mouse_3c_6k_depth0.1_cov0.1_rpm1.valid.h5",
]

MODEL_CONFIG_PATH = SRC_DIR / "config/base_model_expr_384d_16h_12l_128env_32ad.yaml"
CHECKPOINT_PATH = Path(
    "/public-supool/home/annie/translation_model/checkpoint/pretrain/"
    "base_model_expr_384d_16h_12l_128env_32ad-PsiteDensityHead."
    "hs_22c_18c_26c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1.100_0.001.latest.pt"
)

HEAD_HIDDEN_DIM = 384
BATCH_SIZE = 50
NUM_WORKERS = 5
PREFETCH_FACTOR = 5
PRINT_EVERY = 50
RANDOM_SEED = 20260723
MAX_BATCHES = None  # Set an integer for a quick partial-dataset smoke test.

METRIC_NAMES = (
    "total",
    "micro",
    "macro",
    "macro_weighted",
    "ranking",
    "ranking_weighted",
)


def setup_runtime():
    """Initialize single-GPU or single-node multi-GPU validation."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != local_world_size:
        raise RuntimeError("Multi-node validation is not supported by this script.")

    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed:
        dist.init_process_group(backend=backend)
    return device, local_rank, world_size, distributed


def resolve_paths(file_names):
    """Resolve and validate dataset files."""
    paths = [DATASET_DIR / file_name for file_name in file_names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing validation datasets: " + ", ".join(missing))
    return paths


def collate_validation_batch(batch):
    """Pad a validation batch without modifying its RPF target."""
    _, species, _, expr_vectors, meta_info, seq_embs, count_embs = zip(*batch)

    species_list = list(species)
    expr_batch = torch.stack(expr_vectors)
    cds_starts = [int(meta.get("cds_start_pos", -1)) for meta in meta_info]
    cds_stops = [int(meta.get("cds_end_pos", -1)) for meta in meta_info]
    motif_occs = [meta.get("motif_occ", []) for meta in meta_info]

    seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
    count_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
    pad_masks = (seq_padded != -1)[:, :, 0]

    batch_size, max_length = seq_padded.shape[:2]
    cds_masks = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for index, (start, stop) in enumerate(zip(cds_starts, cds_stops)):
        if start != -1 and stop != -1 and stop > start:
            cds_masks[index, start - 1 : min(stop, max_length)] = True
        else:
            cds_masks[index, :] = True

    return {
        "species": species_list,
        "expr": expr_batch,
        "sequence": seq_padded,
        "count_target": count_padded,
        "pad_mask": pad_masks,
        "cds_mask": cds_masks,
        "cds_starts": cds_starts,
        "motif_occs": motif_occs,
    }


def build_validation_loader(dataset_paths, batch_size, world_size, rank):
    """Build the same length-bucketed validation stream used by the Trainer."""
    datasets = [TranslationDataset.from_h5(str(path), lazy=True) for path in dataset_paths]
    dataset = ConcatDataset(datasets)
    lengths = [length for current in datasets for length in current.lengths]
    sampler = DistributedBucketSampler(
        lengths=lengths,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
        balance_classes=False,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_validation_batch,
    }
    if NUM_WORKERS > 0:
        loader_kwargs.update(
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True,
        )
    return DataLoader(**loader_kwargs)


def load_legacy_model(device):
    """Create the historical model and strictly restore its checkpoint."""
    if not MODEL_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing model config: {MODEL_CONFIG_PATH}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    model = TranslationBaseModel.from_config(str(MODEL_CONFIG_PATH))
    model.add_head(
        "count",
        PsiteDensityHead.create_from_model(model, d_pred_h=HEAD_HIDDEN_DIM),
        overwrite=True,
        move_to_model_device=False,
    )
    model.to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary.")
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    state_dict = model._strip_head_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def build_legacy_loss_evaluator():
    """Create only the state required by the legacy Trainer loss method."""
    evaluator = object.__new__(Trainer)
    evaluator.dynamics_criterion = nn.SmoothL1Loss(reduction="none", beta=1)
    evaluator.te_criterion = nn.MSELoss(reduction="none")
    evaluator.alpha_limit = (4.0, 4.0)
    evaluator.current_alpha = 4.0
    return evaluator


def amp_context(device):
    """Match the BF16 autocast behavior used during historical validation."""
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def move_batch_to_device(batch, device):
    """Move tensor fields while preserving string and metadata fields."""
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def evaluate(model, loader, loss_evaluator, device, rank, distributed):
    """Evaluate legacy masking and strict-zero RPF inputs on identical batches."""
    masking_adapter = BatchMaskingAdapter(mask_value=0.0)
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(RANDOM_SEED + rank)

    metric_sums = torch.zeros((2, len(METRIC_NAMES)), dtype=torch.float64, device=device)
    batch_count = torch.zeros(1, dtype=torch.float64, device=device)
    retained_nonzero = torch.zeros(2, dtype=torch.float64, device=device)

    total_batches = len(loader) if MAX_BATCHES is None else min(len(loader), MAX_BATCHES)
    progress = tqdm(total=total_batches, desc="Legacy RPF leakage test") if rank == 0 else None

    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(loader):
            if MAX_BATCHES is not None and batch_index >= MAX_BATCHES:
                break

            legacy_count, legacy_eval_mask = masking_adapter.get_random_masked_batch(
                embeddings=cpu_batch["count_target"],
                cds_starts=cpu_batch["cds_starts"],
                occs=cpu_batch["motif_occs"],
                pad_mask=cpu_batch["pad_mask"],
                mask_perc_range=(1.0, 1.0),
                full_mask_perc=1.0,
                generator=mask_generator,
            )
            if not torch.equal(legacy_eval_mask, cpu_batch["pad_mask"]):
                raise RuntimeError("Full legacy masking did not select every valid position.")

            nonzero_target = (
                cpu_batch["pad_mask"]
                & cpu_batch["count_target"].squeeze(-1).ne(0)
            )
            unchanged_nonzero = (
                nonzero_target
                & legacy_count.squeeze(-1).eq(cpu_batch["count_target"].squeeze(-1))
            )
            retained_nonzero[0] += unchanged_nonzero.sum().to(device=device, dtype=torch.float64)
            retained_nonzero[1] += nonzero_target.sum().to(device=device, dtype=torch.float64)

            batch = move_batch_to_device(cpu_batch, device)
            legacy_count = legacy_count.to(device, non_blocking=device.type == "cuda")
            legacy_eval_mask = legacy_eval_mask.to(device, non_blocking=device.type == "cuda")
            strict_zero_count = torch.zeros_like(batch["count_target"])

            with amp_context(device):
                legacy_output = model(
                    seq_batch=batch["sequence"],
                    count_batch=legacy_count,
                    species=batch["species"],
                    expr_vector=batch["expr"],
                    src_mask=batch["pad_mask"],
                    head_names=["count"],
                )
                _, legacy_components = loss_evaluator.count_task_criterion(
                    legacy_output,
                    batch["count_target"],
                    legacy_eval_mask,
                    batch["cds_mask"],
                    is_eval=True,
                    return_components=True,
                )
            del legacy_output

            with amp_context(device):
                strict_output = model(
                    seq_batch=batch["sequence"],
                    count_batch=strict_zero_count,
                    species=batch["species"],
                    expr_vector=batch["expr"],
                    src_mask=batch["pad_mask"],
                    head_names=["count"],
                )
                _, strict_components = loss_evaluator.count_task_criterion(
                    strict_output,
                    batch["count_target"],
                    batch["pad_mask"],
                    batch["cds_mask"],
                    is_eval=True,
                    return_components=True,
                )
            del strict_output

            for metric_index, metric_name in enumerate(METRIC_NAMES):
                metric_sums[0, metric_index] += legacy_components[metric_name].double()
                metric_sums[1, metric_index] += strict_components[metric_name].double()
            batch_count += 1

            if progress is not None:
                progress.update(1)
                if (batch_index + 1) % PRINT_EVERY == 0:
                    progress.set_postfix(
                        legacy=f"{legacy_components['total'].item():.4f}",
                        zero=f"{strict_components['total'].item():.4f}",
                    )

    if progress is not None:
        progress.close()

    if distributed:
        dist.all_reduce(metric_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(retained_nonzero, op=dist.ReduceOp.SUM)

    if batch_count.item() == 0:
        raise RuntimeError("No validation batches were evaluated.")
    return metric_sums / batch_count, retained_nonzero


def print_results(metric_means, retained_nonzero, checkpoint):
    """Print a compact comparison table and interpretation guidance."""
    epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
    best_val = checkpoint.get("best_val_loss", "unknown") if isinstance(checkpoint, dict) else "unknown"
    retained_fraction = (
        retained_nonzero[0] / retained_nonzero[1]
        if retained_nonzero[1].item() > 0
        else torch.tensor(float("nan"), device=retained_nonzero.device)
    )

    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint epoch: {epoch}; stored best validation loss: {best_val}")
    print(f"Exact retained fraction among non-zero RPF targets: {retained_fraction.item():.4%}")
    print("\nMetric                 legacy_80/10/10    strict_zero       delta       ratio")
    print("-" * 78)
    for metric_index, metric_name in enumerate(METRIC_NAMES):
        legacy_value = metric_means[0, metric_index].item()
        strict_value = metric_means[1, metric_index].item()
        delta = strict_value - legacy_value
        ratio = strict_value / legacy_value if legacy_value != 0 else float("inf")
        print(
            f"{metric_name:<22}"
            f"{legacy_value:>14.6f}"
            f"{strict_value:>15.6f}"
            f"{delta:>12.6f}"
            f"{ratio:>12.3f}x"
        )

    print(
        "\nInterpretation: a large strict-zero increase supports the hypothesis that "
        "the legacy masked-RPF input leaked target information. A small increase means "
        "the sequence-only loss gap must mainly come from another model or training difference."
    )


def main():
    device, rank, world_size, distributed = setup_runtime()
    try:
        torch.manual_seed(RANDOM_SEED + rank)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        dataset_paths = resolve_paths(VALID_DATASET_FILES)
        loader = build_validation_loader(dataset_paths, BATCH_SIZE, world_size, rank)
        model, checkpoint = load_legacy_model(device)
        loss_evaluator = build_legacy_loss_evaluator()

        if rank == 0:
            print(f"Device: {device}; world_size: {world_size}; batches per rank: {len(loader)}")
            print("Mode 1: legacy full selection with 80% zero / 10% random / 10% unchanged")
            print("Mode 2: strict all-zero count_batch")

        metric_means, retained_nonzero = evaluate(
            model=model,
            loader=loader,
            loss_evaluator=loss_evaluator,
            device=device,
            rank=rank,
            distributed=distributed,
        )
        if rank == 0:
            print_results(metric_means, retained_nonzero, checkpoint)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

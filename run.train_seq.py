#!/usr/bin/env python3
"""Editable Python launcher for TRACE sequence-only training."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead
from train.model_trainer_seq import Trainer
from utils import print_param_counts


# -----------------------------------------------------------------------------
# Experiment configuration: edit this section before running the script.
# -----------------------------------------------------------------------------
DATASET_DIR = Path("/public-supool/home/annie/translation_model/dataset")
TRAIN_DATASET_FILES = [
    "human_tissue_22c_6k_depth0.1_cov0.1_rpm1.train.h5",
    # "human_cell_line_18c_6k_depth0.1_cov0.1_rpm1.train.h5",
    # "human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.train.h5",
    "macaque_4c_6k_depth0.1_cov0.1_rpm1.train.h5",
    "mouse_3c_6k_depth0.1_cov0.1_rpm1.train.h5",
]
VALID_DATASET_FILES = [
    "human_tissue_22c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    # "human_cell_line_18c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    # "human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "macaque_4c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "mouse_3c_6k_depth0.1_cov0.1_rpm1.valid.h5",
]
DATASET_NAME = "hs_22c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1"

MODEL_CONFIG_PATH = SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint/train"
LOG_DIR = PROJECT_ROOT / "log/train"

HEAD_HIDDEN_DIM = 384
BATCH_SIZE = 50
EPOCH_NUM = 60
PATIENCE = 8
EARLY_STOPPING_START_EPOCH = None  # None starts monitoring after LR warmup.
ALPHA_LIMIT = (0.2, 4.0)
RANKING_LOSS_WEIGHT = 0.2
LEARNING_RATE = 1e-3
LR_WARMUP_PERC = 0.3
ACCUMULATION_STEPS = 1
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
EPSILON = 1e-9
EXPR_NOISE_STD = 0.1
MASK_PERC = {"species": 0.1, "cell": 0.1}
BALANCE_CLASSES = True
RESUME = True
SAVE_EVERY = 1
PRINT_EVERY = 50


def setup_runtime():
    """Initialize single-GPU or single-node multi-GPU training."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != local_world_size:
        raise RuntimeError("Multi-node training is not supported by this launcher.")
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


def resolve_dataset_paths(file_names):
    paths = [DATASET_DIR / file_name for file_name in file_names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Dataset files were not found. Edit DATASET_DIR and dataset file lists: "
            + ", ".join(missing)
        )
    return [str(path) for path in paths]


def main():
    device, rank, world_size, distributed = setup_runtime()
    train_paths = resolve_dataset_paths(TRAIN_DATASET_FILES)
    valid_paths = resolve_dataset_paths(VALID_DATASET_FILES)

    model = BaseModel.from_config(str(MODEL_CONFIG_PATH))
    model.add_head(
        "count",
        PsiteDensityHead.create_from_model(model, d_pred_h=HEAD_HIDDEN_DIM),
        overwrite=True,
        move_to_model_device=False,
    )
    model.to(device)

    if rank == 0:
        print(model.model_name)
        print(model.list_heads())
        print_param_counts(model)

    if distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[rank], output_device=rank)
        model = DDP(model, **ddp_kwargs)

    trainer = Trainer(
        model=model,
        dataset_paths=train_paths,
        val_dataset_paths=valid_paths,
        dataset_name=DATASET_NAME,
        batch_size=BATCH_SIZE,
        checkpoint_dir=str(CHECKPOINT_DIR),
        log_dir=str(LOG_DIR),
        world_size=world_size,
        rank=rank,
        resume=RESUME,
        print_progress_every=PRINT_EVERY,
        save_every=SAVE_EVERY,
        epoch_num=EPOCH_NUM,
        patience=PATIENCE,
        early_stopping_start_epoch=EARLY_STOPPING_START_EPOCH,
        alpha_limit=ALPHA_LIMIT,
        ranking_loss_weight=RANKING_LOSS_WEIGHT,
        mask_perc=MASK_PERC,
        expr_noise_std=EXPR_NOISE_STD,
        learning_rate=LEARNING_RATE,
        lr_warmup_perc=LR_WARMUP_PERC,
        accumulation_steps=ACCUMULATION_STEPS,
        balance_classes=BALANCE_CLASSES,
        beta=BETAS,
        epsilon=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    trainer.fit()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

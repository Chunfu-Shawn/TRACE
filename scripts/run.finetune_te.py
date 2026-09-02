#!/usr/bin/env python3
"""Editable Python launcher for scalar TE fine-tuning."""

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
from model.prediction_heads import TERegressionHead
from train.model_finetune_te import TEFinetuneTrainer, configure_te_finetuning
from utils import print_param_counts


# -----------------------------------------------------------------------------
# Experiment configuration: edit this section before running the script.
# -----------------------------------------------------------------------------
DATASET_DIR = Path("/path/to/dataset")
TRAIN_DATASET_FILES = ["human.train.h5"]
VALID_DATASET_FILES = ["human.valid.h5"]
DATASET_NAME = "human_te"

MODEL_CONFIG_PATH = SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad.yaml"
PRETRAINED_CHECKPOINT = None
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint/finetune"
LOG_DIR = PROJECT_ROOT / "log/finetune"

FT_MODE = "head"
LORA_R = 8
LORA_ALPHA = 16.0
HEAD_HIDDEN_DIM = 384
HEAD_DROPOUT = 0.1

BATCH_SIZE = 16
EPOCH_NUM = 30
PATIENCE = 5
LEARNING_RATE = 5e-4
LR_WARMUP_PERC = 0.1
ACCUMULATION_STEPS = 1
WEIGHT_DECAY = 0.01
BALANCE_CLASSES = False
NUM_WORKERS = 0
RESUME = True


def setup_runtime():
    """Initialize a single-process or torchrun-distributed runtime."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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
    if PRETRAINED_CHECKPOINT is not None:
        model.load_pretrained_weights(
            str(PRETRAINED_CHECKPOINT), strict=False, map_location="cpu"
        )
    model.add_head(
        "te",
        TERegressionHead.create_from_model(
            model, d_pred_h=HEAD_HIDDEN_DIM, p_drop=HEAD_DROPOUT
        ),
        overwrite=True,
        move_to_model_device=False,
    )
    configure_te_finetuning(
        model, ft_mode=FT_MODE, lora_r=LORA_R, lora_alpha=LORA_ALPHA
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

    trainer = TEFinetuneTrainer(
        model=model,
        dataset_paths=train_paths,
        val_dataset_paths=valid_paths,
        dataset_name=DATASET_NAME,
        batch_size=BATCH_SIZE,
        checkpoint_dir=str(CHECKPOINT_DIR),
        log_dir=str(LOG_DIR),
        world_size=world_size,
        rank=rank,
        ft_mode=FT_MODE,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        epoch_num=EPOCH_NUM,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        lr_warmup_perc=LR_WARMUP_PERC,
        accumulation_steps=ACCUMULATION_STEPS,
        balance_classes=BALANCE_CLASSES,
        weight_decay=WEIGHT_DECAY,
        resume=RESUME,
        num_workers=NUM_WORKERS,
    )
    trainer.finetune()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

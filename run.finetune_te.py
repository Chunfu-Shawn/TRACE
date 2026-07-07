#!/usr/bin/env python3
"""
TE Fine-tuning Launcher — run with torchrun.

Usage:
  torchrun --nproc_per_node=2 run.finetune_te.py

Supported ft_mode:
  - "lora":  LoRA-adapt backbone + train TE head
  - "head":  freeze backbone, train only TE head
  - "full":  full fine-tuning
"""

import sys, os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

sys.path.append("/public-supool/home/annie/translation_model/TRACE/src")

from model.translation_base_model import TranslationBaseModel
from model.prediction_heads import TERegressionHead
from train.model_finetune_te import TEFinetuneTrainer
from utils import print_param_counts

# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------
rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
dataset_dir = "/public-supool/home/annie/translation_model/dataset/"

# Use the same datasets as pretraining (or a TE-specific subset)
human_t_dataset_name = "human_tissue_21c_6k_depth0.1_cov0.1_rpm1"
human_cl_dataset_name = "human_cell_line_18c_6k_depth0.1_cov0.1_rpm1"
human_cl_un_dataset_name = "human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1"
macaque_dataset_name = "macaque_4c_6k_depth0.1_cov0.1_rpm1"
mouse_dataset_name = "mouse_3c_6k_depth0.1_cov0.1_rpm1"

train_paths = [
    os.path.join(dataset_dir, f"{n}.train.h5")
    for n in [human_t_dataset_name]
    # , human_cl_dataset_name, human_cl_un_dataset_name, macaque_dataset_name, mouse_dataset_name]
]
val_paths = [
    os.path.join(dataset_dir, f"{n}.valid.h5")
    for n in [human_t_dataset_name]
    # , human_cl_dataset_name, human_cl_un_dataset_name, macaque_dataset_name, mouse_dataset_name]
]

# ---------------------------------------------------------------------------
# Load pretrained model
# ---------------------------------------------------------------------------
base_model = TranslationBaseModel.from_config(
    "/public-supool/home/annie/translation_model/TRACE/src/config/base_model_expr_384d_16h_12l_128env_32ad.yaml"
).cuda(rank)

# Load pretrained weights
ckpt_path = os.path.join(
    "/public-supool/home/annie/translation_model/checkpoint/pretrain",
    "base_model_expr_384d_16h_12l_128env_32ad-TranslationProfileHead.hs_21c_18c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1.100_0.001.latest.pt",
)
if os.path.isfile(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if rank == 0:
        print(f"[Launcher] Loaded pretrained weights from {ckpt_path}")

# ---------------------------------------------------------------------------
# Add TE regression head
# ---------------------------------------------------------------------------
base_model.add_head(
    "te",
    TERegressionHead.create_from_model(base_model, d_pred_h=128, p_drop=0.3),
    overwrite=True,
)
print(base_model.model_name)
print(base_model.list_heads())
if rank == 0:
    print_param_counts(base_model)

# ---------------------------------------------------------------------------
# Wrap with DDP
# ---------------------------------------------------------------------------
base_model = DDP(base_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

# ---------------------------------------------------------------------------
# Fine-tune
# ---------------------------------------------------------------------------
trainer = TEFinetuneTrainer(
    model=base_model,
    dataset_paths=train_paths,
    val_dataset_paths=val_paths,
    dataset_name="hs_21c_18c_26c_rm_4c_mm_3c_depth0.1_cov0.1_rpm1_te",
    batch_size=50,
    checkpoint_dir="/public-supool/home/annie/translation_model/checkpoint/finetune",
    log_dir="/public-supool/home/annie/translation_model/log/finetune",
    world_size=world_size,
    rank=rank,
    # ---- Strategy ----
    ft_mode="lora",            # "lora" | "head" | "full"
    lora_r=8,
    lora_alpha=16.0,
    # ---- Training ----
    epoch_num=30,
    patience=5,
    learning_rate=5e-4,        # higher LR for head + LoRA
    lr_warmup_perc=0.1,
    accumulation_steps=1,
    balance_classes=True,
    # ---- Resume ----
    resume=True,
    save_every=1,
)
trainer.finetune()

dist.destroy_process_group()

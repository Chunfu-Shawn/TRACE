#!/usr/bin/env python3
"""Fine-tune a BaseModel with one scalar TE prediction per RNA."""

import contextlib
import json
import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from data.translation_dataset import TranslationDataset
from train.distributed_balanced_bucket_sampler import DistributedBucketSampler
from utils import unwrap_model


def _create_lr_lambda(total_steps: int, warmup_steps: int = 0, min_eta: float = 1e-4):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return max(min_eta, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(min_eta, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


def _no_weight_decay(name: str) -> bool:
    lowered = name.lower()
    return any(
        key in lowered
        for key in (".bias", "layernorm", "layer_norm", ".ln", ".embedding", "embed")
    )


def configure_te_finetuning(
    model: nn.Module,
    ft_mode: str = "head",
    lora_r: int = 8,
    lora_alpha: float = 16.0,
) -> nn.Module:
    """Configure trainable parameters before wrapping the model with DDP."""
    if ft_mode not in {"head", "full", "lora"}:
        raise ValueError("ft_mode must be one of: 'head', 'full', 'lora'")
    if hasattr(model, "module"):
        raise ValueError("configure_te_finetuning must be called before DDP wrapping")
    if not hasattr(model, "heads") or "te" not in model.heads:
        raise KeyError("Register a 'te' head before configuring TE fine-tuning")

    if ft_mode == "lora":
        from lora_utils import replace_linear_with_lora, set_trainable_base_and_lora

        replace_linear_with_lora(model, r=lora_r, lora_alpha=lora_alpha)
        set_trainable_base_and_lora(model, train_base=False, train_lora=True)
    elif ft_mode == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True

    for parameter in model.heads["te"].parameters():
        parameter.requires_grad = True
    return model


class TEFinetuneTrainer:
    """Train a registered ``te`` head against ``meta_info['te_scale']``."""

    def __init__(
        self,
        model: nn.Module,
        dataset_paths: Union[str, List[str]],
        val_dataset_paths: Union[str, List[str]],
        dataset_name: str,
        batch_size: int,
        checkpoint_dir: str,
        log_dir: str,
        world_size: int = 1,
        rank: int = 0,
        ft_mode: str = "head",
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        epoch_num: int = 30,
        patience: int = 5,
        learning_rate: float = 5e-4,
        lr_warmup_perc: float = 0.1,
        accumulation_steps: int = 1,
        balance_classes: bool = False,
        beta: Tuple[float, float] = (0.9, 0.98),
        epsilon: float = 1e-9,
        weight_decay: float = 0.01,
        resume: bool = True,
        print_progress_every: int = 20,
        save_every: int = 1,
        num_workers: int = 0,
    ):
        self.model = model
        self.raw_model = unwrap_model(model)
        if "te" not in self.raw_model.heads:
            raise KeyError("The model must have a registered head named 'te'")
        if ft_mode not in {"head", "full", "lora"}:
            raise ValueError("ft_mode must be one of: 'head', 'full', 'lora'")

        self.distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.distributed else int(world_size)
        self.rank = dist.get_rank() if self.distributed else int(rank)
        self.device = next(self.raw_model.parameters()).device
        self.model_name = self.raw_model.model_name
        self.batch_size = int(batch_size)
        self.epoch_num = int(epoch_num)
        self.patience = int(patience)
        self.patience_counter = 0
        self.ac_steps = max(1, int(accumulation_steps))
        self.lr = float(learning_rate)
        self.ft_mode = ft_mode
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.balance_classes = bool(balance_classes)
        self.weight_decay = float(weight_decay)
        self.num_workers = max(0, int(num_workers))
        self._print_progress_every = max(1, int(print_progress_every))
        self._save_every = max(1, int(save_every))

        if not any(parameter.requires_grad for parameter in self.raw_model.parameters()):
            raise RuntimeError(
                "No trainable parameters. Call configure_te_finetuning before DDP wrapping."
            )

        self.dataset, self.sampler = self._build_dataset_and_sampler(
            dataset_paths, is_train=True
        )
        self.val_dataset, self.val_sampler = self._build_dataset_and_sampler(
            val_dataset_paths, is_train=False
        )
        self.steps_per_epoch = len(self.sampler)
        total_steps = max(1, self.epoch_num * self.steps_per_epoch // self.ac_steps)
        warmup_steps = int(float(lr_warmup_perc) * total_steps)

        self.optimizer = torch.optim.AdamW(
            self._get_param_groups(), lr=self.lr, betas=beta, eps=epsilon
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_create_lr_lambda(total_steps, warmup_steps)
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        ft_tag = f"te_ft.{ft_mode}"
        if ft_mode == "lora":
            ft_tag += f".r{self.lora_r}_a{int(self.lora_alpha)}"
        effective_batch = self.batch_size * self.ac_steps * self.world_size
        self.model_full_name = (
            f"{self.model_name}.{dataset_name}.{ft_tag}.{effective_batch}_{self.lr}"
        )
        self.training_epoch_data: List[Dict[str, Any]] = []
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        if resume:
            self._maybe_load_checkpoint()

        if self.rank == 0:
            trainable = sum(
                parameter.numel()
                for parameter in self.raw_model.parameters()
                if parameter.requires_grad
            )
            total = sum(parameter.numel() for parameter in self.raw_model.parameters())
            print(f"[TEFinetune] model={self.model_full_name}")
            print(
                f"[TEFinetune] train={len(self.dataset)}, val={len(self.val_dataset)}, "
                f"device={self.device}, mode={ft_mode}"
            )
            print(
                f"[TEFinetune] trainable parameters: {trainable:,}/{total:,} "
                f"({100.0 * trainable / max(1, total):.2f}%)"
            )

    def _get_param_groups(self):
        decay, no_decay = [], []
        for name, parameter in self.raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if _no_weight_decay(name) else decay).append(parameter)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": self.weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        if not groups:
            raise RuntimeError("No trainable parameters were found")
        return groups

    def _build_dataset_and_sampler(self, paths, is_train):
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError("At least one dataset path is required")
        datasets = [TranslationDataset.from_h5(path, lazy=True) for path in paths]
        combined = ConcatDataset(datasets)
        lengths = []
        labels = []
        for dataset in datasets:
            lengths.extend(dataset.lengths)
            if is_train and self.balance_classes:
                labels.extend(
                    f"{species}_{cell_type}"
                    for species, cell_type in zip(dataset.species, dataset.cell_types)
                )
        sampler = DistributedBucketSampler(
            lengths=lengths,
            batch_size=self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=is_train,
            drop_last=is_train,
            balance_classes=self.balance_classes and is_train,
            cell_types=labels if self.balance_classes and is_train else None,
        )
        return combined, sampler

    def _loader(self, dataset, sampler, is_train):
        options = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.device.type == "cuda",
            "collate_fn": self._collate_batch,
        }
        if self.num_workers > 0:
            options.update(prefetch_factor=2, persistent_workers=True)
        return DataLoader(**options)

    def _to_device(self, tensor):
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")

    def _amp_context(self):
        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _expression_vector(self, vector):
        expected = getattr(self.raw_model, "d_expr", None)
        if expected is None or vector.numel() == expected:
            return vector.float()
        fallback = getattr(self.raw_model, "mean_expr_vector", None)
        if fallback is not None and fallback.numel() == expected:
            return fallback.detach().cpu().float()
        raise ValueError(
            f"Expression vector has {vector.numel()} values, expected {expected}"
        )

    def _collate_batch(self, batch):
        _, species, _, expr_vectors, meta_info, seq_embs, _ = zip(*batch)
        seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        pad_masks = (seq_padded != -1).any(dim=-1)
        expr_batch = torch.stack(
            [self._expression_vector(vector) for vector in expr_vectors]
        )
        te_targets = torch.zeros(len(batch), 1, dtype=torch.float32)
        valid_mask = torch.zeros(len(batch), dtype=torch.bool)
        for index, metadata in enumerate(meta_info):
            value = metadata.get("te_scale")
            if value is not None and np.isfinite(value):
                te_targets[index, 0] = float(value)
                valid_mask[index] = True
        return list(species), expr_batch, seq_padded, te_targets, pad_masks, valid_mask

    def _reduce_error_stats(self, squared_error_sum, valid_count):
        stats = torch.stack(
            [squared_error_sum.to(torch.float64), valid_count.to(torch.float64)]
        )
        if self.distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[1].item() == 0:
            return float("nan")
        return (stats[0] / stats[1]).item()

    def train_epoch(self, epoch):
        self.model.train()
        self.sampler.set_epoch(epoch)
        loader = self._loader(self.dataset, self.sampler, is_train=True)
        error_sum = torch.zeros((), device=self.device)
        valid_count = torch.zeros((), device=self.device)
        local_loss = []
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"Epoch {epoch + 1} train", disable=self.rank != 0)

        for batch_index, batch in enumerate(progress):
            species, expr, seq, target, src_mask, valid = batch
            expr = self._to_device(expr)
            seq = self._to_device(seq)
            target = self._to_device(target)
            src_mask = self._to_device(src_mask)
            valid = self._to_device(valid)
            do_sync = (batch_index + 1) % self.ac_steps == 0 or batch_index + 1 == len(loader)
            sync_context = (
                self.model.no_sync()
                if not do_sync and hasattr(self.model, "no_sync")
                else contextlib.nullcontext()
            )
            with sync_context:
                with self._amp_context():
                    prediction = self.model(
                        seq_batch=seq,
                        species=species,
                        expr_vector=expr,
                        src_mask=src_mask,
                        head_names=["te"],
                    )["te"]
                    batch_errors = (prediction[valid] - target[valid]).square()
                    loss = (
                        batch_errors.mean()
                        if batch_errors.numel() > 0
                        else prediction.sum() * 0.0
                    )
                self.scaler.scale(loss / self.ac_steps).backward()

            if do_sync:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.raw_model.parameters() if p.requires_grad], 2.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            error_sum += batch_errors.detach().sum()
            valid_count += valid.sum()
            local_loss.append(float(loss.detach()))
            if self.rank == 0 and (batch_index + 1) % self._print_progress_every == 0:
                progress.set_postfix(te_mse=f"{float(loss.detach()):.4f}")

        mean_loss = self._reduce_error_stats(error_sum, valid_count)
        if self.rank == 0:
            print(f"Epoch {epoch + 1} train TE MSE: {mean_loss:.6f}")
        return mean_loss, local_loss

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.model.eval()
        self.val_sampler.set_epoch(epoch)
        loader = self._loader(self.val_dataset, self.val_sampler, is_train=False)
        error_sum = torch.zeros((), device=self.device)
        valid_count = torch.zeros((), device=self.device)
        local_loss = []
        progress = tqdm(loader, desc=f"Epoch {epoch + 1} eval", disable=self.rank != 0)

        for batch in progress:
            species, expr, seq, target, src_mask, valid = batch
            expr = self._to_device(expr)
            seq = self._to_device(seq)
            target = self._to_device(target)
            src_mask = self._to_device(src_mask)
            valid = self._to_device(valid)
            with self._amp_context():
                prediction = self.model(
                    seq_batch=seq,
                    species=species,
                    expr_vector=expr,
                    src_mask=src_mask,
                    head_names=["te"],
                )["te"]
            batch_errors = (prediction[valid] - target[valid]).square()
            error_sum += batch_errors.sum()
            valid_count += valid.sum()
            if batch_errors.numel() > 0:
                local_loss.append(float(batch_errors.mean()))

        mean_loss = self._reduce_error_stats(error_sum, valid_count)
        if self.rank == 0:
            print(f"Epoch {epoch + 1} eval TE MSE: {mean_loss:.6f}")
        return mean_loss, local_loss

    def _checkpoint_paths(self):
        latest = os.path.join(self.checkpoint_dir, f"{self.model_full_name}.latest.pt")
        best = os.path.join(self.checkpoint_dir, f"{self.model_full_name}.best.pt")
        return latest, best

    def save_checkpoint(self, epoch, is_best):
        if self.rank != 0:
            return
        latest, best = self._checkpoint_paths()
        checkpoint = {
            "epoch": epoch,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "ft_mode": self.ft_mode,
            "training_epoch_data": self.training_epoch_data,
        }
        torch.save(checkpoint, latest)
        if is_best:
            torch.save(checkpoint, best)
        print(f"[TEFinetune] Saved checkpoint: {latest} (best={is_best})")

    def _maybe_load_checkpoint(self):
        latest, _ = self._checkpoint_paths()
        if not os.path.isfile(latest):
            return
        checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
        state = checkpoint.get("model", checkpoint.get("model_state_dict"))
        if state is None:
            raise ValueError(f"Checkpoint {latest} does not contain model weights")
        state = self.raw_model._strip_head_module_prefix(state)
        self.raw_model.load_state_dict(state, strict=False)
        optimizer_state = checkpoint.get("optimizer", checkpoint.get("optimizer_state_dict"))
        scheduler_state = checkpoint.get("scheduler", checkpoint.get("scheduler_state_dict"))
        scaler_state = checkpoint.get("scaler", checkpoint.get("scaler_state_dict"))
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.training_epoch_data = checkpoint.get("training_epoch_data", [])
        if self.rank == 0:
            print(f"[TEFinetune] Resumed from {latest} at epoch {self.start_epoch}")

    def finetune(self):
        for epoch in range(self.start_epoch, self.epoch_num):
            train_loss, _ = self.train_epoch(epoch)
            val_loss, _ = self.eval_epoch(epoch)
            if not np.isfinite(val_loss):
                raise RuntimeError("No finite TE labels were found in the validation dataset")
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.training_epoch_data.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            )
            if self.rank == 0:
                if is_best or (epoch + 1) % self._save_every == 0:
                    self.save_checkpoint(epoch + 1, is_best)
                log_path = os.path.join(
                    self.log_dir, f"{self.model_full_name}.epoch_data.json"
                )
                with open(log_path, "w", encoding="utf-8") as handle:
                    json.dump(self.training_epoch_data, handle, indent=2)
            if self.patience_counter >= self.patience:
                if self.rank == 0:
                    print(
                        f"[TEFinetune] Early stopping at epoch {epoch + 1}; "
                        f"best validation MSE={self.best_val_loss:.6f}"
                    )
                break

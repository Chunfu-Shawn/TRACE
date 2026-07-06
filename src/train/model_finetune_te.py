#!/usr/bin/env python3
"""
TE Fine-tuning Trainer with LoRA adaptation.

Supports three modes:
  - "lora":   LoRA-adapt the backbone Linear layers, train TE head from scratch.
  - "head":   Freeze backbone, train only the TE regression head.
  - "full":   Full fine-tuning (all parameters).

Training target: ``te_scale`` from dataset meta_info (scalar, ~[-2, 2]).
The count_batch is passed UNMASKED so the encoder sees the real ribosome profile.
"""

import os, time, json, math, contextlib
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils import unwrap_model
from data.translation_dataset import TranslationDataset
from train.distributed_balanced_bucket_sampler import DistributedBucketSampler


# ---------------------------------------------------------------------------
# LR schedule helper (same as pretraining)
# ---------------------------------------------------------------------------
def _create_lr_lambda(total_steps: int, warmup_steps: int = 0, min_eta: float = 1e-4):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return max(min_eta, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_eta, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def _no_weight_decay(name: str) -> bool:
    name = name.lower()
    return any(k in name for k in (".bias", "layernorm", "layer_norm", ".ln", ".embedding", "embed"))


# ---------------------------------------------------------------------------
# TE Fine-tune Trainer
# ---------------------------------------------------------------------------
class TEFinetuneTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset_paths: Union[str, List[str]],
        val_dataset_paths: Union[str, List[str]],
        dataset_name: str,
        batch_size: int,
        checkpoint_dir: str,
        log_dir: str,
        world_size: int,
        rank: int,
        # Fine-tuning strategy
        ft_mode: str = "lora",               # "lora" | "head" | "full"
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        # Training
        epoch_num: int = 30,
        patience: int = 5,
        learning_rate: float = 5e-4,
        lr_warmup_perc: float = 0.1,
        accumulation_steps: int = 1,
        balance_classes: bool = False,
        beta: Tuple[float, float] = (0.9, 0.98),
        epsilon: float = 1e-9,
        weight_decay: float = 0.01,
        # Resume / logging
        resume: bool = True,
        print_progress_every: int = 20,
        save_every: int = 1,
    ):
        self.model = model
        self.model_name = unwrap_model(model).model_name
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.patience = patience
        self.patience_counter = 0
        self.ac_steps = accumulation_steps
        self.lr = learning_rate
        self.lr_warmup_perc = lr_warmup_perc
        self.ft_mode = ft_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        self.device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
        self._print_progress_every = print_progress_every
        self._save_every = save_every

        # ---- Apply LoRA if requested ----
        if ft_mode == "lora":
            from lora_utils import replace_linear_with_lora
            raw = unwrap_model(model)
            replace_linear_with_lora(raw, r=lora_r, lora_alpha=lora_alpha)
            if rank == 0:
                print(f"[TEFinetune] Applied LoRA: r={lora_r}, alpha={lora_alpha}")
            # After LoRA replacement, freeze base weights, train only LoRA + head
            from lora_utils import set_trainable_base_and_lora
            set_trainable_base_and_lora(raw, train_base=False, train_lora=True)
            # Unfreeze the TE head
            for p in raw.heads["te"].parameters():
                p.requires_grad = True
        elif ft_mode == "head":
            raw = unwrap_model(model)
            for p in raw.parameters():
                p.requires_grad = False
            for p in raw.heads["te"].parameters():
                p.requires_grad = True
        # else "full": everything trainable (default)

        # ---- Build dataset ----
        self.dataset, self.sampler = self._build_dataset_and_sampler(dataset_paths, is_train=True)
        self.val_dataset, self.val_sampler = self._build_dataset_and_sampler(val_dataset_paths, is_train=False)

        # ---- Scheduling ----
        self.steps_per_epoch = len(self.sampler)
        self._total_steps = max(1, int(epoch_num * self.steps_per_epoch // max(1, accumulation_steps)))
        warmup_steps = int(lr_warmup_perc * self._total_steps)

        # ---- Optimizer ----
        self.optimizer = torch.optim.AdamW(
            self._get_param_groups(),
            lr=learning_rate, betas=beta, eps=epsilon, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=_create_lr_lambda(self._total_steps, warmup_steps),
        )

        # ---- AMP ----
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ---- Loss ----
        self.criterion = nn.MSELoss()

        # ---- Logging ----
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        ft_tag = f"te_ft.{ft_mode}"
        if ft_mode == "lora":
            ft_tag += f".r{lora_r}_a{int(lora_alpha)}"
        self.model_full_name = f"{self.model_name}.{dataset_name}.{ft_tag}.{batch_size * accumulation_steps * world_size}_{learning_rate}"
        self.training_epoch_data: List[Dict[str, Any]] = []
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        if resume:
            self._maybe_load_checkpoint()

        if rank == 0:
            print(f"[TEFinetune] model={self.model_full_name}")
            print(f"[TEFinetune] ft_mode={ft_mode}, lr={learning_rate}, warmup_steps={warmup_steps}")
            print(f"[TEFinetune] train={len(self.dataset)} samples, val={len(self.val_dataset)}")
            trainable = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
            total = sum(p.numel() for p in unwrap_model(model).parameters())
            print(f"[TEFinetune] trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    # ------------------------------------------------------------------
    # Params / optimizer
    # ------------------------------------------------------------------
    def _get_param_groups(self):
        raw = unwrap_model(self.model)
        decay, no_decay = [], []
        for name, p in raw.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if _no_weight_decay(name) else decay).append(p)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": 0.01})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def _build_dataset_and_sampler(self, paths, is_train):
        if isinstance(paths, str):
            paths = [paths]
        datasets = [TranslationDataset.from_h5(p, lazy=True) for p in paths]
        combined = ConcatDataset(datasets)
        all_lengths, all_cell_types = [], []
        for ds in combined.datasets:
            all_lengths.extend(ds.lengths)
            if is_train and self.balance_classes:
                all_cell_types.extend(f"{sp}_{ct}" for sp, ct in zip(ds.species, ds.cell_types))
        sampler = DistributedBucketSampler(
            lengths=all_lengths,
            batch_size=self.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=is_train,
            drop_last=is_train,
            balance_classes=(self.balance_classes and is_train),
            cell_types=all_cell_types if (self.balance_classes and is_train) else None,
        )
        return combined, sampler

    # ------------------------------------------------------------------
    # Collation / to_device
    # ------------------------------------------------------------------
    def _to_device(self, t):
        return t.to(self.device, non_blocking=self.device.type == "cuda")

    def _amp_context(self):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.use_amp else contextlib.nullcontext()

    def _collate_batch(self, batch, is_eval=False):
        """Pad a raw batch, zero out count_batch, extract TE targets."""
        _, species, _, expr_vectors, meta_info, seq_embs, count_embs = zip(*batch)

        species_list = list(species)
        expr_batch = torch.stack(expr_vectors)

        # Pad sequences
        seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
        count_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
        pad_masks = (seq_padded != -1)[:, :, 0]

        B, L = seq_padded.shape[:2]

        # Mask count_batch entirely: model must predict TE from sequence alone
        count_padded[:] = -1

        # Extract TE targets
        te_targets = torch.zeros(B, 1)
        valid_mask = torch.ones(B, dtype=torch.bool)

        for i, meta in enumerate(meta_info):
            te_val = meta.get("te_scale", None)
            if te_val is None or not np.isfinite(te_val):
                valid_mask[i] = False
                continue
            te_targets[i, 0] = float(te_val)

        return (
            species_list, expr_batch, seq_padded, count_padded,
            te_targets, pad_masks, valid_mask,
        )

    # ------------------------------------------------------------------
    # Train / Eval epochs
    # ------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        loader = DataLoader(
            self.dataset, batch_sampler=self.sampler,
            num_workers=5, prefetch_factor=5, persistent_workers=True,
            pin_memory=self.device.type == "cuda",
            collate_fn=lambda s: self._collate_batch(s, is_eval=False),
        )
        self.sampler.set_epoch(epoch)
        total_loss = torch.zeros(1, device=self.device)
        n_batches = len(loader)
        local_loss = []

        if self.rank == 0:
            pbar = tqdm(total=n_batches, desc=f"Epoch {epoch+1} train")

        for bi, batch_data in enumerate(loader):
            species_list, expr_batch, seq_padded, count_padded, te_targets, pad_masks, valid = batch_data
            if valid.sum() == 0:
                continue

            expr_batch = self._to_device(expr_batch)
            seq_padded = self._to_device(seq_padded)
            # For TE training, pass the FULL count_batch (unmasked)
            count_padded = self._to_device(count_padded)
            te_targets = self._to_device(te_targets)
            pad_masks = self._to_device(pad_masks)

            with self._amp_context():
                out = self.model(
                    seq_batch=seq_padded,
                    count_batch=count_padded,
                    species=species_list,
                    expr_vector=expr_batch,
                    src_mask=pad_masks,
                    head_names=["te"],
                )
                te_pred = out["te"]  # (B, 1)
                loss = self.criterion(te_pred[valid], te_targets[valid])
                acc_loss = loss / self.ac_steps

            do_sync = ((bi + 1) % self.ac_steps == 0) or ((bi + 1) == n_batches)
            ctx = (self.model.no_sync() if not do_sync and hasattr(self.model, "no_sync") else contextlib.nullcontext())
            with ctx:
                self.scaler.scale(acc_loss).backward()

            if do_sync:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.detach()
            local_loss.append([float(loss)])

            if self.rank == 0 and (bi + 1) % self._print_progress_every == 0:
                pbar.update(self._print_progress_every)
                print(f"  loss: {loss.item():.4f}")

        if self.rank == 0:
            pbar.close()

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_loss)
        mean_loss = total_loss.item() / float(n_batches * self.world_size)
        if self.rank == 0:
            print(f"Epoch {epoch+1} train loss: {mean_loss:.4f}")
        return mean_loss, gathered

    def eval_epoch(self, epoch):
        self.model.eval()
        loader = DataLoader(
            self.val_dataset, batch_sampler=self.val_sampler,
            num_workers=5, prefetch_factor=5, persistent_workers=True,
            pin_memory=self.device.type == "cuda",
            collate_fn=lambda s: self._collate_batch(s, is_eval=True),
        )
        total_loss = torch.zeros(1, device=self.device)
        n_batches = len(loader)
        local_loss = []

        if self.rank == 0:
            pbar = tqdm(total=n_batches, desc=f"Epoch {epoch+1} eval")

        for batch_data in loader:
            species_list, expr_batch, seq_padded, count_padded, te_targets, pad_masks, valid = batch_data
            if valid.sum() == 0:
                continue

            expr_batch = self._to_device(expr_batch)
            seq_padded = self._to_device(seq_padded)
            count_padded = self._to_device(count_padded)
            te_targets = self._to_device(te_targets)
            pad_masks = self._to_device(pad_masks)

            with torch.no_grad(), self._amp_context():
                out = self.model(
                    seq_batch=seq_padded,
                    count_batch=count_padded,
                    species=species_list,
                    expr_vector=expr_batch,
                    src_mask=pad_masks,
                    head_names=["te"],
                )
                te_pred = out["te"]
                loss = self.criterion(te_pred[valid], te_targets[valid])

            total_loss += loss.detach()
            local_loss.append([float(loss)])

            if self.rank == 0 and len(local_loss) % self._print_progress_every == 0:
                pbar.update(self._print_progress_every)

        if self.rank == 0:
            pbar.close()

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_loss)
        mean_loss = total_loss.item() / float(n_batches * self.world_size)
        if self.rank == 0:
            print(f"Epoch {epoch+1} eval loss: {mean_loss:.4f}")
        return mean_loss, gathered

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch, is_best):
        raw = unwrap_model(self.model)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "ft_mode": self.ft_mode,
        }
        path = os.path.join(self.checkpoint_dir, f"{self.model_full_name}.ckpt.{epoch}.pth")
        torch.save(ckpt, path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.model_full_name}.best.pth")
            torch.save(ckpt, best_path)
        if self.rank == 0:
            print(f"[TEFinetune] Checkpoint saved: {path}" + (" (best)" if is_best else ""))

    def _maybe_load_checkpoint(self):
        ckpt_path = os.path.join(self.checkpoint_dir, f"{self.model_full_name}.best.pth")
        if not os.path.isfile(ckpt_path):
            return
        if self.rank == 0:
            print(f"[TEFinetune] Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        unwrap_model(self.model).load_state_dict(ckpt["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # ------------------------------------------------------------------
    # Orchestrate
    # ------------------------------------------------------------------
    def finetune(self):
        for epoch in range(self.start_epoch, self.epoch_num):
            if self.rank == 0:
                print(f"\n[TEFinetune] === Epoch {epoch+1}/{self.epoch_num} ===")

            train_loss, _ = self.train_epoch(epoch)
            val_loss, _ = self.eval_epoch(epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.rank == 0:
                    print(f"[TEFinetune] Early stop counter: {self.patience_counter}/{self.patience}")

            self.training_epoch_data.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            if self.rank == 0 and (epoch + 1) % self._save_every == 0:
                self.save_checkpoint(epoch + 1, is_best)
                log_path = os.path.join(self.log_dir, f"{self.model_full_name}.epoch_data.json")
                with open(log_path, "w") as f:
                    json.dump(self.training_epoch_data, f)

            if self.patience_counter >= self.patience:
                if self.rank == 0:
                    print(f"\n[TEFinetune] Early stopping at epoch {epoch+1}. Best val_loss={self.best_val_loss:.4f}")
                    self.save_checkpoint(epoch + 1, is_best=False)
                break

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gc

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def print_param_counts(model: nn.Module) -> None:
    """
    Print total parameter count and number of trainable parameters.
    Useful to verify replacement and trainability settings after freezing/unfreezing.
    """
    model = unwrap_model(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: trainable {trainable:,} / total {total:,} ({100.0 * trainable / total:.2f}% trainable)")
    
# -------------------------
# Utilities for DDP wrappers
# -------------------------

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    If model is wrapped in DistributedDataParallel or DataParallel, return the underlying module,
    otherwise return model itself. This is useful because named_modules() on wrappers includes the wrapper.
    """
    # DDP wrappers in torch usually expose `.module`
    return getattr(model, "module", model)


def clean_up_memory():
    if 'dataset' in globals(): del dataset
    if 'saved_data' in globals(): del saved_data
    if 'dataloader' in globals(): del dataloader
    
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Memory clean up finished.")


def freeze_encoder_for_finetuning(model: nn.Module, trainable_keywords: list = None):
    """
    Freeze most model parameters, only unfreeze layers matching keywords.
    
    Args:
        model: the PyTorch model to freeze
        trainable_keywords: list of layer name substrings to keep trainable.
    """
    # if no keywords specified, provide a sensible default set
    if trainable_keywords is None:
        trainable_keywords = [
            "adaln",           # AdaLN mapping network
            "modulator",       # alternative name for AdaLN network
            "cell_embed",      # cell-type embedding layer
            "cell_type",       # another common cell-type naming
            "head",            # output prediction heads (e.g., seq_head, count_head)
            "out_proj",        # final linear projection layer
            "classifier"       # classifier head
        ]
    
    print("=== Freezing Model Parameters ===")
    
    # Step 1: freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Step 2: iterate over named parameters and unfreeze those matching keywords
    unfrozen_count = 0
    frozen_count = 0
    unfrozen_names = []
    
    for name, param in model.named_parameters():
        # case-insensitive matching
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in trainable_keywords):
            param.requires_grad = True
            unfrozen_count += param.numel()
            unfrozen_names.append(name)
        else:
            frozen_count += param.numel()

    # print freeze/unfreeze statistics
    print(f"-> Frozen parameters: {frozen_count:,} (backbone)")
    print(f"-> Trainable parameters: {unfrozen_count:,} (AdaLN, embeddings, heads)")
    print("-> Unfrozen layers:")
    for name in unfrozen_names:
        print(f"   - {name}")
        
    return model
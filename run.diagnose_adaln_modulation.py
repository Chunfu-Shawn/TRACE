#!/usr/bin/env python3
"""Summarize raw and bounded AdaLN modulation values across cell environments."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.translation_dataset import TranslationDataset
from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead


# -----------------------------------------------------------------------------
# Diagnostic configuration: edit these paths before running on the server.
# -----------------------------------------------------------------------------
DATASET_PATH = (
    PROJECT_ROOT.parent
    / "dataset/human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.test.h5"
)
MODEL_CONFIG_PATH = SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
CHECKPOINT_PATH = (
    PROJECT_ROOT.parent
    / "checkpoint/train/base_model_384d_16h_12l_64env_16ad_bs-"
    "PsiteDensityHead.hs_22c_18c_6k_depth0.1_cov0.1_rpm1_"
    "e50_a2_b02_exp_aug.100_0.001.best_profile.pt"
)
OUTPUT_DIR = PROJECT_ROOT.parent / "results/ablation/adaln_modulation"
HEAD_HIDDEN_DIM = 384
CELL_BATCH_SIZE = 32


def checkpoint_state_dict(checkpoint):
    """Extract a model state dictionary from supported checkpoint formats."""
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    if isinstance(checkpoint, dict) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint, {}
    raise ValueError(f"Unsupported checkpoint format: {CHECKPOINT_PATH}")


def load_model(device):
    """Create BaseModel, attach its density head, and restore the checkpoint."""
    model = BaseModel.from_config(str(MODEL_CONFIG_PATH))
    model.add_head(
        "count",
        PsiteDensityHead.create_from_model(model, d_pred_h=HEAD_HIDDEN_DIM),
        overwrite=True,
        move_to_model_device=False,
    )
    model.to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict, metadata = checkpoint_state_dict(checkpoint)
    state_dict = model._strip_head_module_prefix(state_dict)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module.") :]: value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, metadata


def load_cell_environments(dataset):
    """Return one expression vector and species label per unique cell type."""
    cell_types = sorted(dataset.cell_expr_dict)
    species_by_cell = {}
    for cell_type, species in zip(dataset.cell_types, dataset.species):
        species_by_cell.setdefault(str(cell_type), str(species))
    expression = torch.from_numpy(
        np.stack([dataset.cell_expr_dict[cell_type] for cell_type in cell_types])
    ).float()
    species = [species_by_cell.get(cell_type, "unknown") for cell_type in cell_types]
    return cell_types, species, expression


def summarize_values(raw_values, bound):
    """Calculate the requested raw percentiles and bound-occupancy fractions."""
    raw = torch.cat(raw_values).float()
    absolute_raw = raw.abs()
    row = {
        "Value_N": int(raw.numel()),
        "Raw_Mean": float(raw.mean()),
        "Raw_Std": float(raw.std(unbiased=False)),
        "Raw_Abs_P95": float(torch.quantile(absolute_raw, 0.95)),
        "Raw_Abs_P99": float(torch.quantile(absolute_raw, 0.99)),
        "Raw_Abs_Max": float(absolute_raw.max()),
        "Bound": float(bound) if bound is not None else np.nan,
    }
    if bound is None:
        row["Fraction_Raw_GE_0.9_Bound"] = np.nan
        row["Fraction_Bounded_GE_0.9_Bound"] = np.nan
        return row

    bounded = bound * torch.tanh(raw / bound)
    row["Fraction_Raw_GE_0.9_Bound"] = float(
        (absolute_raw >= 0.9 * bound).float().mean()
    )
    row["Fraction_Bounded_GE_0.9_Bound"] = float(
        (bounded.abs() >= 0.9 * bound).float().mean()
    )
    return row


@torch.inference_mode()
def collect_modulation_statistics(model, species, expression, device):
    """Collect raw gamma, beta, and alpha values from every AdaLN sublayer."""
    names = ("gamma", "beta", "alpha")
    sublayer_names = ("attention", "ffn")
    all_values = {name: [] for name in names}
    layer_values = {}

    for start in range(0, len(species), CELL_BATCH_SIZE):
        stop = min(start + CELL_BATCH_SIZE, len(species))
        batch_expression = expression[start:stop].to(device)
        batch_species = species[start:stop]
        species_indices = model._normalize_species(
            batch_species, len(batch_species)
        ).to(device)
        species_embedding = model.species_embedding(species_indices)
        compact_style = model.expr_projector(
            torch.cat([batch_expression, species_embedding], dim=-1)
        )

        for layer_index, layer in enumerate(model.encoder.encoder_layers, start=1):
            for sublayer_index, sublayer in enumerate(layer.sublayers):
                raw_style = sublayer.adaLN_modulation(compact_style).float()
                chunks = raw_style.chunk(3, dim=-1)
                bounds = sublayer.adaln_modulation_bounds or (None, None, None)
                for name, values, bound in zip(names, chunks, bounds):
                    key = (layer_index, sublayer_names[sublayer_index], name, bound)
                    flattened = values.detach().cpu().reshape(-1)
                    layer_values.setdefault(key, []).append(flattened)
                    all_values[name].append(flattened)

    layer_rows = []
    for (layer, sublayer, name, bound), values in sorted(layer_values.items()):
        row = {
            "Layer": layer,
            "Sublayer": sublayer,
            "Parameter": name,
        }
        row.update(summarize_values(values, bound))
        layer_rows.append(row)

    first_sublayer = model.encoder.encoder_layers[0].sublayers[0]
    overall_bounds = first_sublayer.adaln_modulation_bounds or (None, None, None)
    overall_rows = []
    for name, bound in zip(names, overall_bounds):
        row = {"Parameter": name}
        row.update(summarize_values(all_values[name], bound))
        overall_rows.append(row)
    return pd.DataFrame(layer_rows), pd.DataFrame(overall_rows)


def main():
    """Run the modulation diagnostic and save layer-level and overall tables."""
    for path in (DATASET_PATH, MODEL_CONFIG_PATH, CHECKPOINT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input was not found: {path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TranslationDataset.from_h5(str(DATASET_PATH), lazy=True)
    cell_types, species, expression = load_cell_environments(dataset)
    model, metadata = load_model(device)
    layer_table, overall_table = collect_modulation_statistics(
        model, species, expression, device
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layer_path = OUTPUT_DIR / "adaln_modulation_by_layer.csv"
    summary_path = OUTPUT_DIR / "adaln_modulation_summary.csv"
    metadata_path = OUTPUT_DIR / "adaln_modulation_metadata.json"
    layer_table.to_csv(layer_path, index=False)
    overall_table.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": str(DATASET_PATH),
                "checkpoint": str(CHECKPOINT_PATH),
                "checkpoint_epoch": metadata.get("epoch"),
                "cell_environment_n": len(cell_types),
                "cell_types": cell_types,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Cell environments: {len(cell_types)}")
    print(overall_table.to_string(index=False))
    print(f"Layer statistics: {layer_path}")
    print(f"Overall statistics: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

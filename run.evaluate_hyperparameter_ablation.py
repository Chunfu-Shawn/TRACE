#!/usr/bin/env python3
"""Evaluate hyperparameter ablations on unseen cell environments.

Edit ``MODEL_SPECS`` to register trained checkpoints. The script reuses the
prediction and metric implementations in ``run.evaluate_architecture_ablation.py``
and writes one publication-friendly supplementary-table row per checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_architecture_evaluator():
    """Load the existing evaluator so metric definitions stay identical."""
    module_name = "_trace_architecture_ablation_evaluator"
    script_path = PROJECT_ROOT / "run.evaluate_architecture_ablation.py"
    module_spec = importlib.util.spec_from_file_location(module_name, script_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load evaluation helpers from {script_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


evaluator = load_architecture_evaluator()


# -----------------------------------------------------------------------------
# Paths and evaluation settings
# -----------------------------------------------------------------------------
DATASET_DIR = Path("/public-supool/home/annie/translation_model/dataset")
CHECKPOINT_DIR = Path(
    "/public-supool/home/annie/translation_model/checkpoint/train"
)
OUTPUT_DIR = Path(
    "/public-supool/home/annie/translation_model/results/ablation/"
    "hyperparameter_zero_shot"
)

TRAIN_REFERENCE_DATASET = (
    DATASET_DIR / "human_5c_6k_depth0.1_cov0.1_rpm1.train.h5"
)
TEST_DATASET_PATH = (
    DATASET_DIR
    / "human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.test.h5"
)

# Use the same validation-selected checkpoint type for every ablation.
CHECKPOINT_SUFFIX = ".best_total.pt"
BATCH_SIZE = 30
NUM_TEST_SAMPLES: Optional[int] = None
MIN_RPF_DEPTH: Optional[float] = 0.1
MIN_RNA_PER_CELL = 500
BOOTSTRAP_ITERATIONS = 2000
RANDOM_SEED = 42
REUSE_EXISTING_PREDICTIONS = True
REUSE_EXISTING_METRICS = True
FAIL_FAST = False


MODEL_CLASSES: Dict[str, Type[torch.nn.Module]] = {
    "BaseModel": evaluator.BaseModel,
    "BaseModelHybrid": evaluator.BaseModelHybrid,
    "BaseModelLN": evaluator.BaseModelLN,
    "BaseModelConv": evaluator.BaseModelConv,
}


@dataclass(frozen=True)
class HyperparameterSpec:
    """Model, training, and checkpoint metadata for one table row."""

    model_id: str
    label: str
    model_class: str
    config_path: Path
    checkpoint_glob: str
    head_hidden_dim: int
    macro_alpha_start: float
    macro_alpha_final: float
    ranking_beta: float
    training_environment_n: int
    training_strategy: str
    learning_rate: Optional[float] = None
    expression_mask_probability: Optional[float] = None
    expression_interpolation_probability: Optional[float] = None
    expression_noise_std: Optional[float] = None
    seed: Optional[int] = None
    requested_epochs: Optional[int] = None
    early_stopping_enabled: bool = True
    effective_batch_size: Optional[int] = None
    force_zero_expression: bool = False
    notes: str = ""
    enabled: bool = True
    checkpoint: Optional[Path] = None


# Duplicate an entry for every trained checkpoint. Fields such as alpha and beta
# are explicit because legacy checkpoints do not consistently store them.
MODEL_SPECS = (
    HyperparameterSpec(
        model_id="trace_384d_12l_alpha2_beta02",
        label="TRACE 384d 12L alpha=2 beta=0.2 lr=0.001",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*"
            "hs_5c*a2_b02_exp_aug*"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0.2,
        macro_alpha_final=2.0,
        ranking_beta=0.2,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.001,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=50,
        effective_batch_size=100,
        notes="Reference model",
    ),
    HyperparameterSpec(
        model_id="trace_384d_12l_alpha2_beta0",
        label="TRACE 384d 12L alpha=2 beta=0 lr=0.001",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*"
            "hs_5c*a2_b0_exp_aug*"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0.2,
        macro_alpha_final=2.0,
        ranking_beta=0.0,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.001,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=50,
        effective_batch_size=150,
        notes="Ranking-loss ablation",
    ),
    HyperparameterSpec(
        model_id="trace_384d_12l_alpha1_beta0",
        label="TRACE 384d 12L alpha=1 beta=0 lr=0.001",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*"
            "hs_5c*e25_a1_b0_exp_aug*"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0.2,
        macro_alpha_final=1.0,
        ranking_beta=0.0,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.001,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=25,
        effective_batch_size=150,
        notes="Ranking-loss, alpha ablation",
    ),
    HyperparameterSpec(
        model_id="trace_384d_12l_alpha0.5_beta0",
        label="TRACE 384d 12L alpha=0.5 beta=0 lr=0.001",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*"
            "hs_5c*e25_a05_b0_exp_aug*"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0.2,
        macro_alpha_final=0.5,
        ranking_beta=0.0,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.001,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=25,
        effective_batch_size=150,
        notes="Ranking-loss, alpha ablation",
    ),
    HyperparameterSpec(
        model_id="trace_384d_12l_alpha0_beta0",
        label="TRACE 384d 12L alpha=0 beta=0 lr=0.001",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*"
            "hs_5c*e25_a0_b0_exp_aug*"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0,
        macro_alpha_final=0,
        ranking_beta=0.0,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.001,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=25,
        effective_batch_size=150,
        notes="Ranking-loss, alpha ablation"
    ),
    HyperparameterSpec(
        model_id="trace_384d_12l_alpha1_beta0_lr0005",
        label="TRACE 384d 12L alpha=1 beta=0 lr=0.005",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*"
            "hs_5c_6k_depth0.1_cov0.1_rpm1_e25_a1_b0_exp_aug_i03_m15_lr_sweep*"
            "_0.005"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0.2,
        macro_alpha_final=1.0,
        ranking_beta=0.0,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.005,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=25,
        early_stopping_enabled=False,
        effective_batch_size=150,
        notes="Learning-rate ablation",
    ),
    HyperparameterSpec(
        model_id="trace_384d_12l_alpha1_beta0_lr00001",
        label="TRACE 384d 12L alpha=1 beta=0 lr=0.0001",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_384d_16h_12l_64env_16ad_bs*"
            "hs_5c_6k_depth0.1_cov0.1_rpm1_e25_a1_b0_exp_aug_i03_m15_lr_sweep*"
            "_0.0001"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0.2,
        macro_alpha_final=1.0,
        ranking_beta=0.0,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.0001,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=25,
        early_stopping_enabled=False,
        effective_batch_size=150,
        notes="Learning-rate ablation",
    ),
    HyperparameterSpec(
        model_id="trace_256d_12l_alpha1_beta0",
        label="TRACE 256d 12L alpha=1 beta=0",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_256d_16h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_256d_16h_12l_64env_16ad_bs*"
            "hs_5c_6k_depth0.1_cov0.1_rpm1_e25_a1_b0_exp_aug_i03_m15*"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0.2,
        macro_alpha_final=1.0,
        ranking_beta=0.0,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.001,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=25,
        early_stopping_enabled=False,
        effective_batch_size=150,
        notes="Model-width ablation",
    ),
    HyperparameterSpec(
        model_id="trace_256d_8h_12l_alpha1_beta0",
        label="TRACE 256d 8H 12L alpha=1 beta=0",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_256d_8h_12l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_256d_8h_12l_64env_16ad_bs*"
            "hs_5c_6k_depth0.1_cov0.1_rpm1_e25_a1_b0_exp_aug_i03_m15*"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0.2,
        macro_alpha_final=1.0,
        ranking_beta=0.0,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.001,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=25,
        early_stopping_enabled=False,
        effective_batch_size=150,
        notes="Attention-head ablation",
    ),
    HyperparameterSpec(
        model_id="trace_256d_6l_alpha1_beta0",
        label="TRACE 256d 6L alpha=1 beta=0",
        model_class="BaseModel",
        config_path=(
            SRC_DIR / "config/base_model_256d_16h_6l_64env_16ad_bs.yaml"
        ),
        checkpoint_glob=(
            "base_model_256d_16h_6l_64env_16ad_bs*"
            "hs_5c_6k_depth0.1_cov0.1_rpm1_e25_a1_b0_exp_aug_i03_m15*"
        ),
        head_hidden_dim=384,
        macro_alpha_start=0.2,
        macro_alpha_final=1.0,
        ranking_beta=0.0,
        training_environment_n=5,
        training_strategy="Mask + interpolation",
        learning_rate=0.001,
        expression_mask_probability=0.15,
        expression_interpolation_probability=0.3,
        expression_noise_std=0.1,
        seed=42,
        requested_epochs=25,
        early_stopping_enabled=False,
        effective_batch_size=150,
        notes="Model-width and depth ablation",
    )
    # Add alpha, depth, width, or other ablations here using the same fields.
    # Set checkpoint=Path("/exact/path/model.best_total.pt") when a glob is
    # ambiguous. Set enabled=False to keep a planned run out of the analysis.
)


def configure_evaluator() -> None:
    """Apply this script's paths and filters to the shared evaluator."""
    evaluator.CHECKPOINT_DIR = CHECKPOINT_DIR
    evaluator.OUTPUT_DIR = OUTPUT_DIR
    evaluator.TRAIN_REFERENCE_DATASET = TRAIN_REFERENCE_DATASET
    evaluator.TEST_DATASET_PATH = TEST_DATASET_PATH
    evaluator.CHECKPOINT_SUFFIX = CHECKPOINT_SUFFIX
    evaluator.BATCH_SIZE = BATCH_SIZE
    evaluator.NUM_TEST_SAMPLES = NUM_TEST_SAMPLES
    evaluator.MIN_RPF_DEPTH = MIN_RPF_DEPTH
    evaluator.MIN_RNA_PER_CELL = MIN_RNA_PER_CELL
    evaluator.BOOTSTRAP_ITERATIONS = BOOTSTRAP_ITERATIONS
    evaluator.RANDOM_SEED = RANDOM_SEED
    evaluator.REUSE_EXISTING_PREDICTIONS = REUSE_EXISTING_PREDICTIONS
    evaluator.REUSE_EXISTING_METRICS = REUSE_EXISTING_METRICS


def as_evaluator_spec(spec: HyperparameterSpec):
    """Convert table metadata into the evaluator's model specification."""
    if spec.model_class not in MODEL_CLASSES:
        choices = ", ".join(MODEL_CLASSES)
        raise ValueError(f"Unknown model class {spec.model_class!r}; use {choices}")
    return evaluator.ModelSpec(
        model_id=spec.model_id,
        label=spec.label,
        model_class=MODEL_CLASSES[spec.model_class],
        config_path=spec.config_path,
        checkpoint_glob=spec.checkpoint_glob,
        color="#4477AA",
        force_zero_expression=spec.force_zero_expression,
        enabled=spec.enabled,
        checkpoint=spec.checkpoint,
    )


def scalar_or_nan(value: object) -> float:
    """Convert scalar checkpoint metadata to a CSV-safe float."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return float("nan")
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def model_architecture_metadata(model: torch.nn.Module) -> Dict[str, object]:
    """Read architecture dimensions from the instantiated model."""
    constructor = dict(getattr(model, "_constructor_args", {}))
    number_of_layers = constructor.get(
        "number_of_layers", getattr(model, "number_of_layers", np.nan)
    )
    if isinstance(model, evaluator.BaseModelHybrid):
        number_of_adaln_layers = constructor.get(
            "number_of_adaln_layers",
            getattr(model, "pre_adaln_layers", np.nan),
        )
    elif isinstance(model, evaluator.BaseModel):
        number_of_adaln_layers = number_of_layers
    else:
        number_of_adaln_layers = 0
    bounds = constructor.get(
        "adaln_modulation_bounds",
        getattr(model, "adaln_modulation_bounds", None),
    )
    bounds = bounds if isinstance(bounds, dict) else {}
    return {
        "D_Model": constructor.get("d_model", getattr(model, "d_model", np.nan)),
        "Number_of_Layers": number_of_layers,
        "N_Heads": constructor.get("n_heads", getattr(model, "n_heads", np.nan)),
        "D_FF": constructor.get("d_ff", getattr(model, "d_ff", np.nan)),
        "Adaptive_Dim": constructor.get(
            "adaptive_dim", getattr(model, "adaptive_dim", np.nan)
        ),
        "Number_of_AdaLN_Layers": number_of_adaln_layers,
        "AdaLN_Gamma_Bound": bounds.get("gamma", np.nan),
        "AdaLN_Beta_Bound": bounds.get("beta", np.nan),
        "AdaLN_Alpha_Bound": bounds.get("alpha", np.nan),
        "Parameter_N": int(sum(parameter.numel() for parameter in model.parameters())),
        "Trainable_Parameter_N": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def experiment_metadata(spec: HyperparameterSpec) -> Dict[str, object]:
    """Return user-specified hyperparameters for the supplementary table."""
    return {
        "Model_ID": spec.model_id,
        "Model_Label": spec.label,
        "Model_Class": spec.model_class,
        "Head_Hidden_Dim": spec.head_hidden_dim,
        "Macro_Alpha_Start": spec.macro_alpha_start,
        "Macro_Alpha_Final": spec.macro_alpha_final,
        "Ranking_Loss_Enabled": spec.ranking_beta > 0,
        "Ranking_Beta": spec.ranking_beta,
        "Learning_Rate": spec.learning_rate,
        "Training_Environment_N": spec.training_environment_n,
        "Training_Strategy": spec.training_strategy,
        "Expression_Mask_Probability": spec.expression_mask_probability,
        "Expression_Interpolation_Probability": (
            spec.expression_interpolation_probability
        ),
        "Expression_Noise_STD": spec.expression_noise_std,
        "Force_Zero_Expression_at_Test": spec.force_zero_expression,
        "Seed": spec.seed,
        "Requested_Epochs": spec.requested_epochs,
        "Early_Stopping_Enabled": spec.early_stopping_enabled,
        "Effective_Batch_Size": spec.effective_batch_size,
        "Notes": spec.notes,
    }


def checkpoint_metadata(metadata: dict) -> Dict[str, object]:
    """Extract training-state fields stored by current and legacy trainers."""
    return {
        "Checkpoint_Epoch": metadata.get("epoch", np.nan),
        "Best_Validation_Loss": scalar_or_nan(metadata.get("best_val_loss")),
        "Best_Validation_Profile_Spearman": scalar_or_nan(
            metadata.get("best_profile_spearman")
        ),
        "Best_Validation_Scale_Spearman": scalar_or_nan(
            metadata.get("best_scale_spearman")
        ),
        "Checkpoint_Current_Alpha": scalar_or_nan(metadata.get("current_alpha")),
        "Checkpoint_Learning_Rate": scalar_or_nan(metadata.get("learning_rate")),
    }


SUPPLEMENTARY_COLUMNS = (
    "Model_ID",
    "Model_Label",
    "Model_Class",
    "D_Model",
    "Number_of_Layers",
    "N_Heads",
    "D_FF",
    "Adaptive_Dim",
    "Number_of_AdaLN_Layers",
    "AdaLN_Gamma_Bound",
    "AdaLN_Beta_Bound",
    "AdaLN_Alpha_Bound",
    "Head_Hidden_Dim",
    "Parameter_N",
    "Trainable_Parameter_N",
    "Training_Environment_N",
    "Training_Strategy",
    "Macro_Alpha_Start",
    "Macro_Alpha_Final",
    "Ranking_Loss_Enabled",
    "Ranking_Beta",
    "Learning_Rate",
    "Expression_Mask_Probability",
    "Expression_Interpolation_Probability",
    "Expression_Noise_STD",
    "Force_Zero_Expression_at_Test",
    "Seed",
    "Requested_Epochs",
    "Early_Stopping_Enabled",
    "Effective_Batch_Size",
    "Checkpoint_Selection",
    "Checkpoint_Epoch",
    "Best_Validation_Loss",
    "Best_Validation_Profile_Spearman",
    "Best_Validation_Scale_Spearman",
    "Checkpoint_Current_Alpha",
    "Checkpoint_Learning_Rate",
    "Test_Dataset",
    "Cell_Type_N",
    "Eligible_Cell_Type_N",
    "RNA_N",
    "RNA_Passing_Depth_N",
    "Min_RPF_Depth",
    "Min_RNA_Per_Cell",
    "Mean_RNA_Profile_Spearman",
    "Mean_RNA_Profile_Spearman_CI95_Low",
    "Mean_RNA_Profile_Spearman_CI95_High",
    "Mean_RNA_Profile_Spearman_Cell_N",
    "Mean_CDS_Profile_Spearman",
    "Mean_CDS_Profile_Spearman_CI95_Low",
    "Mean_CDS_Profile_Spearman_CI95_High",
    "Mean_CDS_Profile_Spearman_Cell_N",
    "CDS_Mean_Spearman",
    "CDS_Mean_Spearman_CI95_Low",
    "CDS_Mean_Spearman_CI95_High",
    "CDS_Mean_Spearman_Cell_N",
    "CDS_Mean_MAE",
    "CDS_Mean_MAE_CI95_Low",
    "CDS_Mean_MAE_CI95_High",
    "CDS_Mean_MAE_Cell_N",
    "Notes",
)


def main() -> None:
    """Evaluate all configured checkpoints and write supplementary CSV files."""
    configure_evaluator()
    for path in (TRAIN_REFERENCE_DATASET, TEST_DATASET_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = evaluator.TranslationDataset.from_h5(
        str(TRAIN_REFERENCE_DATASET), lazy=True
    )
    test_dataset = evaluator.TranslationDataset.from_h5(
        str(TEST_DATASET_PATH), lazy=True
    )
    test_cell_types = evaluator.verify_unseen_cell_types(
        train_dataset, test_dataset
    )
    print(
        f"Verified {len(test_cell_types)} unseen test cell types: "
        + ", ".join(test_cell_types)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    specs = [spec for spec in MODEL_SPECS if spec.enabled]
    evaluator_specs = [as_evaluator_spec(spec) for spec in specs]
    cell_tables = []
    model_metadata_rows = []
    manifest_rows = []

    for order, (spec, eval_spec) in enumerate(zip(specs, evaluator_specs), start=1):
        manifest_row: Dict[str, object] = {
            "Model_Order": order,
            "Model_ID": spec.model_id,
            "Model_Label": spec.label,
            "Config": str(spec.config_path),
            "Checkpoint": "",
            "Checkpoint_Match_N": 0,
            "Prediction_PKL": "",
            "Prediction_Reused": False,
            "RNA_Metrics_CSV": "",
            "Metrics_Reused": False,
            "Status": "pending",
            "Error": "",
        }
        try:
            checkpoint_path, match_count = evaluator.resolve_checkpoint(eval_spec)
            manifest_row["Checkpoint"] = str(checkpoint_path)
            manifest_row["Checkpoint_Match_N"] = match_count
            if match_count > 1:
                print(
                    f"[{spec.label}] {match_count} checkpoints matched; "
                    f"using newest: {checkpoint_path}"
                )

            evaluator.HEAD_HIDDEN_DIM = spec.head_hidden_dim
            model, stored_metadata = evaluator.load_model(
                eval_spec, checkpoint_path, device
            )
            metadata_row = {
                "Model_Order": order,
                **experiment_metadata(spec),
                **model_architecture_metadata(model),
                **checkpoint_metadata(stored_metadata),
                "Checkpoint_Selection": CHECKPOINT_SUFFIX.removeprefix("."),
                "Test_Dataset": TEST_DATASET_PATH.name,
            }
            prediction_path, prediction_reused = evaluator.predict_model(
                eval_spec, model, test_dataset, checkpoint_path
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            transcript_metrics, metric_path, metrics_reused = (
                evaluator.load_or_evaluate_metrics(
                    test_dataset, prediction_path, eval_spec
                )
            )
            cell_metrics = evaluator.summarize_cell_types(
                transcript_metrics, eval_spec, test_cell_types
            )
            cell_tables.append(cell_metrics)
            model_metadata_rows.append(metadata_row)
            manifest_row.update(
                {
                    "Prediction_PKL": str(prediction_path),
                    "Prediction_Reused": prediction_reused,
                    "RNA_Metrics_CSV": str(metric_path),
                    "Metrics_Reused": metrics_reused,
                    "Status": "complete",
                }
            )
        except Exception as error:
            manifest_row["Status"] = "failed"
            manifest_row["Error"] = f"{type(error).__name__}: {error}"
            print(f"[{spec.label}] failed: {manifest_row['Error']}")
            if FAIL_FAST:
                raise
        manifest_rows.append(manifest_row)

    manifest = pd.DataFrame.from_records(manifest_rows)
    manifest.to_csv(OUTPUT_DIR / "checkpoint_manifest.csv", index=False)
    if not cell_tables:
        raise RuntimeError("No model produced evaluable zero-shot predictions")

    cell_metrics = pd.concat(cell_tables, ignore_index=True)
    model_metrics = evaluator.summarize_models(cell_metrics, evaluator_specs)
    metadata = pd.DataFrame.from_records(model_metadata_rows)
    supplementary = metadata.merge(
        model_metrics, on=["Model_ID", "Model_Label"], how="inner"
    ).sort_values("Model_Order")
    supplementary = supplementary.drop(columns="Model_Order")
    supplementary = supplementary.reindex(columns=SUPPLEMENTARY_COLUMNS)

    cell_metrics.to_csv(
        OUTPUT_DIR / "zero_shot_metrics_by_cell_type.csv", index=False
    )
    supplementary_path = OUTPUT_DIR / "supplementary_table_zero_shot.csv"
    supplementary.to_csv(supplementary_path, index=False)

    print("\nZero-shot hyperparameter summary:")
    print(
        supplementary[
            [
                "Model_Label",
                "D_Model",
                "Number_of_Layers",
                "Macro_Alpha_Final",
                "Ranking_Beta",
                "Learning_Rate",
                "Mean_RNA_Profile_Spearman",
                "Mean_CDS_Profile_Spearman",
                "CDS_Mean_Spearman",
                "CDS_Mean_MAE",
            ]
        ].to_string(index=False)
    )
    print(f"\nSupplementary table saved to: {supplementary_path}")


if __name__ == "__main__":
    main()

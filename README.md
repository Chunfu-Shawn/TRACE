# TRACE: Translation Resolution Across Cell Environments

A Transformer-based model that decodes full-length transcriptomes into translatomes — predicting per-position ribosome density profiles purely from full-length RNA sequence and cellular context. TRACE integrates multi-omics data — transcript sequence, gene expression, and species identity — through adaptive layer normalization (AdaLN-Zero) to resolve translation regulation across cell types and species.

## Overview

TRACE takes as input:
- **RNA sequence** (one-hot encoded nucleotides)
- **Cellular transcriptome profile** (continuous expression vector, 16840 genes)
- **Species label** (discrete identifier for evolutionary context)

And decodes the full-length transcript into a translatome — the per-position ribosome density profile — enabling:
- Translation efficiency (TE) estimation
- Ribosome dynamics and pausing site identification
- Cross-species and cross-cell-type coding ORF prediction

## Model Architecture

Key architectural features:
- **AdaLN-Zero**: Each transformer sublayer is modulated by a compact style vector derived from the concatenated expression + species features, with a learned gating parameter initialized to zero for stable training.
- **Rotary Position Embedding (RoPE)**: Applied to query/key in self-attention, with NTK-aware scaling for long sequences.
- **Flash Attention**: Automatic dispatch to FlashAttention-2 when available, with graceful fallback to standard PyTorch attention.
- **Pluggable Heads**: Modular prediction heads (density, coding, decoupled shape/scale) that can be added/removed at runtime.

## Project Structure

```
TRACE/
├── environment.yml                        # Conda environment
├── src/
│   ├── model/
│   │   ├── base_model.py                  # BaseModel — sequence-only encoder
│   │   ├── translation_base_model.py      # TranslationBaseModel — encoder + RPF
│   │   ├── model_modules.py              # AdaLN-Zero encoder, LinearEmbedding, RoPE attention
│   │   ├── prediction_heads.py           # PsiteDensityHead, TranslationProfileHead, TERegressionHead
│   │   ├── position_embedding.py         # RoPE (LlamaRotaryEmbeddingExt)
│   │   ├── flash_multi_headed_attention.py # FlashAttention-2 wrapper
│   │   ├── transformer.py                # Core transformer components
│   │   ├── translation_predictor.py      # Inference utilities
│   │   ├── orf_caller.py                 # ORF identification
│   │   ├── eval_RPF_density_TIS_TTS.py   # TIS/TTS density evaluation
│   │   └── generate_cell_env_expr_array.py # Expression vector generation
│   ├── data/
│   │   ├── translation_dataset.py        # TranslationDataset — lazy H5 loader
│   │   ├── translation_dataset_generator.py # H5 dataset generation pipeline
│   │   ├── rpf_counter.py                # Ribo-seq read counting
│   │   ├── psite_counter.py              # P-site footprint extraction
│   │   ├── cell_env_expr_array_generate.py # Cell-type expression matrix
│   │   ├── transcript_sequence_generate.py # Transcript sequence encoding
│   │   ├── transcript_CDS_embedding.py   # CDS-aware embedding generation
│   │   ├── transcript_exon_index.py      # Exon boundary indexing
│   ├── train/
│   │   ├── seq_pretrain.py               # SeqPretrainTrainer (BaseModel pretraining)
│   │   ├── model_pretrain.py             # PretrainingTrainer (TranslationBaseModel)
│   │   ├── model_finetune_te.py          # TEFinetuneTrainer (TE regression)
│   │   ├── distributed_balanced_bucket_sampler.py # Length-bucketed DDP sampler
│   │   └── masking_adapter.py            # BERT-style masking + curriculum
│   ├── config/
│   │   ├── model_config.py               # ModelConfig dataclass (BaseModel)
│   │   ├── model_config_expr.py          # ModelConfig (TranslationBaseModel)
│   │   ├── *.yaml                        # Hyperparameter configs
│   │   └── model_config_*.py             # Ablation config dataclasses
│   ├── lora_utils.py                     # LoRA adapter injection helpers
│   └── utils.py                          # Shared utilities (unwrap_model, etc.)
├── test/
│   ├── run_test.py                       # Basic model forward test
│   └── test_transcripts.py               # Transcript loading test
└── tools/
    └── tumor_neoantigen/                 # Tumor neoantigen identification pipeline
```

## Setup

### Environment

Two options are provided — a full `environment.yml` (reproducible conda env)
and a minimal `requirements.txt` for pip-only installs.

```bash
# Option A: conda (full environment, includes Jupyter + dev tools)
conda env create -f environment.yml
conda activate TRACE

# Option B: pip (core dependencies only)
pip install -r requirements.txt
```

**Core dependencies** (required to train / run inference):

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | — |
| PyTorch | 2.6.0 (CUDA 12.4) | model definition + training |
| flash-attn | 2.8.0 | efficient attention (optional — falls back to PyTorch) |
| h5py | 3.14 | H5 dataset I/O |
| numpy | 2.3 | array operations |
| PyYAML | 6.0 | config file parsing |
| tqdm | 4.67 | progress bars |
| einops | 0.8 | tensor reshaping |
| loralib | 0.1 | LoRA adapter injection (fine-tuning) |
| ninja | 1.13 | JIT build helper (flash-attn) |

**Optional packages** (can be removed if you only need minimal inference):

`pyahocorasick` (motif matching)  `jupyter*` / `notebook*` / `ipython*` (dev environment)
`statsmodels` `scikit-misc` `networkx` `viennarna` `gffutils` `pyfaidx` `conda-pack`

### Hardware

- Training: Multi-GPU (tested on 4–8× A100/H100)
- Inference: Single GPU (>=16GB VRAM recommended for 384d model)

## Dataset

Pre-processed training/validation/test H5 datasets and expression dictionaries
are publicly available on Zenodo:

> **Zenodo**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21469176.svg)](https://doi.org/10.5281/zenodo.21469176)
>
> The archive contains `.train.h5` / `.valid.h5` / `.test.h5` files for each
> species (human, macaque, mouse) as well as per-species expression dictionaries
> (`{species}_expression_dict.pt`).

### H5 File Layout

```
<dataset>.h5
├── .attrs["n_samples"]          int           total number of transcripts
├── .attrs["cell_type_counts"]   JSON str      cell-type distribution
├── /cell_exprs/
│   └── <cell_type>              (d_expr,)     Z-scored expression vector per cell type
├── /sequences/
│   └── <tid>                    (L, d_seq)    continuous sequence features (e.g., one-hot codon)
└── /samples/<uuid>/
    ├── .attrs["species"]        str           human | macaque | mouse
    ├── .attrs["cell_type"]      str           e.g., "heart", "liver", "HepG2"
    ├── .attrs["cds_start_pos"]  int16         CDS start (1-based), -1 if unknown
    ├── .attrs["cds_end_pos"]    int16         CDS end (1-based), -1 if unknown
    ├── .attrs["te_scale"]       float32       translation efficiency (Z-scored), None if missing
    ├── .attrs["rpf_depth"]      float32       ribosome profiling depth
    ├── .attrs["rpf_coverage"]   float32       ribosome profiling coverage
    ├── .attrs["motif_occ"]      list[int]     upstream motif occurrences
    └── count_emb                (L, d_count)  per-position RPF density (target)
```

### Loading Data

The `TranslationDataset` class provides lazy (recommended) or eager loading
from `.h5` files:

```python
from data.translation_dataset import TranslationDataset

# Lazy loading — minimal RAM, reads on-demand per __getitem__
ds = TranslationDataset.from_h5("human.train.h5", lazy=True)

print(f"Samples: {ds.n_samples}")
print(f"Cell types: {ds.cell_type_counts}")
print(f"Cell expr dict keys: {list(ds.cell_expr_dict.keys())}")

# Access a single sample (returns one transcript)
tid, species, cell_type, expr_vector, meta, seq_emb, count_emb = ds[0]

print(f"species: {species}")           # e.g., "human"
print(f"cell_type: {cell_type}")       # e.g., "heart"
print(f"seq_emb shape: {seq_emb.shape}")     # (L, d_seq)
print(f"count_emb shape: {count_emb.shape}") # (L, d_count)
print(f"cds: {meta['cds_start_pos']}–{meta['cds_end_pos']}")
print(f"te_scale: {meta.get('te_scale')}")

# Expression vector for a specific cell type
expr_vec = ds.cell_expr_dict["heart"]  # shape (d_expr,)
```

## Training

### Pretraining

Pretraining uses `torchrun` for distributed multi-GPU training.  The entry
point script below creates a `BaseModel`, attaches a `PsiteDensityHead`,
wraps it with DDP, and launches the `SeqPretrainTrainer`.

```python
import sys
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead
from train.seq_pretrain import SeqPretrainTrainer
from utils import print_param_counts

rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world_size)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# ============================================================
# dataset paths — edit these to point at your .h5 files
# ============================================================
dataset_dir = "/path/to/your/dataset/"

human_train_dataset_path = os.path.join(dataset_dir, "human.train.h5")
human_val_dataset_path   = os.path.join(dataset_dir, "human.valid.h5")

# To add more species (macaque, mouse, ...), append extra .h5 paths
# to the lists below:
train_paths = [human_train_dataset_path]
val_paths   = [human_val_dataset_path]

# ============================================================
# config & checkpoint paths — edit these
# ============================================================
config_path = os.path.join(os.path.dirname(__file__),
                           "src/config/base_model_384d_16h_12l_64env_16ad.yaml")
checkpoint_dir = "./checkpoints/seq_pretrain"
log_dir        = "./logs/seq_pretrain"

# ============================================================
# model
# ============================================================
base_model = BaseModel.from_config(config_path).cuda(rank)

base_model.add_head(
    "count",
    PsiteDensityHead.create_from_model(base_model, d_pred_h=384),
    overwrite=True,
)
print(base_model.model_name)
print(base_model.list_heads())
print_param_counts(base_model)

# DDP
base_model = DDP(base_model, device_ids=[rank], output_device=rank)

# ============================================================
# trainer
# ============================================================
trainer = SeqPretrainTrainer(
    model=base_model,
    dataset_paths=train_paths,
    val_dataset_paths=val_paths,
    dataset_name="seq_pretrain",
    batch_size=50,
    checkpoint_dir=checkpoint_dir,
    log_dir=log_dir,
    world_size=world_size,
    rank=rank,
    resume=True,
    save_every=1,
    epoch_num=60,
    mask_perc={"species": 0.1, "cell": 0.1},
    expr_noise_std=0.1,
    learning_rate=0.001,
    lr_warmup_perc=0.3,
    accumulation_steps=1,
    balance_classes=True,
    beta=(0.9, 0.98),
    epsilon=1e-9,
    weight_decay=0.01,
)
trainer.pretrain()

dist.destroy_process_group()
```

**Key hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epoch_num` | 60 | Total training epochs |
| `batch_size` | 50 | Per-GPU batch size |
| `learning_rate` | 1e-3 | Peak LR (cosine decay with warmup) |
| `lr_warmup_perc` | 0.3 | Fraction of steps for linear warmup |
| `accumulation_steps` | 1 | Gradient accumulation steps |
| `mask_perc` | `{"species": 0.1, "cell": 0.1}` | Randomly mask species/cell-type context |
| `expr_noise_std` | 0.1 | Gaussian noise injected into expression vectors |
| `weight_decay` | 0.01 | AdamW weight decay |
| `balance_classes` | True | Bucket-sampler balances cell types |

The trainer callbacks include early stopping (patience=8), periodic checkpoint
saving, and JSON logging of per-epoch losses.

## Inference

TRACE predicts the translatome purely from transcriptome — only RNA sequence and cellular context are needed.

```python
from model.translation_base_model import TranslationBaseModel
from model.mask_heads import TranslationProfileHead

# Load model
model = TranslationBaseModel.from_config("config.yaml")
model.add_head("count", TranslationProfileHead.create_from_model(model))
model.load_pretrained_weights("checkpoint.pt")

# Predict — decode transcriptome into translatome
result = model.predict(
    seq_batch=seq_array,        # (seq_len, 4) or (bs, seq_len, 4)
    count_batch=None,           # Not needed at deployment (auto-filled with zeros)
    species="human",
    cell_type="heart",          # or expr_vector=torch.Tensor
    head_names=["count"]
)
```

## Quick Start / Test Case

The following example demonstrates how to run a full inference pipeline: load a pretrained model and expression dictionary, provide a transcriptome FASTA file, and predict per-position ribosome density profiles.

### Data Structure

The pre-built expression dictionaries in `config/{species}_expression_dict.pt` have the following structure:

```
{
    "cell_type_name_1": torch.Tensor(shape=(d_expr=16839,), dtype=float16),   # Z-scored expression values
    "cell_type_name_2": torch.Tensor(shape=(d_expr=16839,), dtype=float16),   # aligned to global_anchor_gene_order.txt
    ...
}
```

Each tensor is a dense Z-score vector following the global anchor gene order (`config/global_anchor_gene_order.txt`), where the position in the tensor corresponds to a specific ortholog anchor gene. The model loads these via `model.load_expression_dict()`.

### Test Case: Predict Translation Profiles from FASTA

```python
import torch
from model.translation_base_model import TranslationBaseModel
from model.mask_heads import TranslationProfileHead
from model.translation_predictor import TranslationProfilePredictor

# ==========================================
# Step 1: Load Model
# ==========================================
# Load base model from a YAML config (adjust path as needed)
base_model = TranslationBaseModel.from_config(
    "config/base_model_expr_384d_16h_12l_128env_32ad.yaml"
)
base_model.add_head(
    "count",
    TranslationProfileHead.create_from_model(base_model, d_pred_h=384),
    overwrite=True,
)
base_model.load_pretrained_weights("/path/to/pretrained_checkpoint.pt")

# ==========================================
# Step 2: Load Cell Environment Expression Vectors
# ==========================================
species = "human"  # or "macaque", "mouse"
expr_dict_path = f"config/{species}_expression_dict.pt"
expr_dict = torch.load(expr_dict_path, map_location="cpu")

# Register expression profiles into the model
base_model.load_expression_dict(expr_dict)

print(f"Loaded {len(base_model.cell_expr_dict)} cell types for {species}.")
# Example: base_model.cell_expr_dict keys might include
# "heart", "liver", "brain", "HepG2", "K562", etc.

# ==========================================
# Step 3: Prepare FASTA Input
# ==========================================
# Provide one or more FASTA files containing transcript sequences
fasta_files = ["/path/to/transcriptome.fasta"]

# Optional: filter to specific transcript IDs (e.g., from RNA-seq TPM analysis)
target_tids = ["ENST00000335137", "ENST00000448941"]  # or load from get_active_transcripts()

# ==========================================
# Step 4: Initialize Predictor and Run
# ==========================================
predictor = TranslationProfilePredictor(
    model=base_model,
    fasta_files=fasta_files,
)

# Select a cell type to predict in
cell_type = "heart"  # must be a key in expr_dict

# Get the expression vector for this cell type
cell_expr_vector = base_model.cell_expr_dict[cell_type].numpy()

# Run prediction
output_path = predictor.run(
    species=species,
    cell_type=cell_type,
    cell_expr_vector=cell_expr_vector,
    target_tids=target_tids,      # optional: predict only specific transcripts
    out_dir="./results",
    suffix="heart_test",
    min_len=200,
    max_len=20000,
    batch_size=32,
)

print(f"Predictions saved to: {output_path}")
```

### Output

The prediction is saved as a pickle (`.pkl`) file containing a dictionary:

```python
{
    "cell_type_name": {
        "ENST00000335137": np.ndarray(shape=(seq_len, 1), dtype=float16),  # per-position RPF density
        "ENST00000448941": np.ndarray(shape=(seq_len, 1), dtype=float16),  # per-position RPF density
        ...
    }
}
```

Each entry maps a transcript ID → a 1D per-nucleotide ribosome density profile (float16) of the same length as the transcript sequence.


## Citation
If you use this code, please cite:

```bibtex
@article{trace2025,
  title={TRACE: Translation Resolution Across Cell Environments},
  author={Xiao, Chunfu},
  year={2025}
}
```

## License

This project is licensed for academic research use. Contact the author for commercial licensing.

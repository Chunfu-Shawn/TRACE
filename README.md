# TRACE: Translation Resolution Across Cell Environments

## Overview

TRACE uses the sequence-only `BaseModel` in `src/model/base_model.py` to predict a
full-length, per-nucleotide ribosome-density profile. The model inputs are:

- RNA sequence features `(B, L, 4)`;
- a 16,840-gene expression vector aligned to the human anchor-gene order;
- a species embedding;

The encoder combines RoPE self-attention with AdaLN-Zero cellular conditioning. A
registered translation-profile head converts the encoder output into a non-negative
ribosome-density prediction `(B, L, 1)`. RPF density is used only as the training
target and is not an input to `BaseModel`.

FlashAttention is optional. If the package, compatible CUDA device, or supported
dtype is unavailable, TRACE automatically uses PyTorch self-attention. CPU inference
does not require `flash-attn`.

## Project structure

The project tree is useful for locating the public entry points, but new users only
need the files listed below. Historical and experimental implementations are omitted
from this overview.

```text
TRACE/
├── run.train_seq.py
│   Editable Python launcher for sequence-only model training.
├── environment.yml / requirements.txt
│   Conda and pip dependency definitions.
├── src/model/
│   ├── base_model.py
│   │   Sequence-only BaseModel and its inference/checkpoint API.
│   ├── prediction_heads.py
│   │   Pluggable per-position translation-profile prediction heads.
│   ├── model_modules.py
│   │   Embeddings, AdaLN-Zero encoder layers, and standard self-attention.
│   ├── flash_multi_headed_attention.py
│   │   Optional FlashAttention implementation with runtime fallback.
│   ├── translation_predictor.py
│   │   Batched FASTA-to-ribosome-profile inference utilities.
│   ├── position_embedding.py
│   │   Rotary positional embedding implementation.
│   └── generate_cell_env_expr_array.py
│       Converts featureCounts output into aligned expression vectors.
├── src/train/
│   ├── model_trainer_seq.py
│   │   Sequence-only training loop, losses, logging, and checkpoint resume.
│   └── distributed_balanced_bucket_sampler.py
│       Length-aware and optionally class-balanced distributed sampler.
├── src/data/
│   ├── translation_dataset.py
│   │   Lazy/eager HDF5 dataset reader used by training.
│   ├── translation_dataset_generator.py
│   │   Builds training HDF5 files from processed sequence and RPF data.
│   ├── rpf_counter.py
│   │   Counts compatible Ribo-seq reads on transcript coordinates.
│   └── transcript_exon_index.py
│       Transcript/exon coordinate conversion and indexing utilities.
├── src/config/
│   Model YAML, expression dictionaries, human anchor order, and species ID map.
└── test/run_test.py
    CPU smoke test for BaseModel profile prediction.
```

## Installation

Python 3.11 and PyTorch 2.6 are the reference versions.

```bash
git clone <repository-url>
cd TRACE

# Conda
conda env create -f environment.yml
conda activate TRACE

# Or pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# The repository is not packaged as a wheel yet.
export PYTHONPATH="$PWD/src:${PYTHONPATH}"
```

### Reference GPU training environment

The model-training environment used by the launchers is pinned to the following
reference combination. `flash-attn` is compiled against the installed PyTorch and
CUDA toolkit, so changing PyTorch or CUDA may require reinstalling it.

| Component | Reference version | Purpose |
|---|---:|---|
| Python | 3.11.13 | Runtime |
| PyTorch | 2.6.0 with CUDA 12.4 | Model training and CUDA kernels |
| CUDA toolkit | 12.4, including `nvcc` | Compiles FlashAttention CUDA extensions |
| `flash-attn` | 2.8.0.post2 | Memory-efficient attention used during GPU training |
| `ninja` | 1.13.0 | Parallel extension build system |
| `packaging` | 25.0 | Build-time version parsing |
| `psutil` | 5.9.0 | Build-time CPU and memory detection |

On Linux with a CUDA 12.4-compatible NVIDIA driver, install the CUDA-enabled
PyTorch wheel and the pinned FlashAttention build dependencies as follows:

```bash
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  ninja==1.13.0 packaging==25.0 psutil==5.9.0
MAX_JOBS=4 python -m pip install \
  flash-attn==2.8.0.post2 --no-build-isolation
```

`MAX_JOBS=4` limits compilation parallelism and can be adjusted for the available
host RAM and CPU cores. Verify the installed runtime before training:

```bash
python - <<'PY'
from importlib.metadata import version

import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("flash-attn:", version("flash-attn"))
print("ninja:", version("ninja"))
print("packaging:", version("packaging"))
print("psutil:", version("psutil"))
PY
```

The pinned versions above define the reproducible reference environment, rather
than the only compatible versions. FlashAttention remains optional: if it is not
installed or cannot be used for the current device and dtype, TRACE falls back to
standard PyTorch self-attention. It is not needed for CPU inference.

## Hardware

The following estimates apply to inference with the default 384-dimensional,
16-head, 12-layer model, `batch_size=1`, and FP16 autocast:

| RNA length | Standard attention | FlashAttention |
|---:|---:|---:|
| 2,000 nt | approximately 1–2 GB | approximately 1–1.5 GB |
| 6,000 nt | approximately 3–4 GB | approximately 1.2–2 GB |
| 10,000 nt | approximately 7–9 GB | approximately 1.5–2.5 GB |

A GPU with at least 16 GB is therefore recommended for standard attention and
generally provides sufficient margin for transcripts up to approximately 10,000 nt.
A 4 GB GPU is generally sufficient for FlashAttention inference with `batch_size=1`.
Actual usage varies with transcript length, CUDA context, allocator fragmentation,
and other processes using the GPU.

FlashAttention is activated only on CUDA with FP16 or BF16 inputs. Standard-attention
FP32 inference can require almost twice the attention memory shown above. CPU
inference is supported but is substantially slower.

## Download data and checkpoints

Preprocessed HDF5 datasets and expression dictionaries are available from
[Zenodo (10.5281/zenodo.21469176)](https://doi.org/10.5281/zenodo.21469176).

Place the supplied expression dictionaries under `src/config/`, or pass their actual
paths explicitly. All repository paths in the examples below are relative to the
repository root.

Important configuration files are:

```text
src/config/base_model_384d_16h_12l_64env_16ad.yaml
src/config/global_anchor_gene_order.txt
src/config/global_species_id_mapping.json
src/config/human_expression_dict.pt
src/config/macaque_expression_dict.pt
src/config/mouse_expression_dict.pt
```

## HDF5 dataset contract

`TranslationDataset.from_h5(path, lazy=True)` expects the following structure:

```text
<dataset>.h5
├── .attrs["n_samples"]          int           total number of transcripts
├── .attrs["cell_type_counts"]   JSON str      cell-type distribution
├── /cell_exprs/
│   └── <cell_type>              (d_expr,)     Z-scored expression vector per cell type
├── /sequences/
│   └── <tid>                    (L, d_seq)    sequence features (e.g., one-hot nucleotides)
└── /samples/<uuid>/
    ├── .attrs["tid"]            str           key into the sequences group
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

`count_emb` is the prediction target during training. It is loaded by the trainer
but is never passed into the sequence-only BaseModel.

```python
from data.translation_dataset import TranslationDataset

dataset = TranslationDataset.from_h5("/data/human.train.h5", lazy=True)
uuid, species, cell_type, expression, metadata, sequence, count = dataset[0]

print(uuid, species, cell_type)
print(sequence.shape)    # (L, 4), model input
print(expression.shape)  # (16840,), model input
print(count.shape)       # (L, 1), training target
```

## Quick inference

The following example runs on CPU, Apple MPS, or CUDA. Register the same count head
used by the checkpoint before loading its weights.

```python
import contextlib
import torch

from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = BaseModel.from_config(
    "src/config/base_model_384d_16h_12l_64env_16ad.yaml"
)
model.add_head(
    "count",
    PsiteDensityHead.create_from_model(model, d_pred_h=384),
)
model.load_pretrained_weights(
    "/path/to/pretrained.latest.pt",
    strict=False,
    map_location="cpu",
)
model.load_expression_dict(
    torch.load("src/config/human_expression_dict.pt", map_location="cpu")
)
model.to(device)

def inference_context():
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()

with inference_context():
    prediction = model.predict(
        seq_batch="AUGCCGAUGCAG",
        species="human",
        cell_type="HepG2",
        head_names=["count"],
    )

profile = prediction["count"]
print(profile.shape)
```

`predict()` accepts one RNA string, a list of strings, an `(L, 4)` array, or a
batched `(B, L, 4)` tensor. A direct expression vector can replace `cell_type`:

```python
expression = model.cell_expr_dict["HepG2"]
with inference_context():
    prediction = model.predict(
        seq_batch="AUGCCGAUGCAG",
        species="human",
        expr_vector=expression,
        head_names=["count"],
    )
```

For transcriptome FASTA files, use
`model.translation_predictor.TranslationProfilePredictor`. It uses the same model
interface and automatically disables CUDA autocast on CPU.

### Test Case: Predict Translation Profiles from FASTA

After loading `model` as shown above, provide one or more transcript FASTA files and
select a cell type from the registered expression dictionary:

```python
from model.translation_predictor import TranslationProfilePredictor

species = "human"
cell_type = "liver"
fasta_files = ["/path/to/transcriptome.fasta"]

predictor = TranslationProfilePredictor(
    model=model,
    fasta_files=fasta_files,
)

cell_expression = model.cell_expr_dict[cell_type].cpu().numpy()
output_path = predictor.run(
    species=species,
    cell_type=cell_type,
    cell_expr_vector=cell_expression,
    target_tids=None,
    out_dir="results",
    suffix="liver",
    min_len=200,
    max_len=10000,
    batch_size=1,
)

print(f"Predictions saved to: {output_path}")
```

The output pickle has the following structure:

```python
{
    "liver": {
        "ENST00000335137": profile_array,  # shape: (sequence_length, 1)
        "ENST00000448941": profile_array,
    }
}
```

## Build an expression dictionary from featureCounts

`generate_cell_env_expr_array.py` recognizes gene IDs in the supplied cross-species
mapping automatically. The user does not specify a species. Human, macaque, and
mouse Ensembl gene IDs, with or without version suffixes, are mapped to the human
anchor order in `src/config/global_anchor_gene_order.txt`.

For gene-level featureCounts output:

```bash
python src/model/generate_cell_env_expr_array.py \
  --counts_file /path/to/featureCounts.txt \
  --ref_order src/config/global_anchor_gene_order.txt \
  --mapping_json src/config/global_species_id_mapping.json \
  --quant_level gene \
  --output_pt expression_dict.pt
```

`--quant_level gene` is the default and may be omitted. For transcript-level input,
provide a two-column transcript-to-gene table:

```bash
python src/model/generate_cell_env_expr_array.py \
  --counts_file /path/to/transcript_featureCounts.txt \
  --ref_order src/config/global_anchor_gene_order.txt \
  --mapping_json src/config/global_species_id_mapping.json \
  --quant_level transcript \
  --tx2gene /path/to/tx2gene.tsv \
  --output_pt expression_dict.pt
```

The output is `{sample_name: tensor}`. Every tensor has the same length and order as
the human anchor-gene list. The command prints the detected ID namespaces and mapping
coverage; very low coverage usually means the wrong ID type or a heavily filtered
count matrix was supplied.

## Model training

Edit the experiment configuration at the top of `run.train_seq.py`, including
dataset paths, output directories, batch size, learning rate, and epoch count. The
essential Python workflow is:

```python
import torch

from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead
from train.model_trainer_seq import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_paths = [
    "/path/to/dataset/human_tissue_22c_6k_depth0.1_cov0.1_rpm1.train.h5",
    "/path/to/dataset/human_cell_line_18c_6k_depth0.1_cov0.1_rpm1.train.h5",
    "/path/to/dataset/human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.train.h5",
    "/path/to/dataset/macaque_4c_6k_depth0.1_cov0.1_rpm1.train.h5",
    "/path/to/dataset/mouse_3c_6k_depth0.1_cov0.1_rpm1.train.h5",
]
valid_paths = [
    "/path/to/dataset/human_tissue_22c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "/path/to/dataset/human_cell_line_18c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "/path/to/dataset/human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "/path/to/dataset/macaque_4c_6k_depth0.1_cov0.1_rpm1.valid.h5",
    "/path/to/dataset/mouse_3c_6k_depth0.1_cov0.1_rpm1.valid.h5",
]

model = BaseModel.from_config(
    "src/config/base_model_384d_16h_12l_64env_16ad.yaml"
)
model.add_head(
    "count",
    PsiteDensityHead.create_from_model(model, d_pred_h=384),
    overwrite=True,
)
model.to(device)

trainer = Trainer(
    model=model,
    dataset_paths=train_paths,
    val_dataset_paths=valid_paths,
    dataset_name="hs_22c_18c_26c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1",
    batch_size=50,
    checkpoint_dir="checkpoint/train_seq",
    log_dir="log/train_seq",
    world_size=1,
    rank=0,
    resume=True,
    save_every=1,
    epoch_num=60,
    patience=8,
    mask_perc={"species": 0.1, "cell": 0.1},
    expr_noise_std=0.1,
    learning_rate=1e-3,
    lr_warmup_perc=0.3,
    accumulation_steps=1,
    balance_classes=True,
    beta=(0.9, 0.98),
    epsilon=1e-9,
    weight_decay=0.01,
)
trainer.fit()
```

The provided launcher supports one GPU or multiple GPUs on one machine. Multi-node
training is intentionally not supported. After editing its configuration section,
run it directly on one GPU:

```bash
python run.train_seq.py
```

For multiple GPUs on one machine, use `torchrun` without additional command-line
arguments:

```bash
torchrun --standalone --nproc_per_node=4 run.train_seq.py
```

The trainer prints and records the decomposed count loss:

- `micro`: token-level profile loss;
- `macro`: frame-aware CDS mean loss;
- `ranking`: pairwise CDS-level ranking loss;
- training uses an alpha curriculum from `0.2` to `4.0` during learning-rate
  warmup, while validation always uses
  `total = micro + 4 * macro + 0.2 * ranking`.

Early stopping monitoring starts only after learning-rate warmup unless the launcher
sets a later epoch explicitly. Each validation epoch reports:

- `profile_spearman`: the mean position-wise density Spearman correlation across
  all RNAs with non-constant targets; a constant prediction for an evaluable RNA
  is scored as zero rather than omitted;
- `scale_spearman`: the global Spearman correlation between predicted and target
  CDS mean density across the validation set;
- `cds_mean_mae`: the absolute CDS mean density calibration error;
- `calibration_slope` and `calibration_intercept`: the least-squares calibration
  relation `target = intercept + slope * prediction`, whose ideal values are `1`
  and `0`, respectively.

In addition to `<run>.latest.pt`, the trainer independently maintains
`<run>.best_total.pt`, `<run>.best_profile.pt`, and `<run>.best_scale.pt`.

Checkpoints use the keys `model`, `optimizer`, `scheduler`, and `scaler`. Resume loads
the model using the current unwrapped model's map location.

## Troubleshooting

- `ModuleNotFoundError: model`: run from the repository root and export
  `PYTHONPATH="$PWD/src:${PYTHONPATH}"`.
- CPU inference tries to import FlashAttention: make sure `flash-attn` is not required
  by a custom environment; it is optional in the supplied dependency files.
- Expression vector dimension error: the model config and expression dictionary must
  use the same `d_expr`, normally 16,840.
- Near-zero expression mapping coverage: inspect the cleaned input IDs and use gene
  IDs, not transcript IDs, unless `--quant_level transcript --tx2gene ...` is set.

## Citation

```bibtex
@article{trace2026,
  title={TRACE: Translation Resolution Across Cell Environments},
  author={Xiao, Chunfu},
  year={2026}
}
```

## License

This project is licensed for academic research use. Contact the author for commercial
licensing.

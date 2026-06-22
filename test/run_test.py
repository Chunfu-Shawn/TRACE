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
fasta_files = ["./gencode.v43.pc_transcripts.test_2000.fa"]

# Optional: filter to specific transcript IDs (e.g., from RNA-seq TPM analysis)
target_tids = None # or load from get_active_transcripts()

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
    out_dir="./",
    suffix="heart_test",
    min_len=200,
    max_len=20000,
    batch_size=32,
)

print(f"Predictions saved to: {output_path}")
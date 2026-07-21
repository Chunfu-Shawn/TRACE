#!/bin/bash
# Build cross-species expression dictionaries from featureCounts output.
# Edit the paths below for your environment.

set -e

BASE_DIR="/home/user/data3/rbase"
LIB_DIR="${BASE_DIR}/translation_model/models/lib"
CONFIG_DIR="${BASE_DIR}/translation_model/models/src/config"
ORTHOLOG_CSV="${BASE_DIR}/genome_ref/Homolog/human_macaque_mouse_orthologs.tsv"
HUMAN_COUNTS="/home/user/data3/yaoc/translation_model/rna-seq/matched_counts/matched_samples_RNA-seq.txt"
MACAQUE_COUNTS="/home/user/data3/yaoc/translation_model/rna-seq/counts_gene/macaque_featureCounts.txt"
MOUSE_COUNTS="/home/user/data3/yaoc/translation_model/rna-seq/counts_gene/mouse_featureCounts.txt"

SCRIPT="src/data/cell_env_expr_array_generate.py"
REF_ORDER="${CONFIG_DIR}/global_anchor_gene_order.txt"
REF_MAP="${CONFIG_DIR}/global_species_id_mapping.json"

# ============================================================
# Phase 1 — Human: establish global reference coordinates
# ============================================================
echo "========== Phase 1: Establishing Human Reference Coordinates =========="
python "${SCRIPT}" \
    --counts      "${HUMAN_COUNTS}" \
    --ortholog    "${ORTHOLOG_CSV}" \
    --output_tpm  "${LIB_DIR}/human_expression_tpm.csv" \
    --output_pt   "${CONFIG_DIR}/human_expression_dict.pt" \
    --output_order_txt     "${REF_ORDER}" \
    --output_mapping_json  "${REF_MAP}" \
    --min_tpm 0

# ============================================================
# Phase 2 — Macaque: align to human reference
# ============================================================
echo ""
echo "========== Phase 2: Aligning Macaque Data =========="
python "${SCRIPT}" \
    --counts      "${MACAQUE_COUNTS}" \
    --ortholog    "${ORTHOLOG_CSV}" \
    --output_tpm  "${LIB_DIR}/macaque_expression_tpm.csv" \
    --output_pt   "${CONFIG_DIR}/macaque_expression_dict.pt" \
    --reference_order "${REF_ORDER}" \
    --min_tpm 0

# ============================================================
# Phase 3 — Mouse: align to human reference
# ============================================================
echo ""
echo "========== Phase 3: Aligning Mouse Data =========="
python "${SCRIPT}" \
    --counts      "${MOUSE_COUNTS}" \
    --ortholog    "${ORTHOLOG_CSV}" \
    --output_tpm  "${LIB_DIR}/mouse_expression_tpm.csv" \
    --output_pt   "${CONFIG_DIR}/mouse_expression_dict.pt" \
    --reference_order "${REF_ORDER}" \
    --min_tpm 0

echo ""
echo "Done. Expression dictionaries saved to ${CONFIG_DIR}/"
echo "  human_expression_dict.pt, macaque_expression_dict.pt, mouse_expression_dict.pt"

#!/bin/bash
# Build optimized genome indices from GTF annotations.
# Edit the paths below for your environment.

BASE_DIR="/home/user/data3/rbase"
LIB_DIR="${BASE_DIR}/translation_model/models/lib"

# ---- human (GENCODE v48) ----
python src/data/transcript_exon_index.py \
    --gtf        "${BASE_DIR}/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf" \
    --tmp_db     "temp.db" \
    --out_prefix "${LIB_DIR}/genome_index"

# ---- human in-house (PacBio custom GTF) ----
python src/data/transcript_exon_index.py \
    --gtf        "${BASE_DIR}/lit/project/sORFs/08-Iso-seq-20250717/results/custom.gtf.with_orf.gtf" \
    --tmp_db     "temp.inhouse.db" \
    --out_prefix "${LIB_DIR}/genome_index.inhouse"

# ---- macaque (rheMac10, Ensembl) ----
python src/data/transcript_exon_index.py \
    --gtf        "${BASE_DIR}/genome_ref/Rhesus_macaque/rheMac10/rheMac10.ensGene.gtf" \
    --tmp_db     "temp.macaque.db" \
    --out_prefix "${LIB_DIR}/genome_index.macaque"

# ---- mouse (GENCODE vM32) ----
python src/data/transcript_exon_index.py \
    --gtf        "${BASE_DIR}/genome_ref/Mus_musculus/gencode.vM32.annotation.gtf" \
    --tmp_db     "temp.mouse.db" \
    --out_prefix "${LIB_DIR}/genome_index.mouse"

echo "Done. Genome indices saved."

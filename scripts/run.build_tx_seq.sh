#!/bin/bash
# Build transcript sequence pickles from FASTA + metadata.
# Edit the paths below for your environment.

BASE_DIR="/home/user/data3/rbase"
LIB_DIR="${BASE_DIR}/translation_model/models/lib"

# ---- human (GENCODE v48) ----
python src/data/transcript_sequence_generate.py \
    --tx_meta "${LIB_DIR}/transcript_meta.pkl" \
    --fasta   "${BASE_DIR}/genome_ref/Homo_sapiens/hg38/fasta/transcripts/gencode.v48.transcripts.fa" \
    --output  "${LIB_DIR}/tx_seq.v48.pkl" \
    --id_sep "|" --id_field 0

# ---- macaque (Mmul_10, Ensembl v101) ----
python src/data/transcript_sequence_generate.py \
    --tx_meta "${LIB_DIR}/transcript_meta.macaque.pkl" \
    --fasta   "${BASE_DIR}/genome_ref/Rhesus_macaque/rheMac10/fasta/Macaca_mulatta.Mmul_10.transcripts.fa" \
    --output  "${LIB_DIR}/tx_seq.Mmul_10.v101.pkl" \
    --id_sep " " --id_field 0

# ---- mouse (GENCODE vM32) ----
python src/data/transcript_sequence_generate.py \
    --tx_meta "${LIB_DIR}/transcript_meta.mouse.pkl" \
    --fasta   "${BASE_DIR}/genome_ref/Mus_musculus/gencode.vM32.transcripts.fa" \
    --output  "${LIB_DIR}/tx_seq.mouse.vM32.pkl" \
    --id_sep "|" --id_field 0

echo "Done. Transcript sequence pickles saved."

#!/bin/bash
set -euo pipefail

source /home/user/data2/rbase/env/anaconda3/etc/profile.d/conda.sh

project_name=Li_Luo_liver_tumor

WORK_DIR=/home/user/data3/rbase/translation_model/neoantigen/samples/${project_name}
SCRIPT_DIR=/home/user/data3/rbase/translation_model/models/tools/tumor_neoantigen
FA_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/Homo_sapiens.GRCh38.primary_assembly.genome.fa
GTF_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf
META_FILE=${WORK_DIR}/meta_info_patient_run.csv

echo "=========================================="
echo "==== Somatic Variants Calling by GATK ===="
echo "=========================================="

bash $SCRIPT_DIR/run.rnaseq_variants_calling.sh \
    --bamDir $WORK_DIR/bam \
    --meta $WORK_DIR/meta_info_patient_run.csv \
    --ref_fasta $FA_FILE \
    --out_dir $WORK_DIR/GATK_somatic_variants \
    --threads 2
#!/bin/bash
set -euo pipefail

source /home/user/data2/rbase/env/anaconda3/etc/profile.d/conda.sh

project_name=cohort_2

WORK_DIR=/home/user/data3/rbase/translation_model/neoantigen/samples/${project_name}
SCRIPT_DIR=/home/user/data3/rbase/translation_model/models/tools/tumor_neoantigen
GTF_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf
ADD_GTF_FILE=${WORK_DIR}/assembly/final_filtered_novel_transcripts_enhanced.gtf
COUNTS_IN=${WORK_DIR}/featureCounts_tumor/transcript_counts.txt
META_FILE=${WORK_DIR}/meta_info_patient_run.csv

echo "============================================="
echo "==== Identify tumor-specific transcripts ===="
echo "============================================="
TUMOR_UP_CSV=${WORK_DIR}/tumor_specific_tx/patient_tumor_upregulated_transcripts_all.csv

# 1. Specific junction of transcripts
[ -f ${WORK_DIR}/assembly/specific_junction_mapping.tsv ] || \
python ${SCRIPT_DIR}/extract_specific_junctions.py \
    --ref_gtf $GTF_FILE \
    --tumor_gtfs $ADD_GTF_FILE \
    --output_mapping ${WORK_DIR}/assembly/specific_junction_mapping.tsv \
    --mode tumor_specific

# 2. find tumor specific transcripts by TPM and junction CPM
[ -f ${TUMOR_UP_CSV} ] || \
python ${SCRIPT_DIR}/find_tumor_specific_transcripts.py \
    --counts_file ${COUNTS_IN} \
    --summary_file ${COUNTS_IN}.summary \
    --metadata_file ${META_FILE} \
    --out_tpm_file ${WORK_DIR}/featureCounts_tumor/transcript_tpm_matrix.csv \
    --output_file ${TUMOR_UP_CSV} \
    --jcounts_file ${WORK_DIR}/featureCounts_tumor/junction_counts.txt.jcounts \
    --junc_mapping ${WORK_DIR}/assembly/specific_junction_mapping.tsv \
    --class_mapping ${WORK_DIR}/assembly/transcript_class_mapping.tsv \
    --min_max_tcount 20 \
    --min_max_jcount 5 \
    --pseudo_count 0.1 \
    --min_tumor_tpm 1.0 \
    --strict_normal_max 0.5 \
    --min_tumor_cpm 2.0 \
    --strict_normal_max_cpm 1.0 \
    --min_log2fc 2.0

echo -e "\n"
echo "==================================================="
echo "==== Filter tumor-specific transcripts by GTEx ===="
echo "==================================================="
STEP1_CSV=${WORK_DIR}/tumor_specific_tx/safe_tumor_specific_transcripts_GTEx-step1.csv
STEP2_CSV=${WORK_DIR}/tumor_specific_tx/safe_tumor_specific_transcripts_GTEx-step2.csv

# Run Step 1
[ -f ${STEP1_CSV} ] || \
python ${SCRIPT_DIR}/filter_gtex_step1.py \
    --input ${TUMOR_UP_CSV} \
    --gtex_tpm /home/user/data3/rbase/database/GTEx/GTEx_v11_tissue_median_transcript_tpm.csv \
    --gtex_junc /home/user/data3/rbase/database/GTEx/GTEx_Tissue_Median_Junction_CPM.csv \
    --junc_mapping ${WORK_DIR}/assembly/specific_junction_mapping.tsv \
    --max_tpm 0.5 \
    --max_jcpm 1.0 \
    --output ${STEP1_CSV}

# Run featureCounts for Step 2
[ -f ${WORK_DIR}/featureCounts_gtex/gtex_novel_transcript_counts.txt ] || bash ${SCRIPT_DIR}/run_gtex_novel_quant.sh \
    --bam_dir /home/user/data/share/GTExV8 \
    --work_dir ${WORK_DIR}/featureCounts_gtex \
    --quant_target_gtf ${WORK_DIR}/assembly/final_quantification_targets_enhanced.gtf \
    --threads 40

# Run Step 2
[ -f ${STEP2_CSV} ] || python ${SCRIPT_DIR}/filter_gtex_step2.py \
    --step1_file ${STEP1_CSV} \
    --counts_file ${WORK_DIR}/featureCounts_gtex/gtex_novel_transcript_counts.txt \
    --fc_log ${WORK_DIR}/featureCounts_gtex/featureCounts_gtex.log \
    --anno_file /home/user/data3/rbase/database/GTEx/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt \
    --max_tpm 0.5 \
    --output ${STEP2_CSV}

echo -e "\n"
echo "======================================================"
echo "==== Neoantigen prediction by TRACE and netMHCpan ===="
echo "======================================================"

HLA_CSV=${WORK_DIR}/patient_hla_typing.csv
TRANSCRIPTS_FASTA=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/transcripts/gencode.v48.transcripts.fa
DENOVO_TRANSCRIPTS_FASTA=/home/user/data3/rbase/small_peptide/denovo_genes/nucl_fa/denovo_gene_transcripts.fasta
CONFIG_DIR=/home/user/data3/rbase/translation_model/models/src/config
WEIGHT_DIR=/home/user/data3/rbase/translation_model/models/checkpoint/pretrain
TRACE_MODE="short"

# 跳过 CSV 表头逐行读取
tail -n +2 "$HLA_CSV" | while IFS=',' read -r dataset patient hla_a1 hla_a2 hla_b1 hla_b2 hla_c1 hla_c2 || [ -n "$dataset" ];
do    
    # 格式化患者名：将 "patient 10615" 转为 "patient_10615" 用于建文件和目录
    patient_safe=$(echo "$patient" | tr ' ' '_')
    # 从 Metadata 中动态获取该患者的肿瘤 Run ID
    RUN_ID=$(grep "$patient" "$META_FILE" | grep -i "tumor" | cut -d',' -f1 | head -n 1)
    if [ -z "$RUN_ID" ]; then
        echo "[Warning] 找不到患者 $patient 的肿瘤 Run ID，跳过此患者..."
        continue
    fi

    # 自动格式化 HLA 字符串，去除星号并去重
    alleles=("$hla_a1" "$hla_a2") #"$hla_b1" "$hla_b2" "$hla_c1" "$hla_c2")
    formatted_hlas=""
    for allele in "${alleles[@]}"; do
        [ -z "$allele" ] && continue 
        fmt="HLA-${allele//\*/}"
        formatted_hlas+="$fmt,"
    done
    UNIQUE_HLAS=$(echo "$formatted_hlas" | tr ',' '\n' | sort -u | grep -v "^$" | paste -sd, -)

    echo "---------------------------------------------------------"
    echo "▶ Processing Patient: $patient (Tumor Run: $RUN_ID)"
    echo "▶ HLA Alleles: $UNIQUE_HLAS"
    echo "---------------------------------------------------------"

    # 为每位患者建立专属文件夹
    PATIENT_TRACE_DIR=${WORK_DIR}/translation/${patient_safe}
    PATIENT_MHC_DIR=${WORK_DIR}/HLA_affinity/${patient_safe}
    mkdir -p "$PATIENT_TRACE_DIR"
    mkdir -p "$PATIENT_MHC_DIR"

    echo -e "\n"
    echo "-----------------------------------------"
    echo "=> Translation prediction by TRACE"
    echo "-----------------------------------------"
    
    if [ -s ${PATIENT_TRACE_DIR}/high_confidence_proteins.${patient_safe}.${TRACE_MODE}_mode.fasta ]; then
        echo "[Skip] Translated ORFs already generated for $patient"
    else
        # Activate environmtnt
        conda activate ribo_model
        # run
        python ${SCRIPT_DIR}/run_trace_prediction.py \
            --input_csv ${STEP2_CSV} \
            --out_dir ${PATIENT_TRACE_DIR} \
            --fasta_files ${WORK_DIR}/assembly/novel_transcripts.fasta $TRANSCRIPTS_FASTA $DENOVO_TRANSCRIPTS_FASTA \
            --config_path ${CONFIG_DIR}/base_model_expr_384d_8h_10l_64env_16ad.yaml \
            --weights_path ${WEIGHT_DIR}/base_model_expr_384d_8h_10l_64env_16ad-PsiteDensityHead.human_7c_8k_depth0.1_cov0.1_rpm1.90_0.001.best.pt \
            --patient_counts_file ${WORK_DIR}/featureCounts_tumor/gene_counts.txt \
            --counts_level "gene" \
            --tpm_csv ${WORK_DIR}/featureCounts_tumor/transcript_tpm_matrix.csv \
            --tpm_level "transcript" \
            --ref_order ${CONFIG_DIR}/global_anchor_gene_order.txt \
            --tx2gene_mapping /home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/ens_genes_v115.txt \
            --mapping_json ${CONFIG_DIR}/global_species_id_mapping.json \
            --sample_run_id "$RUN_ID" \
            --patient_id "$patient_safe" \
            --mode ${TRACE_MODE} \
            --batch_size 5 \
            --device cuda \
            1> ${PATIENT_TRACE_DIR}/trace_prediction.log 2>&1
    fi

    echo -e "\n"
    echo "--------------------------------------------"
    echo "=> HLA affinity prediction by netMHCpan"
    echo "--------------------------------------------"

    if [ -s "${PATIENT_MHC_DIR}/netMHCpan.log" ]; then
        echo "[Skip] HLA-affinity already generated for $patient"
    else
        netMHCpan -s -BA -t 10 \
            -a "$UNIQUE_HLAS" \
            -xls -xlsfile ${PATIENT_MHC_DIR}/netMHCpan_results.xls \
            -f ${PATIENT_TRACE_DIR}/high_confidence_proteins.${patient_safe}.${TRACE_MODE}_mode.fasta \
            1> ${PATIENT_MHC_DIR}/netMHCpan.log 2>&1
    fi

    echo -e "\n"
    echo "--------------------------------------------"
    echo "=> Peptide Prioritization & Filtering"
    echo "--------------------------------------------"

    python ${SCRIPT_DIR}/neoantigen_prioritization_report.py \
        --step2_csv ${STEP2_CSV} \
        --netmhcpan_log ${PATIENT_MHC_DIR}/netMHCpan.log \
        --fasta_file ${PATIENT_TRACE_DIR}/high_confidence_proteins.${patient_safe}.${TRACE_MODE}_mode.fasta \
        --translation_csv ${PATIENT_TRACE_DIR}/high_confidence_orfs.${patient_safe}.${TRACE_MODE}_mode.csv \
        --patient_id ${patient_safe} \
        --output_dir ${WORK_DIR}/patient_epitope_reports \
        --bind_levels SB WB \
        --max_aff_nm 2000 \
        --max_rank_el 5.0

    echo -e "\n"
    echo "----------------------------------------------"
    echo "=> Normal proteome by TRACE"
    echo "----------------------------------------------"

    # Get the normal Run ID for this patient
    NORM_RUN_ID=$(grep "$patient" "$META_FILE" | grep -i "normal" | cut -d',' -f1 | head -n 1)

    if [ -z "$NORM_RUN_ID" ]; then
        echo "[Warning] No normal Run ID found for $patient, skipping normal proteome."
    elif [ -s ${PATIENT_TRACE_DIR}/normal/high_confidence_proteins.${patient_safe}.${TRACE_MODE}_mode.fasta ]; then
        echo "[Skip] Normal proteome already predicted for $patient"
    else
        # Generate input CSV: transcripts with normal TPM > 0.5
        NORMAL_TX_CSV=${PATIENT_TRACE_DIR}/normal/normal_expressed_transcripts.csv
        mkdir -p ${PATIENT_TRACE_DIR}/normal
        python ${SCRIPT_DIR}/prepare_normal_transcript_input.py \
            --tpm_csv ${WORK_DIR}/featureCounts_tumor/transcript_tpm_matrix.csv \
            --normal_run ${NORM_RUN_ID} \
            --output ${NORMAL_TX_CSV} \
            --min_tpm 1

        if [ -s "$NORMAL_TX_CSV" ]; then
            conda activate ribo_model
            python ${SCRIPT_DIR}/run_trace_prediction.py \
                --input_csv ${NORMAL_TX_CSV} \
                --out_dir ${PATIENT_TRACE_DIR}/normal \
                --fasta_files ${WORK_DIR}/assembly/novel_transcripts.fasta $TRANSCRIPTS_FASTA $DENOVO_TRANSCRIPTS_FASTA \
                --config_path ${CONFIG_DIR}/base_model_expr_384d_8h_10l_64env_16ad.yaml \
                --weights_path ${WEIGHT_DIR}/base_model_expr_384d_8h_10l_64env_16ad-PsiteDensityHead.human_7c_8k_depth0.1_cov0.1_rpm1.90_0.001.best.pt \
                --patient_counts_file ${WORK_DIR}/featureCounts_tumor/gene_counts.txt \
                --counts_level "gene" \
                --tpm_csv ${WORK_DIR}/featureCounts_tumor/transcript_tpm_matrix.csv \
                --tpm_level "transcript" \
                --ref_order ${CONFIG_DIR}/global_anchor_gene_order.txt \
                --tx2gene_mapping /home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/ens_genes_v115.txt \
                --mapping_json ${CONFIG_DIR}/global_species_id_mapping.json \
                --sample_run_id "$NORM_RUN_ID" \
                --patient_id "$patient_safe" \
                --mode ${TRACE_MODE} \
                --batch_size 5 \
                --device cuda \
                1> ${PATIENT_TRACE_DIR}/normal/trace_prediction.log 2>&1
        fi
    fi
done

echo -e "\n"

echo -e "\n"

echo -e "\n"
echo "----------------------------------------------"
echo "=> Filter against patient normal proteome (TRACE)"
echo "----------------------------------------------"

NORMAL_FILTERED_DIR=${WORK_DIR}/patient_normal_filtered_reports
[ -d ${NORMAL_FILTERED_DIR} ] || python ${SCRIPT_DIR}/filter_normal_proteome_offtargets.py \
    --input_dir ${WORK_DIR}/patient_epitope_reports \
    --trace_base_dir ${WORK_DIR}/translation \
    --trace_mode ${TRACE_MODE} \
    --output_dir ${NORMAL_FILTERED_DIR}

echo -e "\n"
echo "---------------------------"
echo "=> Canonical proteome filtering"
echo "---------------------------"

echo "---------------------------"

[ -d ${WORK_DIR}/patient_neoepitope_reports ] || python ${SCRIPT_DIR}/filter_canonical_offtargets.py \
    --input_dir ${NORMAL_FILTERED_DIR} \
    --fasta /home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/translations/gencode.v49.pc_translations.fa \
    --output_dir ${WORK_DIR}/patient_neoepitope_reports
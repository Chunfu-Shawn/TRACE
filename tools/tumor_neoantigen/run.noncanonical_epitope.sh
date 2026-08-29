#!/bin/bash
set -euo pipefail

source /home/user/data2/rbase/env/anaconda3/etc/profile.d/conda.sh

project_name=cohort_2

WORK_DIR=/home/user/data3/rbase/translation_model/neoantigen/samples/${project_name}
SCRIPT_DIR=/home/user/data3/rbase/translation_model/models/tools/tumor_neoantigen
source "${SCRIPT_DIR}/quantification_config.sh"
GTF_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf
ADD_GTF_FILE=${WORK_DIR}/assembly/final_filtered_novel_transcripts_enhanced.gtf
COMPLETE_GTF=${WORK_DIR}/featureCounts_tumor/complete_transcript_annotation.gtf
COUNTS_IN=${WORK_DIR}/featureCounts_tumor/transcript_counts.complete_gtf.multioverlap.txt
GENE_COUNTS_IN=${WORK_DIR}/featureCounts_tumor/gene_counts.complete_gtf.multioverlap.txt
JUNCTION_COUNTS_IN=${WORK_DIR}/featureCounts_tumor/junction_counts.complete_gtf.multioverlap.txt.jcounts
TPM_MATRIX=${WORK_DIR}/featureCounts_tumor/transcript_true_tpm_matrix.csv
META_FILE=${WORK_DIR}/meta_info_patient_run.csv
# Step 2 requires local GTEx BAMs and is disabled by default on servers without raw data.
RUN_GTEX_STEP2=${RUN_GTEX_STEP2:-yes}
GTEX_BAM_DIR=${GTEX_BAM_DIR:-/home/user/data/share/GTExV8}
# Cohort mode loads TRACE and shared FASTA/expression resources once.
RUN_COHORT_TRACE=${RUN_COHORT_TRACE:-yes}
# Normal-tissue TRACE is an optional second-tier safety screen.
RUN_NORMAL_TRACE=${RUN_NORMAL_TRACE:-no}
TRACE_BATCH_SIZE=${TRACE_BATCH_SIZE:-5}
TRACE_MAX_PATIENTS=${TRACE_MAX_PATIENTS:-}

is_true() {
    case "${1,,}" in
        yes|true|1) return 0 ;;
        *) return 1 ;;
    esac
}

echo "============================================="
echo "==== Identify tumor-specific transcripts ===="
echo "============================================="
JUNCTION_MAPPING=${WORK_DIR}/assembly/specific_junction_mapping.stranded.tsv
TUMOR_UP_CSV=${WORK_DIR}/tumor_specific_tx/patient_tumor_upregulated_transcripts.true_tpm_gene_lib.stranded.csv

# 1. Specific junction of transcripts
[ -f ${JUNCTION_MAPPING} ] || \
python ${SCRIPT_DIR}/extract_specific_junctions.py \
    --ref_gtf $GTF_FILE \
    --tumor_gtfs $ADD_GTF_FILE \
    --output_mapping ${JUNCTION_MAPPING} \
    --mode tumor_specific

# 2. find tumor specific transcripts by TPM and junction CPM
[ -f ${TUMOR_UP_CSV} ] || \
python ${SCRIPT_DIR}/find_tumor_specific_transcripts.py \
    --counts_file ${COUNTS_IN} \
    --gene_counts_file ${GENE_COUNTS_IN} \
    --metadata_file ${META_FILE} \
    --out_tpm_file ${TPM_MATRIX} \
    --output_file ${TUMOR_UP_CSV} \
    --jcounts_file ${JUNCTION_COUNTS_IN} \
    --junc_mapping ${JUNCTION_MAPPING} \
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
STEP1_CSV=${WORK_DIR}/tumor_specific_tx/safe_tumor_specific_transcripts_GTEx-step1.true_tpm_gene_lib.stranded.csv
STEP2_CSV=${WORK_DIR}/tumor_specific_tx/safe_tumor_specific_transcripts_GTEx-step2.true_tpm_gene_lib.stranded.csv

# Run Step 1
[ -f ${STEP1_CSV} ] || \
python ${SCRIPT_DIR}/filter_gtex_step1.py \
    --input ${TUMOR_UP_CSV} \
    --gtex_tpm /home/user/data3/rbase/database/GTEx/GTEx_v11_tissue_median_transcript_tpm.csv \
    --gtex_junc /home/user/data3/rbase/database/GTEx/GTEx_Tissue_Median_Junction_CPM.csv \
    --junc_mapping ${JUNCTION_MAPPING} \
    --max_tpm 0.5 \
    --max_jcpm 1.0 \
    --output ${STEP1_CSV}

FINAL_TARGET_CSV=${STEP1_CSV}
if is_true "${RUN_GTEX_STEP2}"; then
    if [ -d "${GTEX_BAM_DIR}" ]; then
        GTEX_COUNTS=${WORK_DIR}/featureCounts_gtex/gtex_transcript_counts.complete_gtf.multioverlap.txt
        [ -f ${GTEX_COUNTS} ] || bash ${SCRIPT_DIR}/run_gtex_novel_quant.sh \
            --bam_dir ${GTEX_BAM_DIR} \
            --work_dir ${WORK_DIR}/featureCounts_gtex \
            --annotation_gtf ${COMPLETE_GTF} \
            --strand ${STRAND_FLAG} \
            --threads 40

        [ -f ${STEP2_CSV} ] || python ${SCRIPT_DIR}/filter_gtex_step2.py \
            --step1_file ${STEP1_CSV} \
            --counts_file ${GTEX_COUNTS} \
            --anno_file /home/user/data3/rbase/database/GTEx/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt \
            --max_tpm 0.5 \
            --output ${STEP2_CSV}
        FINAL_TARGET_CSV=${STEP2_CSV}
    else
        echo "[Warning] GTEx Step 2 requested but BAM directory is unavailable: ${GTEX_BAM_DIR}"
        echo "[Warning] Continuing without GTEx raw-BAM transcript filtering."
    fi
else
    echo "[Info] GTEx Step 2 is disabled; using precomputed GTEx Step 1 only."
fi

echo -e "\n"
echo "======================================================"
echo "==== Tumor-associated antigen prediction by TRACE and netMHCpan ===="
echo "======================================================"

HLA_CSV=${WORK_DIR}/patient_hla_typing.csv
TRANSCRIPTS_FASTA=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/transcripts/gencode.v48.transcripts.fa
DENOVO_TRANSCRIPTS_FASTA=/home/user/data3/rbase/small_peptide/denovo_genes/fasta/denovo_gene_transcripts.fasta
CONFIG_DIR=/home/user/data3/rbase/translation_model/models/src/config
WEIGHT_DIR=/home/user/data3/rbase/translation_model/models/checkpoint/train
TRACE_MODE=${TRACE_MODE:-balanced}

run_cohort_trace() {
    python "${SCRIPT_DIR}/run_trace_cohort_prediction.py" \
        --input_csv "${FINAL_TARGET_CSV}" \
        --out_dir "${WORK_DIR}/translation" \
        --fasta_files "${WORK_DIR}/assembly/novel_transcripts.fasta" "$TRANSCRIPTS_FASTA" "$DENOVO_TRANSCRIPTS_FASTA" \
        --config_path "${CONFIG_DIR}/base_model_384d_16h_12l_64env_16ad_bs.yaml" \
        --weights_path "${WEIGHT_DIR}/base_model_384d_16h_12l_64env_16ad_bs-PsiteDensityHead.hs_22c_18c_26c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1_e50_a1_b0_exp_aug_i03_m15.200_0.001.best_profile.pt" \
        --patient_counts_file "${GENE_COUNTS_IN}" \
        --counts_level gene \
        --tpm_csv "${TPM_MATRIX}" \
        --tpm_level transcript \
        --ref_order "${CONFIG_DIR}/global_anchor_gene_order.txt" \
        --tx2gene_mapping /home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/ens_genes_v115.txt \
        --mapping_json "${CONFIG_DIR}/global_species_id_mapping.json" \
        --mode "${TRACE_MODE}" \
        --batch_size "${TRACE_BATCH_SIZE}" \
        --device cuda \
        "$@"
}

if is_true "${RUN_COHORT_TRACE}"; then
    echo -e "\n"
    echo "----------------------------------------------"
    echo "=> Cohort TRACE prediction (single model load)"
    echo "----------------------------------------------"
    mkdir -p "${WORK_DIR}/translation"
    conda activate ribo_model
    if [ -n "${TRACE_MAX_PATIENTS}" ]; then
        run_cohort_trace --max_patients "${TRACE_MAX_PATIENTS}" \
            1> "${WORK_DIR}/translation/trace_cohort_prediction.log" 2>&1
    else
        run_cohort_trace \
            1> "${WORK_DIR}/translation/trace_cohort_prediction.log" 2>&1
    fi
    if [ -n "${TRACE_MAX_PATIENTS}" ]; then
        echo "[Info] TRACE benchmark subset complete; inspect trace_cohort_status.csv before the full rerun."
        exit 0
    fi
fi

# Read one patient per row after skipping the CSV header.
tail -n +2 "$HLA_CSV" | while IFS=',' read -r dataset patient hla_a1 hla_a2 hla_b1 hla_b2 hla_c1 hla_c2 || [ -n "$dataset" ];
do    
    # Convert spaces in the patient identifier to underscores for paths.
    patient_safe=$(echo "$patient" | tr ' ' '_')
    # Resolve the tumor run using normalized metadata labels.
    RUN_ID=$(python "${SCRIPT_DIR}/metadata_utils.py" \
        --metadata "$META_FILE" --patient "$patient" --tissue tumor)
    if [ -z "$RUN_ID" ]; then
        echo "[Warning] No tumor run was found for $patient; skipping this patient."
        continue
    fi

    # Format and deduplicate HLA-A alleles for netMHCpan.
    alleles=("$hla_a1" "$hla_a2") #"$hla_b1" "$hla_b2" "$hla_c1" "$hla_c2")
    formatted_hlas=""
    for allele in "${alleles[@]}"; do
        [ -z "$allele" ] && continue 
        fmt="HLA-${allele//\*/}"
        formatted_hlas+="$fmt,"
    done
    UNIQUE_HLAS=$(echo "$formatted_hlas" | tr ',' '\n' | sort -u | sed '/^$/d' | paste -sd, -)
    if [ -z "${UNIQUE_HLAS}" ]; then
        echo "[Warning] No valid HLA-A alleles found for $patient; skipping this patient."
        continue
    fi

    echo "---------------------------------------------------------"
    echo "▶ Processing Patient: $patient (Tumor Run: $RUN_ID)"
    echo "▶ HLA Alleles: $UNIQUE_HLAS"
    echo "---------------------------------------------------------"

    # Create patient-specific TRACE and HLA output directories.
    PATIENT_TRACE_DIR=${WORK_DIR}/translation/${patient_safe}
    PATIENT_MHC_DIR=${WORK_DIR}/HLA_affinity/${patient_safe}
    NETMHCPAN_LOG=${PATIENT_MHC_DIR}/netMHCpan.${TRACE_MODE}.log
    NETMHCPAN_XLS=${PATIENT_MHC_DIR}/netMHCpan_results.${TRACE_MODE}.xls
    mkdir -p "$PATIENT_TRACE_DIR"
    mkdir -p "$PATIENT_MHC_DIR"

    echo -e "\n"
    echo "-----------------------------------------"
    echo "=> Translation prediction by TRACE"
    echo "-----------------------------------------"
    
    if [ -s ${PATIENT_TRACE_DIR}/high_confidence_proteins.${patient_safe}.${TRACE_MODE}_mode.fasta ]; then
        echo "[Skip] Translated ORFs already generated for $patient"
    elif is_true "${RUN_COHORT_TRACE}"; then
        echo "[Warning] Cohort TRACE produced no protein FASTA for $patient; skipping downstream prediction."
        continue
    else
        # Activate environmtnt
        conda activate ribo_model
        # run
        python ${SCRIPT_DIR}/run_trace_prediction.py \
            --input_csv ${FINAL_TARGET_CSV} \
            --out_dir ${PATIENT_TRACE_DIR} \
            --fasta_files ${WORK_DIR}/assembly/novel_transcripts.fasta $TRANSCRIPTS_FASTA $DENOVO_TRANSCRIPTS_FASTA \
            --config_path ${CONFIG_DIR}/base_model_384d_16h_12l_64env_16ad_bs.yaml \
            --weights_path ${WEIGHT_DIR}/base_model_384d_16h_12l_64env_16ad_bs-PsiteDensityHead.hs_22c_18c_26c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1_e50_a1_b0_exp_aug_i03_m15.200_0.001.best_profile.pt \
            --patient_counts_file ${GENE_COUNTS_IN} \
            --counts_level "gene" \
            --tpm_csv ${TPM_MATRIX} \
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

    if [ -s "${NETMHCPAN_LOG}" ]; then
        echo "[Skip] HLA-affinity already generated for $patient"
    else
        netMHCpan -s -BA -t 10 \
            -a "$UNIQUE_HLAS" \
            -xls -xlsfile ${NETMHCPAN_XLS} \
            -f ${PATIENT_TRACE_DIR}/high_confidence_proteins.${patient_safe}.${TRACE_MODE}_mode.fasta \
            1> ${NETMHCPAN_LOG} 2>&1
    fi

    echo -e "\n"
    echo "--------------------------------------------"
    echo "=> Peptide Prioritization & Filtering"
    echo "--------------------------------------------"

    python ${SCRIPT_DIR}/neoantigen_prioritization_report.py \
        --step2_csv ${FINAL_TARGET_CSV} \
        --netmhcpan_log ${NETMHCPAN_LOG} \
        --fasta_file ${PATIENT_TRACE_DIR}/high_confidence_proteins.${patient_safe}.${TRACE_MODE}_mode.fasta \
        --translation_csv ${PATIENT_TRACE_DIR}/high_confidence_orfs.${patient_safe}.${TRACE_MODE}_mode.csv \
        --patient_id ${patient_safe} \
        --tumor_run_id "$RUN_ID" \
        --output_dir ${WORK_DIR}/patient_epitope_reports \
        --bind_levels SB WB \
        --max_aff_nm 2000 \
        --max_rank_el 5.0

    if is_true "${RUN_NORMAL_TRACE}"; then
        echo -e "\n"
        echo "----------------------------------------------"
        echo "=> Optional normal proteome by TRACE"
        echo "----------------------------------------------"

        # Resolve the matched normal run using normalized metadata labels.
        NORM_RUN_ID=$(python "${SCRIPT_DIR}/metadata_utils.py" \
            --metadata "$META_FILE" --patient "$patient" --tissue normal)

        if [ -z "$NORM_RUN_ID" ]; then
            echo "[Warning] No normal Run ID found for $patient, skipping normal proteome."
        elif [ -s ${PATIENT_TRACE_DIR}/normal/high_confidence_proteins.${patient_safe}.${TRACE_MODE}_mode.fasta ]; then
            echo "[Skip] Normal proteome already predicted for $patient"
        else
            # Generate input CSV for transcripts expressed in matched adjacent tissue.
            NORMAL_TX_CSV=${PATIENT_TRACE_DIR}/normal/normal_expressed_transcripts.csv
            mkdir -p ${PATIENT_TRACE_DIR}/normal
            python ${SCRIPT_DIR}/prepare_normal_transcript_input.py \
                --tpm_csv ${TPM_MATRIX} \
                --normal_run ${NORM_RUN_ID} \
                --output ${NORMAL_TX_CSV} \
                --min_tpm 1

            if [ -s "$NORMAL_TX_CSV" ]; then
                conda activate ribo_model
                python ${SCRIPT_DIR}/run_trace_prediction.py \
                    --input_csv ${NORMAL_TX_CSV} \
                    --out_dir ${PATIENT_TRACE_DIR}/normal \
                    --fasta_files ${WORK_DIR}/assembly/novel_transcripts.fasta $TRANSCRIPTS_FASTA $DENOVO_TRANSCRIPTS_FASTA \
                    --config_path ${CONFIG_DIR}/base_model_384d_16h_12l_64env_16ad_bs.yaml \
                    --weights_path ${WEIGHT_DIR}/base_model_384d_16h_12l_64env_16ad_bs-PsiteDensityHead.hs_22c_18c_26c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1_e50_a1_b0_exp_aug_i03_m15.200_0.001.best_profile.pt \
                    --patient_counts_file ${GENE_COUNTS_IN} \
                    --counts_level "gene" \
                    --tpm_csv ${TPM_MATRIX} \
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
    fi
done

echo -e "\n"
echo "----------------------------------------------"
echo "=> Filter against patient normal proteome (TRACE)"
echo "----------------------------------------------"

if is_true "${RUN_NORMAL_TRACE}"; then
    NORMAL_FILTERED_DIR=${WORK_DIR}/patient_normal_filtered_reports
    python ${SCRIPT_DIR}/filter_normal_proteome_offtargets.py \
        --input_dir ${WORK_DIR}/patient_epitope_reports \
        --trace_base_dir ${WORK_DIR}/translation \
        --trace_mode ${TRACE_MODE} \
        --output_dir ${NORMAL_FILTERED_DIR}
else
    echo "[Info] Normal TRACE is disabled; continuing with canonical-proteome filtering."
    NORMAL_FILTERED_DIR=${WORK_DIR}/patient_epitope_reports
fi

echo -e "\n"
echo "---------------------------"
echo "=> Canonical proteome filtering and antigen classification"
echo "---------------------------"

TAA_REPORT_DIR=${WORK_DIR}/patient_tumor_associated_antigen_reports
python ${SCRIPT_DIR}/filter_canonical_offtargets.py \
    --input_dir ${NORMAL_FILTERED_DIR} \
    --fasta /home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/translations/gencode.v49.pc_translations.fa \
    --output_dir ${TAA_REPORT_DIR}

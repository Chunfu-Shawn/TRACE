#!/bin/bash
set -euo pipefail

source /home/user/data2/rbase/env/anaconda3/etc/profile.d/conda.sh

project_name=Li_Luo_liver_tumor

WORK_DIR=/home/user/data3/rbase/translation_model/neoantigen/samples/${project_name}
SCRIPT_DIR=/home/user/data3/rbase/translation_model/models/tools/tumor_neoantigen
FA_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/Homo_sapiens.GRCh38.primary_assembly.genome.fa
GTF_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf
META_FILE=${WORK_DIR}/meta_info_patient_run.csv
TRANSCRIPTS_FASTA=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/transcripts/gencode.v48.transcripts.fa
DENOVO_TRANSCRIPTS_FASTA=/home/user/data3/rbase/small_peptide/denovo_genes/nucl_fa/denovo_gene_transcripts.fasta
CANONICAL_PROTEOME=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/translations/gencode.v49.pc_translations.fa
CONFIG_DIR=/home/user/data3/rbase/translation_model/models/src/config
WEIGHT_DIR=/home/user/data3/rbase/translation_model/models/checkpoint/pretrain
TRACE_MODE="short"
HLA_CSV=${WORK_DIR}/patient_hla_typing.csv
TPM_MATRIX=${WORK_DIR}/featureCounts_tumor/transcript_tpm_matrix.csv

echo "=========================================="
echo "==== Phase 1: Somatic Variants (GATK) ===="
echo "=========================================="

VCF_DIR=${WORK_DIR}/GATK_somatic_variants
bash $SCRIPT_DIR/run.rnaseq_variants_calling.sh \
    --bamDir $WORK_DIR/bam \
    --meta $META_FILE \
    --ref_fasta $FA_FILE \
    --out_dir $VCF_DIR \
    --threads 2

echo -e "\n"
echo "=========================================="
echo "==== Phase 2: Per-patient pipeline ========"
echo "=========================================="

MUT_TRANSCRIPT_DIR=${WORK_DIR}/mutation_transcripts
ANNOTATION_DIR=${WORK_DIR}/mutation_annotation
MUT_PEPTIDE_DIR=${WORK_DIR}/mutation_peptides
MUT_REPORT_DIR=${WORK_DIR}/patient_mutation_neoepitope_reports
mkdir -p ${MUT_TRANSCRIPT_DIR} ${ANNOTATION_DIR} ${MUT_PEPTIDE_DIR} ${MUT_REPORT_DIR}

tail -n +2 "$HLA_CSV" | while IFS=',' read -r dataset patient hla_a1 hla_a2 hla_b1 hla_b2 hla_c1 hla_c2 || [ -n "$patient" ];
do
    patient_safe=$(echo "$patient" | tr ' ' '_')
    TUMOR_RUN=$(grep "$patient" "$META_FILE" | grep -i "tumor" | cut -d',' -f1 | head -n 1)
    if [ -z "$TUMOR_RUN" ]; then
        echo "[Warning] No tumor Run ID for $patient, skipping."
        continue
    fi
    VCF_IN=${VCF_DIR}/${patient_safe}/${patient_safe}_somatic_filtered.vcf.gz

    echo "=========================================="
    echo "=> Patient: $patient_safe  (Tumor Run: $TUMOR_RUN)"
    echo "=========================================="


    # ---------------------------------------------------------------
    # Step 0: Filter RNA editing sites
    # ---------------------------------------------------------------
    VCF_FILTERED=${VCF_DIR}/${patient_safe}/${patient_safe}_no_editing.vcf
    RNA_EDITING_DB=/home/user/data3/rbase/database/RNA_editing/REDIportal_ATLAS_2024.tsv
    if [ -f "$VCF_FILTERED" ]; then
        echo "[Skip] RNA editing filter already applied"
    else
        echo "-> [0/7] Filtering RNA editing sites"
        if [ -f "$RNA_EDITING_DB" ]; then
            python ${SCRIPT_DIR}/filter_rna_editing.py \
                --vcf ${VCF_IN} \
                --output ${VCF_FILTERED} \
                --editing_db ${RNA_EDITING_DB}
        else
            echo "[Warning] RNA editing DB not found: ${RNA_EDITING_DB}"
            echo "         Using heuristic A>G / T>C filter as fallback"
            python ${SCRIPT_DIR}/filter_rna_editing.py \
                --vcf ${VCF_IN} \
                --output ${VCF_FILTERED} \
                --filter_a_to_g
        fi
    fi
    # Use filtered VCF for downstream steps
    VCF_IN=${VCF_FILTERED}

    # ---------------------------------------------------------------
    # Step 1: Extract transcripts overlapping any variant (exon-level)
    # ---------------------------------------------------------------
    MUT_TX_CSV=${MUT_TRANSCRIPT_DIR}/${patient_safe}_mutation_transcripts.csv
    if [ -f "$MUT_TX_CSV" ]; then
        echo "[Skip] Mutation transcript list already exists"
    else
        echo "-> [1/7] Extracting transcripts with mutations"
        python ${SCRIPT_DIR}/extract_mutation_transcripts.py \
            --vcf ${VCF_IN} \
            --gtf ${GTF_FILE} \
            --output ${MUT_TX_CSV} \
            --tumor_run ${TUMOR_RUN}
    fi

    # ---------------------------------------------------------------
    # Step 2: TRACE translation prediction on mutated transcripts
    # ---------------------------------------------------------------
    PATIENT_MUT_DIR=${WORK_DIR}/translation_mutation/${patient_safe}
    mkdir -p ${PATIENT_MUT_DIR}
    TRACE_ORF_CSV=${PATIENT_MUT_DIR}/high_confidence_orfs.${patient_safe}.${TRACE_MODE}_mode.csv
    TRACE_PROT_FASTA=${PATIENT_MUT_DIR}/high_confidence_proteins.${patient_safe}.${TRACE_MODE}_mode.fasta

    if [ -s "$TRACE_PROT_FASTA" ]; then
        echo "[Skip] TRACE already done"
    elif [ -s "$MUT_TX_CSV" ]; then
        echo "-> [2/7] TRACE prediction on mutated transcripts"
        conda activate ribo_model
        python ${SCRIPT_DIR}/run_trace_prediction.py \
            --input_csv ${MUT_TX_CSV} \
            --out_dir ${PATIENT_MUT_DIR} \
            --fasta_files ${WORK_DIR}/assembly/novel_transcripts.fasta $TRANSCRIPTS_FASTA $DENOVO_TRANSCRIPTS_FASTA \
            --config_path ${CONFIG_DIR}/base_model_expr_384d_8h_10l_64env_16ad.yaml \
            --weights_path ${WEIGHT_DIR}/base_model_expr_384d_8h_10l_64env_16ad-PsiteDensityHead.human_7c_8k_depth0.1_cov0.1_rpm1.90_0.001.best.pt \
            --patient_counts_file ${WORK_DIR}/featureCounts_tumor/gene_counts.txt \
            --counts_level "gene" \
            --tpm_csv ${TPM_MATRIX} \
            --tpm_level "transcript" \
            --ref_order ${CONFIG_DIR}/global_anchor_gene_order.txt \
            --tx2gene_mapping /home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/ens_genes_v115.txt \
            --mapping_json ${CONFIG_DIR}/global_species_id_mapping.json \
            --sample_run_id "$TUMOR_RUN" \
            --patient_id "$patient_safe" \
            --mode ${TRACE_MODE} \
            --batch_size 5 \
            --device cuda \
            1> ${PATIENT_MUT_DIR}/trace_prediction.log 2>&1
    else
        echo "[Skip] No mutation transcripts found for $patient_safe"
    fi

    # ---------------------------------------------------------------
    # Step 3: Add TRACE ORFs to GTF
    # ---------------------------------------------------------------
    ENHANCED_GTF=${ANNOTATION_DIR}/${patient_safe}_enhanced.gtf
    if [ -f "$ENHANCED_GTF" ]; then
        echo "[Skip] Enhanced GTF already exists"
    elif [ -s "$TRACE_ORF_CSV" ]; then
        echo "-> [3/7] Adding TRACE ORFs to GTF"
        python ${SCRIPT_DIR}/add_trace_orfs_to_gtf.py \
            --trace_orf_csv ${TRACE_ORF_CSV} \
            --extra_gtf ${WORK_DIR}/assembly/final_filtered_novel_transcripts_enhanced.gtf \            --ref_gtf ${GTF_FILE} \
            --output_gtf ${ENHANCED_GTF}
    else
        echo "[Skip] No TRACE ORFs; using reference GTF"
        ENHANCED_GTF=${GTF_FILE}
    fi

    # ---------------------------------------------------------------
    # Step 4: Annotate variants against enhanced GTF
    # ---------------------------------------------------------------
    ANNOTATED_CSV=${ANNOTATION_DIR}/${patient_safe}_annotated_variants.csv
    if [ -f "$ANNOTATED_CSV" ]; then
        echo "[Skip] Variants already annotated"
    else
        echo "-> [4/7] Annotating variants (coding impact)"
        python ${SCRIPT_DIR}/annotate_mutation_variants.py \
            --vcf ${VCF_IN} \
            --gtf ${ENHANCED_GTF} \
            --output ${ANNOTATED_CSV}
    fi

    # ---------------------------------------------------------------
    # Step 5: Generate 21aa mutant peptide windows
    # ---------------------------------------------------------------
    PATIENT_PEP_DIR=${MUT_PEPTIDE_DIR}/${patient_safe}
    mkdir -p ${PATIENT_PEP_DIR}
    MUT_PEP_FASTA=${PATIENT_PEP_DIR}/mutant_peptides.fasta
    MUT_PEP_CSV=${PATIENT_PEP_DIR}/mutant_peptide_map.csv

    if [ -f "$MUT_PEP_FASTA" ]; then
        echo "[Skip] Mutant peptides already generated"
    elif [ -s "$ANNOTATED_CSV" ]; then
        echo "-> [5/7] Generating 21aa mutant peptide windows"
        python ${SCRIPT_DIR}/generate_mutant_peptides.py \
            --annotated_csv ${ANNOTATED_CSV} \
            --trace_fasta ${TRACE_PROT_FASTA} \
            --canonical_fasta ${CANONICAL_PROTEOME} \
            --output_fasta ${MUT_PEP_FASTA} \
            --output_csv ${MUT_PEP_CSV} \
            --window 21
    else
        echo "[Skip] No annotated variants"
    fi

    # ---------------------------------------------------------------
    # Step 6: netMHCpan HLA binding prediction
    # ---------------------------------------------------------------
    alleles=("$hla_a1" "$hla_a2" "$hla_b1" "$hla_b2" "$hla_c1" "$hla_c2")
    formatted_hlas=""
    for allele in "${alleles[@]}"; do
        [ -z "$allele" ] && continue
        fmt="HLA-${allele//\*/}"
        formatted_hlas+="$fmt,"
    done
    UNIQUE_HLAS=$(echo "$formatted_hlas" | tr ',' '\n' | sort -u | grep -v "^$" | paste -sd, -)

    if [ -s "${PATIENT_PEP_DIR}/netMHCpan.log" ]; then
        echo "[Skip] netMHCpan already done"
    elif [ -s "$MUT_PEP_FASTA" ]; then
        echo "-> [6/7] netMHCpan prediction (HLA: $UNIQUE_HLAS)"
        netMHCpan -s -BA -t 10 \
            -a "$UNIQUE_HLAS" \
            -xls -xlsfile ${PATIENT_PEP_DIR}/netMHCpan_results.xls \
            -f ${MUT_PEP_FASTA} \
            1> ${PATIENT_PEP_DIR}/netMHCpan.log 2>&1
    else
        echo "[Skip] No mutant peptides for netMHCpan"
    fi

    # ---------------------------------------------------------------
    # Step 7: Integration report
    # ---------------------------------------------------------------
    PATIENT_REPORT=${MUT_REPORT_DIR}/${patient_safe}.csv
    if [ -f "$PATIENT_REPORT" ]; then
        echo "[Skip] Report already exists"
    elif [ -s "${PATIENT_PEP_DIR}/netMHCpan.log" ] && [ -s "$TRACE_ORF_CSV" ]; then
        echo "-> [7/7] Generating neoantigen report"
        python ${SCRIPT_DIR}/mutation_neoantigen_report.py \
            --mutation_csv ${ANNOTATED_CSV} \
            --peptide_csv ${MUT_PEP_CSV} \
            --trace_csv ${TRACE_ORF_CSV} \
            --netmhcpan_log ${PATIENT_PEP_DIR}/netMHCpan.log \
            --tpm_csv ${TPM_MATRIX} \
            --patient_id ${patient_safe} \
            --tumor_run_id ${TUMOR_RUN} \
            --output ${PATIENT_REPORT} \
            --bind_levels SB WB \
            --max_aff_nm 2000 \
            --max_rank_el 5.0
    else
        echo "[Skip] Missing inputs for report"
    fi

    echo "=> $patient_safe complete."
    echo ""
done

echo "=============================================="
echo "==== All mutation neoantigen pipelines done =="
echo "=============================================="
echo "Reports: ${MUT_REPORT_DIR}/"

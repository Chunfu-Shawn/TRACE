#!/bin/bash

WORK_DIR=/home/user/data3/rbase/translation_model/neoantigen/samples
SCRIPT_DIR=/home/user/data3/rbase/translation_model/models/tools/tumor_neoantigen
GTF_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf
FA_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/Homo_sapiens.GRCh38.primary_assembly.genome.fa
GENOME_INDEX=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/genome_index_v48_150nt
BED_FILE=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.bed
projects=(cohort_2)

for project in ${projects[@]};
do
    echo "##### $project #####"

    echo "----- Quality control for fastq data -----"
    bash $SCRIPT_DIR/run.quality_control.sh \
        --fastqDir $WORK_DIR/$project/fastq --file_suffix .fastq.gz \
        --outputDir $WORK_DIR/$project/fastqc \
        1>$WORK_DIR/$project/run.quality_control.log 2>&1

    echo "----- Filter low-quality reads and remove adapters and QC -----"
    bash $SCRIPT_DIR/run.fastp.sh \
        --fastqDir $WORK_DIR/$project/fastq --outputDir $WORK_DIR/$project/fastq_clean \
        1>$WORK_DIR/$project/run.fastp.log 2>&1
    
    echo "----- Quality control for clean fastq data -----"
    bash $SCRIPT_DIR/run.quality_control.sh \
        --fastqDir $WORK_DIR/$project/fastq_clean --file_suffix .clean.fastq.gz \
        --outputDir $WORK_DIR/$project/fastqc_clean \
        1>$WORK_DIR/$project/run.quality_control_clean.log 2>&1
    
    echo "----- Align reads to genome -----"
    bash $SCRIPT_DIR/run.mapping_uniq.sh \
        --fastqDir $WORK_DIR/$project/fastq_clean --file_suffix _1.clean.fastq.gz \
        --outputDir $WORK_DIR/$project/bam \
        --annoIndex  $GENOME_INDEX --removeRawBam yes \
        1>$WORK_DIR/$project/run.mapping_uniq.log 2>&1

    echo "----- Transcript assembly (StringTie) -----"
    bash $SCRIPT_DIR/run.stringtie.sh \
        --bamDir $WORK_DIR/$project/bam \
        --outputDir $WORK_DIR/$project/assembly \
        --refGTF $GTF_FILE \
        --refFasta $FA_FILE \
        --threads_per_job 10 \
        1>$WORK_DIR/$project/run.stringtie.log 2>&1

    echo "----- Add transcripts of de novo genes  -----"
    bash $SCRIPT_DIR/prepare_combined_targets.sh \
        --denovo_enst_gtf /home/user/data3/rbase/small_peptide/denovo_genes/gtf/denovo_genes.hominoid_NEE_CG.gtf \
        --pacbio_gtf /home/user/data3/rbase/small_peptide/denovo_genes/gtf/denovo_genes.hominoid.gtf \
        --pacbio_class /home/user/data3/lit/project/sORFs/08-Iso-seq-20250717/processed/classify/collapsed_classification.filtered_lite_classification.txt \
        --ref_gtf $GTF_FILE \
        --quant_target $WORK_DIR/$project/assembly/final_quantification_targets.gtf \
        --intact_target $WORK_DIR/$project/assembly/final_filtered_novel_transcripts.gtf \
        1> $WORK_DIR/$project/run.prepare_targets.log 2>&1

    echo "----- Transcript expression quantification (featureCounts) -----"
    bash $SCRIPT_DIR/run.featurecounts.sh \
        --bamDir $WORK_DIR/$project/bam \
        --refGTF /home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro_denovo_removed.gtf \
        --intactNovelGTF $WORK_DIR/$project/assembly/final_filtered_novel_transcripts_enhanced.gtf \
        --outputDir $WORK_DIR/$project/featureCounts_tumor \
        --threads 20 \
        --auto_strand_bed $BED_FILE \
        1> $WORK_DIR/$project/run.featureCounts.log 2>&1
done
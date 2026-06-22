#!/bin/bash 
set -euo pipefail

# Default parameters
threads=40

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --bam_dir)        bamDir="$2"; shift ;;
        --work_dir)       workDir="$2"; shift ;;
        --quant_target_gtf) quantTargetGTF="$2"; shift ;; 
        --threads)        threads="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "${bamDir:-}" ] || [ -z "${workDir:-}" ] || [ -z "${quantTargetGTF:-}" ]; then
    echo "Usage: $0 --bam_dir <dir> --work_dir <dir> --quant_target_gtf <file> [--threads <int>]"
    exit 1
fi

[ -d "$workDir" ] || mkdir -p "$workDir"

echo "=========================================================="
echo "### Step 1: Collecting GTEx BAM files (Top 50/Tissue)  ###"
echo "=========================================================="
tissues=(Brain_Cortex Brain_Frontal_Cortex_BA9 brain_hippocampus brain_amygdala Brain_Hypothalamus Brain_Cerebellum Brain_Cerebellar_Hemisphere Esophagus_Gastroesophageal_Junction esophagus_mucosa thyroid Heart_Left_Ventricle Breast_Mammary_Tissue Lung stomach pancreas Liver Kidney_Cortex Kidney_Medulla Colon_Sigmoid Colon_Transverse bladder skin_sun_exposed whole_blood Ovary Cervix_Ectocervix cervix_endocervix uterus Prostate Testis)

bams=""
total_bams=0

for tissue in "${tissues[@]}"; do
    if [ -d "$bamDir/$tissue/uniq_bam" ]; then
        bam_list=($(ls -S "$bamDir/$tissue/uniq_bam/"*_uniq.sorted.bam | head -n 50))
        for bam in "${bam_list[@]}"; do
            bams+="$bam "
            total_bams=$((total_bams + 1))
        done
        echo " -> Collected ${bam_list[@]} BAMs for $tissue"
    else
        echo " -> [Warning] Directory not found: $bamDir/$tissue/uniq_bam"
    fi
done

echo "Total GTEx BAM files collected: $total_bams"


echo "=========================================================="
echo "### Step 2: Ultra-Fast featureCounts (Strictly Novel Features Only) ###"
echo "=========================================================="
counts_out="$workDir/gtex_novel_transcript_counts.txt"

if [ -s "$counts_out" ]; then
    echo "[Skip] Transcript counts already generated: $counts_out"
else
    echo "Starting ultra-high-speed featureCounts against pure abnormal fragments..."
    
    time featureCounts \
        -p --countReadPairs \
        -t exon \
        -g transcript_id \
        -s 0 \
        -T "$threads" \
        -a "$quantTargetGTF" \
        -o "$counts_out" \
        $bams > "$workDir/featureCounts_gtex.log" 2>&1
        
    echo "✅ GTEx featureCounts completed successfully!" 
fi
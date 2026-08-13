#!/bin/bash 
set -euo pipefail

# Default parameters
threads=40
strand=0

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --bam_dir)        bamDir="$2"; shift ;;
        --work_dir)       workDir="$2"; shift ;;
        --annotation_gtf) annotationGTF="$2"; shift ;;
        --quant_target_gtf) annotationGTF="$2"; shift ;;
        --threads)        threads="$2"; shift ;;
        --strand)         strand="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "${bamDir:-}" ] || [ -z "${workDir:-}" ] || [ -z "${annotationGTF:-}" ]; then
    echo "Usage: $0 --bam_dir <dir> --work_dir <dir> --annotation_gtf <file> [--threads <int>] [--strand 0|1|2]"
    exit 1
fi

case "$strand" in
    0|1|2) ;;
    *) echo "Error: --strand must be 0, 1, or 2."; exit 1;;
esac

[ -d "$workDir" ] || mkdir -p "$workDir"

echo "=========================================================="
echo "### Step 1: Collecting GTEx BAM files (Top 50/Tissue)  ###"
echo "=========================================================="
tissues=(Brain_Cortex Brain_Frontal_Cortex_BA9 brain_hippocampus brain_amygdala Brain_Hypothalamus Brain_Cerebellum Brain_Cerebellar_Hemisphere Esophagus_Gastroesophageal_Junction esophagus_mucosa thyroid Heart_Left_Ventricle Breast_Mammary_Tissue Lung stomach pancreas Liver Kidney_Cortex Kidney_Medulla Colon_Sigmoid Colon_Transverse bladder skin_sun_exposed whole_blood Ovary Cervix_Ectocervix cervix_endocervix uterus Prostate Testis)

bams=""
total_bams=0

for tissue in "${tissues[@]}"; do
    if [ -d "$bamDir/$tissue/uniq_bam" ]; then
        bam_list=()
        while IFS= read -r bam; do
            bam_list+=("$bam")
        done < <(find "$bamDir/$tissue/uniq_bam" -maxdepth 1 -type f \
            -name '*_uniq.sorted.bam' -printf '%s\t%p\n' | sort -rn | head -n 50 | cut -f2-)
        for bam in "${bam_list[@]:-}"; do
            [ -n "$bam" ] || continue
            bams+="$bam "
            total_bams=$((total_bams + 1))
        done
        echo " -> Collected ${#bam_list[@]} BAMs for $tissue"
    else
        echo " -> [Warning] Directory not found: $bamDir/$tissue/uniq_bam"
    fi
done

echo "Total GTEx BAM files collected: $total_bams"
if (( total_bams == 0 )); then
    echo "Error: No GTEx BAM files were collected." >&2
    exit 1
fi


echo "=========================================================="
echo "### Step 2: Complete-GTF featureCounts with Multi-Overlap ###"
echo "=========================================================="
counts_out="$workDir/gtex_transcript_counts.complete_gtf.multioverlap.txt"

if [ -s "$counts_out" ]; then
    echo "[Skip] Transcript counts already generated: $counts_out"
else
    echo "Starting featureCounts against the complete transcript annotation with -O..."
    
    time featureCounts \
        -p -O -B --countReadPairs \
        -t exon \
        -g transcript_id \
        -s "$strand" \
        -T "$threads" \
        -a "$annotationGTF" \
        -o "$counts_out" \
        $bams > "$workDir/featureCounts_gtex.complete_gtf.multioverlap.log" 2>&1
        
    echo "✅ GTEx featureCounts completed successfully!" 
fi

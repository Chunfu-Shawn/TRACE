#!/bin/sh 
set -euo pipefail

## Argument Parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --bamDir)         bamDir=$2;shift;;        
        --refGTF)         refGTF=$2;shift;;
        --quantTargetGTF) quantTargetGTF=$2;shift;; # Track A: Subtracted fragments
        --intactNovelGTF) intactNovelGTF=$2;shift;; # Track B: Structurally intact transcripts
        --outputDir)      outputDir=$2;shift;;     
        --threads)        threads=$2;shift;;       
        --)               shift; break;;
        *)                echo -e "\n[ERR] $(date) Unknown option: $1"; exit 1;;
    esac
    shift
done

threads=${threads:-20}

if [ -z "${bamDir:-}" ] || [ -z "${refGTF:-}" ] || [ -z "${quantTargetGTF:-}" ] || [ -z "${intactNovelGTF:-}" ] || [ -z "${outputDir:-}" ]; then
    echo "Error: Missing required parameters."
    echo "Usage: bash run_featurecounts.sh --bamDir <dir> --refGTF <gtf> --quantTargetGTF <gtf> --intactNovelGTF <gtf> --outputDir <dir>"
    exit 1
fi

[ -d $outputDir ] || mkdir -p $outputDir

# ---------------------------------------------------------
# Phase 1: Build Dual-Track Annotations
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 1: Preparing Dual-Track GTFs ###"
echo "=========================================================="
# Track A GTF: Used for Expression Quantification (TPM) to avoid overlap penalties
trackA_gtf="${outputDir}/TrackA_Quantification.gtf"
# Track B GTF: Used for Junction Annotation to preserve exact biological exon boundaries
trackB_gtf="${outputDir}/TrackB_IntactStructure.gtf"

if [ ! -s "$trackA_gtf" ]; then
    echo "Building Track A GTF (Ref + Subtracted Novel)..."
    cat $refGTF $quantTargetGTF > $trackA_gtf
fi

if [ ! -s "$trackB_gtf" ]; then
    echo "Building Track B GTF (Ref + Intact Novel)..."
    cat $refGTF $intactNovelGTF > $trackB_gtf
fi

echo "Dual-Track GTFs generated successfully."

# ---------------------------------------------------------
# Phase 2: Gather all BAM files
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 2: Gathering BAM files ###"
echo "=========================================================="
bam_files=$(find ${bamDir} -type f -name "*.uniq.sorted.bam" | sort)
bam_count=$(echo "$bam_files" | wc -w)

if [ "$bam_count" -eq 0 ]; then
    echo "Error: No .uniq.sorted.bam files found in $bamDir"
    exit 1
fi
echo "Found $bam_count BAM files to process."

# ---------------------------------------------------------
# Phase 3: Execute Dual-Track featureCounts
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 3: Running Dual-Track featureCounts ###"
echo "=========================================================="
counts_tx="${outputDir}/transcript_counts.txt"
gene_counts_tx="${outputDir}/gene_counts.txt"
counts_junc="${outputDir}/junction_counts.txt"

# ---------------------------------------------------------
# [Run 1] Track A: Expression Quantification (No Junctions)
# ---------------------------------------------------------
if [ -s "$counts_tx" ]; then
    echo "[Skip] Track A (TPM) counts already exist."
else
    echo "-> Running Track A: Transcript-level quantification..."
    # allow multi-overlap
    time featureCounts \
        -p -O -B --countReadPairs \
        -T $threads \
        -t exon \
        -g transcript_id \
        -s 2 \
        -a $trackA_gtf \
        -o $counts_tx \
        $bam_files
fi

if [ -s "$gene_counts_tx" ]; then
    echo "[Skip] Gene counts already exist."
else
    echo "-> Gene-level quantification..."
    time featureCounts \
        -p -B --countReadPairs \
        -T $threads \
        -t exon \
        -g gene_id \
        -s 2 \
        -a $trackB_gtf \
        -o $gene_counts_tx \
        $bam_files
fi

# ---------------------------------------------------------
# [Run 2] Track B: Junction Extraction & Annotation
# ---------------------------------------------------------
if [ -s "${counts_junc}.jcounts" ]; then
    echo "[Skip] Track B (Junction) counts already exist."
else
    echo "-> Running Track B: Junction Extraction (Using Intact GTF)..."
    time featureCounts \
        -p -B \
        -T $threads \
        -t exon \
        -g transcript_id \
        -s 2 \
        -J \
        -a $trackB_gtf \
        -o $counts_junc \
        $bam_files
        
    # We only care about the .jcounts file from this run. 
    # The normal counts output from this intact GTF will suffer from overlap penalties, so we delete it.
    rm -f $counts_junc
    rm -f "${counts_junc}.summary"
fi

echo "=========================================================="
echo "All done! Pipeline finished successfully."
echo "Track A (Read Counts) saved to: ${counts_tx}"
echo "Track B (Junction Counts) saved to: ${counts_junc}.jcounts"
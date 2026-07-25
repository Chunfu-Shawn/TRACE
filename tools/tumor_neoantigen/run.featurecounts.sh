#!/bin/bash
set -euo pipefail

## Argument Parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --bamDir)         bamDir=$2;shift;;        
        --refGTF)         refGTF=$2;shift;;
        --intactNovelGTF) intactNovelGTF=$2;shift;; # Intact tumor transcript annotation
        --outputDir)      outputDir=$2;shift;;     
        --threads)        threads=$2;shift;;       
        --strand)         STRAND_FLAG=$2;shift;;
        --auto_strand_bed) auto_strand_bed=$2;shift;;
        --)               shift; break;;
        *)                echo -e "\n[ERR] $(date) Unknown option: $1"; exit 1;;
    esac
    shift
done

threads=${threads:-20}

if [ -z "${bamDir:-}" ] || [ -z "${refGTF:-}" ] || [ -z "${intactNovelGTF:-}" ] || [ -z "${outputDir:-}" ]; then
    echo "Error: Missing required parameters."
    echo "Usage: bash run.featurecounts.sh --bamDir <dir> --refGTF <gtf> --intactNovelGTF <gtf> --outputDir <dir> [--strand 0|1|2]"
    exit 1
fi

[ -d $outputDir ] || mkdir -p $outputDir

# ---------------------------------------------------------
# Phase 1: Build one complete annotation for every counting layer
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 1: Preparing Complete Transcript Annotation ###"
echo "=========================================================="
complete_gtf="${outputDir}/complete_transcript_annotation.gtf"

if [ ! -s "$complete_gtf" ]; then
    echo "Building complete GTF (reference + intact tumor targets)..."
    cat "$refGTF" "$intactNovelGTF" > "$complete_gtf"
fi

echo "Complete annotation generated successfully: $complete_gtf"

# ---------------------------------------------------------
# Phase 2: Gather all BAM files
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 2: Gathering BAM files ###"
echo "=========================================================="
mapfile -t discovered_bam_files < <(find "${bamDir}" -type f -name "*.uniq.sorted.bam" | sort)
bam_files=()
for bam_file in "${discovered_bam_files[@]}"; do
    paired_alignment_count=$(samtools view -c -f 1 "$bam_file")
    if (( paired_alignment_count > 0 )); then
        bam_files+=("$bam_file")
    else
        echo "[Skip] Single-end BAM is excluded by preprocessing policy: $bam_file"
    fi
done
bam_count=${#bam_files[@]}

if [ "$bam_count" -eq 0 ]; then
    echo "Error: No paired-end .uniq.sorted.bam files found in $bamDir"
    exit 1
fi
echo "Found $bam_count paired-end BAM files to process."
library_count_args=(-p -B --countReadPairs)


# ---------------------------------------------------------
# Phase 2.5: Resolve one strand mode for all counting layers
# ---------------------------------------------------------
if [ -n "${STRAND_FLAG:-}" ]; then
    case "$STRAND_FLAG" in
        0|1|2) ;;
        *) echo "Error: --strand must be 0, 1, or 2."; exit 1;;
    esac
    echo "-> Using explicit strandness: featureCounts -s $STRAND_FLAG"
elif [ -n "${auto_strand_bed:-}" ]; then
    echo "=========================================================="
    echo "### Phase 2.5: Auto-detecting strandness ###"
    echo "=========================================================="
    first_bam="${bam_files[0]}"
    echo "-> Running infer_experiment.py on: $first_bam"
    strand_output=$(infer_experiment.py -r "$auto_strand_bed" -i "$first_bam" -s 1000000 2>/dev/null || true)
    fwd_frac=$(echo "$strand_output" | grep '1++,1--,2+-,2-+' | sed 's/.*: //' || echo "0")
    rev_frac=$(echo "$strand_output" | grep '1+-,1-+,2++,2--' | sed 's/.*: //' || echo "0")
    STRAND_FLAG=$(awk -v fwd="$fwd_frac" -v rev="$rev_frac" 'BEGIN {
        if (fwd + rev > 0) { ratio = fwd / (fwd + rev); }
        else { ratio = 0.5; }
        if (ratio > 0.7) print 1;
        else if (ratio < 0.3) print 2;
        else print 0;
    }')
    echo "-> Strand fractions: fwd=$fwd_frac rev=$rev_frac → featureCounts -s $STRAND_FLAG"
else
    STRAND_FLAG=0
    echo "-> Using default strandness: featureCounts -s $STRAND_FLAG"
fi

# ---------------------------------------------------------
# Phase 3: Execute Dual-Track featureCounts
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 3: Running Dual-Track featureCounts ###"
echo "=========================================================="
counts_tx="${outputDir}/transcript_counts.complete_gtf.multioverlap.txt"
gene_counts_tx="${outputDir}/gene_counts.complete_gtf.multioverlap.txt"
counts_junc="${outputDir}/junction_counts.complete_gtf.multioverlap.txt"

# ---------------------------------------------------------
# [Run 1] Track A: Expression Quantification (No Junctions)
# ---------------------------------------------------------
if [ -s "$counts_tx" ]; then
    echo "[Skip] Track A (TPM) counts already exist."
else
    echo "-> Running Track A: Transcript-level quantification..."
    # allow multi-overlap
    time featureCounts \
        "${library_count_args[@]}" -O \
        -T $threads \
        -t exon \
        -g transcript_id \
        -s $STRAND_FLAG \
        -a "$complete_gtf" \
        -o $counts_tx \
        "${bam_files[@]}"
fi

if [ -s "$gene_counts_tx" ]; then
    echo "[Skip] Gene counts already exist."
else
    echo "-> Gene-level quantification..."
    time featureCounts \
        "${library_count_args[@]}" -O \
        -T $threads \
        -t exon \
        -g gene_id \
        -s $STRAND_FLAG \
        -a "$complete_gtf" \
        -o $gene_counts_tx \
        "${bam_files[@]}"
fi

# ---------------------------------------------------------
# [Run 2] Track B: Junction Extraction & Annotation
# ---------------------------------------------------------
if [ -s "${counts_junc}.jcounts" ]; then
    echo "[Skip] Track B (Junction) counts already exist."
else
    echo "-> Running Track B: Junction Extraction (Using Intact GTF)..."
    time featureCounts \
        "${library_count_args[@]}" -O \
        -T $threads \
        -t exon \
        -g transcript_id \
        -s $STRAND_FLAG \
        -J \
        -a "$complete_gtf" \
        -o $counts_junc \
        "${bam_files[@]}"
        
    # Only the .jcounts file is consumed downstream; remove the auxiliary feature table.
    rm -f $counts_junc
    rm -f "${counts_junc}.summary"
fi

echo "=========================================================="
echo "All done! Pipeline finished successfully."
echo "Track A (Read Counts) saved to: ${counts_tx}"
echo "Track B (Junction Counts) saved to: ${counts_junc}.jcounts"

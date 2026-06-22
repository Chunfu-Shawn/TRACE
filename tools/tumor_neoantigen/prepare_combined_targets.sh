#!/bin/bash
set -euo pipefail

# =================================================================
# Script: prepare_combined_targets.sh
# Purpose: Integrate de novo genes and PacBio Iso-Seq transcripts.
#          Generates TWO distinct target GTFs for Dual-Track pipeline:
#          1. Track A (Quantification): Subtracted & fragmented.
#          2. Track B (Intact Structure): Structurally complete for junctions.
#          Features dual-key locus eviction (ENSG + Gene Symbol).
# =================================================================

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --denovo_enst_gtf) DENOVO_ENST_GTF=$2; shift;;
        --pacbio_gtf)      PACBIO_GTF=$2; shift;;
        --pacbio_class)    PACBIO_CLASS=$2; shift;;
        --ref_gtf)         REF_GTF_IN=$2; shift;;
        --quant_target)    QUANT_TARGET_IN=$2; shift;;  # Track A Input (StringTie subtracted)
        --intact_target)   INTACT_TARGET_IN=$2; shift;; # Track B Input (StringTie intact)
        --)                shift; break;;
        *)                 echo -e "Unknown option: $1"; exit 1;;
    esac
    shift
done

if [ -z "${DENOVO_ENST_GTF:-}" ] || [ -z "${PACBIO_GTF:-}" ] || [ -z "${PACBIO_CLASS:-}" ] || [ -z "${INTACT_TARGET_IN:-}" ]; then
    echo "Error: Missing required parameters."
    exit 1
fi

REF_GTF_OUT="${REF_GTF_IN%.gtf}_denovo_removed.gtf"
QUANT_TARGET_OUT="${QUANT_TARGET_IN%.gtf}_enhanced.gtf"   # Goes to Track A
INTACT_TARGET_OUT="${INTACT_TARGET_IN%.gtf}_enhanced.gtf" # Goes to Track B

echo "=========================================================="
echo "### Module 1: Building the Master Reference Blacklist ###"
echo "=========================================================="

echo "Step 1.1: Extracting de novo ENSG IDs..."
awk '$3 == "transcript" || $3 == "exon" {
    for(i=9; i<=NF; i++) {
        if ($i == "gene_id") {
            gid=$(i+1); gsub(/"|;/, "", gid); 
            split(gid, a, "."); print a[1]; break;
        }
    }
}' "$DENOVO_ENST_GTF" | sort | uniq > "blacklist_part1.tmp.txt"

echo "Step 1.2: Identifying valid PacBio transcripts present in GTF..."
awk '$3 == "exon" || $3 == "transcript" {
    for(i=9; i<=NF; i++) {
        if ($i == "transcript_id") {
            tid=$(i+1); gsub(/"|;/, "", tid);
            split(tid, a, ":"); print a[1]; break;
        }
    }
}' "$PACBIO_GTF" | sort | uniq > "valid_pb_in_gtf.tmp.txt"

echo "Step 1.3: Extracting PacBio FSM corresponding Gene IDs/Symbols..."
# Extract 'associated_gene' (Column 7). This could be an ENSG or a Gene Symbol (e.g., GAPDH).
awk 'BEGIN{FS="\t"} 
NR==FNR {valid_pb[$1]=1; next} 
FNR>1 && $6 == "full-splice_match" {
    if ($1 in valid_pb) {
        gid=$7; 
        split(gid, a, "."); print a[1];
    }
}' "valid_pb_in_gtf.tmp.txt" "$PACBIO_CLASS" | sort | uniq > "blacklist_part2.tmp.txt"

echo "Step 1.4: Merging Master Blacklist..."
cat "blacklist_part1.tmp.txt" "blacklist_part2.tmp.txt" | sort | uniq > "master_blacklist.tmp.txt"
echo " -> Master Blacklist contains $(wc -l < "master_blacklist.tmp.txt") unique keys (ENSGs or Symbols) to evict."

echo "=========================================================="
echo "### Module 2: Updating the Reference Background ###"
echo "=========================================================="
# [CRITICAL UPGRADE]: Dual-Key Eviction. Checks BOTH 'gene_id' and 'gene_name' against the blacklist.
awk 'NR==FNR {blacklist[$1]=1; next} 
{
    if ($0 ~ /^#/) { print $0; next; } # Keep headers
    
    gid=""; gname="";
    for(i=9; i<=NF; i++) {
        if ($i == "gene_id") {
            gid_full=$(i+1); gsub(/"|;/, "", gid_full); 
            split(gid_full, a, "."); gid=a[1];
        }
        else if ($i == "gene_name") {
            gname_full=$(i+1); gsub(/"|;/, "", gname_full); 
            gname=gname_full;
        }
    }
    
    # If EITHER the gene_id OR the gene_name is in the blacklist, we drop the line.
    if (!(gid in blacklist) && !(gname in blacklist)) { 
        print $0; 
    }
}' "master_blacklist.tmp.txt" "$REF_GTF_IN" > "$REF_GTF_OUT"
echo " -> Dual-Key Gene-level eviction complete! Cleaned reference GTF saved."

echo "=========================================================="
echo "### Module 3: Extracting Target Sequences ###"
echo "=========================================================="

echo "Step 3.1: Extracting ENST de novo exons (Intact by default)..."
awk '$3 == "exon"' "$DENOVO_ENST_GTF" > "targets_enst_denovo_intact.tmp.gtf"

echo "Step 3.2: Parsing PacBio Subgroups (FSM vs Non-FSM)..."
awk 'BEGIN{FS="\t"} 
NR==FNR {valid_pb[$1]=1; next} 
FNR>1 && $1 in valid_pb {
    if ($6 == "full-splice_match") print $1 > "pb_fsm_ids.tmp.txt";
    else print $1 > "pb_non_fsm_ids.tmp.txt";
}' "valid_pb_in_gtf.tmp.txt" "$PACBIO_CLASS"

touch "pb_fsm_ids.tmp.txt" "pb_non_fsm_ids.tmp.txt"

echo "Step 3.3: Promoting PacBio FSM targets (Intact by default)..."
awk 'NR==FNR {wanted[$1]=1; next}
$3 == "exon" {
    for(i=9; i<=NF; i++) {
        if ($i == "transcript_id") {
            tid=$(i+1); gsub(/"|;/, "", tid); 
            split(tid, a, ":"); base_tid=a[1];
            if (base_tid in wanted) print $0;
            break;
        }
    }
}' "pb_fsm_ids.tmp.txt" "$PACBIO_GTF" > "targets_pb_fsm_intact.tmp.gtf"

echo "Step 3.4: Processing PacBio Non-FSM targets..."
awk 'NR==FNR {wanted[$1]=1; next}
$3 == "exon" {
    for(i=9; i<=NF; i++) {
        if ($i == "transcript_id") {
            tid=$(i+1); gsub(/"|;/, "", tid); 
            split(tid, a, ":"); base_tid=a[1];
            if (base_tid in wanted) print $0;
            break;
        }
    }
}' "pb_non_fsm_ids.tmp.txt" "$PACBIO_GTF" > "targets_pb_non_fsm_intact.tmp.gtf"

# Now we perform subtraction ONLY for Track A's version
if [ -s "targets_pb_non_fsm_intact.tmp.gtf" ]; then
    bedtools subtract \
        -a "targets_pb_non_fsm_intact.tmp.gtf" \
        -b "$REF_GTF_OUT" \
        -s > "pb_non_fsm_subtracted.tmp.gtf"

    awk -F'\t' 'BEGIN {OFS="\t"} 
    { 
        feat_len = $5 - $4 + 1;
        if (feat_len >= 50) {
            $3="exon"; print $0;
        }
    }' "pb_non_fsm_subtracted.tmp.gtf" > "targets_pb_non_fsm_subtracted.tmp.gtf"
else
    touch "targets_pb_non_fsm_subtracted.tmp.gtf"
fi

echo "=========================================================="
echo "### Module 4: Dual-Track Final Consolidation ###"
echo "=========================================================="

echo "Merging Track A Targets (Subtracted/Fragmented)..."
cat "$QUANT_TARGET_IN" \
    "targets_enst_denovo_intact.tmp.gtf" \
    "targets_pb_fsm_intact.tmp.gtf" \
    "targets_pb_non_fsm_subtracted.tmp.gtf" > "$QUANT_TARGET_OUT"

echo "Merging Track B Targets (Intact/Structural)..."
cat "$INTACT_TARGET_IN" \
    "targets_enst_denovo_intact.tmp.gtf" \
    "targets_pb_fsm_intact.tmp.gtf" \
    "targets_pb_non_fsm_intact.tmp.gtf" > "$INTACT_TARGET_OUT"

# Clean up
rm -f *.tmp.txt *.tmp.gtf

echo "✅ Pipeline Complete! Dual-Track target files successfully generated."
echo "----------------------------------------------------------"
echo "Use in run_featurecounts.sh:"
echo "  --refGTF         : $REF_GTF_OUT"
echo "  --quantTargetGTF : $QUANT_TARGET_OUT (For TPM Run)"
echo "  --intactNovelGTF : $INTACT_TARGET_OUT (For Junction Run)"
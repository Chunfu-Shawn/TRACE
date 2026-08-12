#!/bin/bash
set -Eeuo pipefail

################################################
#File Name: run.fastqc.analysis.sh
#Author: rbase    
#Mail: xiaochunfu@stu.pku.edu.cn
#Modified: Added auto-detection for SE/PE compatibility
################################################

# Run a bounded number of FastQC jobs without a FIFO token pool.
# A killed worker cannot permanently block this batch-based scheduler.
thread_num=3

## Argument
file_suffix=".fastq.gz" # Default suffix when the option is omitted.
while [[ $# -gt 0 ]]; do
    case $1 in
        --fastqDir)              fastqDir=$2;shift;;
        --file_suffix)           file_suffix=$2;shift;;
        --outputDir)             outputDir=$2;shift;;
        --)                      shift; break;;
        *)                       echo -e "\n[ERR] $(date) Unknown option: $1"; exit 1;;
    esac
    shift
done

if [ -z "${fastqDir:-}" ] || [ -z "${outputDir:-}" ]; then
    echo "[Error] --fastqDir and --outputDir are required." >&2
    exit 1
fi
if [ ! -d "$fastqDir" ]; then
    echo "[Error] FASTQ directory does not exist: $fastqDir" >&2
    exit 1
fi
if ! command -v fastqc >/dev/null 2>&1; then
    echo "[Error] fastqc is not available in PATH." >&2
    exit 1
fi
if ! command -v multiqc >/dev/null 2>&1; then
    echo "[Error] multiqc is not available in PATH." >&2
    exit 1
fi

mkdir -p "$outputDir"

# Discover sample prefixes without parsing ls output.
shopt -s nullglob
fastq_files=("${fastqDir}"/*"${file_suffix}")
if (( ${#fastq_files[@]} == 0 )); then
    echo "[Error] No *${file_suffix} files found in ${fastqDir}." >&2
    exit 1
fi

sample_names=()
for fastq_path in "${fastq_files[@]}"; do
    fastq_name=$(basename "$fastq_path")
    sample_name=${fastq_name%"${file_suffix}"}
    sample_name=$(printf '%s\n' "$sample_name" | sed -E 's/(_1|_2|_R1|_R2)$//')
    sample_names+=("$sample_name")
done
samples=()
while IFS= read -r sample_name; do
    samples+=("$sample_name")
done < <(printf '%s\n' "${sample_names[@]}" | sort -u)

report_path_for_fastq() {
    local fastq_path=$1
    local report_name
    report_name=$(basename "$fastq_path" | sed -E 's/\.(fastq|fq)(\.gz)?$//')_fastqc.html
    printf '%s/%s\n' "$outputDir" "$report_name"
}

run_fastqc_sample() {
    local sample=$1
    local fq1=""
    local fq2=""
    local report1
    local report2
    local missing_inputs=()

    echo "-- FastQC for $sample --"

    if [ -f "$fastqDir/${sample}_1${file_suffix}" ]; then
        fq1="$fastqDir/${sample}_1${file_suffix}"
        if [ ! -f "$fastqDir/${sample}_2${file_suffix}" ]; then
            echo "[Error] Missing mate file for $fq1" >&2
            return 1
        fi
        fq2="$fastqDir/${sample}_2${file_suffix}"
    elif [ -f "$fastqDir/${sample}_R1${file_suffix}" ]; then
        fq1="$fastqDir/${sample}_R1${file_suffix}"
        if [ ! -f "$fastqDir/${sample}_R2${file_suffix}" ]; then
            echo "[Error] Missing mate file for $fq1" >&2
            return 1
        fi
        fq2="$fastqDir/${sample}_R2${file_suffix}"
    elif [ -f "$fastqDir/${sample}_2${file_suffix}" ] || \
         [ -f "$fastqDir/${sample}_R2${file_suffix}" ]; then
        echo "[Error] Read-2 file exists without its read-1 mate for $sample" >&2
        return 1
    elif [ -f "$fastqDir/${sample}${file_suffix}" ]; then
        echo "[Skip] Single-end sample $sample is excluded by preprocessing policy."
        return 0
    else
        echo "[Error] Could not resolve FASTQ files for $sample." >&2
        return 1
    fi

    report1=$(report_path_for_fastq "$fq1")
    report2=$(report_path_for_fastq "$fq2")
    if [ -s "$report1" ] && [ -s "$report2" ]; then
        echo "[Skip] Both FastQC reports already exist for $sample."
        return 0
    fi

    echo "[Info] Detected PE files for $sample"
    [ -s "$report1" ] || missing_inputs+=("$fq1")
    [ -s "$report2" ] || missing_inputs+=("$fq2")
    fastqc -o "$outputDir" -t 10 "${missing_inputs[@]}"
}

wait_for_batch() {
    local pid
    local batch_failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            batch_failed=1
        fi
    done
    pids=()
    if (( batch_failed != 0 )); then
        return 1
    fi
}

pids=()
for sample in "${samples[@]}"; do
    run_fastqc_sample "$sample" &
    pids+=("$!")
    if (( ${#pids[@]} == thread_num )); then
        if ! wait_for_batch; then
            echo "[Error] At least one FastQC job failed in the current batch." >&2
            exit 1
        fi
    fi
done

if ! wait_for_batch; then
    echo "[Error] At least one FastQC job failed in the final batch." >&2
    exit 1
fi

echo "-- Running MultiQC --"
multiqc "$outputDir" --outdir "$outputDir" --force
echo "All tasks finished!"

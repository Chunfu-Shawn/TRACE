#!/bin/bash
set -Eeuo pipefail

## Argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --fastqDir)       fastqDir=$2;shift;;
        --file_suffix)    file_suffix=$2;shift;; # Deprecated; input layout is auto-detected.
        --outputDir)      outputDir=$2;shift;;
        --annoIndex)      annoIndex=$2;shift;;
        --removeRawBam)   removeRawBam=$2;shift;;
        --)               shift; break;;
        *)                echo -e "\n[ERR] $(date) Unknown option: $1"; exit 1;;
    esac
    shift
done

# 并发数
thread_num=1

if [ -z "${fastqDir:-}" ] || [ -z "${outputDir:-}" ] || [ -z "${annoIndex:-}" ]; then
    echo "Error: Missing required directories."
    exit 1
fi

echo "### Mapping to genome by STAR (2-pass mode) ###"

[ -d $outputDir ] || mkdir -p $outputDir

# Discover paired-end and single-end cleaned FASTQ layouts in one pass.
shopt -s nullglob
fastq_files=("${fastqDir}"/*.clean.fastq.gz)
if (( ${#fastq_files[@]} == 0 )); then
    echo "Error: No *.clean.fastq.gz files found in ${fastqDir}." >&2
    exit 1
fi

sample_names=()
for fastq_path in "${fastq_files[@]}"; do
    fastq_name=$(basename "$fastq_path")
    case "$fastq_name" in
        *_1.clean.fastq.gz)  sample_names+=("${fastq_name%_1.clean.fastq.gz}") ;;
        *_2.clean.fastq.gz)  sample_names+=("${fastq_name%_2.clean.fastq.gz}") ;;
        *_R1.clean.fastq.gz) sample_names+=("${fastq_name%_R1.clean.fastq.gz}") ;;
        *_R2.clean.fastq.gz) sample_names+=("${fastq_name%_R2.clean.fastq.gz}") ;;
        *.clean.fastq.gz)    sample_names+=("${fastq_name%.clean.fastq.gz}") ;;
    esac
done
samples=()
while IFS= read -r sample_name; do
    samples+=("$sample_name")
done < <(printf '%s\n' "${sample_names[@]}" | sort -u)

pids=()
active_jobs=0

wait_for_batch() {
    local pid
    local batch_failed=0
    if (( active_jobs == 0 )); then
        return 0
    fi
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            batch_failed=1
        fi
    done
    pids=()
    active_jobs=0
    if (( batch_failed != 0 )); then
        return 1
    fi
}

for sample in "${samples[@]}";
do
    final_output="${outputDir}/${sample}/${sample}.uniq.sorted.bam"
    
    if [ -s "$final_output" ] && [ -s "${final_output}.bai" ]; then
        echo "Skip ${sample}, output exists."
        continue
    fi
    if [ -e "$final_output" ] || [ -e "${final_output}.bai" ]; then
        echo "[Warning] Incomplete BAM checkpoint found for ${sample}; rebuilding mapping output."
    fi
    
    {
        echo "-- processing $sample --"

        [ -d $outputDir/${sample} ] || mkdir -p $outputDir/${sample}
        cd $outputDir/${sample}

        input_files=()
        if [ -f "${fastqDir}/${sample}_1.clean.fastq.gz" ] && \
           [ -f "${fastqDir}/${sample}_2.clean.fastq.gz" ]; then
            input_files=(
                "${fastqDir}/${sample}_1.clean.fastq.gz"
                "${fastqDir}/${sample}_2.clean.fastq.gz"
            )
            echo "Detected paired-end data for $sample (_1/_2)."
        elif [ -f "${fastqDir}/${sample}_R1.clean.fastq.gz" ] && \
             [ -f "${fastqDir}/${sample}_R2.clean.fastq.gz" ]; then
            input_files=(
                "${fastqDir}/${sample}_R1.clean.fastq.gz"
                "${fastqDir}/${sample}_R2.clean.fastq.gz"
            )
            echo "Detected paired-end data for $sample (_R1/_R2)."
        elif [ -f "${fastqDir}/${sample}.clean.fastq.gz" ]; then
            echo "[Skip] Single-end sample $sample is excluded by preprocessing policy."
            exit 0
        else
            echo "Error: Incomplete or unsupported cleaned FASTQ layout for $sample." >&2
            exit 1
        fi

        ### Perform 2-pass mapping.
        # 优化点：直接利用 STAR 管道输出 BAM 并通过 samtools 排序
        temp_bam="${sample}.uniq.sorted.bam.incomplete"
        temp_bai="${temp_bam}.bai"
        trap 'rm -f "$temp_bam" "$temp_bai"' EXIT
        time STAR \
            --genomeDir $annoIndex \
            --readFilesIn "${input_files[@]}" \
            --readFilesCommand zcat \
            --twopassMode Basic \
            --runThreadN 20 \
            --outFilterMultimapScoreRange 1 \
            --outFilterMultimapNmax 1 \
            --outFilterMismatchNmax 10 \
            --outSJfilterOverhangMin 20 6 6 6 \
            --alignSJoverhangMin 4 \
            --alignSJDBoverhangMin 3 \
            --alignIntronMax 500000 \
            --sjdbScore 2 \
            --limitBAMsortRAM 30000000000 \
            --outFilterMatchNminOverLread 0.33 \
            --outFilterScoreMinOverLread 0.33 \
            --sjdbOverhang 149 \
            --outSAMstrandField intronMotif \
            --outSAMattributes All \
            --outSAMtype BAM Unsorted \
            --outStd BAM_Unsorted | \
            samtools sort -@ 20 -m 2G -o "$temp_bam"
        
        # index
        samtools index -@ 20 "$temp_bam"
        mv "$temp_bam" "${sample}.uniq.sorted.bam"
        mv "$temp_bai" "${sample}.uniq.sorted.bam.bai"
        
        # stat
        samtools flagstat -@ 20 ${sample}.uniq.sorted.bam > ${sample}.uniq.sorted.bam.flagstat
        trap - EXIT

    }&
    pids+=("$!")
    ((active_jobs += 1))
    if (( active_jobs == thread_num )); then
        if ! wait_for_batch; then
            echo "[ERROR] At least one STAR mapping job failed." >&2
            exit 1
        fi
    fi
done

if ! wait_for_batch; then
    echo "[ERROR] At least one STAR mapping job failed." >&2
    exit 1
fi
echo "All done!"

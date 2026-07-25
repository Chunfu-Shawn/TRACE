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

# mkfifo (并发控制)
tempfifo="my_temp_fifo_$$"
mkfifo ${tempfifo}
exec 6<>${tempfifo} 
rm -f ${tempfifo}

for ((i=1;i<=${thread_num};i++)); do
    echo
done >&6 

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
mapfile -t samples < <(printf '%s\n' "${sample_names[@]}" | sort -u)

pids=()
for sample in "${samples[@]}";
do
    final_output="${outputDir}/${sample}/${sample}.uniq.sorted.bam"
    
    if [ -f "$final_output" ]; then
        echo "Skip ${sample}, output exists."
        continue
    fi
    
    read -u6
    {
        trap 'echo >&6' EXIT
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
            samtools sort -@ 20 -m 2G -o ${sample}.uniq.sorted.bam
        
        # index
        samtools index -@ 20 ${sample}.uniq.sorted.bam
        
        # stat
        samtools flagstat -@ 20 ${sample}.uniq.sorted.bam > ${sample}.uniq.sorted.bam.flagstat

    }&
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed != 0 )); then
    echo "[ERROR] At least one STAR mapping job failed." >&2
    exit 1
fi
echo "All done!"

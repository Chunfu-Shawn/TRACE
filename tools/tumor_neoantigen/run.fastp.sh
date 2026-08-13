#!/bin/bash
set -Eeuo pipefail

################################################
#File Name: run.ribo-seq.analysis.sh
#Author: rbase    
#Mail: xiaochunfu@stu.pku.edu.cn
#Modified: Added auto-detection for SE/PE compatibility
################################################

# Run a bounded number of fastp jobs without a FIFO token pool.
thread_num=3

## Argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --fastqDir)          fastqDir=$2;shift;;
        --outputDir)         outputDir=$2;shift;;
        --)                  shift; break;;
        *)                   echo -e "\n[ERR] $(date) Unknown option: $1"; exit 1;;
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

# Discover sample prefixes without parsing ls output.
shopt -s nullglob
fastq_files=("${fastqDir}"/*.fastq.gz)
if (( ${#fastq_files[@]} == 0 )); then
    echo "[Error] No *.fastq.gz files found in ${fastqDir}." >&2
    exit 1
fi
sample_names=()
for fastq_path in "${fastq_files[@]}"; do
    fastq_name=$(basename "$fastq_path")
    sample_names+=("$(printf '%s\n' "$fastq_name" | sed -E 's/(_1|_2|_R1|_R2)?\.fastq\.gz$//')")
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
    {
        [ -d $outputDir ] || mkdir -p $outputDir
        cd $outputDir
        
        echo "-- Processing $sample --"
        
        # 【修改点 2】: 检测文件路径，判断是单端还是双端
        fq1=""
        fq2=""
        
        # 优先匹配常见的双端命名格式 (_1/_2 或 _R1/_R2)
        if [ -f "$fastqDir/${sample}_1.fastq.gz" ]; then
            fq1="$fastqDir/${sample}_1.fastq.gz"
            if [ ! -f "$fastqDir/${sample}_2.fastq.gz" ]; then
                echo "[Error] Missing mate file for $fq1" >&2
                exit 1
            fi
            fq2="$fastqDir/${sample}_2.fastq.gz"
        elif [ -f "$fastqDir/${sample}_R1.fastq.gz" ]; then
            fq1="$fastqDir/${sample}_R1.fastq.gz"
            if [ ! -f "$fastqDir/${sample}_R2.fastq.gz" ]; then
                echo "[Error] Missing mate file for $fq1" >&2
                exit 1
            fi
            fq2="$fastqDir/${sample}_R2.fastq.gz"
        elif [ -f "$fastqDir/${sample}_2.fastq.gz" ] || \
             [ -f "$fastqDir/${sample}_R2.fastq.gz" ]; then
            echo "[Error] Read-2 file exists without its read-1 mate for $sample" >&2
            exit 1
        elif [ -f "$fastqDir/${sample}.fastq.gz" ]; then
            # 匹配纯单端命名格式
            fq1="$fastqDir/${sample}.fastq.gz"
        fi

        # 根据检测结果执行对应的 fastp 命令
        if [ -n "$fq1" ] && [ -n "$fq2" ]; then
            # ================== 双端 (PE) 模式 ==================
            echo "[Info] Detected Paired-End data for $sample"
            if [ -s "${sample}_1.clean.fastq.gz" ] && [ -s "${sample}_2.clean.fastq.gz" ]; then
                echo "[Skip] Both cleaned FASTQ files already exist for $sample."
            else
                fastp \
                -i $fq1 -I $fq2 \
                -o ${sample}_1.clean.fastq.gz -O ${sample}_2.clean.fastq.gz \
                -w 16 --qualified_quality_phred 20 --length_required 50 \
                -h ${sample}_fastp.html -j ${sample}_fastp.json
            fi
                
        elif [ -n "$fq1" ]; then
            echo "[Skip] Single-end sample $sample is excluded by preprocessing policy."
                
        else
            echo "[WARN] Could not find valid fastq files for $sample in $fastqDir"
        fi

    }&
    pids+=("$!")
    ((active_jobs += 1))
    if (( active_jobs == thread_num )); then
        if ! wait_for_batch; then
            echo "[ERROR] At least one fastp job failed." >&2
            exit 1
        fi
    fi
done

if ! wait_for_batch; then
    echo "[ERROR] At least one fastp job failed." >&2
    exit 1
fi
echo "All tasks finished!"

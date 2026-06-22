#!/bin/bash
set -euo pipefail

# ==============================================================================
# Script: run.rnaseq_variants_calling.sh
# Purpose: Auto-pair Tumor/Normal RNA-seq BAMs from CSV and run GATK4 Mutect2.
#          Features mkfifo concurrency and fixes Async I/O writer bugs.
# ==============================================================================

# ==============================================================================
# Helper Function: Checkpoint Look-ahead
# 只要参数里提供的任何一个文件（当前步骤或未来步骤的产物）存在，就返回跳过(1)
# ==============================================================================
function step_needs_run() {
    for file in "$@"; do
        if [ -f "$file" ]; then
            return 1 # False: 发现下游或当前文件已存在，跳过该步骤
        fi
    done
    return 0 # True: 全都没找到，必须执行该步骤
}

# 1. Parse Arguments
THREAD_NUM=1 # 默认并发数为1
while [[ $# -gt 0 ]]; do
    case $1 in
        --bamDir)    BAM_DIR=$2; shift;;
        --meta)      META_FILE=$2; shift;;
        --ref_fasta) REF_FASTA=$2; shift;;
        --out_dir)   OUT_DIR=$2; shift;;
        --threads)   THREAD_NUM=$2; shift;; # 新增并发控制参数
        --)          shift; break;;
        *)           echo "Unknown option: $1"; exit 1;;
    esac
    shift
done

if [ -z "${BAM_DIR:-}" ] || [ -z "${META_FILE:-}" ] || [ -z "${REF_FASTA:-}" ] || [ -z "${OUT_DIR:-}" ]; then
    echo "Usage: bash run.rnaseq_variants_calling.sh --bamDir <dir> --meta <csv> --ref_fasta <fa> --out_dir <dir> [--threads 4]"
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "=========================================================="
echo " Phase 1: Parsing Metadata and Pairing Samples"
echo "=========================================================="

declare -A tumor_runs
declare -A normal_runs

while IFS=',' read -r run ind tissue || [[ -n "$run" ]]; do
    run=$(echo "$run" | tr -d '\r"')
    ind=$(echo "$ind" | tr -d '\r"')
    tissue=$(echo "$tissue" | tr -d '\r"' | tr '[:upper:]' '[:lower:]')
    
    if [[ "$run" == "Run" || -z "$run" ]]; then continue; fi 
    
    safe_pid=$(echo "$ind" | tr ' ' '_')
    
    if [[ "$tissue" == "tumor" || "$tissue" == "cancer" ]]; then
        tumor_runs["$safe_pid"]="$run"
    elif [[ "$tissue" == "normal" || "$tissue" == "adjacent" ]]; then
        normal_runs["$safe_pid"]="$run"
    fi
done < "$META_FILE"

echo " -> Successfully loaded metadata mapping."
echo " -> Concurrency set to: $THREAD_NUM patients at a time."

# ==============================================================================
# Phase 2: Concurrent Processing Initialization (mkfifo setup)
# ==============================================================================
tempfifo="my_temp_fifo_$$"
mkfifo ${tempfifo}
exec 6<>${tempfifo} 
rm -f ${tempfifo}

# 为 fifo 注入初始的令牌 (tokens)
for ((i=1; i<=${THREAD_NUM}; i++)); do
    echo >&6
done 

shopt -s nullglob 

echo "=========================================================="
echo " Phase 3: Executing GATK Pipeline"
echo "=========================================================="

for pid in "${!tumor_runs[@]}"; do
    # 领取一个并发令牌，如果没有令牌则在此阻塞等待
    read -u6
    {
        t_run="${tumor_runs[$pid]:-}"
        n_run="${normal_runs[$pid]:-}"
        
        if [[ -z "$t_run" || -z "$n_run" ]]; then
            echo "[Warning] Patient $pid is missing either a Tumor or Normal run. Skipping..."
            echo >&6 # 归还令牌
            continue
        fi
        
        t_bam_array=("$BAM_DIR/$t_run/"*.uniq.sorted.bam)
        n_bam_array=("$BAM_DIR/$n_run/"*.uniq.sorted.bam)
        
        if [ ${#t_bam_array[@]} -eq 0 ] || [ ${#n_bam_array[@]} -eq 0 ]; then
            echo "[Warning] Could not find .uniq.sorted.bam files for patient $pid. Skipping..."
            echo >&6 # 归还令牌
            continue
        fi
        
        TUMOR_BAM="${t_bam_array[0]}"
        NORMAL_BAM="${n_bam_array[0]}"
        
        WORK_DIR="$OUT_DIR/$pid"
        mkdir -p "$WORK_DIR/tmp"
        
        echo " [Thread Start] Processing Patient: $pid"
        
        # 预先定义各个步骤的终极标志性输出文件
        OUT_CLEAN="$WORK_DIR/tumor.clean.bam"
        OUT_RG="$WORK_DIR/tumor.rg.bam"
        OUT_MD="$WORK_DIR/tumor.md.bam"
        OUT_SPLIT="$WORK_DIR/tumor.split.bam"
        OUT_RAW_VCF="$WORK_DIR/${pid}_somatic_raw.vcf.gz"
        OUT_FLT_VCF="$WORK_DIR/${pid}_somatic_filtered.vcf.gz"

        # ----------------------------------------------------------------------
        # Step 2.0: The Bulletproof Vest (Filter Toxic Reads)
        # ----------------------------------------------------------------------
        FILTERED_TUMOR="$WORK_DIR/tumor.clean.bam"
        FILTERED_NORMAL="$WORK_DIR/normal.clean.bam"
        
        if step_needs_run "$OUT_CLEAN" "$OUT_RG" "$OUT_MD" "$OUT_SPLIT" "$OUT_RAW_VCF" "$OUT_FLT_VCF"; then
            echo " -> [0/5] Pre-filtering toxic reads (Removing flags 2308)..."
            samtools view -@ 4 -b -F 2308 "$TUMOR_BAM" > "$FILTERED_TUMOR"
            samtools view -@ 4 -b -F 2308 "$NORMAL_BAM" > "$FILTERED_NORMAL"
        else
            echo " -> [0/5] Skipped (Checkpoint reached)"
        fi
        
        # ----------------------------------------------------------------------
        # Step 2.1: AddOrReplaceReadGroups 
        # ----------------------------------------------------------------------
        if step_needs_run "$OUT_RG" "$OUT_MD" "$OUT_SPLIT" "$OUT_RAW_VCF" "$OUT_FLT_VCF"; then
            echo " -> [1/5] Injecting Read Groups..."
            gatk --java-options "-Djava.io.tmpdir=$WORK_DIR/tmp" AddOrReplaceReadGroups -I "$FILTERED_TUMOR" -O "$WORK_DIR/tumor.rg.bam" -RGID 1 -RGSM Tumor -RGPL illumina -RGLB lib1 -RGPU unit1
            gatk --java-options "-Djava.io.tmpdir=$WORK_DIR/tmp" AddOrReplaceReadGroups -I "$FILTERED_NORMAL" -O "$WORK_DIR/normal.rg.bam" -RGID 1 -RGSM Normal -RGPL illumina -RGLB lib1 -RGPU unit1
            
            # [阅后即焚]
            rm -f "$WORK_DIR"/*.clean.bam
        else
            echo " -> [1/5] Skipped (Checkpoint reached)"
        fi
        
        # ----------------------------------------------------------------------
        # Step 2.2: MarkDuplicates
        # ----------------------------------------------------------------------
        if step_needs_run "$OUT_MD" "$OUT_SPLIT" "$OUT_RAW_VCF" "$OUT_FLT_VCF"; then
            echo " -> [2/5] Marking Duplicates..."
            gatk --java-options "-Xmx8G -Djava.io.tmpdir=$WORK_DIR/tmp" MarkDuplicates -I "$WORK_DIR/tumor.rg.bam" -O "$WORK_DIR/tumor.md.bam" -M "$WORK_DIR/tumor.md.metrics.txt"
            gatk --java-options "-Xmx8G -Djava.io.tmpdir=$WORK_DIR/tmp" MarkDuplicates -I "$WORK_DIR/normal.rg.bam" -O "$WORK_DIR/normal.md.bam" -M "$WORK_DIR/normal.md.metrics.txt"
            
            # [阅后即焚]
            rm -f "$WORK_DIR"/*.rg.bam
        else
            echo " -> [2/5] Skipped (Checkpoint reached)"
        fi
        
        # ----------------------------------------------------------------------
        # Step 2.3: SplitNCigarReads (The Boss Fight)
        # ----------------------------------------------------------------------
        if step_needs_run "$OUT_SPLIT" "$OUT_RAW_VCF" "$OUT_FLT_VCF"; then
            echo " -> [3/5] Splitting N-CIGAR reads..."
            gatk --java-options "-Dsamjdk.use_async_io_write_samtools=false -Xmx15G -XX:ParallelGCThreads=4 -Djava.io.tmpdir=$WORK_DIR/tmp" SplitNCigarReads \
                -R "$REF_FASTA" -I "$WORK_DIR/tumor.md.bam" -O "$WORK_DIR/tumor.split.bam" \
                --max-mismatches-in-overhang 100 --tmp-dir "$WORK_DIR/tmp" --verbosity ERROR
                
            gatk --java-options "-Dsamjdk.use_async_io_write_samtools=false -Xmx15G -XX:ParallelGCThreads=4 -Djava.io.tmpdir=$WORK_DIR/tmp" SplitNCigarReads \
                -R "$REF_FASTA" -I "$WORK_DIR/normal.md.bam" -O "$WORK_DIR/normal.split.bam" \
                --max-mismatches-in-overhang 100 --tmp-dir "$WORK_DIR/tmp" --verbosity ERROR
                
            # [阅后即焚]
            rm -f "$WORK_DIR"/*.md.bam
        else
            echo " -> [3/5] Skipped (Checkpoint reached)"
        fi
        
        # ----------------------------------------------------------------------
        # Step 2.4: Mutect2 (The Final Boss)
        # ----------------------------------------------------------------------
        if step_needs_run "$OUT_RAW_VCF" "$OUT_RAW_VCF.tbi" "$OUT_FLT_VCF"; then
            echo " -> [4/5] Running Mutect2..."
            
            # [终极修复]: 加入 -Ovi false 禁止边写边建索引，防止封口失败
            gatk --java-options "-Dsamjdk.use_async_io_write_samtools=false -Dsamjdk.use_async_io_write_tribble=false -Xmx15G -XX:ParallelGCThreads=4 -Djava.io.tmpdir=$WORK_DIR/tmp" Mutect2 \
                -R "$REF_FASTA" \
                -I "$WORK_DIR/tumor.split.bam" \
                -I "$WORK_DIR/normal.split.bam" \
                -tumor Tumor \
                -normal Normal \
                -O "$WORK_DIR/${pid}_somatic_raw.vcf.gz" \
                --create-output-variant-index false \
                --tmp-dir "$WORK_DIR/tmp" --verbosity ERROR \
                --native-pair-hmm-threads 10
                
            echo " -> [4.5/5] Manually indexing VCF to prevent BGZF corruption..."
            # 独立手动建索引，绝对安全
            gatk IndexFeatureFile -I "$WORK_DIR/${pid}_somatic_raw.vcf.gz"
                
            # [阅后即焚] 
            rm -f "$WORK_DIR"/*.split.bam "$WORK_DIR"/*.bai
        else
            echo " -> [4/5] Skipped (Checkpoint reached)"
        fi
        
        # ----------------------------------------------------------------------
        # Step 2.5: FilterMutectCalls
        # ----------------------------------------------------------------------
        if step_needs_run "$OUT_FLT_VCF" "$OUT_FLT_VCF.tbi"; then
            echo " -> [5/5] Filtering somatic variants..."
            
            gatk --java-options "-Dsamjdk.use_async_io_write_tribble=false -Xmx8G -Djava.io.tmpdir=$WORK_DIR/tmp" FilterMutectCalls \
                -R "$REF_FASTA" \
                -V "$WORK_DIR/${pid}_somatic_raw.vcf.gz" \
                -O "$WORK_DIR/${pid}_somatic_filtered.vcf.gz" \
                --create-output-variant-index false \
                --tmp-dir "$WORK_DIR/tmp" --verbosity ERROR
                
            gatk IndexFeatureFile -I "$WORK_DIR/${pid}_somatic_filtered.vcf.gz"
        else
            echo " -> [5/5] Skipped (Checkpoint reached)"
        fi
        
        # 清除临时目录
        rm -rf "$WORK_DIR/tmp"
        echo " [Thread Finish] Patient $pid Complete!"
        
        # 归还并发令牌，允许下一个患者进入队列
        echo >&6 
        
    } & # '&' 符号使得大括号内的代码块进入后台运行
done

# ==============================================================================
# Phase 4: Wait for completion
# ==============================================================================
echo " -> All patient jobs dispatched. Waiting for background threads to finish..."
wait
exec 6>&- # 关闭文件描述符

echo "🎉 All pipelines executed successfully!"
#!/bin/sh 
set -euo pipefail

## Argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --bamDir)         bamDir=$2;shift;;      
        --outputDir)      outputDir=$2;shift;;   
        --refGTF)         refGTF=$2;shift;;      
        --refFasta)       refFasta=$2;shift;;    
        --threads_per_job) threads_per_job=$2;shift;; 
        --)               shift; break;;
        *)                echo -e "\n[ERR] $(date) Unknown option: $1"; exit 1;;
    esac
    shift
done

threads_per_job=${threads_per_job:-10}
job_num=3 # 并发处理的样本数量

if [ -z "${bamDir:-}" ] || [ -z "${outputDir:-}" ] || [ -z "${refGTF:-}" ] || [ -z "${refFasta:-}" ]; then
    echo "Error: Missing required parameters."
    exit 1
fi

[ -d $outputDir ] || mkdir -p $outputDir

# ---------------------------------------------------------
# Phase 1: Per-sample Transcript Assembly (StringTie)
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 1: Per-sample Transcript Assembly ###"
echo "=========================================================="

tempfifo="my_temp_fifo_$$"
mkfifo ${tempfifo}
exec 6<>${tempfifo} 
rm -f ${tempfifo}

for ((i=1;i<=${job_num};i++)); do echo; done >&6 

samples=($(ls ${bamDir}/*/*.uniq.sorted.bam | awk -F'/' '{print $(NF-1)}'))

for sample in ${samples[@]}; do
    sample_outdir="${outputDir}/${sample}"
    input_bam="${bamDir}/${sample}/${sample}.uniq.sorted.bam"
    output_gtf="${sample_outdir}/${sample}.gtf"
    
    if [ -s "$output_gtf" ]; then
        echo "[Skip] ${sample} assembly already completed."
        continue
    fi
    
    read -u6
    {
        echo "-- assembling $sample --"
        [ -d $sample_outdir ] || mkdir -p $sample_outdir
        stringtie ${input_bam} -G ${refGTF} --rf -p ${threads_per_job} -o ${output_gtf}
        echo >&6
    }&
done
wait
exec 6>&-

mergelist="${outputDir}/mergelist.txt"
find ${outputDir} -mindepth 2 -name "*.gtf" > $mergelist
echo "[Phase 1] All samples assembled. Generated mergelist with $(wc -l < $mergelist) files."


# ---------------------------------------------------------
# Phase 2: Merge all transcriptomes into a master GTF
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 2: Merge all transcriptomes ###"
echo "=========================================================="
merged_gtf="${outputDir}/stringtie_merged.gtf"

if [ -s "$merged_gtf" ]; then
    echo "[Skip] $merged_gtf already exists."
else
    echo "Running StringTie Merge..."
    time stringtie --merge -p 20 -G ${refGTF} -o ${merged_gtf} $mergelist
fi

# ---------------------------------------------------------
# Phase 3: Identify novel transcripts & Record Class Codes
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 3: Identify novel transcripts (Gffcompare) ###"
echo "=========================================================="
gffcmp_prefix="${outputDir}/gffcompare_out"
novel_gtf="${outputDir}/novel_transcripts.gtf"
target_ids="${outputDir}/target_tcons_ids.txt"
class_mapping="${outputDir}/transcript_class_mapping.tsv" 

if [ -s "$novel_gtf" ]; then
    echo "[Skip] $novel_gtf already exists."
else
    echo "Running Gffcompare..."
    time gffcompare -r ${refGTF} -G -o ${gffcmp_prefix} ${merged_gtf}
    
    echo "Step 1: Extracting Novel IDs and Class Codes (u, i, x, j, m, n, e, o, k)..."
    echo -e "Transcript_ID\tClass_Code\tRef_Gene_Name" > ${class_mapping}
    
    awk 'BEGIN {OFS="\t"} 
    $3=="transcript" {
        if ($0 ~ /class_code "[uixjmneok]"/ && $1 ~ /^chr([1-9]|1[0-9]|2[0-2]|[XYM])$/) {
            tid=""; cc=""; gname="Unknown";
            for(i=9; i<=NF; i++) {
                if ($i == "transcript_id") tid=$(i+1);
                if ($i == "class_code") cc=$(i+1);
                if ($i == "cmp_ref_gene" || $i == "gene_name") gname=$(i+1);
            }
            gsub(/"|;/, "", tid);
            gsub(/"|;/, "", cc);
            gsub(/"|;/, "", gname);
            
            # 【修改 1：源头拦截】彻底剔除所有以 ENS 开头的已知转录本，只保留 StringTie 生成的 MSTRG/STRG
            if (tid !~ /^ENS/) {
                print tid, cc, gname;
            }
        }
    }' ${gffcmp_prefix}.annotated.gtf | sort | uniq >> ${class_mapping}

    tail -n +2 ${class_mapping} | cut -f1 > ${target_ids}
    valid_count=$(wc -l < ${target_ids})
    echo "Found $valid_count strictly NOVEL transcripts (ENSTs excluded)."

    echo "Step 2: Extracting full GTF records (Robustly avoiding cross-contamination)..."
    # 【修改 2：精准提取】弃用 grep -Fwf，改用 awk 精准解析 transcript_id 属性
    # NR==FNR 处理 target_ids.txt，将目标 ID 存入字典 wanted
    # 接着处理 annotated.gtf，只有当行内的 transcript_id 存在于 wanted 字典时，才输出该行
    awk 'NR==FNR {wanted[$1]=1; next} 
    {
        tid=""; 
        for(i=9; i<=NF; i++) {
            if ($i == "transcript_id") {
                tid=$(i+1); 
                gsub(/"|;/, "", tid); 
                break;
            }
        }
        if (tid in wanted) print $0;
    }' ${target_ids} ${gffcmp_prefix}.annotated.gtf > ${novel_gtf}
    
    echo "Pure novel transcripts GTF with intact exons generated successfully."
fi

# ---------------------------------------------------------
# Phase 4: Filter transcripts by length (For Fasta/TRACE)
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 4: Filter transcripts by length ###"
echo "=========================================================="
final_novel_gtf="${outputDir}/final_filtered_novel_transcripts.gtf"
novel_fasta="${outputDir}/novel_transcripts.fasta"

if [ -s "$final_novel_gtf" ]; then
    echo "[Skip] Length filtering already completed."
else
    echo "Extracting FASTA and filtering by length (300bp - 20000bp)..."
    gffread -w ${novel_fasta} -g ${refFasta} ${novel_gtf}

    awk '/^>/ {if (seqlen){print id, seqlen}; id=$1; seqlen=0; next} {seqlen += length($0)} END {print id, seqlen}' \
        ${novel_fasta} | \
        awk '$2 >= 300 && $2 <= 20000 {gsub(/^>/, "", $1); print $1}' > ${outputDir}/valid_transcript_ids.txt

    grep -Fwf ${outputDir}/valid_transcript_ids.txt ${novel_gtf} > ${final_novel_gtf}
fi

# ---------------------------------------------------------
# Phase 5: Isolate Unique Regions for Quantification
# ---------------------------------------------------------
echo "=========================================================="
echo "### Phase 5: Extract strictly novel features (Bedtools)###"
echo "=========================================================="
quant_target_gtf="${outputDir}/final_quantification_targets.gtf"

if [ -s "$quant_target_gtf" ]; then
    echo "[Skip] Quantification targets already extracted."
else
    echo "Step 1: Extracting canonical exons from reference GTF..."
    awk '$3 == "exon"' ${refGTF} > ${outputDir}/ref_exons.tmp.gtf

    echo "Step 2: Extracting exons from assembled novel GTF..."
    # 必须使用经过长度过滤的 final_novel_gtf
    awk '$3 == "exon"' ${final_novel_gtf} > ${outputDir}/novel_exons.tmp.gtf

    echo "Step 3: Subtracting reference exons to isolate truly novel segments..."
    # 使用 bedtools subtract 挖去已知区域 (-s 确保同链相减)
    bedtools subtract \
        -a ${outputDir}/novel_exons.tmp.gtf \
        -b ${outputDir}/ref_exons.tmp.gtf \
        -s > ${outputDir}/strictly_novel_pieces.tmp.gtf

    echo "Step 4: Filtering out micro-fragments (< 50bp) and formatting..."
    # 计算特征长度 ($5 终止位点 - $4 起始位点 + 1)，只保留 >= 50bp 的区间
    awk -F'\t' 'BEGIN {OFS="\t"} 
    { 
        feat_len = $5 - $4 + 1;
        if (feat_len >= 50) {
            $3="exon"; 
            print $0;
        }
    }' ${outputDir}/strictly_novel_pieces.tmp.gtf > ${quant_target_gtf}

    # 清理临时文件
    rm -f ${outputDir}/*.tmp.gtf
    echo "Physical deduplication complete! Safe targets saved to: ${quant_target_gtf}"
fi

echo "=========================================================="
echo "All done! Pipeline finished successfully."
echo "Use [ ${final_novel_gtf} ] for Fasta extraction / TRACE modeling."
echo "Use [ ${quant_target_gtf} ] for featureCounts expression quantification."
echo "Metadata saved to [ ${class_mapping} ]."
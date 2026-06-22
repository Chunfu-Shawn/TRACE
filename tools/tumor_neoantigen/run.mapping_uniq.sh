#!/bin/sh 
set -euo pipefail

## Argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --fastqDir)       fastqDir=$2;shift;;
        --file_suffix)    file_suffix=$2;shift;; # e.g., _R1.fastq.gz or .fastq.gz
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

if [ -z "$fastqDir" ] || [ -z "$outputDir" ]; then
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

# 假设 file_suffix 是类似 _1.fq.gz 或者 _R1.fastq.gz
samples=(`cd $fastqDir && ls *${file_suffix} | sed "s/${file_suffix}//g"`)

for sample in ${samples[@]};
do
    final_output="${outputDir}/${sample}/${sample}.uniq.sorted.bam"
    
    if [ -f "$final_output" ]; then
        echo "Skip ${sample}, output exists."
        continue
    fi
    
    read -u6
    {
        echo "-- processing $sample --"

        [ -d $outputDir/${sample} ] || mkdir -p $outputDir/${sample}
        cd $outputDir/${sample}

        # 判断是单端还是双端测序数据
        R1_file="${fastqDir}/${sample}${file_suffix}"
        # 猜测 R2 后缀，例如 _R1.fastq.gz 对应 _R2.fastq.gz
        R2_suffix=$(echo $file_suffix | sed 's/1/2/') 
        R2_file="${fastqDir}/${sample}${R2_suffix}"

        if [ -f "$R2_file" ]; then
            input_files="$R1_file $R2_file"
            echo "Detected Paired-End data for $sample"
        else
            input_files="$R1_file"
            echo "Detected Single-End data for $sample"
        fi

        ### Perform 2-pass mapping.
        # 优化点：直接利用 STAR 管道输出 BAM 并通过 samtools 排序
        time STAR \
            --genomeDir $annoIndex \
            --readFilesIn $input_files \
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


        echo >&6
    }&
done
wait
echo "All done!"
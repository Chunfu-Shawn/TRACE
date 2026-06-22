################################################
#File Name: run.ribo-seq.analysis.sh
#Author: rbase    
#Mail: xiaochunfu@stu.pku.edu.cn
#Modified: Added auto-detection for SE/PE compatibility
################################################

#!/bin/sh 

##并发运行脚本，并控制并发数
# 设置并发的进程数
thread_num=3
a=$(date +%H%M%S)
# mkfifo
tempfifo="my_temp_fifo"
mkfifo ${tempfifo}
# 使文件描述符为非阻塞式
exec 6<>${tempfifo}
rm -f ${tempfifo}

# 为文件描述符创建占位信息
for ((i=1;i<=${thread_num};i++))
do
{
    echo 
}
done >&6 #事实上就是在fd6中放置了$thread个回车符

## Argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --fastqDir)          fastqDir=$2;shift;;
        --outputDir)         outputDir=$2;shift;;
        --)                  shift; break;;
        *)                   usage; echo -e "\n[ERR] $(date) Unkonwn option: $1"; exit 1;;
    esac
    shift
done

# 【修改点 1】: 智能提取样本前缀
# 兼容后缀包含 _1.fastq.gz, _2.fastq.gz, _R1.fastq.gz, _R2.fastq.gz 或单纯的 .fastq.gz
# sort -u 用于去除双端文件产生的重复前缀
samples=(`cd $fastqDir && ls *.fastq.gz | sed -E 's/(_1|_2|_R1|_R2)?\.fastq\.gz//g' | sort -u`)

for sample in ${samples[@]};
do
    read -u6
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
            [ -f "$fastqDir/${sample}_2.fastq.gz" ] && fq2="$fastqDir/${sample}_2.fastq.gz"
        elif [ -f "$fastqDir/${sample}_R1.fastq.gz" ]; then
            fq1="$fastqDir/${sample}_R1.fastq.gz"
            [ -f "$fastqDir/${sample}_R2.fastq.gz" ] && fq2="$fastqDir/${sample}_R2.fastq.gz"
        elif [ -f "$fastqDir/${sample}.fastq.gz" ]; then
            # 匹配纯单端命名格式
            fq1="$fastqDir/${sample}.fastq.gz"
        fi

        # 根据检测结果执行对应的 fastp 命令
        if [ -n "$fq1" ] && [ -n "$fq2" ]; then
            # ================== 双端 (PE) 模式 ==================
            echo "[Info] Detected Paired-End data for $sample"
            [ -f ${sample}_1.clean.fastq.gz ] || fastp \
                -i $fq1 -I $fq2 \
                -o ${sample}_1.clean.fastq.gz -O ${sample}_2.clean.fastq.gz \
                -w 16 --qualified_quality_phred 20 --length_required 50 \
                -h ${sample}_fastp.html -j ${sample}_fastp.json
                
        elif [ -n "$fq1" ]; then
            # ================== 单端 (SE) 模式 ==================
            echo "[Info] Detected Single-End data for $sample"
            [ -f ${sample}.clean.fastq.gz ] || fastp \
                -i $fq1 \
                -o ${sample}.clean.fastq.gz \
                -w 16 --qualified_quality_phred 20 --length_required 50 \
                -h ${sample}_fastp.html -j ${sample}_fastp.json
                
        else
            echo "[WARN] Could not find valid fastq files for $sample in $fastqDir"
        fi

        # 当进程结束以后，再向FD6中加上一个回车符，即补上了read -u6减去的那个
        echo >&6
    }&
done
wait
echo "All tasks finished!"
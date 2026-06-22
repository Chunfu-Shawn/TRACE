################################################
#File Name: run.fastqc.analysis.sh
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
file_suffix=".fastq.gz" # 增加一个默认值以防未输入
while [[ $# -gt 0 ]]; do
    case $1 in
        --fastqDir)              fastqDir=$2;shift;;
        --file_suffix)           file_suffix=$2;shift;;
        --outputDir)             outputDir=$2;shift;;
        --)                      shift; break;;
        *)                       usage; echo -e "\n[ERR] $(date) Unkonwn option: $1"; exit 1;;
    esac
    shift
done

# 【修改点 1】：智能提取样本前缀
# 先去掉文件后缀 (比如 .clear.fastq.gz)，然后再去掉 _1, _2, _R1, _R2 结尾
samples=(`cd $fastqDir && ls *${file_suffix} | sed 's/'"${file_suffix}"'$//g' | sed -E 's/(_1|_2|_R1|_R2)$//g' | sort -u`)

for sample in ${samples[@]};
do
    read -u6
    {
        [ -d $outputDir ] || mkdir -p $outputDir
        
        echo "-- FastQC for $sample --"
        
        fq1=""
        fq2=""
        
        # 【修改点 2】：根据前缀检测文件，匹配双端或单端
        if [ -f "$fastqDir/${sample}_1${file_suffix}" ]; then
            fq1="$fastqDir/${sample}_1${file_suffix}"
            [ -f "$fastqDir/${sample}_2${file_suffix}" ] && fq2="$fastqDir/${sample}_2${file_suffix}"
        elif [ -f "$fastqDir/${sample}_R1${file_suffix}" ]; then
            fq1="$fastqDir/${sample}_R1${file_suffix}"
            [ -f "$fastqDir/${sample}_R2${file_suffix}" ] && fq2="$fastqDir/${sample}_R2${file_suffix}"
        elif [ -f "$fastqDir/${sample}${file_suffix}" ]; then
            fq1="$fastqDir/${sample}${file_suffix}"
        fi

        # 预测 FastQC 输出的 html 文件名（用于判断是否已经跑过）
        # FastQC 会自动去掉 .fastq.gz 或 .fq.gz 然后加上 _fastqc.html
        report_check=$(basename $fq1 | sed -E 's/\.fastq\.gz|\.fq\.gz|\.fastq|\.fq//')_fastqc.html

        # 【修改点 3】：运行 FastQC
        # FastQC 支持同时输入多个文件，我们将找到的该样本的所有文件一起传给它
        if [ -n "$fq1" ] && [ -n "$fq2" ]; then
            echo "[Info] Detected PE files for $sample"
            [ -f "$outputDir/$report_check" ] || fastqc -o $outputDir -t 10 $fq1 $fq2
        elif [ -n "$fq1" ]; then
            echo "[Info] Detected SE file for $sample"
            [ -f "$outputDir/$report_check" ] || fastqc -o $outputDir -t 10 $fq1
        else
            echo "[WARN] Could not find valid files for $sample with suffix ${file_suffix}"
        fi
        
        # 当进程结束以后，再向FD6中加上一个回车符，即补上了read -u6减去的那个
        echo >&6
    }&
done

wait
## merge
echo "-- Running MultiQC --"
[ -f $outputDir/multiqc_report.html ] || multiqc $outputDir --outdir $outputDir
echo "All tasks finished!"
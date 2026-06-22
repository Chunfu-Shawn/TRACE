################################################
#File Name: run.mapping.sh
#Author: rbase    
#Mail: xiaochunfu@stu.pku.edu.cn
#Created Time: Wed 23 Aug 2023 11:23:38 PM CST
################################################

#!/bin/sh 

HUMAN_REF_FA=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/fasta/Homo_sapiens.GRCh38.primary_assembly.genome.fa
HUMAN_REF_GTF=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.gtf
HUMAN_INDEX=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/genome_index_v48_150nt

[ -d $HUMAN_INDEX ] || STAR --runMode genomeGenerate \
    --runThreadN 20 \
    --genomeDir $HUMAN_INDEX \
    --genomeFastaFiles $HUMAN_REF_FA \
    --sjdbGTFfile $HUMAN_REF_GTF \
    --sjdbOverhang 149
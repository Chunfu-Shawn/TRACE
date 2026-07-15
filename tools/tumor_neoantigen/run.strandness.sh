#!/bin/bash

WORK_DIR=/home/user/data3/rbase/translation_model/neoantigen/samples
SCRIPT_DIR=/home/user/data3/rbase/translation_model/models/tools/tumor_neoantigen
GENOME_INDEX=/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/genome_index_v48_150nt

projects=(LiZH_liver_tumor)

for project in ${projects[@]};
do
    echo "##### $project #####"
    BAM_DIR=$WORK_DIR/$project/bam
    samples=(`cd $BAM_DIR && ls | sort -u`)

    for sample in ${samples[@]};
    do  
        echo "-- Processing $sample --"
        infer_experiment.py -r /home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/gencode.v48.comp_annotation_chro.bed \
            -i $BAM_DIR/$sample/$sample.uniq.sorted.bam
    done
    
    echo "All tasks finished!"
done
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import time

def build_gtex_median_matrix(tpm_file, anno_file, output_file, chunk_size=50000):
    start_time = time.time()
    print("1. 加载样本注释文件，构建 组织映射字典...")
    anno_df = pd.read_csv(anno_file, sep='\t', low_memory=False)
    sample_to_tissue = dict(zip(anno_df['SAMPID'], anno_df['SMTSD']))

    print("2. 读取 GTEx TPM 表头，匹配有效样本...")
    with open(tpm_file, 'r') as f:
        header = f.readline().strip().split('\t')
    
    sample_cols = header[2:] # 跳过 transcript_id 和 gene_id
    
    # 构建 组织 -> 样本列表 的映射
    tissue_to_samples = defaultdict(list)
    for s in sample_cols:
        if s in sample_to_tissue:
            tissue = sample_to_tissue[s]
            tissue_to_samples[tissue].append(s)
            
    print(f"成功识别 {len(tissue_to_samples)} 种组织类型。开始处理海量矩阵...")

    # 指定列类型以节约内存
    dtype_dict = {'transcript_id': str, 'gene_id': str}
    for col in sample_cols:
        dtype_dict[col] = np.float32

    # 分块读取并聚合
    chunks = pd.read_csv(tpm_file, sep='\t', chunksize=chunk_size, dtype=dtype_dict)
    
    first_chunk = True
    processed_rows = 0
    
    for chunk in chunks:
        # 建立用于保存中位数的空 DataFrame
        median_df = pd.DataFrame()
        
        # 核心：自动去除转录本的版本号 (如 ENST00000373020.9 -> ENST00000373020)
        median_df['Transcript_ID'] = chunk['transcript_id'].str.split('.').str[0]
        
        # 遍历每种组织，计算沿行的中位数
        for tissue, samps in tissue_to_samples.items():
            valid_samps = [s for s in samps if s in chunk.columns]
            if valid_samps:
                # 计算该组织所有样本的中位数 (axis=1 代表同行跨列计算)
                median_df[tissue] = chunk[valid_samps].median(axis=1)
        
        # 追加写入 CSV (如果是第一块则写入表头)
        mode = 'w' if first_chunk else 'a'
        header_flag = True if first_chunk else False
        median_df.to_csv(output_file, mode=mode, index=False, header=header_flag)
        
        first_chunk = False
        processed_rows += len(chunk)
        print(f"已处理 {processed_rows} 个转录本...")

    elapsed = (time.time() - start_time) / 60
    print(f"\n🎉 降维完成！耗时: {elapsed:.2f} 分钟.")
    print(f"生成的组织中位数矩阵已保存至: {output_file}")

if __name__ == "__main__":
    # 请根据实际路径修改以下变量
    GTEX_TPM = '/home/user/data3/rbase/database/GTEx/GTEx_Analysis_2025-08-22_v11_RSEMv1.3.3_transcripts_tpm.txt'
    GTEX_ANNO = '/home/user/data3/rbase/database/GTEx/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt'
    OUTPUT_MATRIX = '/home/user/data3/rbase/database/GTEx/GTEx_Tissue_Median_TPM.csv'
    
    build_gtex_median_matrix(GTEX_TPM, GTEX_ANNO, OUTPUT_MATRIX)
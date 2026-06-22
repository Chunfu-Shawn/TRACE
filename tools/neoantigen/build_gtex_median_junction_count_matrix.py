#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import gzip
import time
from collections import defaultdict

def build_gtex_junction_median_matrix(jcounts_file, anno_file, output_file, chunk_size=50000):
    start_time = time.time()
    print("1. 加载样本注释文件，构建 组织映射字典...")
    anno_df = pd.read_csv(anno_file, sep='\t', low_memory=False)
    sample_to_tissue = dict(zip(anno_df['SAMPID'], anno_df['SMTSD']))

    print("2. 读取 GTEx Junction GCT 表头，匹配有效样本...")
    # GCT 格式兼容处理 (.gct 或 .gct.gz)
    fopen = gzip.open if jcounts_file.endswith('.gz') else open
    mode = 'rt' if jcounts_file.endswith('.gz') else 'r'
    
    with fopen(jcounts_file, mode) as f:
        f.readline() # 跳过第一行 (版本号, e.g., #1.2)
        f.readline() # 跳过第二行 (维度, e.g., 523817 19788)
        header = f.readline().strip().split('\t')
    
    # GCT 文件的特征列是 Name 和 Description
    sample_cols = header[2:] 
    
    # 构建 组织 -> 样本列表 的映射
    tissue_to_samples = defaultdict(list)
    for s in sample_cols:
        if s in sample_to_tissue:
            tissue = sample_to_tissue[s]
            tissue_to_samples[tissue].append(s)
            
    print(f"成功识别 {len(tissue_to_samples)} 种组织类型。开始处理海量矩阵...")

    # 指定特征列为字符串，其余样本列使用 float32 节省内存
    dtype_dict = {'Name': str, 'Description': str}

    # skiprows=2 完美跳过 GCT 的前两行元数据
    chunks = pd.read_csv(
        jcounts_file, 
        sep='\t', 
        skiprows=2, 
        chunksize=chunk_size, 
        dtype=dtype_dict,
        low_memory=False
    )
    
    first_chunk = True
    processed_rows = 0
    
    for chunk in chunks:
        # 建立用于保存中位数的空 DataFrame
        median_df = pd.DataFrame()
        
        # 提取 Junction 的绝对坐标 (e.g., chr1:11212-12009:+)
        median_df['Junction_ID'] = chunk['Name']
        
        # 可选：保留 Description 列(相关的已知基因) 方便日后追溯
        median_df['Associated_Gene'] = chunk['Description']
        
        # 保存该 Junction 在所有组织中的中位数，以便最后求全局最大值
        tissue_medians = []
        
        # 遍历每种组织，计算沿行的中位数
        for tissue, samps in tissue_to_samples.items():
            valid_samps = [s for s in samps if s in chunk.columns]
            if valid_samps:
                # 转换类型为 float32 计算中位数，保留两位小数减小文件体积
                med_series = chunk[valid_samps].astype(np.float32).median(axis=1).round(2)
                median_df[tissue] = med_series
                tissue_medians.append(med_series)
        
        # 核心辅助列：计算该 Junction 在所有正常组织中位数的最大值
        # 方便下游极其暴力地过滤：如果 Max_Median_Count == 0，即为纯天然肿瘤特异连接点！
        if tissue_medians:
            median_df['Max_Median_Count'] = pd.concat(tissue_medians, axis=1).max(axis=1)
        else:
            median_df['Max_Median_Count'] = 0.0

        # 追加写入 CSV
        mode_write = 'w' if first_chunk else 'a'
        header_flag = True if first_chunk else False
        median_df.to_csv(output_file, mode=mode_write, index=False, header=header_flag)
        
        first_chunk = False
        processed_rows += len(chunk)
        print(f"已处理 {processed_rows} 个剪接点 (Junctions)...")

    elapsed = (time.time() - start_time) / 60
    print(f"\n🎉 降维完成！耗时: {elapsed:.2f} 分钟.")
    print(f"生成的组织中位数矩阵已保存至: {output_file}")

if __name__ == "__main__":
    GTEX_JUNCTIONS = '/home/user/data3/rbase/database/GTEx/GTEx_Analysis_2025-08-22_v11_STARv2.7.11b_junctions.gct'
    GTEX_ANNO = '/home/user/data3/rbase/database/GTEx/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt'
    OUTPUT_MATRIX = '/home/user/data3/rbase/database/GTEx/GTEx_Tissue_Median_Junction_Counts.csv'
    
    build_gtex_junction_median_matrix(GTEX_JUNCTIONS, GTEX_ANNO, OUTPUT_MATRIX)
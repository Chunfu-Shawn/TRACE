#!/usr/bin/env python3
import pandas as pd
import sys

def main():
    # 文件路径配置
    gene_reads_file = "/home/user/data3/rbase/database/GTEx/GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_reads.gct"
    junctions_file = "/home/user/data3/rbase/database/GTEx/GTEx_Analysis_2025-08-22_v11_STARv2.7.11b_junctions.gct"
    metadata_file = "/home/user/data3/rbase/database/GTEx/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"
    output_file = "/home/user/data3/rbase/database/GTEx/GTEx_Tissue_Median_Junction_CPM.csv"

    print("1. Loading Metadata and mapping samples to tissues...")
    # 提取映射字典
    meta_df = pd.read_csv(metadata_file, sep='\t', usecols=['SAMPID', 'SMTSD']).dropna()
    sample_to_tissue = dict(zip(meta_df['SAMPID'], meta_df['SMTSD']))

    print("\n2. Streaming Gene Reads to calculate exact Assigned Library Sizes...")
    # 流式读取 gene_reads，避免爆内存 (尽管 gene_reads 比 junctions 小，但也很大)
    lib_sizes = pd.Series(dtype=float)
    gene_reads_iter = pd.read_csv(gene_reads_file, sep='\t', skiprows=2, index_col=0, chunksize=20000)
    
    for chunk in gene_reads_iter:
        # 去掉 Description 列，沿列方向求和
        chunk_sum = chunk.drop(columns=['Description'], errors='ignore').sum(axis=0)
        if lib_sizes.empty:
            lib_sizes = chunk_sum
        else:
            lib_sizes = lib_sizes.add(chunk_sum, fill_value=0)
            
    print(f" -> Calculated exact assigned library sizes for {len(lib_sizes)} samples.")

    print("\n3. Preparing Junctions Header and Common Samples...")
    # 极速读取 junctions 文件的表头，确定共有样本
    junc_header = pd.read_csv(junctions_file, sep='\t', skiprows=2, index_col=0, nrows=0)
    junc_samples = [c for c in junc_header.columns if c != 'Description']
    
    # 获取有效的共有样本 (并且在 Metadata 中有组织注释的)
    common_samples = [s for s in junc_samples if s in lib_sizes.index and s in sample_to_tissue]
    print(f" -> Found {len(common_samples)} valid common samples with tissue annotations.")

    # 构建 Tissue 到 Samples 的分组列表，大幅加速后续计算
    tissue_groups = {}
    for col in common_samples:
        tissue = sample_to_tissue[col]
        if tissue not in tissue_groups:
            tissue_groups[tissue] = []
        tissue_groups[tissue].append(col)
    print(f" -> Grouped into {len(tissue_groups)} specific tissue types.")

    print("\n4. Streaming Junctions: Calculate CPM -> Median -> Save to Disk...")
    # 核心优化：每次只处理 25,000 个剪接点，算完就扔，永不爆内存
    chunk_size = 25000 
    first_chunk = True
    processed_rows = 0

    junc_iter = pd.read_csv(junctions_file, sep='\t', skiprows=2, index_col=0, chunksize=chunk_size)
    
    for chunk in junc_iter:
        junc_desc = chunk['Description']
        
        # 立即提取需要的样本列，过滤掉无效样本，节省内存
        chunk_data = chunk[common_samples]
        
        # 将原始 counts 转换为 CPM
        # chunk_data.div(lib_sizes) 会自动通过列名对齐运算
        chunk_cpm = chunk_data.div(lib_sizes[common_samples], axis=1) * 1e6
        
        # 初始化结果块
        res_chunk = pd.DataFrame(index=chunk.index)
        res_chunk['Associated_Gene'] = junc_desc
        
        # 按组织器官计算中位数
        for tissue, samples in tissue_groups.items():
            res_chunk[tissue] = chunk_cpm[samples].median(axis=1)
            
        # [关键一步] 将处理完的压缩数据直接追加写入 CSV，释放内存
        res_chunk.index.name = 'Junction_ID'
        res_chunk.to_csv(output_file, mode='w' if first_chunk else 'a', header=first_chunk)
        
        first_chunk = False
        processed_rows += len(chunk)
        print(f"    ... Processed {processed_rows} junctions ...")

    print("\n✅ All chunks processed successfully!")
    print(f"Output saved safely to: {output_file}")

if __name__ == "__main__":
    main()
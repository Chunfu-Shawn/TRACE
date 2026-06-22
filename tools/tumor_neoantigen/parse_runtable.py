import pandas as pd
import sys
import os

run_table = sys.argv[1]
bam_dir = sys.argv[2]
b1_out = sys.argv[3]
b2_out = sys.argv[4]

print(f"Reading metadata from {run_table}...")
# 支持 CSV 或 TSV 格式
df = pd.read_csv(run_table, sep=None, engine='python')
df.columns = df.columns.str.strip() # 去除列名可能的多余空格

# 关键：按照患者 (Individual) 排序，确保 Tumor 和 Normal 在列表中的顺序完全一一对应
df_sorted = df.sort_values('Individual')

tumor_df = df_sorted[df_sorted['tissue'].str.lower() == 'tumor']
normal_df = df_sorted[df_sorted['tissue'].str.lower() == 'non-tumor']

if len(tumor_df) != len(normal_df):
    print(f"WARNING: Unequal pairs! Tumor: {len(tumor_df)}, Normal: {len(normal_df)}")

tumor_runs = tumor_df['Run'].tolist()
normal_runs = normal_df['Run'].tolist()

# 拼接绝对路径 (假设格式为 bamDir/SRRXXXX/SRRXXXX.uniq.sorted.bam)
tumor_bams = [os.path.join(bam_dir, r, f"{r}.uniq.sorted.bam") for r in tumor_runs]
normal_bams = [os.path.join(bam_dir, r, f"{r}.uniq.sorted.bam") for r in normal_runs]

# 检查文件是否存在
for b in tumor_bams + normal_bams:
    if not os.path.exists(b):
        print(f"WARNING: BAM file not found: {b}")

with open(b1_out, 'w') as f1:
    f1.write(",".join(tumor_bams))
with open(b2_out, 'w') as f2:
    f2.write(",".join(normal_bams))

print(f"Successfully generated paired lists for {len(tumor_runs)} individuals.")
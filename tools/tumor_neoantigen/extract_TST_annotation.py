import sys

target_file = sys.argv[1]
ref_gtf = sys.argv[2]
novel_gtf = sys.argv[3]
out_gtf = sys.argv[4]

# 1. 读取目标 ID，去除潜在的版本号并存入 Set (哈希查找，极快)
targets = set()
with open(target_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            # 兼容带有版本号的输入 (e.g. ENST00000123.4 -> ENST00000123)
            targets.add(line.split('.')[0])

print(f"Loaded {len(targets)} unique target IDs.")

# 2. 遍历两个 GTF 文件，精准过滤
extracted_lines = 0
with open(out_gtf, 'w') as fout:
    for gtf_file in [ref_gtf, novel_gtf]:
        with open(gtf_file, 'r') as fin:
            for line in fin:
                if line.startswith('#'):
                    continue
                
                # 快速定位 transcript_id 属性
                if 'transcript_id "' in line:
                    # 截取 transcript_id 的值
                    tid_part = line.split('transcript_id "')[1].split('"')[0]
                    tid_base = tid_part.split('.')[0] # 去除参考 GTF 中的版本号
                    
                    if tid_base in targets:
                        fout.write(line)
                        extracted_lines += 1

print(f"Successfully extracted {extracted_lines} lines into Mini-GTF.")
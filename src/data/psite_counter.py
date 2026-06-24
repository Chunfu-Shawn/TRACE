import time
import pysam
import pickle
import numpy as np
from collections import defaultdict, namedtuple
from pathlib import Path

# 定义片段结构 (tx_offset: 距离转录本 5' 端的距离)
Segment = namedtuple("Segment", "chrom start end strand tx_offset tid")

def _parse_gtf_as_exon(annot_path, key_attr="transcript_id"):
    """解析 GTF 提取全长 exon"""
    exons = defaultdict(list)
    with open(annot_path, 'r') as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip(): continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "exon": continue
            
            chrom, start, end, strand, attrs = p[0], int(p[3])-1, int(p[4]), p[6], p[8]
            tid = None
            
            for kv in attrs.split(";"):
                kv = kv.strip()
                if not kv: continue
                parts = kv.split(None, 1)
                if len(parts) < 2: continue
                k, v = parts
                if k == key_attr:
                    tid = v.strip('"').strip("'")
                    break
            
            if tid:
                exons[tid].append((chrom, start, end, strand))
    return exons

def _build_transcript_segments(exons_by_tid):
    """构建相对坐标映射片段"""
    tx2len = {} 
    chrom2segs = defaultdict(list)
    chrom_min = {}
    chrom_max = {}
    
    for tid, blocks in exons_by_tid.items():
        if not blocks: continue
        
        strands = {b[3] for b in blocks}
        if len(strands) != 1: continue
        strand = list(strands)[0]
        
        blocks_sorted = sorted(blocks, key=lambda x: x[1])
        transcript_order = blocks_sorted[::-1] if strand == "-" else blocks_sorted
            
        total_len = sum(e - s for _, s, e, _ in transcript_order)
        tx2len[tid] = total_len
        
        current_offset = 0 
        for chrom, s, e, st in transcript_order:
            chrom2segs[chrom].append(Segment(chrom, s, e, st, current_offset, tid))
            chrom_min[chrom] = s if chrom not in chrom_min else min(chrom_min[chrom], s)
            chrom_max[chrom] = e if chrom not in chrom_max else max(chrom_max[chrom], e)
            current_offset += (e - s)

    for chrom in chrom2segs:
        chrom2segs[chrom].sort(key=lambda x: x.start)
        
    return chrom2segs, tx2len, chrom_min, chrom_max


def get_target_psite_distribution(
    bam_path: str,
    annot_path: str,
    out_dir: str,                 # [NEW] 输出文件夹路径
    out_prefix: str = "target",   # [NEW] 输出文件前缀
    target_tids: list = None,
    key_attr: str = "transcript_id",
    threads: int = 4
):
    """
    在 Notebook 中提取指定转录本的 P-site 读段分布 (Dense Numpy Array)。
    结果会作为字典返回，并同时落盘保存为 .pkl 文件。
    """
    print(f"1. Parsing GTF Annotation from {annot_path} ...")
    exons_by_tid = _parse_gtf_as_exon(annot_path, key_attr)
    
    if target_tids is not None:
        target_set = set(target_tids)
        exons_by_tid = {tid: blocks for tid, blocks in exons_by_tid.items() if tid in target_set}
        print(f"   -> Filtered down to {len(exons_by_tid)} target transcripts.")
    else:
        print(f"   -> Found {len(exons_by_tid)} transcripts total.")

    if not exons_by_tid:
        raise ValueError("No matching transcripts found in the annotation. Please check target_tids.")

    print("2. Mapping Genomic Coordinates to Transcript Vectors...")
    chrom2segs, tx2len, chrom_min, chrom_max = _build_transcript_segments(exons_by_tid)
    
    # 提前分配连续 Numpy 内存
    rc_dist = {tid: np.zeros(L, dtype=np.float32) for tid, L in tx2len.items()}
    
    print(f"3. Fetching P-site reads from BAM (Threads: {threads})...")
    t0 = time.time()
    
    with pysam.AlignmentFile(bam_path, "rb", threads=threads) as bam:
        for chrom, segs in chrom2segs.items():
            if not segs: continue
            
            start_region = chrom_min[chrom]
            end_region = chrom_max[chrom]
            
            active_segs = []
            seg_idx = 0
            
            iter_pileup = bam.pileup(chrom, start_region, end_region, truncate=True, 
                                     stepper="all", min_base_quality=0, max_depth=0)
            
            for col in iter_pileup:
                pos = col.reference_pos
                
                if pos < start_region or pos >= end_region: continue
                
                while seg_idx < len(segs) and segs[seg_idx].start <= pos:
                    active_segs.append(segs[seg_idx])
                    seg_idx += 1
                
                if active_segs:
                    active_segs = [s for s in active_segs if s.end > pos]
                
                if not active_segs: continue

                count_pos = count_neg = 0
                for pileread in col.pileups:
                    aln = pileread.alignment
                    if aln.is_unmapped or pileread.is_del or pileread.is_refskip:
                        continue
                    if aln.is_reverse:
                        count_neg += 1
                    else:
                        count_pos += 1

                if count_pos == 0 and count_neg == 0: continue

                # 分发 Count 到各个转录本的 Numpy 数组
                for seg in active_segs:
                    nreads = count_pos if seg.strand == "+" else count_neg
                    if nreads == 0: continue
                    
                    dist = pos - seg.start if seg.strand == "+" else seg.end - 1 - pos
                    offset = seg.tx_offset + dist
                    
                    L = tx2len.get(seg.tid, 0)
                    if 0 <= offset < L:
                        rc_dist[seg.tid][offset] += nreads

    elapsed = time.time() - t0
    print(f"✅ Finished Processing BAM in {elapsed:.2f} seconds.")
    
    # ==========================================
    # [NEW] 存入 pkl 文件的逻辑
    # ==========================================
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    save_file = out_path / f"{out_prefix}_transcript_read_distribution.pkl"
    print(f"4. Saving distribution pickle to: {save_file}")
    
    with open(save_file, 'wb') as f:
        pickle.dump(rc_dist, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("✅ All done!")
    return rc_dist
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from tqdm import tqdm

# =================================================================
# 安全清理 ID 的辅助函数 (复用之前的稳健逻辑)
# =================================================================
def safe_clean_id(raw_id: str) -> str:
    """仅去除 ENST/ENSG 的版本号，保留 PacBio/MSTRG 的完整结构"""
    clean_id = str(raw_id).split('|')[0]
    if (clean_id.startswith('ENST') or clean_id.startswith('ENSG')) and '.' in clean_id:
        clean_id = clean_id.split('.')[0]
    return clean_id


# =================================================================
# Util: Fast Fasta Parser
# =================================================================
def read_fasta(file_path: str) -> Dict[str, str]:
    """Read Fasta file and return a {tid: sequence} dictionary. (Turbo Version)"""
    seq_dict = {}
    curr_id = ""
    curr_seq = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if curr_id:
                    seq_dict[curr_id] = "".join(curr_seq).replace('U', 'T')
                
                raw_id = line[1:].split()[0]
                curr_id = safe_clean_id(raw_id)
                curr_seq = []
            else:
                curr_seq.append(line.upper())
        if curr_id:
            seq_dict[curr_id] = "".join(curr_seq).replace('U', 'T')
            
    print(f"Loaded {len(seq_dict)} sequences from {file_path}")
    return seq_dict

# =================================================================
# Core Algorithm: Pure Sequence-Based Baseline ORF Caller
# =================================================================
class BaselineSequenceORFCaller:
    def __init__(self, 
                 start_codons: List[str] = ['ATG', 'CTG', 'GTG', 'TTG', 'ACG'], 
                 stop_codons: List[str] = ['TAA', 'TAG', 'TGA'], 
                 min_len: int = 30):
        
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.min_len = min_len
        
        self.codon_weights = {
            'ATG': 1.0,
            'CTG': 0.8,
            'GTG': 0.6,
            'TTG': 0.4,
            'ACG': 0.2
        }
        
        self.stop_re = re.compile(f"(?=({'|'.join(stop_codons)}))")
        self.start_re = re.compile(f"(?=({'|'.join(start_codons)}))")

    def extract_and_score_candidates(self, sequence: str) -> List[dict]:
        candidates = []
        stop_positions = {0: [], 1: [], 2: []}
        
        for match in self.stop_re.finditer(sequence):
            pos = match.start()
            stop_positions[pos % 3].append(pos)
            
        for match in self.start_re.finditer(sequence):
            start_pos = match.start()
            frame = start_pos % 3
            
            for stop_pos in stop_positions[frame]:
                if stop_pos > start_pos:
                    orf_len = stop_pos - start_pos + 3 
                    if orf_len >= self.min_len:
                        start_codon = sequence[start_pos:start_pos+3]
                        
                        weight = self.codon_weights.get(start_codon, 0.1)
                        length_bonus = np.log10(orf_len + 1)
                        baseline_score = weight * length_bonus
                        
                        candidates.append({
                            'start': start_pos,
                            'stop': stop_pos,
                            'length': orf_len,
                            'start_codon': start_codon,
                            'score': float(baseline_score) 
                        })
                    break 
        return candidates

    def fast_nms(self, cands: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        keep = []
        cands.sort(key=lambda x: x['score'], reverse=True)
        
        for i, cand in enumerate(cands):
            if cand.get('suppressed', False): continue
            keep.append(cand)
            
            s1, e1, l1 = cand['start'], cand['stop'], cand['length']
            
            for j in range(i + 1, len(cands)):
                if cands[j].get('suppressed', False): continue
                    
                s2, e2, l2 = cands[j]['start'], cands[j]['stop'], cands[j]['length']
                overlap_l = max(0, min(e1, e2) - max(s1, s2))
                
                if overlap_l > 0:
                    iou = overlap_l / (l1 + l2 - overlap_l)
                    if iou > iou_threshold:
                        cands[j]['suppressed'] = True
                        
        for k in keep:
            k.pop('suppressed', None)
        return keep

    def collapse_and_nms(self, cands: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        if not cands: return []
        
        cands_by_stop = {}
        for cand in cands:
            e = cand['stop']
            if e not in cands_by_stop:
                cands_by_stop[e] = []
            cands_by_stop[e].append(cand)
            
        resolved_cands = []
        for e, group in cands_by_stop.items():
            if len(group) == 1:
                resolved_cands.append(group[0])
            else:
                atg_cands = [c for c in group if c['start_codon'] == 'ATG']
                if atg_cands:
                    best_cand = max(atg_cands, key=lambda x: x['length'])
                else:
                    best_cand = max(group, key=lambda x: x['length'])
                resolved_cands.append(best_cand)

        return self.fast_nms(resolved_cands, iou_threshold=iou_threshold)


# =================================================================
# Main Pipeline: Batch Processing & Saving
# =================================================================
class BaselineORFIdentifier:
    def __init__(self, 
                 fasta_file: str,
                 cell_types: List[str], 
                 tpm_csv_path: Optional[str] = None,
                 tpm_level: str = 'gene',
                 mapping_csv_path: Optional[str] = None):
                 
        self.fasta_file = fasta_file
        self.cell_types = cell_types
        self.tpm_level = tpm_level.lower()
        self.has_tpm = tpm_csv_path is not None and os.path.exists(tpm_csv_path)
        
        print("Loading Fasta File for Baseline Analysis...")
        self.seq_dict = read_fasta(self.fasta_file)
        
        # 建立 Transcript -> Gene Mapping
        self.tx2gene = {}
        if mapping_csv_path and os.path.exists(mapping_csv_path):
            print(f"Loading Gene-Transcript Mapping from {mapping_csv_path}...")
            try:
                m_df = pd.read_csv(mapping_csv_path, sep='\t')
                g_col, t_col = 'Gene stable ID', 'Transcript stable ID'
                if g_col in m_df.columns and t_col in m_df.columns:
                    for _, r in m_df.iterrows():
                        g_id = safe_clean_id(str(r[g_col]))
                        t_id = safe_clean_id(str(r[t_col]))
                        self.tx2gene[t_id] = g_id
            except Exception as e:
                print(f"  [Error] Failed to load Mapping CSV: {e}")
                
        # =================================================================
        # [MODIFIED] 为所有指定的细胞系独立加载 TPM 字典
        # =================================================================
        print(f"Loading TPM Expression Matrix (Level: {self.tpm_level})...")
        self.tpm_dicts = {ct: {} for ct in self.cell_types}
        
        if self.has_tpm:
            try:
                t_df = pd.read_csv(tpm_csv_path, index_col=0)
                t_df.index = t_df.index.to_series().astype(str).apply(safe_clean_id)
                t_df = t_df.groupby(t_df.index).mean() # 去重合并
                
                for ct in self.cell_types:
                    if ct in t_df.columns:
                        self.tpm_dicts[ct] = t_df[ct].to_dict()
                        print(f"  -> Loaded TPM for {len(self.tpm_dicts[ct])} {self.tpm_level}s in '{ct}'.")
                    else:
                        print(f"  [Warning] Cell type '{ct}' not found in TPM matrix.")
            except Exception as e:
                print(f"  [Error] Failed to load TPM matrix: {e}")
                self.has_tpm = False

    def run(self, 
            out_dir: str = "./results/baseline", 
            start_codons: List[str] = ['ATG', 'CTG', 'GTG', 'TTG', 'ACG'],
            # [MODIFIED] 支持接收单一列表，或按细胞系分类的字典 {cell_type: [tids]}
            target_tids: Optional[Union[List[str], Dict[str, List[str]]]] = None, 
            min_len: int = 30) -> pd.DataFrame:
        
        os.makedirs(out_dir, exist_ok=True)
        caller = BaselineSequenceORFCaller(start_codons=start_codons, min_len=min_len)
        all_records = []
        
        # =================================================================
        # [MODIFIED] 构建每个细胞系的需要处理的转录本集合，提高后续分发效率
        # =================================================================
        active_tids_per_cell = {ct: set(self.seq_dict.keys()) for ct in self.cell_types}
        
        if target_tids is not None:
            if isinstance(target_tids, dict):
                # 传入的是 {cell_type: [tids]} 格式
                for ct in self.cell_types:
                    if ct in target_tids:
                        active_tids_per_cell[ct] = set(safe_clean_id(t) for t in target_tids[ct])
            else:
                # 传入的是统一的 [tids] 列表，所有细胞系共用
                common_set = set(safe_clean_id(t) for t in target_tids)
                for ct in self.cell_types:
                    active_tids_per_cell[ct] = common_set
                    
        # 计算全局需要处理的转录本并集 (只对包含在任何一个细胞系里的转录本进行序列扫描)
        global_target_tids = set.union(*active_tids_per_cell.values())
        seq_dict_to_process = {tid: seq for tid, seq in self.seq_dict.items() if tid in global_target_tids}
        
        if not seq_dict_to_process:
            print("Warning: No matching sequences found to process!")
            return pd.DataFrame()

        print(f"\nStarting Sequence-Based Baseline Calling for {len(seq_dict_to_process)} unique transcripts...")
        
        for tid, sequence in tqdm(seq_dict_to_process.items()):
            
            # 1. 序列层面的提取和 NMS 对同一条转录本只需做一次
            cands = caller.extract_and_score_candidates(sequence)
            final_cands = caller.collapse_and_nms(cands, iou_threshold=0.3)
            
            if not final_cands: continue
            
            gene_id = self.tx2gene.get(tid, tid)
            query_id = tid if self.tpm_level == 'transcript' else gene_id
            
            # =================================================================
            # [MODIFIED] 将计算出的 ORF 分发给所有包含该转录本的细胞系
            # =================================================================
            for ct in self.cell_types:
                if tid not in active_tids_per_cell[ct]: 
                    continue # 如果这个细胞系不表达这个转录本，跳过
                    
                if self.has_tpm:
                    tpm_val = float(self.tpm_dicts[ct].get(query_id, 0.0))
                    if pd.isna(tpm_val) or tpm_val < 0: tpm_val = 0.0
                    log_tpm = np.log2(tpm_val + 1.0)
                else:
                    tpm_val = np.nan
                    log_tpm = 0.0
                
                # 为该细胞系追加记录
                for cand in final_cands:
                    # 使用 copy 避免字典引用污染
                    cell_cand = cand.copy()
                    cell_cand['Tid'] = tid
                    cell_cand['Cell_Type'] = ct           # [NEW] 标记细胞系
                    cell_cand['seq_score'] = cand['score']
                    cell_cand['transcription_score'] = log_tpm
                    cell_cand['tpm'] = tpm_val
                    
                    # 清理不需要的内部临时分数
                    cell_cand.pop('score', None)
                    all_records.append(cell_cand)
                
        if not all_records:
            print("No valid ORFs were found.")
            return pd.DataFrame()
            
        final_df = pd.DataFrame(all_records)
        
        # [MODIFIED] 增加 Cell_Type 列
        cols = ['Cell_Type', 'Tid', 'start', 'stop', 'length', 'start_codon', 'seq_score', 'transcription_score', 'tpm']
        final_df = final_df[cols].sort_values(by=['Cell_Type', 'transcription_score', 'seq_score'], ascending=[True, False, False])
        
        save_path = os.path.join(out_dir, f"baseline_called_orfs.multicell.csv")
        final_df.to_csv(save_path, index=False)
        
        print(f"\n🎉 Baseline Calling Completed! Found {len(final_df)} cell-specific ORFs.")
        print(f"Results saved to: {save_path}")
        
        return final_df
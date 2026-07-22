import os
import pickle
import contextlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List, Union
from tqdm import tqdm

from eval.save_prediction_results import _prepare_prediction_dataloader
from utils import unwrap_model, clean_up_memory


def get_active_transcripts(
    tpm_csv_path: str, 
    mapping_csv_path: str, 
    # accept a single string or a list of strings
    cell_type: Union[str, List[str]], 
    min_tpm: float = 0.5
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Reads the TPM CSV matrix and a Gene-to-Transcript mapping table.
    - If cell_type is a string, returns an array of active Transcript IDs.
    - If cell_type is a list, returns a dict of {cell_type: array_of_active_Transcript_IDs}.
    """
    # =================================================================
    # 1. Normalize input types
    # =================================================================
    is_single_input = isinstance(cell_type, str)
    target_cells = [cell_type] if is_single_input else cell_type

    # =================================================================
    # 2. Load data once (shared across all cell types for speed)
    # =================================================================
    print(f"Loading TPM matrix from: {tpm_csv_path}")
    try:
        df = pd.read_csv(tpm_csv_path, index_col=0)
    except Exception as e:
        raise RuntimeError(f"Failed to load TPM CSV. Ensure the path is correct: {e}")

    print(f"Loading mapping table from: {mapping_csv_path}")
    try:
        mapping_df = pd.read_csv(mapping_csv_path, sep='\t')
    except Exception as e:
        raise RuntimeError(f"Failed to load Mapping CSV: {e}")
        
    gene_col = 'Gene stable ID'
    tx_col = 'Transcript stable ID'
    if gene_col not in mapping_df.columns or tx_col not in mapping_df.columns:
        raise ValueError(f"Mapping table must contain '{gene_col}' and '{tx_col}' columns.")

    # =================================================================
    # 3. Iterate over cell types
    # =================================================================
    results = {}
    print(f"Extracting active transcripts (TPM > {min_tpm}) for {len(target_cells)} cell types...")
    
    for ct in target_cells:
        if ct not in df.columns:
            print(f"  [Warning] Cell type '{ct}' not found in TPM matrix. Skipping.")
            results[ct] = np.array([])
            continue

        # filter active genes
        active_mask = df[ct] > min_tpm
        active_gene_ids = df[active_mask].index.values
        
        # map to transcripts
        active_mapping = mapping_df[mapping_df[gene_col].isin(active_gene_ids)]
        active_transcript_ids = active_mapping[tx_col].unique()
        
        results[ct] = active_transcript_ids
        print(f"  -> {ct}: {len(active_gene_ids)} active genes mapped to {len(active_transcript_ids)} unique transcripts.")
    
    # =================================================================
    # 4. Return the appropriate structure based on input type
    # =================================================================
    if is_single_input:
        return results[cell_type]
    
    return results

# =================================================================
# Utility: FASTA parser
# =================================================================
def read_fasta(file_paths: Union[str, List[str]]) -> Dict[str, str]:
    """Read one or more FASTA files, return merged {tid: sequence} dict."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        
    seq_dict = {}
    total_files = len(file_paths)
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"[Warning] Fasta file not found: {file_path}. Skipping...")
            continue
            
        print(f"Reading Fasta File: {file_path}")
        curr_tid = ""
        curr_seq = []
        file_seq_count = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if curr_tid:
                        seq_dict[curr_tid] = "".join(curr_seq)
                        file_seq_count += 1
                    # extract the ID after '>', typically the first space-delimited token
                    curr_tid = line[1:].split()[0]
                    curr_seq = []
                else:
                    curr_seq.append(line.upper())
                    
            if curr_tid:
                seq_dict[curr_tid] = "".join(curr_seq)
                file_seq_count += 1
                
        print(f"  -> Loaded {file_seq_count} sequences from this file.")
        
    print(f"✅ Successfully loaded a total of {len(seq_dict)} unique sequences from {total_files} file(s).")
    return seq_dict

# =================================================================
# Zero-shot inference Dataset
# =================================================================
class DeNovoSequenceDataset(Dataset):
    """
    Lightweight in-memory Dataset for translation profile prediction from RNA sequence only.
    """
    def __init__(self, 
                 seq_dict: Dict[str, str], 
                 species: str,
                 cell_type: str, 
                 cell_expr_vector: np.ndarray,
                 min_len: int = 200,
                 max_len: int = 20000):
        self.tids = list(seq_dict.keys())
        self.seq_dict = seq_dict
        self.species = species
        self.cell_type = cell_type
        self.cell_expr_vector = np.array(cell_expr_vector, dtype=np.float32)
        
        self.nt_mapping = dict(zip("ACGTN", range(5)))
        self.min_len = min_len
        self.max_len = max_len  # maximum supported transcript length
        
        self.uuids = []
        self.seq_embs = []
        self.lengths = []
        
        # pre-compute one-hot encoding and record lengths
        for tid in tqdm(self.tids, desc="Encoding Sequences"):
            seq = self.seq_dict[tid].upper()
            
            # discard transcripts exceeding the maximum length
            if len(seq) > self.max_len or len(seq) < self.min_len:
                continue
                
            self.lengths.append(len(seq))
            
            # clean IDs to avoid version-number or compound-format issues in downstream mapping
            tid_clean = str(tid).split('|')[0]
            if tid_clean.startswith('ENST') and '.' in tid_clean:
                tid_clean = tid_clean.split('.')[0]
                
            uuid = f"{tid_clean}-{self.species}-{self.cell_type}-Prediction"
            self.uuids.append(uuid)
            
            seq_idx = [self.nt_mapping.get(nt, 4) for nt in seq]
            seq_emb = np.eye(5)[seq_idx, :4]
            self.seq_embs.append(seq_emb)
            
        self.n_samples = len(self.lengths)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        species = str(self.species)
        cell_expr_tensor = torch.from_numpy(self.cell_expr_vector)
        seq_emb = torch.from_numpy(self.seq_embs[idx]).float()
        
        # generate a placeholder count matrix (all zeros) matching the truncated length
        count_emb = torch.zeros((self.lengths[idx], 1), dtype=torch.float32)
        
        # return an empty meta_info dict as placeholder
        meta_info = {}
        
        return uuid, species, cell_expr_tensor, meta_info, seq_emb, count_emb

def collate_fn_denovo(batch):
    """Dedicated collation function."""
    uuids, species, cell_exprs, meta_infos, seq_embs, count_embs = zip(*batch)
    lengths = [s.shape[0] for s in seq_embs]
    
    seq_padded = pad_sequence(seq_embs, batch_first=True, padding_value=-1)
    count_padded = pad_sequence(count_embs, batch_first=True, padding_value=-1)
    species_list = list(species)
    cell_exprs = torch.stack(cell_exprs)
    
    return uuids, species_list, cell_exprs, meta_infos, seq_padded, count_padded, lengths


class TranslationProfilePredictor:
    def __init__(self, 
                 model: torch.nn.Module, 
                 fasta_files: Union[str, List[str]]):
        self.model = model
        self.fasta_files = fasta_files

        print(f"\nInitializing Fasta parsing pipeline...")
        # supports both a single string and a list of strings
        self.seq_dict = read_fasta(self.fasta_files)

    def run(
            self,
            species: str,
            cell_type: str,
            cell_expr_vector: np.ndarray, 
            target_tids: Optional[list] = None, 
            out_dir: str = "./results",
            suffix: str = "results",
            min_len: int = 200,
            max_len: int = 20000,
            batch_size: int = 32,
            rank: Optional[int] = None, 
            world_size: Optional[int] = None):
        """
        Run FASTA reading and prediction.
        If target_tids is provided, only predict transcripts present in that list.
        """

        os.makedirs(out_dir, exist_ok=True)
        model_name = unwrap_model(self.model).model_name
        pred_pkl_path = os.path.join(out_dir, f"predictions_count.{model_name}.{suffix}.pkl")

        # ========================================================
        # Filter logic for target transcripts (strip ENST version numbers)
        # ========================================================
        if target_tids is not None:
            # 1. Preprocess target_tids: strip version suffix if ENST-prefixed
            cleaned_target_tids = []
            for t in target_tids:
                t_str = str(t).split('|')[0]
                if t_str.startswith("ENST") and "." in t_str:
                    cleaned_target_tids.append(t_str.split(".")[0])
                else:
                    cleaned_target_tids.append(t_str)
                    
            target_set = set(cleaned_target_tids)
            filtered_seq_dict = {}
            
            # 2. Preprocess FASTA dictionary keys
            for tid, seq in self.seq_dict.items():
                # remove pipe separators
                clean_tid = str(tid).split('|')[0]
                
                # If the FASTA ID is ENST-prefixed, also strip the version suffix for matching
                if clean_tid.startswith("ENST") and "." in clean_tid:
                    clean_tid = clean_tid.split(".")[0]
                    
                if clean_tid in target_set:
                    # Note: keep the original tid in the dictionary key to preserve consistency with downstream results and pickle output
                    filtered_seq_dict[tid] = seq
            
            print(f"Filtered Fasta: Keeping {len(filtered_seq_dict)} sequences matching target Tids "
                  f"(out of {len(self.seq_dict)} total).")
            seq_dict = filtered_seq_dict
            
            if not seq_dict:
                print("Warning: No matching sequences found! Please check if your Tids match the Fasta headers.")
                return None
        else:
            seq_dict = self.seq_dict
                
        print("\nRunning Deep Learning Translation Prediction...")
        self.model.eval()
        base_model = unwrap_model(self.model)
        device = base_model.device
        
        # build Dataset and DataLoader
        dataset = DeNovoSequenceDataset(seq_dict, species, cell_type, cell_expr_vector, min_len, max_len)
        dataloader, run_rank, run_world_size = _prepare_prediction_dataloader(
            dataset, collate_fn_denovo, num_samples=None, batch_size=batch_size,
            rank=rank, world_size=world_size
        )
        
        saved_data = {cell_type: {}}
        iterator = tqdm(dataloader, desc=f"Predicting") if (run_rank == 0 or run_world_size == 1) else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                b_uuids, species_list, b_cell_exprs, b_meta, b_seq, b_count, b_lengths = batch
                
                b_cell_exprs = b_cell_exprs.to(device)
                b_seq = b_seq.to(device)
                src_mask = (b_seq != -1).any(dim=-1)
                amp_context = (
                    torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )

                with amp_context:
                    out = base_model.predict(
                        seq_batch=b_seq,
                        species=species_list,
                        expr_vector=b_cell_exprs,
                        src_mask=src_mask,
                        head_names=["count"],
                    )
                
                probs_batch = out["count"]
                
                # parse and store in the target dictionary
                for i, uuid in enumerate(b_uuids):
                    valid_len = b_lengths[i]
                    pred_sample = probs_batch[i, :valid_len].cpu().numpy().astype(np.float16)
                    # restore original transcript ID
                    tid = str(uuid).split('-')[0]
                    saved_data[cell_type][tid] = pred_sample
                    
        total_preds = len(saved_data[cell_type])
        print(f"Saving {total_preds} predictions to {pred_pkl_path}")
        with open(pred_pkl_path, 'wb') as f:
            pickle.dump(saved_data, f)
            
        clean_up_memory()
        print("🎉 Translation Profile Prediction Completed Successfully!")
        
        return pred_pkl_path

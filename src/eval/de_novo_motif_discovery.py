"""
De novo motif and positional feature discovery for translation regulation.

All position metrics are mapped to a Metagene Coordinate System:
  - 5' UTR: True nucleotide distance from CDS start (< 0)
  - CDS: Length-proportionally mapped to a fixed length (e.g., 900 nt), strictly preserving reading frame.
  - 3' UTR: True nucleotide distance from CDS stop (>= fixed_cds_len)
"""

import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Subset

from eval.environment_gene_attribution import (
    _load_gene_names,
    compute_adaLN_gene_attribution,
)
from model.base_model import BaseModel
from plot.de_novo_motif_discovery import (
    _assign_frame_colors,
    _cds_rect_data,
    _infer_attention_focus,
    _prepare_attention_heatmap_matrix,
    plot_attention_profile,
    plot_attention_profile_heatmap,
    plot_cluster_sequence_logos,
    plot_motif_metagene_heatmap,
    plot_regional_attention_dynamics,
    plot_saliency_profile,
    plot_sequence_logo,
)
from plot.environment_gene_attribution import plot_gene_attribution
from utils import unwrap_model

# ============================================================
# Global Parameter
# ============================================================
FIXED_CDS_LEN = 600  # The normalized length for all CDS regions (must be a multiple of 3)

# ============================================================
# Metagene Mapping Utilities
# ============================================================
def _map_to_metagene(pos, cds_start, cds_end, fixed_cds_len=FIXED_CDS_LEN):
    """
    Map absolute physical position to a unified metagene coordinate (x_pos),
    preserving exact nucleotide distance in UTRs, and proportional length in CDS.
    Strictly preserves the 0/1/2 reading frame periodicity.
    """
    rel_start = pos - cds_start
    rel_stop = pos - cds_end

    if rel_start < 0: # 5' UTR
        x_pos = rel_start
    elif rel_stop >= 0: # 3' UTR
        x_pos = fixed_cds_len + rel_stop
    else: # CDS Internal proportional sampling
        cds_len = cds_end - cds_start
        codon_idx = rel_start // 3
        frame = rel_start % 3
        total_codons = cds_len // 3
        target_codons = fixed_cds_len // 3
        
        if total_codons > 0:
            # Proportional mapping at codon level to preserve frame
            mapped_codon = int(np.round((codon_idx / total_codons) * target_codons))
            mapped_codon = min(mapped_codon, target_codons - 1)
        else:
            mapped_codon = 0
            
        x_pos = mapped_codon * 3 + frame

    return x_pos, rel_start, rel_stop

# ============================================================
# Helper functions
# ============================================================
def _unwrap(model):
    """Return and validate the unwrapped BaseModel instance."""
    raw = unwrap_model(model)
    if type(raw) is not BaseModel:
        raise TypeError(
            "de_novo_motif_discovery requires an exact "
            "model.base_model.BaseModel instance."
        )
    return raw


def _model_device(model, device=None):
    """Resolve an explicit or model-owned torch device."""
    model_device = next(model.parameters()).device
    if device is None:
        return model_device
    requested_device = torch.device(device)
    same_device = (
        requested_device.type == model_device.type
        and (
            requested_device.index is None
            or requested_device.index == model_device.index
        )
    )
    if not same_device:
        raise ValueError(
            f"Requested device {requested_device} does not match the model "
            f"device {model_device}. Move the model before running analysis."
        )
    return model_device


def _extract_count_profile(output):
    """Extract a single-channel positional profile from the count head."""
    if not isinstance(output, dict) or 'count' not in output:
        raise KeyError("BaseModel output must contain a 'count' head.")
    profile = output['count']
    if isinstance(profile, dict):
        if 'profile' not in profile:
            raise KeyError("The count head dictionary has no 'profile' tensor.")
        profile = profile['profile']
    if not torch.is_tensor(profile):
        raise TypeError("The count head must return a tensor.")
    if profile.ndim != 3 or profile.shape[-1] != 1:
        raise ValueError(
            "The count head must return shape (batch, length, 1), "
            f"got {tuple(profile.shape)}."
        )
    return profile


def _sequence_mask(seq_tensor):
    """Return an all-valid mask for one unpadded dataset sequence."""
    return torch.ones(
        seq_tensor.shape[:2],
        dtype=torch.bool,
        device=seq_tensor.device,
    )


def _resolve_base_model_context(raw, sample, seq_tensor, device):
    """Build BaseModel sequence embeddings and environmental conditioning."""
    expr = sample['ev']
    expr_tensor = (
        torch.from_numpy(expr).float().unsqueeze(0).to(device)
        if expr is not None and len(expr) > 0
        else None
    )
    resolved_expr = raw._resolve_expr_vector(
        cell_type=sample['ct'],
        expr_vector=expr_tensor,
        batch_size=1,
    ).to(device)
    species_idx = raw._normalize_species(sample['species'], 1).to(device)
    species_emb = raw.species_embedding(species_idx)
    compact_style = raw.expr_projector(
        torch.cat([resolved_expr, species_emb], dim=-1)
    )
    src_reps = raw.seq_embedding(seq_tensor)
    return expr_tensor, compact_style, src_reps, _sequence_mask(seq_tensor)


def _adaln_parameters(sublayer, compact_style):
    """Resolve AdaLN parameters exactly as BaseModel does during forward."""
    gamma, beta, alpha = sublayer.adaLN_modulation(
        compact_style
    ).chunk(3, dim=-1)
    bounds = getattr(sublayer, 'adaln_modulation_bounds', None)
    if bounds is not None:
        gamma_bound, beta_bound, alpha_bound = bounds
        gamma = sublayer._smooth_bound(gamma, gamma_bound)
        beta = sublayer._smooth_bound(beta, beta_bound)
        alpha = sublayer._smooth_bound(alpha, alpha_bound)
    return gamma, beta, alpha


def _prepare_adaln_input(sublayer, reps, compact_style):
    """Prepare the standard BaseModel Pre-AdaLN sublayer input."""
    gamma, beta, alpha = _adaln_parameters(sublayer, compact_style)
    normalized = (
        (1 + gamma.unsqueeze(1)) * sublayer.LN(reps)
        + beta.unsqueeze(1)
    )
    return normalized, alpha


def _apply_adaln_residual(sublayer, reps, output, gate):
    """Apply the standard BaseModel gated AdaLN residual update."""
    return reps + gate.unsqueeze(1) * sublayer.dropout(output)

def _extract_sample(dataset, idx):
    item = dataset[idx]
    if len(item) < 6:
        raise ValueError(
            "Each dataset sample must provide uuid, species, cell_type, "
            "expr_vector, meta_info, and seq_emb."
        )
    uuid, species, ct, ev, mi, se = item[:6]

    se_np = (
        se.detach().cpu().numpy()
        if torch.is_tensor(se)
        else np.asarray(se)
    )
    if se_np.ndim != 2 or se_np.shape[-1] != 4:
        raise ValueError(
            "seq_emb must have shape (length, 4), "
            f"got {se_np.shape}."
        )

    if torch.is_tensor(ev):
        ev_np = ev.detach().cpu().numpy()
    else:
        ev_np = np.array(ev) if ev is not None else None
    if ev_np is not None and ev_np.ndim == 1 and ev_np.shape[0] == 0:
        ev_np = None

    cds_start = int(mi.get('cds_start_pos', -1)) - 1 if isinstance(mi, dict) else -1
    cds_end = int(mi.get('cds_end_pos', -1)) if isinstance(mi, dict) else -1
    uuid_text = str(uuid)
    tid = uuid_text.rsplit('-', 2)[0] if '-' in uuid_text else uuid_text
    if tid.startswith('ENST'):
        tid = tid.split('.', 1)[0]

    return {
        'se': se_np, 'ev': ev_np,
        'meta_info': mi,
        'cds_start_0': cds_start, 'cds_end_0': cds_end,
        'L': se_np.shape[0], 'ct': ct, 'species': species, 'tid': tid,
        'valid': cds_start >= 0 and cds_end > cds_start
    }


def _select_unique_transcript_samples(
        dataset,
        n_samples,
        min_len=None,
        max_len=None,
        random_state=42):
    """Select eligible dataset samples with at most one row per transcript."""
    if n_samples is not None and n_samples < 1:
        raise ValueError("n_samples must be positive or None.")
    if min_len is not None and min_len < 0:
        raise ValueError("min_len must be non-negative or None.")
    if max_len is not None and max_len < 1:
        raise ValueError("max_len must be positive or None.")
    if (
        min_len is not None
        and max_len is not None
        and min_len > max_len
    ):
        raise ValueError("min_len must not exceed max_len.")

    rng = np.random.default_rng(random_state)
    representatives = {}
    occurrence_counts = defaultdict(int)
    eligible_count = 0
    ineligible_count = 0

    for idx in range(len(dataset)):
        sample = _extract_sample(dataset, idx)
        if (
            not sample['valid']
            or (min_len is not None and sample['L'] < min_len)
            or (max_len is not None and sample['L'] > max_len)
        ):
            ineligible_count += 1
            continue

        eligible_count += 1
        tid = sample['tid']
        occurrence_counts[tid] += 1
        if (
            tid not in representatives
            or rng.random() < 1.0 / occurrence_counts[tid]
        ):
            representatives[tid] = (idx, sample)

    unique_tids = np.asarray(list(representatives), dtype=object)
    if n_samples is not None and len(unique_tids) > n_samples:
        selected_tids = rng.choice(unique_tids, n_samples, replace=False)
    else:
        selected_tids = unique_tids
        rng.shuffle(selected_tids)
    selected = [representatives[tid] for tid in selected_tids]
    duplicate_count = eligible_count - len(representatives)

    print(
        f"Selected {len(selected)} unique transcripts; skipped "
        f"{duplicate_count} duplicate transcript rows and "
        f"{ineligible_count} ineligible rows."
    )
    return selected

# ============================================================
# Phase 1A: Attention positional importance
# ============================================================
def extract_attention_positional_importance(
        model,
        dataset,
        n_samples=200,
        min_len=500,
        max_len=1200,
        device=None,
        random_state=42):
    """Aggregate attention using at most one dataset row per transcript."""
    raw = _unwrap(model)
    device = _model_device(raw, device)
    raw.eval()

    n_heads = raw.n_heads
    head_dim = raw.encoder.encoder_layers[0].multi_headed_attention.head_dim

    # Now using a unified metagene accumulator
    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0, 'rel_start_sum': 0.0, 'rel_stop_sum': 0.0})
    selected_samples = _select_unique_transcript_samples(
        dataset,
        n_samples=n_samples,
        min_len=min_len,
        max_len=max_len,
        random_state=random_state,
    )
    valid_count = 0

    for _, s in tqdm(
            selected_samples,
            desc="Attention positional importance"):

        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device)
        
        cds_start, cds_end = s['cds_start_0'], s['cds_end_0']

        with torch.no_grad():
            _, compact_style, src_reps, src_mask = (
                _resolve_base_model_context(raw, s, se, device)
            )

            for layer_idx, enc_layer in enumerate(raw.encoder.encoder_layers):
                sub = enc_layer.sublayers[0]
                normed, attention_gate = _prepare_adaln_input(
                    sub,
                    src_reps,
                    compact_style,
                )

                attn_mod = enc_layer.multi_headed_attention
                bs_, Lc, d = normed.shape

                q = attn_mod.toqueries(normed).view(bs_, Lc, n_heads, head_dim).transpose(1, 2)
                k = attn_mod.tokeys(normed).view(bs_, Lc, n_heads, head_dim).transpose(1, 2)
                v = attn_mod.tovalues(normed).view(bs_, Lc, n_heads, head_dim).transpose(1, 2)

                if hasattr(attn_mod, 'RoPE'):
                    q = attn_mod.RoPE(q)
                    k = attn_mod.RoPE(k)

                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
                mask = src_mask[:, :Lc].unsqueeze(1).unsqueeze(2)
                scores.masked_fill_(~mask, float('-inf'))
                attn_w = torch.softmax(scores, dim=-1)
                received_by_head = attn_w.sum(dim=2)[0].cpu().numpy()

                for head_idx in range(n_heads):
                    for pos in range(Lc):
                        x_pos, rel_start, rel_stop = _map_to_metagene(
                            pos, cds_start, cds_end, FIXED_CDS_LEN
                        )
                        attention_value = float(
                            received_by_head[head_idx, pos]
                        )
                        key = (layer_idx, head_idx, x_pos)
                        accum[key]['sum'] += attention_value
                        accum[key]['sum_sq'] += attention_value ** 2
                        accum[key]['n'] += 1
                        accum[key]['rel_start_sum'] += rel_start
                        accum[key]['rel_stop_sum'] += rel_stop

                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.transpose(1, 2).reshape(bs_, Lc, n_heads * head_dim)
                attn_out = attn_mod.unifyheads(attn_out)
                if hasattr(attn_mod, 'dropout'):
                    attn_out = attn_mod.dropout(attn_out)
                src_reps = _apply_adaln_residual(
                    sub,
                    src_reps,
                    attn_out,
                    attention_gate,
                )

                sub2 = enc_layer.sublayers[1]
                normed2, ffn_gate = _prepare_adaln_input(
                    sub2,
                    src_reps,
                    compact_style,
                )
                src_reps = _apply_adaln_residual(
                    sub2,
                    src_reps,
                    enc_layer.ffn(normed2),
                    ffn_gate,
                )

        valid_count += 1

    records = []
    for (layer, head, x_pos), v in accum.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records.append({
                'layer': layer,
                'head': head,
                'x_pos': x_pos,
                'mean_attn': mean,
                'std_attn': std,
                'pos_from_cds_start': v['rel_start_sum'] / v['n'], # Averaged physical relative start
                'pos_from_cds_stop': v['rel_stop_sum'] / v['n'],   # Averaged physical relative stop
                'n_contrib': v['n'],
            })

    df = pd.DataFrame(records, columns=[
        'layer', 'head', 'x_pos', 'mean_attn', 'std_attn',
        'pos_from_cds_start', 'pos_from_cds_stop', 'n_contrib',
    ])
    if not df.empty:
        df = df.sort_values(['layer', 'head', 'x_pos'])
    print(
        f"Attention aggregated: {len(df)} layer-head metagene positions "
        f"from {valid_count} samples."
    )
    return df

# ============================================================
# Phase 1B: Input saliency for mean CDS output
# ============================================================
def compute_saliency_profile(
        model,
        dataset,
        n_samples=100,
        max_len=1200,
        device=None,
        random_state=42):
    """Aggregate input saliency using one representative row per transcript."""
    raw = _unwrap(model)
    device = _model_device(raw, device)
    raw.eval()

    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0, 'rel_start_sum': 0.0, 'rel_stop_sum': 0.0})
    selected_samples = _select_unique_transcript_samples(
        dataset,
        n_samples=n_samples,
        max_len=max_len,
        random_state=random_state,
    )
    valid_count = 0

    for _, s in tqdm(
            selected_samples,
            desc="Input saliency (Mean CDS Output)"):
            
        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device).requires_grad_(True)
        ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device) if s['ev'] is not None and len(s['ev']) > 0 else None
        
        L, cds_start, cds_end = s['L'], s['cds_start_0'], s['cds_end_0']

        raw.eval()
        with torch.enable_grad():
            out = raw.forward(
                seq_batch=se,
                cell_type=s['ct'],
                expr_vector=ev,
                species=s['species'],
                src_mask=_sequence_mask(se),
                head_names=['count'],
            )
            pred = _extract_count_profile(out)
            te = pred[0, cds_start:cds_end, 0].mean()

        te.backward()
        grad = se.grad[0].detach().cpu().numpy()
        sal = np.abs(grad).sum(axis=-1)

        for pos in range(L):
            x_pos, rel_start, rel_stop = _map_to_metagene(pos, cds_start, cds_end, FIXED_CDS_LEN)
            accum[x_pos]['sum'] += float(sal[pos])
            accum[x_pos]['sum_sq'] += float(sal[pos]) ** 2
            accum[x_pos]['n'] += 1
            accum[x_pos]['rel_start_sum'] += rel_start
            accum[x_pos]['rel_stop_sum'] += rel_stop

        se.grad = None
        valid_count += 1

    records = []
    for x_pos, v in accum.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records.append({
                'x_pos': x_pos,
                'mean_saliency': mean, 'std_saliency': std,
                'pos_from_cds_start': v['rel_start_sum'] / v['n'],
                'pos_from_cds_stop': v['rel_stop_sum'] / v['n'],
                'n_contrib': v['n'],
            })

    df = pd.DataFrame(records, columns=[
        'x_pos', 'mean_saliency', 'std_saliency',
        'pos_from_cds_start', 'pos_from_cds_stop', 'n_contrib',
    ])
    if not df.empty:
        df = df.sort_values('x_pos')
    print(f"Saliency aggregated: {len(df)} metagene positions from {valid_count} samples")
    return df



# ============================================================
# Notebook Cell 1: Transcript-Level Physical Sequence Slicing
# ============================================================
import logging
# Silence matplotlib warnings about unavailable fonts.
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def _slice_and_append(seq, pos, r, target_list, tid, cds_start, cds_end,
                       attn_score=None):
    """Extract a window around a peak position and record all positional metadata."""
    if pos - r >= 0 and pos + r + 1 <= len(seq):
        fragment = seq[pos - r : pos + r + 1]
        if 'N' not in fragment:
            rel_start = pos - cds_start
            rel_stop = pos - cds_end
            x_pos, _, _ = _map_to_metagene(pos, cds_start, cds_end, FIXED_CDS_LEN)

            record = {
                'tid': tid,
                'sequence': fragment,
                'abs_pos': pos,
                'rel_to_cds_start': rel_start,
                'rel_to_cds_end': rel_stop,
                'x_pos': x_pos,
            }
            if attn_score is not None:
                record['mean_attn'] = attn_score
            target_list.append(record)



def split_and_extract_contrastive_peaks(
        model, dataset, seq_dict, out_dir,
        min_len=500, max_len=4000, attn_perc=75,
        top_ratio=0.20, window_radius=10, device=None):
    """
    [Updated]: Automatically parses `te_scale` from dataset's meta_info to split transcripts 
    into High TE and Low TE groups. 
    Added strict sequence length filtering (min_len, max_len) to prevent GPU OOM caused by O(L^2) attention matrices.
    Generates `transcript_te_dict` for downstream RBP scanning.
    """
    _unwrap(model)
    if not 0 < top_ratio <= 0.5:
        raise ValueError("top_ratio must be in the interval (0, 0.5].")
    print(f"--- Step 1: Parsing TE scales and filtering by length ({min_len} - {max_len} nt) ---")
    
    te_records = []
    transcript_te_dict = {}
    
    # Iterate through the dataset and extract the true TE scale
    for i in range(len(dataset)):
        try:
            sample = _extract_sample(dataset, i)
            tid = sample['tid']
            
            # [CRITICAL UPDATE]: Pre-filter by sequence length to prevent OOM
            if tid not in seq_dict:
                continue
            
            seq_length = len(seq_dict[tid])
            if seq_length < min_len or seq_length > max_len:
                continue # Skip transcripts that are too long or too short
                
            meta_info = sample['meta_info']
            te_val = (
                meta_info.get("te_scale", None)
                if isinstance(meta_info, dict)
                else None
            )
            
            if te_val is not None:
                transcript_te_dict[tid] = float(te_val)
                te_records.append((i, float(te_val)))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
            
    # Sort in descending order based on te_scale (from ~2.0 down to ~-2.0)
    te_records.sort(key=lambda x: x[1], reverse=True)
    n_total = len(te_records)
    
    if n_total == 0:
        print("Error: No valid transcripts found after length filtering!")
        return {}, {}, {}
        
    n_extreme = max(1, int(n_total * top_ratio))
    
    # Get original indices for the extreme groups
    high_te_indices = [x[0] for x in te_records[:n_extreme]]
    low_te_indices = [x[0] for x in te_records[-n_extreme:]]
    
    high_te_dataset = Subset(dataset, high_te_indices)
    low_te_dataset = Subset(dataset, low_te_indices)
    
    print(f"Data split successful: {n_total} total valid transcripts parsed within length limits.")
    print(f"  -> Top {top_ratio*100}% High TE: {len(high_te_dataset)} transcripts.")
    print(f"  -> Bottom {top_ratio*100}% Low TE: {len(low_te_dataset)} transcripts.")
    
    # Empty CUDA cache to clear memory fragmentation before heavy extraction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Extract peaks for contrastive groups
    print("\n--- Step 2: Extracting Attention Peaks for [High TE] Group ---")
    high_te_dfs = extract_attn_peaks_by_region(
        model, high_te_dataset, seq_dict, 
        out_dir=os.path.join(out_dir, "High_TE"), 
        n_samples=len(high_te_dataset), 
        window_radius=window_radius, 
        min_len=min_len,      # Pass length filters down
        max_len=max_len,
        perc=attn_perc,
        device=device
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("\n--- Step 3: Extracting Attention Peaks for [Low TE] Group ---")
    low_te_dfs = extract_attn_peaks_by_region(
        model, low_te_dataset, seq_dict, 
        out_dir=os.path.join(out_dir, "Low_TE"), 
        n_samples=len(low_te_dataset), 
        window_radius=window_radius, 
        min_len=min_len,      # Pass length filters down
        max_len=max_len,
        perc=attn_perc,
        device=device
    )
    
    return high_te_dfs, low_te_dfs, transcript_te_dict


def extract_attn_peaks_by_region(
        model, dataset, seq_dict, out_dir, 
        n_samples=300, window_radius=10, perc=75,
        min_len=0, max_len=float('inf'), device=None):
    """
    Iterate through transcripts, compute absolute physical position Attention scores,
    split by 5' UTR, CDS, 3' UTR, and extract the strongest peak context based on internal attention.
    
    [Optimized]: 
    1. Switched from Saliency (gradients) to Internal Multi-Head Attention weights.
    2. CDS region is expanded by `window_radius` upstream and downstream to naturally 
       capture boundary motifs (e.g., Kozak and Stop codon contexts) within the CDS group.
    3. Implements 1D Non-Maximum Suppression (NMS) to prevent overlapping extraction.
    
    Automatically saves individual raw dataframes to CSV files in the out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    raw = _unwrap(model)
    device = _model_device(raw, device)
    raw.eval()

    region_sequences = {"5UTR": [], "CDS": [], "3UTR": []}
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Extracting raw physical ATTN peaks"):
        s = _extract_sample(dataset, idx)
        tid = s['tid']
        if not s['valid'] or tid not in seq_dict:
            continue

        seq = seq_dict[tid].upper()
        L = len(seq)
        
        if L < min_len or L > max_len:
            continue
            
        cds_start = s['cds_start_0']
        cds_end = s['cds_end_0']

        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, compact_style, src_reps, src_mask = (
                _resolve_base_model_context(raw, s, se, device)
            )
            
            Lc = se.shape[1] # Length with padding
            n_heads = raw.n_heads
            head_dim = raw.encoder.encoder_layers[0].multi_headed_attention.head_dim
            
            attn_track = np.zeros(Lc)

            # 3. Manually step through encoder layers to intercept Attention Weights.
            # [Memory-safe]: process one head at a time to avoid (B, H, L, L)
            # attention matrices that can consume >1 GB each for long sequences.
            for enc_layer in raw.encoder.encoder_layers:
                sub = enc_layer.sublayers[0]
                normed, attention_gate = _prepare_adaln_input(
                    sub,
                    src_reps,
                    compact_style,
                )

                attn_mod = enc_layer.multi_headed_attention
                bs_, _, d = normed.shape

                q = attn_mod.toqueries(normed)  # (1, Lc, n_heads*head_dim)
                k = attn_mod.tokeys(normed)
                v = attn_mod.tovalues(normed)

                # Reshape to per-head views without transposing into a big tensor
                q_h = q.view(bs_, Lc, n_heads, head_dim)  # (1, Lc, H, D)
                k_h = k.view(bs_, Lc, n_heads, head_dim)
                v_h = v.view(bs_, Lc, n_heads, head_dim)

                if hasattr(attn_mod, 'RoPE'):
                    q_h = attn_mod.RoPE(q_h.transpose(1, 2)).transpose(1, 2)
                    k_h = attn_mod.RoPE(k_h.transpose(1, 2)).transpose(1, 2)

                scale = np.sqrt(head_dim)
                mask_2d = src_mask[:, :Lc].to(device)  # (1, Lc)
                received_head = torch.zeros(Lc, device=device)

                all_attn_outs = []
                for h in range(n_heads):
                    qh = q_h[0, :, h, :]  # (Lc, D)
                    kh = k_h[0, :, h, :]  # (Lc, D)
                    vh = v_h[0, :, h, :]  # (Lc, D)

                    scores_h = torch.matmul(qh, kh.T) / scale  # (Lc, Lc)
                    scores_h.masked_fill_(~mask_2d, float('-inf'))
                    attn_w_h = torch.softmax(scores_h, dim=-1)

                    # Accumulate received attention for this head
                    received_head += attn_w_h.sum(dim=0)  # sum over queries → (Lc,)

                    # Compute attention output for this head
                    out_h = torch.matmul(attn_w_h, vh)  # (Lc, D)
                    all_attn_outs.append(out_h)

                    del scores_h, attn_w_h

                # Average received attention across heads
                attn_track += (received_head / n_heads).cpu().numpy()

                # Concatenate per-head outputs
                attn_out = torch.stack(all_attn_outs, dim=0)  # (H, Lc, D)
                attn_out = attn_out.transpose(0, 1).reshape(bs_, Lc, n_heads * head_dim)
                attn_out = attn_mod.unifyheads(attn_out)
                if hasattr(attn_mod, 'dropout'):
                    attn_out = attn_mod.dropout(attn_out)
                src_reps = _apply_adaln_residual(
                    sub,
                    src_reps,
                    attn_out,
                    attention_gate,
                )

                del q_h, k_h, v_h, all_attn_outs, received_head, attn_out

                # FFN sublayer
                sub2 = enc_layer.sublayers[1]
                normed2, ffn_gate = _prepare_adaln_input(
                    sub2,
                    src_reps,
                    compact_style,
                )
                src_reps = _apply_adaln_residual(
                    sub2,
                    src_reps,
                    enc_layer.ffn(normed2),
                    ffn_gate,
                )

        # Truncate to actual transcript length and calculate mean attention per layer
        attn_track = attn_track[:L] / len(raw.encoder.encoder_layers)

        # ==========================================
        # Step 1: Define Expanded Boundaries
        # ==========================================
        # [Fixed]: Re-added the -/+ window_radius to correctly capture boundary motifs into CDS
        utr5_bound = max(0, cds_start - window_radius)
        utr3_bound = min(L, cds_end + window_radius)

        # ==========================================
        # Step 2: Collect Candidate Peaks per Region
        # ==========================================
        candidate_peaks = []

        if utr5_bound > 0:
            utr5_attn = attn_track[:utr5_bound]
            if len(utr5_attn) > 0:
                peak_5 = np.argmax(utr5_attn)
                if utr5_attn[peak_5] > np.percentile(attn_track, perc): 
                    candidate_peaks.append((peak_5, utr5_attn[peak_5], "5UTR"))

        if utr3_bound > utr5_bound:
            cds_attn = attn_track[utr5_bound:utr3_bound]
            if len(cds_attn) > 0:
                peak_cds = np.argmax(cds_attn) + utr5_bound
                if attn_track[peak_cds] > np.percentile(attn_track, perc):
                    candidate_peaks.append((peak_cds, attn_track[peak_cds], "CDS"))

        if utr3_bound < L:
            utr3_attn = attn_track[utr3_bound:]
            if len(utr3_attn) > 0:
                peak_3 = np.argmax(utr3_attn) + utr3_bound
                if attn_track[peak_3] > np.percentile(attn_track, perc):
                    candidate_peaks.append((peak_3, attn_track[peak_3], "3UTR"))

        # ==========================================
        # Step 3: 1D Non-Maximum Suppression (NMS)
        # ==========================================
        candidate_peaks.sort(key=lambda x: x[1], reverse=True)
        
        selected_peaks = []
        for pos, score, region in candidate_peaks:
            conflict = False
            for sel_pos, _, _ in selected_peaks:
                if abs(pos - sel_pos) <= 2 * window_radius:
                    conflict = True
                    break
            
            if not conflict:
                selected_peaks.append((pos, score, region))

        # ==========================================
        # Step 4: Slice and Append Validated Peaks
        # ==========================================
        for pos, _, region in selected_peaks:
            _slice_and_append(seq, pos, window_radius, region_sequences[region], tid, cds_start, cds_end, attn_score=score)

    # Convert to DataFrames and automatically save to disk
    region_dfs = {}
    for region, data in region_sequences.items():
        if data:
            df = pd.DataFrame(data)
            df['Region'] = region
        else:
            df = pd.DataFrame(columns=['tid', 'sequence', 'abs_pos', 'rel_to_cds_start', 'rel_to_cds_end', 'x_pos', 'mean_attn', 'Region'])
        
        csv_filename = os.path.join(out_dir, f"raw_ATTN_peaks_{region}.csv")
        df.to_csv(csv_filename, index=False)
        region_dfs[region] = df

    print(f"\nSuccessfully sliced & saved raw ATTN CSVs: 5'UTR ({len(region_dfs['5UTR'])}), CDS ({len(region_dfs['CDS'])}), 3'UTR ({len(region_dfs['3UTR'])})")
    return region_dfs

def cluster_and_visualize_region_motifs(region_dfs, region_name, out_dir, min_clusters=4, max_clusters=10):
    """
    Perform unsupervised KMeans clustering on short physical sequences of a region.
    [Auto-K with Bounds]: Dynamically determines the optimal number of clusters 
    using Silhouette Score within a user-defined range [min_clusters, max_clusters].
    Appends cluster identifiers, saves to CSV, and exports PDF sequence logos.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    os.makedirs(out_dir, exist_ok=True)

    df = region_dfs.get(region_name, pd.DataFrame())
    
    # Check if the dataframe contains enough sequences to support the minimum clustering request
    if len(df) < min_clusters * 3:
        print(f"Skipping cluster for {region_name}: Insufficient sequences ({len(df)}) to form {min_clusters} clusters.")
        return df.copy()

    sequences = df['sequence'].tolist()

    # Digitization of bases via One-Hot encoding
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded_list = []
    for seq in sequences:
        int_meta = [char_map[c] for c in seq]
        one_hot = np.eye(4)[int_meta].flatten() 
        encoded_list.append(one_hot)
    
    X = np.array(encoded_list)
    
    # =========================================================
    # Auto-K Determination with Strict Bounds Check
    # =========================================================
    # Determine the maximum safe upper limit based on available sample size
    safe_max_limit = min(max_clusters + 1, len(X) // 3 + 1)
    
    best_k = min_clusters
    best_score = -1.0
    best_labels = None
    
    # If the database size allows searching within the bound
    if safe_max_limit > min_clusters + 1:
        for k in range(min_clusters, safe_max_limit):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            score = silhouette_score(X, kmeans.labels_)
            
            # Keep track of the highest Silhouette Score
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = kmeans.labels_
    else:
        # Fallback to min_clusters if the data size strictly restricts search space
        print(f"[{region_name}] Sample size is too tightly constrained. Forcing min_clusters={min_clusters}.")
        kmeans = KMeans(n_clusters=min_clusters, random_state=42, n_init=10).fit(X)
        best_k = min_clusters
        best_labels = kmeans.labels_

    print(f"\n========================================================")
    print(f"  [Auto-K Bound] {region_name} Optimal Clusters: {best_k} (Range: {min_clusters}-{max_clusters}, Score: {best_score:.3f})")
    print(f"========================================================")

    # Append results back to the dataframe copy
    df = df.copy()
    df['Cluster_ID'] = best_labels
    df['Motif_Name'] = region_name + "_Cluster" + df['Cluster_ID'].astype(str)

    # Save labeled dataset
    csv_save_path = os.path.join(out_dir, f"clustered_data_{region_name}.csv")
    df.to_csv(csv_save_path, index=False)

    plot_cluster_sequence_logos(df, region_name, out_dir)
    return df


# ============================================================
# Notebook Cell 3: Motif Spatial Probability Heatmap
# ============================================================

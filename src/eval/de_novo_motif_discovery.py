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
import logomaker
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from model.base_model import BaseModel
from utils import unwrap_model

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

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

def _inverse_metagene(x_pos, cds_start, cds_end, fixed_cds_len=FIXED_CDS_LEN):
    """
    Inverse map a unified metagene coordinate back to the absolute physical position 
    for a specific transcript (used for targeted mutagenesis).
    """
    cds_len = cds_end - cds_start
    
    if x_pos < 0:
        rel_start = x_pos
    elif x_pos >= fixed_cds_len:
        rel_start = cds_len + (x_pos - fixed_cds_len)
    else:
        target_codon = x_pos // 3
        frame = x_pos % 3
        total_codons = cds_len // 3
        target_codons = fixed_cds_len // 3
        
        if target_codons > 0:
            codon_idx = int(np.round((target_codon / target_codons) * total_codons))
            codon_idx = min(codon_idx, total_codons - 1)
        else:
            codon_idx = 0
        rel_start = codon_idx * 3 + frame
        
    return cds_start + rel_start

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
    tid = str(uuid).rsplit('-', 2)[0] if '-' in str(uuid) else str(uuid).split('.')[0]

    return {
        'se': se_np, 'ev': ev_np,
        'meta_info': mi,
        'cds_start_0': cds_start, 'cds_end_0': cds_end,
        'L': se_np.shape[0], 'ct': ct, 'species': species, 'tid': tid,
        'valid': cds_start >= 0 and cds_end > cds_start
    }

# ============================================================
# Phase 1A: Attention positional importance
# ============================================================
def extract_attention_positional_importance(
        model,
        dataset,
        n_samples=200,
        min_len=500,
        max_len=1200,
        device=None):
    """Aggregate received attention by layer, head, and metagene position."""
    raw = _unwrap(model)
    device = _model_device(raw, device)
    raw.eval()

    n_heads = raw.n_heads
    head_dim = raw.encoder.encoder_layers[0].multi_headed_attention.head_dim

    # Now using a unified metagene accumulator
    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0, 'rel_start_sum': 0.0, 'rel_stop_sum': 0.0})
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    valid_count = 0

    for idx in tqdm(indices, desc="Attention positional importance"):
        s = _extract_sample(dataset, idx)
        if not s['valid'] or s['L'] > max_len or s['L'] < min_len:
            continue

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
def compute_saliency_profile(model, dataset, n_samples=100, max_len=1200, device=None):
    raw = _unwrap(model)
    device = _model_device(raw, device)
    raw.eval()

    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0, 'rel_start_sum': 0.0, 'rel_stop_sum': 0.0})
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    valid_count = 0

    for idx in tqdm(indices, desc="Input saliency (Mean CDS Output)"):
        s = _extract_sample(dataset, idx)
        if not s['valid'] or s['L'] > max_len:
            continue
            
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


def run_differential_saliency(
        model,
        dataset,
        cell_type_A,
        cell_type_B,
        max_len=1200,
        n_samples=500,
        device=None):
    # 1. 从数据集中筛出特定细胞系的样本索引
    indices_A = [i for i, d in enumerate(dataset) if d[2] == cell_type_A]
    indices_B = [i for i, d in enumerate(dataset) if d[2] == cell_type_B]
    
    # 构建临时的小型 Dataset 子集 (方便传给 compute_saliency_profile)
    subset_A = Subset(dataset, indices_A)
    subset_B = Subset(dataset, indices_B)

    # 2. 分别计算
    print(f"Running Saliency for {cell_type_A}...")
    sal_A = compute_saliency_profile(model, subset_A, n_samples=min(n_samples, len(indices_A)), max_len=max_len, device=device)
    
    print(f"Running Saliency for {cell_type_B}...")
    sal_B = compute_saliency_profile(model, subset_B, n_samples=min(n_samples, len(indices_B)), max_len=max_len, device=device)
    
    # 3. 计算差分 (Delta Saliency)
    # 假设我们只关心在 A 中起作用而在 B 中不起作用的位点
    merged = pd.merge(sal_A[['x_pos', 'mean_saliency']], sal_B[['x_pos', 'mean_saliency']], on='x_pos', suffixes=('_A', '_B'))
    merged['delta_saliency'] = merged['mean_saliency_A'] - merged['mean_saliency_B']
    
    # 取 Delta 最大的前 50 个 Metagene 坐标
    top_diff_hotspots = merged.nlargest(50, 'delta_saliency')['x_pos'].tolist()
    print(f"Top Differential Hotspots (Active in {cell_type_A}, Silent in {cell_type_B}): {top_diff_hotspots}")
    
    return merged, top_diff_hotspots

# ============================================================
# Phase 1C: AdaLN gene attribution (Unchanged logic, just compacted)
# ============================================================
def _load_gene_names(gene_order_path=None, gene_annot_path="/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/ens_genes_v112.txt"):
    import os as _os
    if gene_order_path is None:
        gene_order_path = "/home/user/data3/rbase/translation_model/models/src/config/global_anchor_gene_order.txt"
    with open(gene_order_path) as f:
        ensg_list = [line.strip() for line in f if line.strip()]

    ensg2name = {}
    if gene_annot_path is not None:
        with open(gene_annot_path) as f:
            header = f.readline().strip().split('\t')
            gid_col = header.index('Gene stable ID') if 'Gene stable ID' in header else 0
            gname_col = header.index('Gene name') if 'Gene name' in header else 2
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) > max(gid_col, gname_col):
                    ensg2name[cols[gid_col]] = cols[gname_col]

    gene_names = [ensg2name.get(e, e) for e in ensg_list]
    return gene_names


def compute_adaLN_gene_attribution(
        model,
        gene_names=None,
        top_k=50,
        gene_annot_path=None,
        gene_order_path=None):
    raw = _unwrap(model)
    n_layers, d_expr = len(raw.encoder.encoder_layers), raw.d_expr
    if gene_names is None:
        gene_names = (
            _load_gene_names(gene_order_path=gene_order_path)
            if gene_annot_path is None
            else _load_gene_names(
                gene_order_path=gene_order_path,
                gene_annot_path=gene_annot_path,
            )
        )
    gene_names = list(gene_names)
    if len(gene_names) != d_expr:
        raise ValueError(
            f"gene_names length {len(gene_names)} does not match d_expr={d_expr}."
        )

    W_proj1 = raw.expr_projector[1].weight.detach().cpu().numpy()
    W_proj1_expr = np.abs(W_proj1[:, :d_expr])
    W_proj2 = np.abs(raw.expr_projector[4].weight.detach().cpu().numpy())

    all_attr = []
    for layer_idx in range(n_layers):
        for sub_idx, sub_name in [(0, 'attn'), (1, 'ffn')]:
            sublayer = raw.encoder.encoder_layers[layer_idx].sublayers[sub_idx]
            if not hasattr(sublayer, 'adaLN_modulation'):
                continue
            mod = sublayer.adaLN_modulation[1]
            W_ada = np.abs(mod.weight.detach().cpu().numpy())
            ada_imp = W_ada.sum(axis=0)
            gene_scores = ada_imp @ W_proj2 @ W_proj1_expr

            top_idx = np.argsort(gene_scores)[::-1][:top_k]
            for gi in top_idx:
                name = gene_names[gi] if gi < len(gene_names) else f"GENE_{gi}"
                all_attr.append({
                    'layer': layer_idx, 'layer_module': f'L{layer_idx}-{sub_name}',
                    'gene': name, 'gene_idx': gi, 'score': float(gene_scores[gi]),
                })

    df = pd.DataFrame(all_attr)
    if df.empty:
        return pd.DataFrame(columns=[
            'layer', 'layer_module', 'gene', 'gene_idx', 'score',
            'score_norm',
        ])
    df['score_norm'] = df.groupby('layer_module')['score'].transform(
        lambda values: (
            values / values.max()
            if values.max() > 0
            else np.zeros(len(values))
        )
    )
    return df

    
# ============================================================
# Plotting utilities — Single continuous axis, per-layer color
# ============================================================

def _assign_frame_colors(df):
    """
    Since the metagene coordinate x_pos intrinsically preserves the frame,
    we can safely calculate frame directly from x_pos.
    Frame 0: Red (#E41A1C), Frame 1: Blue (#377EB8), Frame 2: Gray (gray)
    """
    df['frame'] = df['x_pos'].astype(int) % 3
    # [Mod]: Frame 2 updated to 'gray' based on user preference
    color_map = {0: '#E41A1C', 1: '#377EB8', 2: 'gray'}
    df['frame_color'] = df['frame'].map(color_map)
    df['Frame'] = df['frame'].map({0: 'Frame 0', 1: 'Frame 1', 2: 'Frame 2'})
    df['Frame'] = pd.Categorical(df['Frame'], categories=['Frame 0', 'Frame 1', 'Frame 2'])
    return df

def _cds_rect_data():
    """Build geom_rect data for a single continuous CDS shading."""
    return pd.DataFrame({
        'xmin': [0], 
        'xmax': [FIXED_CDS_LEN], 
        'ymin': [-float('inf')], 
        'ymax': [float('inf')], 
        'fill': ['lightgray']
    })


def plot_attention_profile(attn_df, out_path="attention_profile.pdf", up_len=300, down_len=300, 
                           color_by_frame=True, xlim=None, show_xaxis=False, show_cds=True, 
                           weight=6, height=5):
    
    # [Mod]: Imported geom_point for scatter overlay
    from plotnine import (ggplot, aes, geom_line, geom_point, geom_rect,
                          labs, theme, facet_grid, scale_color_manual, scale_fill_identity,
                          element_text, theme_classic, element_blank, element_line)
    import pandas as pd
    import numpy as np
    
    # 1. Bounds filtering (support explicit xlim for zooming in)
    if xlim is not None:
        df_plot = attn_df[(attn_df['x_pos'] >= xlim[0]) & (attn_df['x_pos'] <= xlim[1])].copy()
    else:
        df_plot = attn_df[(attn_df['x_pos'] >= -up_len) & (attn_df['x_pos'] <= FIXED_CDS_LEN + down_len - 1)].copy()
        
    if df_plot.empty:
        raise ValueError("No attention positions remain within the plot range.")
    df_plot['layer'] = df_plot['layer'].astype(int)
    has_head_profiles = 'head' in df_plot.columns
    if has_head_profiles:
        df_plot['head'] = df_plot['head'].astype(int)

    # 2. Aggregation logic based on whether we group by Frame
    if color_by_frame:
        df_plot = _assign_frame_colors(df_plot)
        group_cols = ['layer', 'x_pos', 'Frame']
        head_group_cols = ['layer', 'head', 'x_pos', 'Frame']
    else:
        group_cols = ['layer', 'x_pos']
        head_group_cols = ['layer', 'head', 'x_pos']

    if has_head_profiles:
        df_head_plot = df_plot.groupby(
            head_group_cols, as_index=False, observed=True
        )[['mean_attn']].mean().dropna(subset=['mean_attn'])
        df_head_plot['log2_mean_attn'] = np.log2(
            df_head_plot['mean_attn'] + 1
        )
    else:
        df_head_plot = pd.DataFrame()

    df_plot = df_plot.groupby(group_cols, as_index=False, observed=True)[['mean_attn']].mean().dropna(subset=['mean_attn'])
    df_plot['log2_mean_attn'] = np.log2(df_plot['mean_attn'] + 1)

    base_out = out_path.replace('.pdf', '')
    rect_cds = _cds_rect_data()
    
    # Dynamic axis styling based on show_xaxis
    x_axis_text = element_text() if show_xaxis else element_blank()
    x_axis_ticks = element_line() if show_xaxis else element_blank()
    x_axis_title = element_text() if show_xaxis else element_blank()
    x_label_str = 'Metagene Position (x_pos)' if show_xaxis else ''

    # ==================================
    # Combined Plot
    # ==================================
    if color_by_frame:
        df_combined = df_plot.groupby(['x_pos', 'Frame'], as_index=False, observed=True)[['mean_attn']].mean()
    else:
        df_combined = df_plot.groupby(['x_pos'], as_index=False, observed=True)[['mean_attn']].mean()
        
    df_combined['log2_mean_attn'] = np.log2(df_combined['mean_attn'] + 1)

    p_comb = (
        ggplot(df_combined, aes(x='x_pos', y='log2_mean_attn'))
        + scale_fill_identity()
    )

    if show_cds:
        p_comb += geom_rect(data=rect_cds, mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'), alpha=0.3, inherit_aes=False, show_legend=False)

    if color_by_frame:
        frame_palette = {'Frame 0': '#D73027', 'Frame 1': '#4575B4', 'Frame 2': 'darkgray'}
        # [Mod]: Continuous single base-line underlying the points
        p_comb += geom_line(size=0.6, alpha=0.4, color='#333333') 
        # [Mod]: Overlay colored points to indicate reading frames
        p_comb += geom_point(aes(color='Frame'), size=3, alpha=1, stroke=0)
        p_comb += scale_color_manual(values=frame_palette)
    else:
        # Monocolor style
        p_comb += geom_line(size=0.6, alpha=0.4, color='#333333')

    p_comb += labs(x=x_label_str, y='log2(Mean attention + 1)') 
    p_comb += theme_classic()
    p_comb += theme(axis_text_x=x_axis_text, axis_ticks_major_x=x_axis_ticks, axis_title_x=x_axis_title, figure_size=(weight, height))
    
    p_comb.save(f"{base_out}.combined.pdf")
    print(f"Combined attention profile saved to {base_out}.combined.pdf")

    # ==================================
    # Per-layer Plot
    # ==================================
    n_layers = df_plot['layer'].nunique()
    df_plot['Layer'] = pd.Categorical([f'L{li}' for li in df_plot['layer']], categories=[f'L{i}' for i in range(n_layers)])

    rect_per_layer = pd.DataFrame({
        'Layer': pd.Categorical([f'L{i}' for i in range(n_layers)], categories=[f'L{i}' for i in range(n_layers)]),
        'xmin': [0] * n_layers, 'xmax': [FIXED_CDS_LEN] * n_layers,
        'ymin': [-float('inf')] * n_layers, 'ymax': [float('inf')] * n_layers,
        'fill': ['lightgray'] * n_layers
    })

    p_layers = (
        ggplot(df_plot, aes(x='x_pos', y='log2_mean_attn'))
        + scale_fill_identity()
    )

    if show_cds:
        p_layers += geom_rect(data=rect_per_layer, mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'), alpha=0.3, inherit_aes=False, show_legend=False)
    
    if color_by_frame:
        # [Mod]: Continuous base-line for faceted layers
        p_layers += geom_line(size=0.4, alpha=0.4, color='#333333')
        # [Mod]: Colored points overlay
        p_layers += geom_point(aes(color='Frame'), size=2, alpha=1, stroke=0)
        p_layers += scale_color_manual(values=frame_palette)
    else:
        p_layers += geom_line(size=0.4, alpha=0.4, color='#333333')

    p_layers += facet_grid('Layer ~ .', scales='free_y')
    p_layers += labs(x=x_label_str, y='log2(Mean attention + 1)')
    p_layers += theme_classic()
    p_layers += theme(axis_text_x=x_axis_text, axis_ticks_major_x=x_axis_ticks, axis_title_x=x_axis_title, 
                      strip_background=element_blank(), strip_text=element_text(size=12), figure_size=(weight, height*3))
    
    p_layers.save(f"{base_out}.per_layer.pdf")
    print(f"Per-layer attention profile saved to {base_out}.per_layer.pdf")

    output_paths = [
        f"{base_out}.combined.pdf",
        f"{base_out}.per_layer.pdf",
    ]
    if has_head_profiles and not df_head_plot.empty:
        head_values = sorted(df_head_plot['head'].unique())
        head_labels = [f'H{head}' for head in head_values]
        layer_labels = [f'L{layer}' for layer in range(n_layers)]
        df_head_plot['Layer'] = pd.Categorical(
            [f'L{layer}' for layer in df_head_plot['layer']],
            categories=layer_labels,
        )
        df_head_plot['Head'] = pd.Categorical(
            [f'H{head}' for head in df_head_plot['head']],
            categories=head_labels,
        )

        per_head_group_columns = ['head', 'x_pos']
        if color_by_frame:
            per_head_group_columns.append('Frame')
        per_head_df = df_head_plot.groupby(
            per_head_group_columns, as_index=False, observed=True
        )[['mean_attn']].mean()
        per_head_df['log2_mean_attn'] = np.log2(
            per_head_df['mean_attn'] + 1
        )
        per_head_df['Head'] = pd.Categorical(
            [f'H{head}' for head in per_head_df['head']],
            categories=head_labels,
        )
        rect_standalone_head = pd.DataFrame({
            'Head': pd.Categorical(head_labels, categories=head_labels),
            'xmin': [0] * len(head_labels),
            'xmax': [FIXED_CDS_LEN] * len(head_labels),
            'ymin': [-float('inf')] * len(head_labels),
            'ymax': [float('inf')] * len(head_labels),
            'fill': ['lightgray'] * len(head_labels),
        })

        per_head_plot = (
            ggplot(per_head_df, aes(x='x_pos', y='log2_mean_attn'))
            + scale_fill_identity()
        )
        if show_cds:
            per_head_plot += geom_rect(
                data=rect_standalone_head,
                mapping=aes(
                    xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax',
                    fill='fill',
                ),
                alpha=0.3,
                inherit_aes=False,
                show_legend=False,
            )
        per_head_plot += geom_line(
            size=0.4, alpha=0.4, color='#333333'
        )
        if color_by_frame:
            per_head_plot += geom_point(
                aes(color='Frame'), size=1.5, alpha=1, stroke=0
            )
            per_head_plot += scale_color_manual(values=frame_palette)
        per_head_plot += facet_grid('Head ~ .', scales='free_y')
        per_head_plot += labs(
            x=x_label_str,
            y='log2(Mean attention across layers + 1)',
        )
        per_head_plot += theme_classic()
        per_head_plot += theme(
            axis_text_x=x_axis_text,
            axis_ticks_major_x=x_axis_ticks,
            axis_title_x=x_axis_title,
            strip_background=element_blank(),
            strip_text=element_text(size=10),
            figure_size=(
                weight,
                max(height, 1.6 * len(head_labels) + 2),
            ),
        )
        standalone_head_path = f"{base_out}.per_head.pdf"
        per_head_plot.save(standalone_head_path)
        print(f"Per-head attention profile saved to {standalone_head_path}")
        output_paths.append(standalone_head_path)

        facet_pairs = pd.MultiIndex.from_product(
            [layer_labels, head_labels], names=['Layer', 'Head']
        ).to_frame(index=False)
        facet_pairs['Layer'] = pd.Categorical(
            facet_pairs['Layer'], categories=layer_labels
        )
        facet_pairs['Head'] = pd.Categorical(
            facet_pairs['Head'], categories=head_labels
        )
        rect_per_layer_head = facet_pairs.assign(
            xmin=0,
            xmax=FIXED_CDS_LEN,
            ymin=-float('inf'),
            ymax=float('inf'),
            fill='lightgray',
        )

        p_heads = (
            ggplot(df_head_plot, aes(x='x_pos', y='log2_mean_attn'))
            + scale_fill_identity()
        )
        if show_cds:
            p_heads += geom_rect(
                data=rect_per_layer_head,
                mapping=aes(
                    xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax',
                    fill='fill',
                ),
                alpha=0.3,
                inherit_aes=False,
                show_legend=False,
            )
        p_heads += geom_line(size=0.35, alpha=0.4, color='#333333')
        if color_by_frame:
            p_heads += geom_point(
                aes(color='Frame'), size=1.2, alpha=1, stroke=0
            )
            p_heads += scale_color_manual(values=frame_palette)
        p_heads += facet_grid('Layer ~ Head', scales='free_y')
        p_heads += labs(x=x_label_str, y='log2(Mean attention + 1)')
        p_heads += theme_classic()
        p_heads += theme(
            axis_text_x=x_axis_text,
            axis_ticks_major_x=x_axis_ticks,
            axis_title_x=x_axis_title,
            strip_background=element_blank(),
            strip_text=element_text(size=9),
            figure_size=(
                max(weight, 2.2 * len(head_labels) + 2),
                max(height, 1.8 * n_layers + 2),
            ),
        )
        head_path = f"{base_out}.per_layer_head.pdf"
        p_heads.save(head_path)
        print(f"Layer-by-head attention profile saved to {head_path}")
        output_paths.append(head_path)
    elif not has_head_profiles:
        print(
            "Head-specific attention was not plotted because the input table "
            "has no 'head' column. Re-run "
            "extract_attention_positional_importance with the updated code."
        )

    return output_paths


def _prepare_attention_heatmap_matrix(
        attn_df,
        row_columns,
        up_len=300,
        down_len=300,
        xlim=None,
        position_bin_size=10,
        normalization='row_zscore'):
    """Aggregate an attention table into a row-by-position heatmap matrix."""
    if position_bin_size < 1:
        raise ValueError("position_bin_size must be a positive integer.")
    if xlim is not None:
        if len(xlim) != 2 or xlim[0] > xlim[1]:
            raise ValueError("xlim must contain two ordered coordinates.")
        lower_bound, upper_bound = xlim
    else:
        lower_bound = -up_len
        upper_bound = FIXED_CDS_LEN + down_len - 1

    required_columns = {'x_pos', 'mean_attn', *row_columns}
    missing_columns = required_columns.difference(attn_df.columns)
    if missing_columns:
        raise ValueError(
            f"Attention table is missing columns: {sorted(missing_columns)}"
        )
    working_df = attn_df[
        attn_df['x_pos'].between(lower_bound, upper_bound)
    ].copy()
    if working_df.empty:
        raise ValueError("No attention positions remain within the plot range.")

    working_df['position_bin'] = (
        np.floor(working_df['x_pos'] / position_bin_size).astype(int)
        * position_bin_size
    )
    grouped_df = working_df.groupby(
        [*row_columns, 'position_bin'], as_index=False, observed=True
    )['mean_attn'].mean()
    raw_matrix = grouped_df.pivot_table(
        index=row_columns,
        columns='position_bin',
        values='mean_attn',
        aggfunc='mean',
    ).sort_index(axis=1)
    number_of_bins_before = raw_matrix.shape[1]
    raw_matrix = raw_matrix.dropna(axis=1, how='any')
    number_of_bins_dropped = number_of_bins_before - raw_matrix.shape[1]
    if number_of_bins_dropped:
        print(
            f"[Attention heatmap] Dropped {number_of_bins_dropped} of "
            f"{number_of_bins_before} position bins because at least one "
            "row lacked a measurement."
        )
    if raw_matrix.empty:
        raise ValueError(
            "No position bins have complete attention measurements across "
            "all requested rows. Increase n_samples or position_bin_size."
        )

    valid_normalizations = {'none', 'row_fraction', 'row_zscore', 'row_minmax'}
    normalization = str(normalization).lower()
    if normalization not in valid_normalizations:
        raise ValueError(
            f"normalization must be one of {sorted(valid_normalizations)}."
        )
    if normalization == 'none':
        display_matrix = raw_matrix.copy()
    elif normalization == 'row_fraction':
        row_sums = raw_matrix.sum(axis=1).replace(0, np.nan)
        display_matrix = raw_matrix.div(row_sums, axis=0).fillna(0.0)
    elif normalization == 'row_minmax':
        row_min = raw_matrix.min(axis=1)
        row_range = (raw_matrix.max(axis=1) - row_min).replace(0, np.nan)
        display_matrix = raw_matrix.sub(row_min, axis=0).div(
            row_range, axis=0
        ).fillna(0.0)
    else:
        row_mean = raw_matrix.mean(axis=1)
        row_std = raw_matrix.std(axis=1, ddof=0).replace(0, np.nan)
        display_matrix = raw_matrix.sub(row_mean, axis=0).div(
            row_std, axis=0
        ).fillna(0.0)

    return raw_matrix, display_matrix


def _infer_attention_focus(raw_matrix, enrichment_threshold=1.15):
    """Classify each attention row by its dominant transcript region."""
    if enrichment_threshold <= 1:
        raise ValueError("enrichment_threshold must be greater than 1.")
    positions = np.asarray(raw_matrix.columns, dtype=float)
    region_masks = {
        "5' UTR": positions < 0,
        'CDS': (positions >= 0) & (positions < FIXED_CDS_LEN),
        "3' UTR": positions >= FIXED_CDS_LEN,
    }
    available_regions = [
        region for region, mask in region_masks.items() if mask.any()
    ]
    focus_labels = {}
    for row_label, row_values in raw_matrix.iterrows():
        region_means = {
            region: float(row_values.iloc[np.flatnonzero(region_masks[region])].mean())
            for region in available_regions
        }
        if len(region_means) < 2 or not any(
                np.isfinite(list(region_means.values()))):
            focus_labels[row_label] = 'Full length'
            continue
        best_region = max(region_means, key=region_means.get)
        other_values = [
            value for region, value in region_means.items()
            if region != best_region
        ]
        reference = float(np.mean(other_values))
        enrichment = (
            (region_means[best_region] + 1e-12) / (reference + 1e-12)
        )
        focus_labels[row_label] = (
            best_region if enrichment >= enrichment_threshold
            else 'Full length'
        )
    return pd.Series(focus_labels, name='Attention focus')


def plot_attention_profile_heatmap(
        attn_df,
        out_path="attention_profile_heatmap.pdf",
        up_len=300,
        down_len=300,
        xlim=None,
        position_bin_size=10,
        normalization='row_zscore',
        cluster_layers=True,
        cluster_heads=True,
        cluster_metric='correlation',
        cluster_method='average',
        head_mode='head',
        enrichment_threshold=1.15,
        cmap=None,
        vmin=None,
        vmax=None,
        layer_width=12,
        layer_height=6,
        head_width=12,
        head_height=7,
        font_size=8,
        show_region_colors=True,
        show_focus_colors=True):
    """Plot clustered layer and head attention-position heatmaps.

    Rows are clustered while metagene columns retain their biological 5'-to-3'
    order. ``head_mode='head'`` averages the same head index across layers;
    ``head_mode='layer_head'`` treats every layer-head pair independently.
    """
    import seaborn as sns
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    if 'head' not in attn_df.columns:
        raise ValueError(
            "The attention table has no 'head' column. Re-run "
            "extract_attention_positional_importance with the updated code."
        )
    head_mode = str(head_mode).lower()
    normalization = str(normalization).lower()
    cluster_metric = str(cluster_metric).lower()
    cluster_method = str(cluster_method).lower()
    if head_mode not in {'head', 'layer_head'}:
        raise ValueError("head_mode must be 'head' or 'layer_head'.")
    for dimension_name, dimension_value in {
            'layer_width': layer_width,
            'layer_height': layer_height,
            'head_width': head_width,
            'head_height': head_height,
    }.items():
        if dimension_value <= 0:
            raise ValueError(f"{dimension_name} must be positive.")

    layer_raw, layer_matrix = _prepare_attention_heatmap_matrix(
        attn_df=attn_df,
        row_columns=['layer'],
        up_len=up_len,
        down_len=down_len,
        xlim=xlim,
        position_bin_size=position_bin_size,
        normalization=normalization,
    )
    head_row_columns = ['head'] if head_mode == 'head' else ['layer', 'head']
    head_raw, head_matrix = _prepare_attention_heatmap_matrix(
        attn_df=attn_df,
        row_columns=head_row_columns,
        up_len=up_len,
        down_len=down_len,
        xlim=xlim,
        position_bin_size=position_bin_size,
        normalization=normalization,
    )

    layer_raw.index = [f'L{int(value)}' for value in layer_raw.index]
    layer_matrix.index = layer_raw.index
    if head_mode == 'head':
        head_raw.index = [f'H{int(value)}' for value in head_raw.index]
    else:
        head_raw.index = [
            f'L{int(layer)}-H{int(head)}'
            for layer, head in head_raw.index
        ]
    head_matrix.index = head_raw.index

    focus_palette = {
        "5' UTR": '#E69F00',
        'CDS': '#D55E00',
        "3' UTR": '#7B61A8',
        'Full length': '#8A8A8A',
    }
    region_palette = {
        "5' UTR": '#E69F00',
        'CDS': '#56B4E9',
        "3' UTR": '#7B61A8',
    }

    def build_linkage(matrix, enabled):
        if not enabled or len(matrix) < 2:
            return None
        if cluster_method == 'ward' and cluster_metric != 'euclidean':
            raise ValueError(
                "Ward linkage requires cluster_metric='euclidean'."
            )
        distances = pdist(matrix.to_numpy(dtype=float), metric=cluster_metric)
        distances = np.nan_to_num(
            distances, nan=1.0, posinf=1.0, neginf=0.0
        )
        if not np.any(distances > 0):
            return None
        return linkage(
            distances,
            method=cluster_method,
            optimal_ordering=True,
        )

    def draw_heatmap(
            raw_matrix,
            display_matrix,
            cluster_rows,
            width,
            height,
            title,
            output_path):
        focus = _infer_attention_focus(
            raw_matrix,
            enrichment_threshold=enrichment_threshold,
        )
        row_colors = (
            focus.map(focus_palette) if show_focus_colors else None
        )
        position_values = np.asarray(display_matrix.columns, dtype=float)
        position_regions = pd.Series(
            np.select(
                [
                    position_values < 0,
                    (position_values >= 0)
                    & (position_values < FIXED_CDS_LEN),
                    position_values >= FIXED_CDS_LEN,
                ],
                ["5' UTR", 'CDS', "3' UTR"],
                default='CDS',
            ),
            index=display_matrix.columns,
            name='Transcript region',
        )
        column_colors = (
            position_regions.map(region_palette)
            if show_region_colors else None
        )
        row_linkage = build_linkage(display_matrix, cluster_rows)
        use_row_clustering = row_linkage is not None
        selected_cmap = cmap or (
            'vlag' if normalization == 'row_zscore' else 'mako'
        )
        center = 0 if normalization == 'row_zscore' else None
        grid = sns.clustermap(
            display_matrix,
            row_cluster=use_row_clustering,
            row_linkage=row_linkage,
            col_cluster=False,
            row_colors=row_colors,
            col_colors=column_colors,
            cmap=selected_cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            xticklabels=False,
            yticklabels=True,
            linewidths=0,
            figsize=(width, height),
            dendrogram_ratio=(0.14, 0.05),
            colors_ratio=(0.025, 0.025),
            cbar_pos=(0.02, 0.80, 0.02, 0.15),
            cbar_kws={'label': normalization.replace('_', ' ').title()},
        )
        grid.ax_heatmap.set_title(title, fontsize=font_size + 2, pad=10)
        grid.ax_heatmap.set_xlabel('Metagene position', fontsize=font_size)
        grid.ax_heatmap.set_ylabel('')
        grid.ax_heatmap.tick_params(axis='y', labelsize=font_size)

        number_of_ticks = min(9, len(display_matrix.columns))
        tick_indices = np.linspace(
            0, len(display_matrix.columns) - 1, number_of_ticks
        ).astype(int)
        grid.ax_heatmap.set_xticks(tick_indices + 0.5)
        grid.ax_heatmap.set_xticklabels(
            [
                str(int(display_matrix.columns[index]))
                for index in tick_indices
            ],
            rotation=0,
            fontsize=font_size,
        )

        legend_handles = []
        if show_focus_colors:
            legend_handles.extend([
                Patch(color=color, label=label)
                for label, color in focus_palette.items()
            ])
        if show_region_colors:
            legend_handles.extend([
                Patch(
                    facecolor=color,
                    edgecolor='none',
                    label=f'Position: {label}',
                )
                for label, color in region_palette.items()
            ])
        if legend_handles:
            grid.ax_heatmap.legend(
                handles=legend_handles,
                title='Annotations',
                frameon=False,
                fontsize=font_size,
                title_fontsize=font_size,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0,
            )
        grid.fig.savefig(output_path, bbox_inches='tight')
        row_order = (
            grid.dendrogram_row.reordered_ind
            if use_row_clustering else list(range(len(display_matrix)))
        )
        ordered_labels = display_matrix.index[row_order].tolist()
        plt.close(grid.fig)
        return ordered_labels, focus

    base_out = os.path.splitext(os.fspath(out_path))[0]
    layer_path = f"{base_out}.layers.pdf"
    head_path = f"{base_out}.heads.pdf"
    layer_order, layer_focus = draw_heatmap(
        raw_matrix=layer_raw,
        display_matrix=layer_matrix,
        cluster_rows=cluster_layers,
        width=layer_width,
        height=layer_height,
        title='Layer attention profiles',
        output_path=layer_path,
    )
    head_title = (
        'Head attention profiles (averaged across layers)'
        if head_mode == 'head'
        else 'Layer-head attention profiles'
    )
    head_order, head_focus = draw_heatmap(
        raw_matrix=head_raw,
        display_matrix=head_matrix,
        cluster_rows=cluster_heads,
        width=head_width,
        height=head_height,
        title=head_title,
        output_path=head_path,
    )
    print(f"Layer attention heatmap saved to {layer_path}")
    print(f"Head attention heatmap saved to {head_path}")
    return {
        'paths': [layer_path, head_path],
        'layer_matrix': layer_matrix,
        'head_matrix': head_matrix,
        'layer_order': layer_order,
        'head_order': head_order,
        'layer_focus': layer_focus,
        'head_focus': head_focus,
    }


def plot_regional_attention_dynamics(attn_df, out_path="regional_attention_dynamics.pdf", up_len=300, down_len=300):
    """
    Plots the layer-by-layer dynamic shifts in attention across 5 specific regions:
    5' UTR, CDS (Frame 0), CDS (Frame 1), CDS (Frame 2), and 3' UTR.
    Produces both an absolute mean attention line plot and a 100% relative proportion bar chart.
    """
    from plotnine import (ggplot, aes, geom_line, geom_point, geom_col, position_stack,
                          labs, theme_classic, scale_color_manual, scale_fill_manual, 
                          scale_x_continuous, theme, element_text)
    import pandas as pd
    import numpy as np

    # 1. Filter sequences based on the upstream/downstream boundaries
    df = attn_df[(attn_df['x_pos'] >= -up_len) & (attn_df['x_pos'] <= FIXED_CDS_LEN + down_len - 1)].copy()

    # 2. Annotate the 5 regions
    conditions = [
        df['x_pos'] < 0,
        (df['x_pos'] >= 0) & (df['x_pos'] < FIXED_CDS_LEN) & (df['x_pos'] % 3 == 0),
        (df['x_pos'] >= 0) & (df['x_pos'] < FIXED_CDS_LEN) & (df['x_pos'] % 3 == 1),
        (df['x_pos'] >= 0) & (df['x_pos'] < FIXED_CDS_LEN) & (df['x_pos'] % 3 == 2),
        df['x_pos'] >= FIXED_CDS_LEN
    ]
    choices = ["5' UTR", "CDS (Frame 0)", "CDS (Frame 1)", "CDS (Frame 2)", "3' UTR"]
    
    # [Fix]: Added explicit string default to satisfy Numpy's strict type promotion
    df['Region'] = np.select(conditions, choices, default="Unknown")

    # Convert to Categorical to maintain a strict legend order
    region_order = ["5' UTR", "CDS (Frame 0)", "CDS (Frame 1)", "CDS (Frame 2)", "3' UTR"]
    df['Region'] = pd.Categorical(df['Region'], categories=region_order)

    # 3. Aggregate Mean Attention per Region per Layer
    # Using 'mean' perfectly balances the varying lengths of UTRs and CDS subsets
    agg_df = df.groupby(['layer', 'Region'], as_index=False, observed=True)[['mean_attn']].mean()
    
    # Scale up by 1000 for cleaner Y-axis numbers in the absolute plot
    agg_df['mean_attn_scaled'] = agg_df['mean_attn']

    # Define color map to perfectly match your previous plots
    color_map = {
        "5' UTR": "#FF7F00",       # Orange for 5' UTR
        "CDS (Frame 0)": "#E41A1C", # Red
        "CDS (Frame 1)": "#377EB8", # Blue
        "CDS (Frame 2)": "gray",    # Gray
        "3' UTR": "#984EA3"         # Purple for 3' UTR
    }

    base_out = out_path.replace('.pdf', '')
    max_layer = int(agg_df['layer'].max())

    # ============================================================
    # Plot 1: Absolute Mean Attention per Nucleotide (Line Plot)
    # ============================================================
    p_line = (
        ggplot(agg_df, aes(x='layer', y='mean_attn_scaled', color='Region', group='Region'))
        + geom_line(size=1.2, alpha=0.9)
        + geom_point(size=3)
        + scale_color_manual(values=color_map)
        + scale_x_continuous(breaks=range(0, max_layer + 1))
        + labs(x='Transformer Layer', 
               y='Mean Attention per nt', 
               title='Layer-wise Absolute Attention Dynamics')
        + theme_classic()
        + theme(figure_size=(7, 5),
                axis_text=element_text(size=10),
                title=element_text(size=12, face="bold"))
    )
    p_line.save(f"{base_out}.line.pdf")
    print(f"Regional dynamics (Line) saved to {base_out}.line.pdf")

    # ============================================================
    # Plot 2: Relative Contribution Proportion (100% Stacked Bar)
    # ============================================================
    # Calculate relative proportion for each layer
    layer_sums = agg_df.groupby('layer')['mean_attn'].transform('sum')
    agg_df['relative_prop'] = agg_df['mean_attn'] / layer_sums

    p_bar = (
        ggplot(agg_df, aes(x='layer', y='relative_prop', fill='Region'))
        # Reverse the stack so 5'UTR is at the bottom, matching 5'->3' direction intuitively
        + geom_col(position=position_stack(reverse=True), color='white', size=0.2)
        + scale_fill_manual(values=color_map)
        + scale_x_continuous(breaks=range(0, max_layer + 1))
        + labs(x='Transformer Layer', 
               y='Relative Regional Contribution (100%)', 
               title='Layer-wise Relative Attention Shift')
        + theme_classic()
        + theme(figure_size=(7, 5),
                axis_text=element_text(size=10),
                title=element_text(size=12, face="bold"))
    )
    p_bar.save(f"{base_out}.proportion.pdf")
    print(f"Regional dynamics (Proportion Bar) saved to {base_out}.proportion.pdf")

def plot_saliency_profile(
        sal_df,
        out_path="saliency_profile.pdf",
        up_len=300,
        down_len=300,
        color_by_frame=True,
        xlim=None,
        show_xaxis=False,
        show_cds=True,
        weight=6,
        height=5):
    """Plot a saliency profile with attention-profile-compatible controls."""
    from plotnine import (
        ggplot, aes, geom_point, geom_line, geom_rect, labs, theme_classic,
        theme, scale_color_manual, scale_fill_identity, element_blank,
        element_line, element_text,
    )

    required_columns = {'x_pos', 'mean_saliency'}
    missing_columns = required_columns.difference(sal_df.columns)
    if missing_columns:
        raise ValueError(
            f"Saliency table is missing columns: {sorted(missing_columns)}"
        )

    if xlim is not None:
        if len(xlim) != 2 or xlim[0] > xlim[1]:
            raise ValueError("xlim must contain two ordered coordinates.")
        lower_bound, upper_bound = xlim
    else:
        lower_bound = -up_len
        upper_bound = FIXED_CDS_LEN + down_len - 1
    df_plot = sal_df[
        sal_df['x_pos'].between(lower_bound, upper_bound)
    ].copy()
    if df_plot.empty:
        raise ValueError("No saliency positions remain within the plot range.")

    group_columns = ['x_pos']
    if color_by_frame:
        df_plot = _assign_frame_colors(df_plot)
        group_columns.append('Frame')
    df_plot = df_plot.groupby(
        group_columns, as_index=False, observed=True
    )[['mean_saliency']].mean().dropna(subset=['mean_saliency'])
    df_plot['log2_saliency'] = np.log2(df_plot['mean_saliency'] + 1)

    frame_palette = {
        'Frame 0': '#E41A1C',
        'Frame 1': '#377EB8',
        'Frame 2': 'gray',
    }
    rect_cds = _cds_rect_data()
    x_axis_text = element_text() if show_xaxis else element_blank()
    x_axis_ticks = element_line() if show_xaxis else element_blank()
    x_axis_title = element_text() if show_xaxis else element_blank()
    x_label = 'Metagene Position (x_pos)' if show_xaxis else ''

    plot = (
        ggplot(df_plot, aes(x='x_pos', y='log2_saliency'))
        + scale_fill_identity()
    )
    if show_cds:
        plot += geom_rect(
            data=rect_cds,
            mapping=aes(
                xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax',
                fill='fill',
            ),
            alpha=0.3,
            inherit_aes=False,
            show_legend=False,
        )
    plot += geom_line(size=0.6, alpha=0.4, color='#333333')
    if color_by_frame:
        plot += geom_point(
            aes(color='Frame'), size=2, alpha=1, stroke=0
        )
        plot += scale_color_manual(values=frame_palette)
    plot += labs(
        x=x_label,
        y='log2(Mean |d(profile)/d(base)| + 1)',
    )
    plot += theme_classic()
    plot += theme(
        axis_text_x=x_axis_text,
        axis_ticks_major_x=x_axis_ticks,
        axis_title_x=x_axis_title,
        figure_size=(weight, height),
    )

    requested_path = os.fspath(out_path)
    base_path, extension = os.path.splitext(requested_path)
    pdf_path = (
        requested_path if extension.lower() == '.pdf'
        else f"{base_path or requested_path}.pdf"
    )
    plot.save(pdf_path)
    print(f"Saliency profile saved to {pdf_path}")
    return pdf_path


def plot_mutagenesis_profile(pos_agg, out_path="mutagenesis_profile.pdf", up_len=300, down_len=300):
    from plotnine import (ggplot, aes, geom_point, geom_line, geom_rect,
                          labs, theme_classic, theme, scale_color_manual, scale_fill_identity, element_blank)

    df_plot = pos_agg[(pos_agg['x_pos'] >= -up_len) & (pos_agg['x_pos'] <= FIXED_CDS_LEN + down_len - 1)].copy()
    df_plot = _assign_frame_colors(df_plot)
    
    df_plot = df_plot.groupby(['x_pos', 'Frame'], as_index=False, observed=True)[['mean_abs_delta']].mean().dropna(subset=['mean_abs_delta'])
    df_plot['log2_abs_delta'] = np.log2(df_plot['mean_abs_delta'] + 1)
    
    frame_palette = {'Frame 0': '#E41A1C', 'Frame 1': '#377EB8', 'Frame 2': 'gray'}
    rect_cds = _cds_rect_data()

    p = (
        ggplot(df_plot, aes(x='x_pos', y='log2_abs_delta'))
        + geom_rect(data=rect_cds, mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'), alpha=0.3, inherit_aes=False, show_legend=False)
        + scale_fill_identity()
        + geom_line(aes(color='Frame', group='Frame'), size=0.8, alpha=0.9)
        + geom_point(aes(color='Frame', group='Frame'), size=0.3, alpha=0.3)
        + scale_color_manual(values=frame_palette)
        + labs(x='', y='log2(Mean |Delta profile| + 1)')
        + theme_classic()
        + theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), axis_title_x=element_blank(), figure_size=(6, 4))
    )
    p.save(out_path)
    print(f"Mutagenesis profile saved to {out_path}")


def plot_gene_attribution(attr_df, out_path="gene_attribution.pdf", top_n=30):
    """
    Plots a heatmap of the top contributing environmental genes across different model layers.
    Args:
        attr_df: DataFrame generated by compute_adaLN_gene_attribution.
        top_n: Number of globally top contributing genes to display.
    """
    from plotnine import (ggplot, aes, geom_tile, scale_fill_cmap, labs, 
                          theme_classic, theme, element_text, element_blank)
    import pandas as pd

    # 1. 寻找全局贡献最大的 Top N 基因
    gene_totals = attr_df.groupby('gene')['score'].sum().reset_index()
    top_genes = gene_totals.nlargest(top_n, 'score')['gene'].tolist()

    # 2. 筛选数据
    df_plot = attr_df[attr_df['gene'].isin(top_genes)].copy()

    # 3. 设置分类排序逻辑以保证画图顺序美观
    # Y轴：总分越高的基因排在越上面
    df_plot['gene'] = pd.Categorical(df_plot['gene'], categories=top_genes[::-1])
    
    # X轴：按照网络深度的顺序 (L0-attn, L0-ffn, L1-attn, ...)
    layer_order = []
    for l in sorted(attr_df['layer'].unique()):
        layer_order.extend([f"L{l}-attn", f"L{l}-ffn"])
    # 只保留实际存在的列
    layer_order = [l for l in layer_order if l in attr_df['layer_module'].unique()]
    df_plot['layer_module'] = pd.Categorical(df_plot['layer_module'], categories=layer_order)

    p = (
        ggplot(df_plot, aes(x='layer_module', y='gene', fill='score_norm'))
        + geom_tile(color='white', size=0.2)
        # 使用炽热色系 (YlOrRd) 呈现基因调控的重要程度
        + scale_fill_cmap(cmap_name='YlOrRd') 
        + labs(
            x='Model Layer & Module (AdaLN)', 
            y='Environmental Gene (Cell Context)', 
            fill='Normalized\nAttribution'
        )
        + theme_classic()
        + theme(
            axis_text_x=element_text(rotation=45, hjust=1),
            axis_text_y=element_text(size=10),
            # 动态调整图片高度以适配基因数量
            figure_size=(max(8, len(layer_order)*0.4), max(5, top_n*0.25))
        )
    )
    p.save(out_path)
    print(f"Gene attribution heatmap saved to {out_path}")


def compute_ovr_differential_saliency(model, dataset, target_cell_type, all_cell_types, 
                                      n_samples_per_group=300, max_len=2000, device=None):
    """
    计算特定细胞系 (Target) 相较于其他所有细胞系 (Rest) 的差分 Saliency。
    """
    from torch.utils.data import Subset
    
    # 1. 构建 Target 和 Rest 的子集
    idx_target = [i for i, d in enumerate(dataset) if d[2] == target_cell_type]
    idx_rest = [i for i, d in enumerate(dataset) if d[2] in all_cell_types and d[2] != target_cell_type]
    
    if len(idx_target) == 0 or len(idx_rest) == 0:
        print(f"Skipping {target_cell_type} due to insufficient samples.")
        return None, []
        
    subset_target = Subset(dataset, idx_target)
    subset_rest = Subset(dataset, idx_rest)
    
    print(f"\n--- OvR Analysis for {target_cell_type} vs Rest ---")
    
    # 2. 分别计算 Saliency
    sal_target = compute_saliency_profile(
        model, subset_target, 
        n_samples=min(n_samples_per_group, len(idx_target)), 
        max_len=max_len, device=device
    )
    
    sal_rest = compute_saliency_profile(
        model, subset_rest, 
        n_samples=min(n_samples_per_group, len(idx_rest)), 
        max_len=max_len, device=device
    )
    
    # 3. 合并计算差分
    merged = pd.merge(sal_target[['x_pos', 'mean_saliency']], 
                      sal_rest[['x_pos', 'mean_saliency']], 
                      on='x_pos', suffixes=('_Target', '_Rest'))
    
    merged['delta_saliency'] = merged['mean_saliency_Target'] - merged['mean_saliency_Rest']
    
    # 提取差异最大的前 30 个宏观热点
    top_diff_hotspots = merged.nlargest(30, 'delta_saliency')['x_pos'].tolist()
    
    return merged, top_diff_hotspots


def extract_context_with_saliency_filter(model, dataset, seq_dict, tx_cds, 
                                         target_cell_type, x_pos_hotspots, 
                                         context_radius=15, max_seqs=300, device=None):
    """
    微观切片：拿着宏观热点去真实转录本上切出短序列。
    加入 Saliency 过滤器：只保留那些在该物理位点上确实有显著响应的转录本片段。
    """
    raw = _unwrap(model)
    device = _model_device(raw, device)
        
    raw.eval()
    
    # 找到目标细胞系的所有样本
    target_samples = [(i, d) for i, d in enumerate(dataset) if d[2] == target_cell_type]
    selected_samples = np.random.choice(len(target_samples), min(max_seqs, len(target_samples)), replace=False)
    
    valid_contexts = []
    
    print(f"Extracting physical sequence contexts for {target_cell_type}...")
    for idx in tqdm(selected_samples, desc="Micro-slicing"):
        real_idx, d = target_samples[idx]
        s = _extract_sample(dataset, real_idx)
        tid = s['tid']
        
        if not s['valid'] or tid not in seq_dict:
            continue
            
        cds_start, cds_end = s['cds_start_0'], s['cds_end_0']
        seq = seq_dict[tid].upper()
        L = len(seq)
        
        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device).requires_grad_(True)
        ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device) if s['ev'] is not None else None
        
        with torch.enable_grad():
            out = raw.forward(
                seq_batch=se, 
                cell_type=s['ct'], 
                expr_vector=ev,
                species=s['species'], 
                src_mask=_sequence_mask(se),
                head_names=['count']
            )
            pred = _extract_count_profile(out)
            profile_loss = (pred[0, :, 0] ** 2).sum()
            
        profile_loss.backward()
        grad = se.grad[0].detach().cpu().numpy()
        sal_track = np.abs(grad).sum(axis=-1)  # 单条转录本的碱基级 Saliency 曲线
        
        # 寻找上下文
        for x_pos in x_pos_hotspots:
            # 反向映射找真实物理位置
            abs_pos = _inverse_metagene(x_pos, cds_start, cds_end, FIXED_CDS_LEN)
            
            if 0 <= abs_pos < L:
                # 关键过滤：只有当这个特定的物理位置在这个细胞系里确实“亮起”了，才切它！
                if sal_track[abs_pos] > np.percentile(sal_track, 80): 
                    ctx_s = max(0, abs_pos - context_radius)
                    ctx_e = min(L, abs_pos + context_radius + 1)
                    ctx_seq = seq[ctx_s:ctx_e]
                    
                    # 补齐边界序列长度，保证聚类对齐
                    if len(ctx_seq) == (context_radius * 2 + 1) and 'N' not in ctx_seq:
                        valid_contexts.append(ctx_seq)
                        
    print(f"Harvested {len(valid_contexts)} highly salient context sequences.")
    return valid_contexts


def plot_sequence_logo(
        sequences,
        title="Cell-Type Specific Motif",
        out_path="sequence_logo.pdf"):
    """Plot an information-content sequence logo and save it as PDF."""
    if not sequences:
        print(f"No sequences found for {title}.")
        return None
        
    # 计算每个位置的信息熵矩阵 (PWM)
    df = logomaker.alignment_to_matrix(sequences=sequences, to_type='information')
    
    fig, ax = plt.subplots(figsize=(8, 3))
    logo = logomaker.Logo(df, ax=ax, font_name='Arial Rounded MT Bold')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.set_ylabel('Information (bits)')
    ax.set_xlabel('Relative Position')
    plt.title(title)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Sequence logo saved to {out_path}")
    return out_path

# ============================================================
# Notebook Cell 1: Transcript-Level Physical Sequence Slicing
# ============================================================
import logging
# [Fix]: 静音 matplotlib 缺失字体的警告
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
    import os
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

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
    import os
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import numpy as np
    import logomaker
    import pandas as pd
    
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

    # Visual rendering of sequence logos
    for cluster_id in range(best_k):
        sub_seqs = df[df['Cluster_ID'] == cluster_id]['sequence'].tolist()
        if len(sub_seqs) < 5: 
            continue
            
        counts_df = logomaker.alignment_to_matrix(sequences=sub_seqs, to_type='information')
        
        fig, ax = plt.subplots(figsize=(7, 2.2))
        logo = logomaker.Logo(counts_df, ax=ax)
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        
        ax.set_ylabel('Information (bits)')
        ax.set_xlabel('Relative Offset from Peak')
        ax.set_title(f"{region_name} - Cluster {cluster_id} (Support: n={len(sub_seqs)})")
        
        pdf_filename = os.path.join(out_dir, f"motif_logo_{region_name}_cluster_{cluster_id}.pdf")
        plt.savefig(pdf_filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
    return df


# ============================================================
# Notebook Cell 3: Motif Spatial Probability Heatmap
# ============================================================
def plot_motif_metagene_heatmap(
        all_motifs_df, out_path="motif_metagene_heatmap.pdf", 
        bin_size=20, up_len=300, down_len=300, max_prob=None,
        weight=8, height=10
        ):
    """
    Plots a heatmap of motif spatial distribution along the metagene.
    Y-axis motifs are dynamically sorted from 5' to 3' based on their peak enrichment positions.
    """
    import pandas as pd
    import numpy as np
    from plotnine import (ggplot, aes, geom_tile, geom_vline, scale_fill_gradient, 
                          labs, theme_classic, theme, element_text, element_blank, element_line)
    
    if 'Motif_Name' not in all_motifs_df.columns or all_motifs_df.empty:
        print("No motif clustering data available to plot.")
        return

    # 1. Filter coordinate boundaries
    df_plot = all_motifs_df[(all_motifs_df['x_pos'] >= -up_len) & (all_motifs_df['x_pos'] <= FIXED_CDS_LEN + down_len)].copy()
    
    # 2. Perform positional binning
    df_plot['x_bin'] = (df_plot['x_pos'] // bin_size) * bin_size + (bin_size / 2)
    
    # 3. Calculate raw frequencies
    heatmap_data = df_plot.groupby(['Motif_Name', 'x_bin']).size().reset_index(name='count')
    
    # 4. Impute empty grid tiles to guarantee a continuous matrix background
    unique_motifs = heatmap_data['Motif_Name'].unique()
    min_bin = (-up_len // bin_size) * bin_size + (bin_size / 2)
    max_bin = ((FIXED_CDS_LEN + down_len) // bin_size) * bin_size + (bin_size / 2)
    all_bins = np.arange(min_bin, max_bin + bin_size, bin_size)
    
    full_index = pd.MultiIndex.from_product([unique_motifs, all_bins], names=['Motif_Name', 'x_bin'])
    full_df = pd.DataFrame(index=full_index).reset_index()
    
    heatmap_data = pd.merge(full_df, heatmap_data, on=['Motif_Name', 'x_bin'], how='left').fillna({'count': 0})
    
    # 5. Standardize via row-wise probability
    motif_totals = heatmap_data.groupby('Motif_Name')['count'].transform('sum')
    heatmap_data['Probability'] = heatmap_data['count'] / (motif_totals + 1e-9)

    # 6. [核心重构]: 按照 5' -> 3' 空间富集峰值 (Peak) 对 Motif 进行动态排序
    # 找出每个 Motif 概率最大的 x_bin
    peak_bins = heatmap_data.loc[heatmap_data.groupby('Motif_Name')['Probability'].idxmax()]
    
    # 按 x_bin 降序排列 (因为 plotnine 的 Y 轴是从下往上画的，降序能让 5' 端最小的值排在图表最上面)
    # 如果有多个 Motif 在同一个位置 Peak，使用 Motif_Name 作为次级排序条件保持稳定
    ordered_motifs = peak_bins.sort_values(
        ['x_bin', 'Motif_Name'], 
        ascending=[False, False]
    )['Motif_Name'].tolist()

    # 应用严格排序
    heatmap_data['Motif_Name'] = pd.Categorical(
        heatmap_data['Motif_Name'], 
        categories=ordered_motifs
    )
    
    # 7. Automatically upscale the color scale if the global probability is too low
    if max_prob is None:
        max_prob = heatmap_data['Probability'].max()
        if max_prob == 0:
            max_prob = 1.0
    
    # 8. Render plot configurations
    p = (
        ggplot(heatmap_data, aes(x='x_bin', y='Motif_Name', fill='Probability'))
        + geom_tile(color='white', size=0.1) 
        + scale_fill_gradient(low='#EFF3FF', high='#08306B', limits=(0, max_prob)) 
        + geom_vline(xintercept=[0, FIXED_CDS_LEN], linetype='dashed', color='red', size=0.6)
        + labs(
            x=f'Metagene Position (Bin Size = {bin_size} nt)', 
            y='Discovered Motifs (Sorted 5\' \u2192 3\')', 
            fill='Spatial\nProbability', 
            title='Motif Spatial Distribution along Metagene'
        )
        + theme_classic()
        + theme(
            figure_size=(weight, height),
            axis_text_y=element_text(size=10),
            axis_text_x=element_text(size=10),
            axis_ticks_major_x=element_line(color='#333333', size=0.5),
            axis_ticks_major_y=element_line(color='#333333', size=0.5),
            axis_line_x=element_blank(),
            axis_line_y=element_blank()
        )
    )

    p.save(out_path)
    print(f"Motif Metagene Heatmap saved to {out_path}")
